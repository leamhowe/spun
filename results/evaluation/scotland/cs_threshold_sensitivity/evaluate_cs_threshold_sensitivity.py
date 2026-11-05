"""
Sensitivity of evaluation results to the Cloud Score+ (CS+) clear-sky threshold,
and performance on clear vs partially affected pixels.

The reference label is read from the dataset zarrs on disk (--data_root), where cloud
(cs_cdf < 0.5) is stored as 200. The dataset zarrs and the prediction zarrs can be stored in
different orientations, so each tile's label and cs_cdf are first transformed into the
prediction frame, using the flip/rotation that makes the dataset label (cloud -> 0) match the
'label' array saved in the SPUN prediction zarr exactly.

Analysis A: CS+ threshold sweep
    Pixels with cs_cdf < t are excluded, for t in --thresholds.
    t = 0.5 is the reference mask, i.e. Tier 1 (10 m) and Tier 2b (20 m).

Analysis B: pixel categories at the reference threshold
    clear:    cs_cdf >= 0.5 and LIS clear      (= Tier 2a at 20 m)
    partial:  cs_cdf >= 0.5 but LIS (MAJA) cloud ("partially affected but potentially usable")
    LIS is filled with 0 under its cloud mask (as in Tier 2b), so its scores in 'partial'
    show what the operational product loses, not a retrieval.

Metrics match the offline comparison script (binary at >50 %, snow-only on the >=1 % union).
Statistics are accumulated per tile, so memory use is constant.
"""
import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

# --- PARAMETERS (as in the offline comparison script) ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255
NDSI_THRESH = 0.4
PIXEL_AREA  = {'10m': 100, '20m': 400}


def get_args():
    p = argparse.ArgumentParser(description="CS+ threshold sensitivity and clear vs partially affected evaluation.")
    p.add_argument('--data_root', required=True, help='Root of the dataset (chip zarrs with label and cs_cdf)')
    p.add_argument('--split_file', default='dataset_metadata.csv', help='Metadata CSV, relative to data_root')
    p.add_argument('--spun_dir', required=True)
    p.add_argument('--lis_dir', required=True)
    p.add_argument('--ndsi_dir', required=True)
    p.add_argument('--wang_dir', required=True)
    p.add_argument('--cs_var', default='cs_cdf', help='Name of the CS+ variable in the chip zarrs')
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 0.55, 0.6, 0.65],
                   help='CS+ thresholds; the lowest is the reference (Tier 1 / 2a / 2b)')
    p.add_argument('--output_prefix', default='cs_threshold_sensitivity',
                   help='Writes <prefix>.txt (report) and <prefix>.csv (all metrics)')
    return p.parse_args()


# --- HELPERS (as in the offline comparison script) ---
def downscale_label_masked(arr, factor=2, invalid_val=200):
    """Downscales 10m -> 20m, ignoring invalid values within each block."""
    arr_masked = np.where(arr == invalid_val, np.nan, arr)
    h, w = arr_masked.shape
    h_new, w_new = h // factor, w // factor
    blocks = arr_masked[:h_new * factor, :w_new * factor].reshape(h_new, factor, w_new, factor)
    sum_ds = np.nansum(blocks, axis=(1, 3))
    count_ds = np.count_nonzero(~np.isnan(blocks), axis=(1, 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ds = sum_ds / count_ds
    return np.where(count_ds == 0, invalid_val, mean_ds).astype(np.float32)


def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    return fname.split('_')[0], fname.split('_')[-1].split('.')[0]


class MetricAccumulator:
    """Running sums giving the same metrics as compute_metrics_package, without storing pixels."""

    def __init__(self):
        self.n = 0
        self.sad = self.ssd = self.sd = 0.0                     # global abs, squared, signed error sums
        self.n_s = 0
        self.sad_s = self.ssd_s = self.sd_s = 0.0               # snow-only (union >= 1 %)
        self.tp = self.fp = self.fn = self.tn = 0               # binary at > 50 %
        self.sum_true = self.sum_pred = 0.0                     # for snow area

    def update(self, y_true, y_pred):
        if y_true.size == 0:
            return
        y_true = y_true.astype(np.float64)
        y_pred = y_pred.astype(np.float64)
        d = y_pred - y_true
        self.n += d.size
        self.sad += np.abs(d).sum(); self.ssd += (d ** 2).sum(); self.sd += d.sum()

        snow = (y_true >= 1.0) | (y_pred >= 1.0)
        ds = d[snow]
        self.n_s += ds.size
        self.sad_s += np.abs(ds).sum(); self.ssd_s += (ds ** 2).sum(); self.sd_s += ds.sum()

        t, p = y_true > 50, y_pred > 50
        self.tp += int((t & p).sum()); self.fp += int((~t & p).sum())
        self.fn += int((t & ~p).sum()); self.tn += int((~t & ~p).sum())
        self.sum_true += y_true.sum(); self.sum_pred += y_pred.sum()

    def result(self, pixel_area):
        if self.n == 0:
            return None
        prec = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        rec = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        ns = max(self.n_s, 1)
        true_km2 = self.sum_true / 100.0 * pixel_area / 1e6
        pred_km2 = self.sum_pred / 100.0 * pixel_area / 1e6
        return {
            'n_pixels': self.n,
            'prec': prec, 'rec': rec, 'f1': f1,
            'acc': (self.tp + self.tn) / self.n,
            'mae': self.sad / self.n, 'rmse': np.sqrt(self.ssd / self.n), 'bias': self.sd / self.n,
            'mae_snow': self.sad_s / ns, 'rmse_snow': np.sqrt(self.ssd_s / ns), 'bias_snow': self.sd_s / ns,
            'ref_area_km2': true_km2, 'pred_area_km2': pred_km2,
            'area_dev_pct': (pred_km2 - true_km2) / true_km2 * 100.0 if true_km2 > 0 else np.nan,
        }


# Candidate orientations between the dataset zarrs and the prediction zarrs
DIHEDRAL = {
    'identity':      lambda a: a,
    'flipud':        np.flipud,
    'fliplr':        np.fliplr,
    'rot180':        lambda a: np.rot90(a, 2),
    'transpose':     lambda a: a.T,
    'rot90':         lambda a: np.rot90(a, 1),
    'rot270':        lambda a: np.rot90(a, 3),
    'antitranspose': lambda a: np.rot90(a, 2).T,
}


def determine_alignments(test_files, data_root, spun_dir):
    """
    For each tile, find the transform T such that T(dataset label) equals the 'label' stored
    in the SPUN prediction zarr (the prediction frame) everywhere outside cloud. Tiles where
    several transforms match (e.g. no snow) use the most common unambiguous transform.
    """
    info, votes = {}, defaultdict(int)
    for label_path in tqdm(test_files, desc='Aligning'):
        stem = '_'.join(extract_ids_from_path(label_path))
        chip_path = label_path if os.path.isabs(label_path) else os.path.join(data_root, label_path)
        spun_path = os.path.join(spun_dir, f"{stem}_SPUN.zarr")
        if not (os.path.exists(chip_path) and os.path.exists(spun_path)):
            continue
        disk = np.squeeze(zarr.open(chip_path, mode='r')['label'][:])
        ev = np.squeeze(zarr.open(spun_path, mode='r')['label'][:])
        scores = {}
        for n, f in DIHEDRAL.items():
            d = f(disk)
            if d.shape != ev.shape:
                scores[n] = -1.0
                continue
            clear = d != LABEL_CLOUD          # compare outside cloud only
            scores[n] = float(np.mean(d[clear] == ev[clear])) if clear.any() else 1.0
        perfect = [n for n, s in scores.items() if s == 1.0]
        best = max(scores, key=scores.get)
        info[stem] = {'best': best, 'score': scores[best], 'perfect': perfect}
        if len(perfect) == 1:
            votes[perfect[0]] += 1

    majority = max(votes, key=votes.get) if votes else 'identity'
    resolved, n_ambiguous, imperfect = {}, 0, []
    for stem, d in info.items():
        if len(d['perfect']) == 1:
            resolved[stem] = d['perfect'][0]
        elif len(d['perfect']) > 1:
            n_ambiguous += 1
            resolved[stem] = majority if majority in d['perfect'] else d['perfect'][0]
        else:
            imperfect.append((stem, d['best'], d['score']))
            resolved[stem] = d['best']
    summary = {'votes': dict(votes), 'majority': majority, 'n_ambiguous': n_ambiguous, 'imperfect': imperfect}
    return resolved, summary


def load_cs_cdf(z, var, path):
    if var not in z:
        raise KeyError(f"'{var}' not found in {path}")
    cs = np.squeeze(z[var][:]).astype(np.float32)
    if np.nanmax(cs) > 1.5:          # stored as 0-100
        cs = cs / 100.0
    return np.where(np.isnan(cs), 1.0, cs)   # NaN -> not cloud


# --- MAIN ---
def main():
    args = get_args()
    thresholds = sorted(args.thresholds)
    ref = thresholds[0]

    df = pd.read_csv(os.path.join(args.data_root, args.split_file))
    path_col = 'relative_path' if 'relative_path' in df.columns else 'filepath'
    split_col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df.loc[df[split_col] == 'Test', path_col].tolist()

    # acc[(analysis, resolution, subset)][model] -> MetricAccumulator
    acc = defaultdict(lambda: defaultdict(MetricAccumulator))
    skipped, errors = 0, []
    skip_reasons, skip_examples = defaultdict(int), {}
    diag = defaultdict(int)

    # --- Align dataset zarrs (label, cs_cdf) to the prediction frame ---
    alignment, align_summary = determine_alignments(test_files, args.data_root, args.spun_dir)
    print(f"\nAlignment of dataset zarrs to prediction zarrs: unambiguous matches {align_summary['votes']}, "
          f"ambiguous (no snow/cloud) {align_summary['n_ambiguous']} -> '{align_summary['majority']}', "
          f"no exact match {len(align_summary['imperfect'])}")
    for stem, best, score in align_summary['imperfect'][:5]:
        print(f"  no exact match: {stem} (best '{best}', {score:.2%} of pixels equal)")

    print(f"\nEvaluating {len(test_files)} test tiles at CS+ thresholds {thresholds} (reference {ref})...")
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            stem = f"{date_id}_{tile_id}"
            spun_path = os.path.join(args.spun_dir, f"{stem}_SPUN.zarr")
            lis_path  = os.path.join(args.lis_dir,  f"{stem}_LIS.zarr")
            ndsi_path = os.path.join(args.ndsi_dir, f"{stem}_NDSI.zarr")
            wang_path = next((p for p in (os.path.join(args.wang_dir, f"{stem}_Wang2022.zarr"),
                                          os.path.join(args.wang_dir, f"{stem}_SPUN.zarr"))
                              if os.path.exists(p)), None)
            chip_path = label_path if os.path.isabs(label_path) else os.path.join(args.data_root, label_path)
            paths = {'chip (label, cs_cdf)': chip_path, 'SPUN': spun_path, 'LIS': lis_path,
                     'NDSI': ndsi_path, 'Wang': wang_path}
            missing = [n for n, p in paths.items() if p is None or not os.path.exists(p)]
            if missing:
                skipped += 1
                for n in missing:
                    skip_reasons[n] += 1
                    skip_examples.setdefault(n, paths[n] or os.path.join(args.wang_dir, f"{stem}_Wang2022.zarr"))
                continue

            # --- Load 10 m data: reference label and cs_cdf from the dataset zarr ---
            z_chip = zarr.open(chip_path, mode='r')
            T = DIHEDRAL[alignment.get(stem, 'identity')]
            gt_10 = np.ascontiguousarray(T(np.squeeze(z_chip['label'][:]).astype(np.float32)))
            cs = np.ascontiguousarray(T(load_cs_cdf(z_chip, args.cs_var, chip_path)))

            z_spun = zarr.open(spun_path, mode='r')
            spun_10 = z_spun['spun_scf'][:]
            ndsi_10 = (zarr.open(ndsi_path, mode='r')['ndsi'][:] > NDSI_THRESH).astype(np.float32) * 100.0
            wang_10 = zarr.open(wang_path, mode='r')['wang_scf'][:]
            lis_20 = zarr.open(lis_path, mode='r')['lis_snow'][:]

            if cs.shape != gt_10.shape or spun_10.shape != gt_10.shape:
                raise ValueError(f"shape mismatch: label {gt_10.shape}, cs_cdf {cs.shape}, SPUN {spun_10.shape}")
            spun_20 = downscale_label_masked(spun_10)
            if spun_20.shape != lis_20.shape:
                skipped += 1
                skip_reasons['SPUN/LIS shape mismatch'] += 1
                skip_examples.setdefault('SPUN/LIS shape mismatch', f"{stem}: {spun_20.shape} vs {lis_20.shape}")
                continue
            ndsi_20 = downscale_label_masked(ndsi_10, invalid_val=-999)
            wang_20 = downscale_label_masked(wang_10, invalid_val=-999)
            lis_20_filled = np.nan_to_num(np.where(lis_20 >= 200, 0, lis_20), nan=0.0)

            # --- Diagnostics ---
            label_cloud = gt_10 == LABEL_CLOUD
            diag['pixels'] += gt_10.size
            diag['label_cloud'] += int(label_cloud.sum())
            diag['cs_cloud'] += int((cs < ref).sum())
            diag['cloud_disagree'] += int((label_cloud != (cs < ref)).sum())
            eval_label = z_spun['label'][:]
            diag['spun_label_differs_outside_cloud'] += int((~label_cloud & (eval_label != gt_10)).sum())

            preds_10 = {'spun': spun_10, 'ndsi': ndsi_10, 'wang': wang_10}
            preds_20 = {'spun': spun_20, 'lis': lis_20_filled, 'ndsi': ndsi_20, 'wang': wang_20}

            def mask_cs(t):
                """Reference label with cloud (200) wherever cs_cdf < t."""
                return np.where((cs < t) | label_cloud, LABEL_CLOUD, gt_10)

            # --- A. Threshold sweep (reference threshold = Tier 1 / Tier 2b) ---
            for t in thresholds:
                gt_t = mask_cs(t)
                m10 = gt_t != LABEL_CLOUD
                for k, v in preds_10.items():
                    acc[('sweep', '10m', t)][k].update(gt_t[m10], v[m10])
                gt20_t = downscale_label_masked(gt_t, invalid_val=LABEL_CLOUD)
                m20 = gt20_t != LABEL_CLOUD
                for k, v in preds_20.items():
                    acc[('sweep', '20m', t)][k].update(gt20_t[m20], v[m20])

            # --- B. Clear (Tier 2a) vs partially affected, at the reference threshold ---
            gt_ref = mask_cs(ref)
            valid10 = gt_ref != LABEL_CLOUD
            lis_10 = np.repeat(np.repeat(lis_20, 2, axis=0), 2, axis=1)
            lis_10_filled = np.nan_to_num(np.where(lis_10 >= 200, 0, lis_10), nan=0.0)
            lis10_cloud = lis_10 == LIS_CLOUD
            lis10_bad = lis10_cloud | (lis_10 == LIS_NODATA) | np.isnan(lis_10)
            for name, m in {'clear': valid10 & ~lis10_bad, 'partial': valid10 & lis10_cloud}.items():
                for k, v in {**preds_10, 'lis': lis_10_filled}.items():
                    acc[('category', '10m', name)][k].update(gt_ref[m], v[m])

            gt20_ref = downscale_label_masked(gt_ref, invalid_val=LABEL_CLOUD)
            valid20 = gt20_ref != LABEL_CLOUD
            lis20_cloud = lis_20 == LIS_CLOUD
            lis20_bad = lis20_cloud | (lis_20 == LIS_NODATA) | np.isnan(lis_20)
            for name, m in {'clear': valid20 & ~lis20_bad, 'partial': valid20 & lis20_cloud}.items():
                for k, v in preds_20.items():
                    acc[('category', '20m', name)][k].update(gt20_ref[m], v[m])

        except Exception as e:
            tile = os.path.basename(label_path).split('.')[0]
            errors.append((label_path, f"{type(e).__name__}: {str(e).replace(tile, '<tile>')}"))

    # --- DIAGNOSTICS ---
    n_ok = len(test_files) - skipped - len(errors)
    err_counts = defaultdict(int)
    for _, e in errors:
        err_counts[e] += 1
    print(f"\nTiles evaluated: {n_ok} | skipped: {skipped} | errors: {len(errors)}")
    for reason, n in skip_reasons.items():
        print(f"  skipped, {reason} not found/valid: {n} tiles, e.g. {skip_examples[reason]}")
    for e, n in sorted(err_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  error ({n} tiles): {e}")
    if n_ok == 0:
        raise SystemExit("\nNo tiles could be evaluated - see the reasons above.")

    # --- COLLECT RESULTS ---
    rows = []
    for (analysis, res, subset), models in acc.items():
        for model, a in models.items():
            r = a.result(PIXEL_AREA[res])
            if r is not None:
                rows.append({'analysis': analysis, 'resolution': res, 'subset': subset, 'model': model, **r})
    results = pd.DataFrame(rows)

    # Pixels as % of the reference evaluation (Tier 1 at 10 m, Tier 2b at 20 m)
    for res in ('10m', '20m'):
        base = results[(results.analysis == 'sweep') & (results.resolution == res) &
                       (results.subset == ref) & (results.model == 'spun')]['n_pixels'].iloc[0]
        sel = results.resolution == res
        results.loc[sel, 'pct_of_reference'] = results.loc[sel, 'n_pixels'] / base * 100.0
    results.to_csv(f"{args.output_prefix}.csv", index=False)

    # --- REPORT ---
    with open(f"{args.output_prefix}.txt", 'w') as f:
        def log(s=''):
            print(s)
            f.write(s + '\n')

        log(f"Tiles evaluated: {n_ok} | skipped: {skipped} | errors: {len(errors)}")
        log(f"Alignment to prediction frame: unambiguous {align_summary['votes']} | ambiguous "
            f"{align_summary['n_ambiguous']} (-> '{align_summary['majority']}') | no exact match "
            f"{len(align_summary['imperfect'])}")
        for reason, n in skip_reasons.items():
            log(f"  skipped, {reason} not found/valid: {n} tiles")
        for e, n in err_counts.items():
            log(f"  error ({n} tiles): {e}")
        n = diag['pixels']
        log(f"10 m pixels: label cloud {diag['label_cloud'] / n:.2%} | cs_cdf < {ref}: {diag['cs_cloud'] / n:.2%} | "
            f"disagree: {diag['cloud_disagree'] / n:.3%}")
        log(f"SPUN-zarr label differs from aligned dataset label outside cloud: "
            f"{diag['spun_label_differs_outside_cloud']:,} pixels (should be 0)")

        fmt = "{:<8} {:<6} {:>11} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7}"
        header = fmt.format('Subset', 'Model', 'N pixels', 'Rec', 'Prec', 'F1', 'MAE', 'RMSE',
                            'MAE(S)', 'RMSE(S)', 'Bias(S)', 'Area dev', '% ref')

        def table(title, analysis, res, subsets):
            log(f"\n{title}")
            log('=' * len(header))
            log(header)
            log('-' * len(header))
            for s in subsets:
                for model in ['spun', 'lis', 'ndsi', 'wang']:
                    r = results[(results.analysis == analysis) & (results.resolution == res) &
                                (results.subset == s) & (results.model == model)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    label = f"{s:.2f}" if isinstance(s, float) else s
                    log(fmt.format(label, model.upper(), f"{int(r.n_pixels):,}",
                                   *[f"{r[c]:.3f}" for c in ['rec', 'prec', 'f1']],
                                   *[f"{r[c]:.2f}" for c in ['mae', 'rmse', 'mae_snow', 'rmse_snow', 'bias_snow']],
                                   f"{r.area_dev_pct:+.1f}%", f"{r.pct_of_reference:.1f}"))
                log('-' * len(header))

        table(f"A. CS+ THRESHOLD SWEEP - 10 m ({ref:.2f} = Tier 1)", 'sweep', '10m', thresholds)
        table(f"A. CS+ THRESHOLD SWEEP - 20 m ({ref:.2f} = Tier 2b)", 'sweep', '20m', thresholds)
        table(f"B. CLEAR vs PARTIALLY AFFECTED - 10 m (LIS upsampled, 0 under its cloud)",
              'category', '10m', ['clear', 'partial'])
        table(f"B. CLEAR (= Tier 2a) vs PARTIALLY AFFECTED - 20 m", 'category', '20m', ['clear', 'partial'])

        log("\nNotes: binary metrics at >50 %; snow-only metrics on pixels where reference or prediction >= 1 %.")
        log("'% ref' = pixels as a percentage of the reference evaluation (Tier 1 at 10 m, Tier 2b at 20 m).")
        log(f"clear = cs_cdf >= {ref} and LIS clear; partial = cs_cdf >= {ref} but LIS cloud. "
            f"LIS is filled with 0 under its cloud mask.")

    print(f"\nSaved {args.output_prefix}.txt and {args.output_prefix}.csv")


if __name__ == '__main__':
    main()
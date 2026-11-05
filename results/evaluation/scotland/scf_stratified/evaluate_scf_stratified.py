"""
Performance stratified by reference SCF: is SPUN retrieving fractional cover, or mainly snow extent?

Evaluations (same files, masks and 20 m aggregation as the offline comparison script):
  - SPUN, 10 m, all pixels (Tier 1)
  - SPUN and LIS, 20 m, pixels cloud-free according to LIS (Tier 2a). LIS only provides
    fractional values where it sees the surface, so Tier 2a is the like-for-like comparison.
  - SPUN and LIS, 20 m, all pixels (Tier 2b), with LIS assigned 0 % under its cloud mask.
    Gives a much larger sample of mixed pixels for SPUN; for LIS it mainly reflects masking.

For each reference-SCF bin (0, 1-25, 26-50, 51-75, 76-99, 100 %) it reports the number of pixels,
mean reference and predicted SCF, MAE, bias, RMSE, and the share of predictions that are
<1 %, fractional (1-99 %) or >=99 %. For mixed reference pixels (1-99 %) it also reports the
Pearson correlation and OLS slope of predicted on reference SCF.

Note: in the all-sky evaluation, pixels flagged as cloud by CS+ have reference SCF 0 %, so they
fall in the 0 % bin. The mixed bins (1-99 %), which test fractional retrieval, are unaffected.

Outputs: <prefix>.csv, <prefix>.txt and <prefix>.png. Statistics are accumulated per tile.
"""
import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255

BIN_LABELS = ['0', '1-25', '26-50', '51-75', '76-99', '100']
EDGES_2D = np.linspace(0, 100, 51)          # 2 % bins for the density plots
EDGES_1D = np.linspace(0, 100, 51)


def get_args():
    p = argparse.ArgumentParser(description="SCF performance stratified by reference SCF.")
    p.add_argument('--data_root', required=True, help='Root directory of the dataset (metadata CSV)')
    p.add_argument('--split_file', default='dataset_metadata.csv')
    p.add_argument('--spun_dir', required=True)
    p.add_argument('--lis_dir', required=True)
    p.add_argument('--output_prefix', default='scf_stratified')
    return p.parse_args()


def downscale_label_masked(arr, factor=2, invalid_val=200):
    """Downscales 10m -> 20m, handling invalid values (as in the offline comparison script)."""
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


def ref_bin(ref):
    """0 -> '0', (0,25] -> '1-25', ..., (75,100) -> '76-99', 100 -> '100'."""
    b = np.digitize(ref, [0, 25, 50, 75], right=True)          # 0 for ref==0, 1..4 above
    b = np.where(ref >= 100, 5, b)
    return b


class StratAccumulator:
    """Per-bin running sums, mixed-pixel regression sums and histograms."""

    def __init__(self):
        k = len(BIN_LABELS)
        self.n = np.zeros(k); self.s_ref = np.zeros(k); self.s_pred = np.zeros(k)
        self.s_abs = np.zeros(k); self.s_sq = np.zeros(k)
        self.n_lo = np.zeros(k); self.n_frac = np.zeros(k); self.n_hi = np.zeros(k)
        self.mx = defaultdict(float)                         # mixed-pixel regression sums
        self.h2d = np.zeros((len(EDGES_2D) - 1, len(EDGES_2D) - 1))
        self.h_ref = np.zeros(len(EDGES_1D) - 1); self.h_pred = np.zeros(len(EDGES_1D) - 1)

    def update(self, ref, pred):
        if ref.size == 0:
            return
        ref = ref.astype(np.float64); pred = np.clip(pred.astype(np.float64), 0, 100)
        b = ref_bin(ref)
        d = pred - ref
        k = len(BIN_LABELS)
        self.n += np.bincount(b, minlength=k)
        self.s_ref += np.bincount(b, ref, minlength=k)
        self.s_pred += np.bincount(b, pred, minlength=k)
        self.s_abs += np.bincount(b, np.abs(d), minlength=k)
        self.s_sq += np.bincount(b, d ** 2, minlength=k)
        self.n_lo += np.bincount(b, pred < 1, minlength=k)
        self.n_frac += np.bincount(b, (pred >= 1) & (pred < 99), minlength=k)
        self.n_hi += np.bincount(b, pred >= 99, minlength=k)

        mixed = (ref > 0) & (ref < 100)
        x, y = ref[mixed], pred[mixed]
        for name, v in (('n', x.size), ('x', x.sum()), ('y', y.sum()), ('xx', (x * x).sum()),
                        ('yy', (y * y).sum()), ('xy', (x * y).sum())):
            self.mx[name] += v

        self.h2d += np.histogram2d(ref, pred, bins=[EDGES_2D, EDGES_2D])[0]
        snow = (ref >= 1) | (pred >= 1)
        self.h_ref += np.histogram(ref[snow], bins=EDGES_1D)[0]
        self.h_pred += np.histogram(pred[snow], bins=EDGES_1D)[0]

    def table(self):
        n = np.maximum(self.n, 1)
        return pd.DataFrame({
            'ref_bin': BIN_LABELS, 'n_pixels': self.n.astype(int),
            'mean_ref': self.s_ref / n, 'mean_pred': self.s_pred / n,
            'mae': self.s_abs / n, 'bias': (self.s_pred - self.s_ref) / n, 'rmse': np.sqrt(self.s_sq / n),
            'pct_pred_lt1': self.n_lo / n * 100, 'pct_pred_fractional': self.n_frac / n * 100,
            'pct_pred_ge99': self.n_hi / n * 100,
        })

    def mixed_stats(self):
        m = self.mx
        if m['n'] < 2:
            return {'n_mixed': int(m['n']), 'pearson_r': np.nan, 'slope': np.nan, 'intercept': np.nan}
        n = m['n']
        sxx = m['xx'] - m['x'] ** 2 / n
        syy = m['yy'] - m['y'] ** 2 / n
        sxy = m['xy'] - m['x'] * m['y'] / n
        slope = sxy / sxx if sxx > 0 else np.nan
        return {'n_mixed': int(n),
                'pearson_r': sxy / np.sqrt(sxx * syy) if sxx > 0 and syy > 0 else np.nan,
                'slope': slope, 'intercept': (m['y'] - slope * m['x']) / n}


def main():
    args = get_args()
    df = pd.read_csv(os.path.join(args.data_root, args.split_file))
    path_col = 'relative_path' if 'relative_path' in df.columns else 'filepath'
    split_col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df.loc[df[split_col] == 'Test', path_col].tolist()

    acc = {'SPUN 10 m (Tier 1, all-sky)': StratAccumulator(),
           'SPUN 20 m (Tier 2a, clear-sky)': StratAccumulator(),
           'LIS 20 m (Tier 2a, clear-sky)': StratAccumulator(),
           'SPUN 20 m (Tier 2b, all-sky)': StratAccumulator(),
           'LIS 20 m (Tier 2b, all-sky)': StratAccumulator()}
    skipped, errors = 0, defaultdict(int)

    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            spun_path = os.path.join(args.spun_dir, f"{date_id}_{tile_id}_SPUN.zarr")
            lis_path = os.path.join(args.lis_dir, f"{date_id}_{tile_id}_LIS.zarr")
            if not (os.path.exists(spun_path) and os.path.exists(lis_path)):
                skipped += 1
                continue

            z = zarr.open(spun_path, mode='r')
            spun_10 = z['spun_scf'][:]
            gt_10 = z['label'][:]
            lis = zarr.open(lis_path, mode='r')['lis_snow'][:].astype(np.float32)

            # Tier 1: 10 m, all pixels
            m10 = gt_10 != LABEL_CLOUD
            acc['SPUN 10 m (Tier 1, all-sky)'].update(gt_10[m10], spun_10[m10])

            # Tier 2a: 20 m, LIS cloud-free pixels
            spun_20 = downscale_label_masked(spun_10)
            gt_20 = downscale_label_masked(gt_10, invalid_val=LABEL_CLOUD)
            if spun_20.shape != lis.shape:
                skipped += 1
                continue
            clear = (gt_20 != LABEL_CLOUD) & (lis != LIS_CLOUD) & (lis != LIS_NODATA) & ~np.isnan(lis)
            acc['SPUN 20 m (Tier 2a, clear-sky)'].update(gt_20[clear], spun_20[clear])
            acc['LIS 20 m (Tier 2a, clear-sky)'].update(gt_20[clear], lis[clear])

            # Tier 2b: 20 m, all pixels; LIS assigned 0 % under its cloud mask / no data
            allsky = gt_20 != LABEL_CLOUD
            lis_filled = np.nan_to_num(np.where(lis >= 200, 0, lis), nan=0.0)
            acc['SPUN 20 m (Tier 2b, all-sky)'].update(gt_20[allsky], spun_20[allsky])
            acc['LIS 20 m (Tier 2b, all-sky)'].update(gt_20[allsky], lis_filled[allsky])
        except Exception as e:
            errors[f"{type(e).__name__}: {e}"] += 1

    n_ok = len(test_files) - skipped - sum(errors.values())
    print(f"\nTiles evaluated: {n_ok} | skipped: {skipped} | errors: {sum(errors.values())}")
    for e, n in list(errors.items())[:5]:
        print(f"  error ({n} tiles): {e}")
    if n_ok == 0:
        raise SystemExit("No tiles could be evaluated.")

    # --- Tables ---
    rows, mixed_rows = [], []
    for name, a in acc.items():
        t = a.table(); t.insert(0, 'evaluation', name); rows.append(t)
        mixed_rows.append({'evaluation': name, **a.mixed_stats()})
    table = pd.concat(rows, ignore_index=True)
    mixed = pd.DataFrame(mixed_rows)
    table.to_csv(f"{args.output_prefix}.csv", index=False)
    mixed.to_csv(f"{args.output_prefix}_mixed.csv", index=False)

    with open(f"{args.output_prefix}.txt", 'w') as f:
        def log(s=''):
            print(s); f.write(s + '\n')
        fmt = "{:<7} {:>10} {:>8} {:>8} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8}"
        hdr = fmt.format('Ref', 'N pixels', 'MeanRef', 'MeanPred', 'MAE', 'Bias', 'RMSE',
                         '%pred<1', '%frac', '%pred>=99')
        for name in acc:
            log(f"\n{name}\n" + '=' * len(hdr) + f"\n{hdr}\n" + '-' * len(hdr))
            for _, r in table[table.evaluation == name].iterrows():
                log(fmt.format(r.ref_bin, f"{r.n_pixels:,}", f"{r.mean_ref:.1f}", f"{r.mean_pred:.1f}",
                               f"{r.mae:.1f}", f"{r.bias:+.1f}", f"{r.rmse:.1f}", f"{r.pct_pred_lt1:.1f}",
                               f"{r.pct_pred_fractional:.1f}", f"{r.pct_pred_ge99:.1f}"))
            m = mixed[mixed.evaluation == name].iloc[0]
            log(f"Mixed reference pixels (1-99 %): n = {m.n_mixed:,} | Pearson r = {m.pearson_r:.3f} | "
                f"slope = {m.slope:.3f} | intercept = {m.intercept:.1f}")
        log("\n%frac = share of predictions that are fractional (1-99 %). A slope of 1 and r of 1 would mean "
            "fractional cover is tracked perfectly; a model doing only extent segmentation would show predictions "
            "concentrated at <1 % and >=99 % in the mixed bins.")

    # --- Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    axes = axes.ravel()
    for ax, (name, a) in zip(axes[:5], acc.items()):
        h = a.h2d.T                                      # rows = pred, cols = ref
        col_tot = h.sum(axis=0, keepdims=True)
        h_norm = np.divide(h, col_tot, out=np.zeros_like(h), where=col_tot > 0)   # P(pred | ref)
        im = ax.pcolormesh(EDGES_2D, EDGES_2D, np.ma.masked_equal(h_norm, 0),
                           norm=mcolors.LogNorm(vmin=1e-3, vmax=1), cmap='viridis', shading='flat')
        t = table[table.evaluation == name]
        centres = [0, 12.5, 37.5, 62.5, 87.5, 100]
        ax.plot([0, 100], [0, 100], 'k--', lw=1, label='1:1')
        ax.plot(centres, t.mean_pred, 'o-', color='#eb5252', lw=2, ms=6, label='Mean prediction per bin')
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect('equal')
        ax.set_xlabel('Reference SCF (%)'); ax.set_ylabel('Predicted SCF (%)')
        m = mixed[mixed.evaluation == name].iloc[0]
        ax.set_title(f"{name}\nmixed pixels: r = {m.pearson_r:.2f}, slope = {m.slope:.2f}", fontsize=11)
        ax.legend(loc='upper left', fontsize=9, frameon=False)
    fig.subplots_adjust(right=0.88, wspace=0.3, hspace=0.35)
    top = axes[2].get_position()
    cax = fig.add_axes([0.9, top.y0, 0.012, top.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Fraction of pixels in reference bin\n(column-normalised)')

    a = acc['SPUN 10 m (Tier 1, all-sky)']
    centres1d = (EDGES_1D[:-1] + EDGES_1D[1:]) / 2
    w = EDGES_1D[1] - EDGES_1D[0]
    axes[5].bar(centres1d, a.h_ref / a.h_ref.sum(), width=w, color='0.6', alpha=0.7, label='Reference')
    axes[5].step(centres1d, a.h_pred / a.h_pred.sum(), where='mid', color='#21918c', lw=2, label='SPUN')
    axes[5].set_yscale('log'); axes[5].set_xlim(0, 100)
    axes[5].set_xlabel('SCF (%)'); axes[5].set_ylabel('Fraction of snow pixels')
    axes[5].set_title('SCF distribution, 10 m\n(pixels with reference or prediction $\\geq$ 1 %)', fontsize=11)
    axes[5].legend(frameon=False)

    plt.savefig(f"{args.output_prefix}.png", dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved {args.output_prefix}.txt, .csv, _mixed.csv and .png")


if __name__ == '__main__':
    main()
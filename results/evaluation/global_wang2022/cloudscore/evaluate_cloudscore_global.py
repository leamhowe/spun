"""
CloudScore+ (CS+) performance on the Wang et al. (2022) global dataset, and propagation of
cloud-mask errors into the final SPUN snow product.

Inputs (same as the CloudScore+ sensitivity script):
  --gt_dir     label TIFFs (0 = no data, 1 = background, 2 = cloud, 3 = snow)
  --spun_dir   raw (unmasked) SPUN SCF predictions, pred_<name>.tif or <name>.tif (0-100)
  --cloud_dir  continuous CS+ cs_cdf files, cs_prob_<name>.tif (0-1, low = cloudy)

For each cs_cdf threshold, a pixel is masked as cloud if cs_cdf < threshold. Reported:
  A. CS+ as a cloud detector against the label cloud class (precision, recall, F1, IoU),
     plus the share of snow and of background that CS+ masks.
  B. Error propagation into the final snow product (SPUN SCF > 50 % and not masked):
     - snow false negatives: masked by CS+ (irrecoverable) vs retained but missed by SPUN
     - snow false positives: cloud retained by CS+ and called snow vs background called snow
Label no-data pixels are excluded. Results for the 6-tile test subset and all tiles.
"""
import os
import glob
import argparse
import numpy as np
import rasterio
from PIL import Image

SNOW_THRESHOLD = 50.0          # SPUN SCF (%) above which a pixel is snow
LABEL_NODATA, LABEL_BG, LABEL_CLOUD, LABEL_SNOW = 0, 1, 2, 3
REFERENCE_THRESHOLD = 0.5      # threshold used in the operational pipeline

SUBSET_IDS = [
    "20191123T111259_20191123T112151_T29RNQ_-8-3_31-22",
    "20200713T103031_20200713T103026_T33VWH_15-31_60-46",
    "20200804T223709_20200804T223712_T59GLM_169-47_-44-02",
    "20200805T001109_20200805T001647_T55GCN_145-04_-42-94",
    "20201126T041121_20201126T041842_T47SNT_100-09_33-93",
    "20210714T081609_20210714T083805_T34HCJ_19-16_-33-13",
]


def get_args():
    p = argparse.ArgumentParser(description="CloudScore+ performance and error propagation on the global dataset")
    p.add_argument('--gt_dir', required=True)
    p.add_argument('--spun_dir', required=True, help='Raw (unmasked) SPUN predictions')
    p.add_argument('--cloud_dir', required=True, help='CS+ cs_cdf files named cs_prob_<name>.tif')
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7])
    p.add_argument('--output', default='cloudscore_global_report.txt')
    return p.parse_args()


def scores(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else np.nan
    iou = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    return prec, rec, f1, iou


def load_tiles(spun_files, args):
    """Load (label, SCF, cs_cdf) for each tile, cropped to a common shape as in the sensitivity script."""
    tiles, skipped, n200 = [], [], 0
    for sp_path in spun_files:
        base = os.path.basename(sp_path).replace('pred_', '')
        gt_path = os.path.join(args.gt_dir, base)
        cs_path = os.path.join(args.cloud_dir, f"cs_prob_{base}")
        if not os.path.exists(gt_path) or not os.path.exists(cs_path):
            skipped.append(f"{base}: missing {'label' if not os.path.exists(gt_path) else 'CS+ file'}")
            continue
        with rasterio.open(sp_path) as src:
            scf = src.read(1).astype(np.float32)
        with rasterio.open(cs_path) as src:
            cs = src.read(1).astype(np.float32)
        gt = np.array(Image.open(gt_path).convert('L'))
        shapes = {scf.shape, cs.shape, gt.shape}
        if len(shapes) > 1:
            skipped.append(f"{base}: shapes differ {sorted(shapes)} (cropped to common extent)")
        h, w = min(a.shape[0] for a in (scf, cs, gt)), min(a.shape[1] for a in (scf, cs, gt))
        scf, cs, gt = scf[:h, :w], cs[:h, :w], gt[:h, :w]
        if np.nanmax(cs) > 1.5:
            cs = cs / 100.0
        if np.nanmax(scf) <= 1.0:
            scf = scf * 100.0
        n200 += int((scf == 200).sum())
        tiles.append((gt, scf, cs))
    return tiles, skipped, n200


def evaluate(tiles, t):
    k = dict.fromkeys(['tp_c', 'fp_c', 'fn_c', 'snow', 'snow_masked', 'snow_hit', 'snow_miss',
                       'cloud', 'cloud_kept', 'cloud_kept_snow', 'bg', 'bg_masked', 'bg_snow'], 0)
    for gt, scf, cs in tiles:
        valid = gt != LABEL_NODATA
        masked = (cs < t) | np.isnan(cs)
        snow_pred = ~masked & (np.nan_to_num(scf) > SNOW_THRESHOLD)
        t_snow, t_cloud, t_bg = (gt == LABEL_SNOW) & valid, (gt == LABEL_CLOUD) & valid, (gt == LABEL_BG) & valid
        k['tp_c'] += int((masked & t_cloud).sum())
        k['fp_c'] += int((masked & valid & ~t_cloud).sum())
        k['fn_c'] += int((~masked & t_cloud).sum())
        k['snow'] += int(t_snow.sum()); k['snow_masked'] += int((t_snow & masked).sum())
        k['snow_hit'] += int((t_snow & snow_pred).sum()); k['snow_miss'] += int((t_snow & ~masked & ~snow_pred).sum())
        k['cloud'] += int(t_cloud.sum()); k['cloud_kept'] += int((t_cloud & ~masked).sum())
        k['cloud_kept_snow'] += int((t_cloud & snow_pred).sum())
        k['bg'] += int(t_bg.sum()); k['bg_masked'] += int((t_bg & masked).sum())
        k['bg_snow'] += int((t_bg & snow_pred).sum())
    return k


def report(title, spun_files, args):
    tiles, skipped, n200 = load_tiles(spun_files, args)
    out = [f"\n{'=' * 100}\n{title}: {len(tiles)} tiles\n{'=' * 100}"]
    out += [f"  note: {s}" for s in skipped]
    if n200:
        out.append(f"  WARNING: {n200:,} SPUN pixels equal 200 - are these masked (pipeline) outputs rather than raw?")
    if not tiles:
        return '\n'.join(out)

    out.append("\nA. CS+ AS A CLOUD DETECTOR (label cloud class)")
    out.append(f"  {'cs_cdf <':>8} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'IoU':>6} | {'cloud missed':>12} | "
               f"{'snow masked':>11} | {'bg masked':>9}")
    rows_b = []
    for t in args.thresholds:
        k = evaluate(tiles, t)
        p, r, f1, iou = scores(k['tp_c'], k['fp_c'], k['fn_c'])
        mark = ' *' if abs(t - REFERENCE_THRESHOLD) < 1e-9 else '  '
        out.append(f"  {t:>6.2f}{mark} | {p:>6.3f} | {r:>6.3f} | {f1:>6.3f} | {iou:>6.3f} | "
                   f"{k['cloud_kept'] / max(k['cloud'], 1):>12.1%} | {k['snow_masked'] / max(k['snow'], 1):>11.1%} | "
                   f"{k['bg_masked'] / max(k['bg'], 1):>9.1%}")
        rows_b.append((t, mark, k))

    out.append("\nB. ERROR PROPAGATION INTO THE FINAL SNOW PRODUCT")
    out.append(f"  {'cs_cdf <':>8} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'FN: masked by CS+':>17} | "
               f"{'FN: missed by SPUN':>18} | {'FP: retained cloud':>18} | {'FP: background':>14}")
    for t, mark, k in rows_b:
        fn = k['snow_masked'] + k['snow_miss']
        fp = k['cloud_kept_snow'] + k['bg_snow']
        sp, sr, sf1, _ = scores(k['snow_hit'], fp, fn)
        out.append(f"  {t:>6.2f}{mark} | {sp:>6.3f} | {sr:>6.3f} | {sf1:>6.3f} | "
                   f"{k['snow_masked'] / max(fn, 1):>17.1%} | {k['snow_miss'] / max(fn, 1):>18.1%} | "
                   f"{k['cloud_kept_snow'] / max(fp, 1):>18.1%} | {k['bg_snow'] / max(fp, 1):>14.1%}")
    k = next((kk for tt, _, kk in rows_b if abs(tt - REFERENCE_THRESHOLD) < 1e-9), None)
    if k:
        out.append(f"\n  At cs_cdf < {REFERENCE_THRESHOLD} (*): CS+ misses {k['cloud_kept'] / max(k['cloud'], 1):.1%} of "
                   f"cloud pixels, of which SPUN predicts {k['cloud_kept_snow'] / max(k['cloud_kept'], 1):.1%} as snow; "
                   f"CS+ masks {k['snow_masked'] / max(k['snow'], 1):.1%} of snow pixels.")
    out.append("  FN/FP columns are shares of the final snow false negatives / false positives.")
    return '\n'.join(out)


def main():
    args = get_args()
    all_files = sorted(f for f in glob.glob(os.path.join(args.spun_dir, '*.tif')) if not f.endswith('_s2.tif'))
    subset = [f for f in all_files if any(i in os.path.basename(f) for i in SUBSET_IDS)]
    text = "CLOUDSCORE+ PERFORMANCE AND ERROR PROPAGATION (GLOBAL DATASET)\n"
    text += report("TEST SUBSET (N=6)", subset, args)
    text += report("FULL DATASET", all_files, args)
    print(text)
    with open(args.output, 'w') as f:
        f.write(text + '\n')
    print(f"\nSaved {args.output}")


if __name__ == '__main__':
    main()
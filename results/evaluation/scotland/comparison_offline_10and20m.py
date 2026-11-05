import pandas as pd
import numpy as np
import os
import argparse
import zarr
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --- PARAMETERS ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255
NDSI_THRESH = 0.4 
PIXEL_AREA_10M = 100  # 10m x 10m
PIXEL_AREA_20M = 400  # 20m x 20m

def get_args():
    parser = argparse.ArgumentParser(description="Generate multi-resolution evaluation report against baselines.")
    
    # Dataset Metadata
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the Zenodo dataset')
    parser.add_argument('--split_file', type=str, default='dataset_metadata.csv', help='Name of the metadata CSV')
    
    # Predictions Directories
    parser.add_argument('--spun_dir', type=str, required=True, help='Directory containing SPUN prediction Zarrs')
    parser.add_argument('--lis_dir', type=str, required=True, help='Directory containing LIS patches')
    parser.add_argument('--ndsi_dir', type=str, required=True, help='Directory containing NDSI Zarrs')
    parser.add_argument('--wang_dir', type=str, required=True, help='Directory containing Wang2022 Zarrs')
    
    # Output
    parser.add_argument('--output_report', type=str, default='final_multires_report_with_cloud_separation.txt', help='Path to save the output text report')
    
    return parser.parse_args()

# --- HELPER FUNCTIONS ---

def downscale_label_masked(arr, factor=2, invalid_val=200):
    """Downscales 10m -> 20m (100x100 -> 50x50), handling invalid values."""
    arr_masked = np.where(arr == invalid_val, np.nan, arr)
    h, w = arr_masked.shape
    h_new, w_new = h // factor, w // factor
    
    arr_masked = arr_masked[:h_new*factor, :w_new*factor]
    blocks = arr_masked.reshape(h_new, factor, w_new, factor)
    sum_ds = np.nansum(blocks, axis=(1, 3))
    count_ds = np.count_nonzero(~np.isnan(blocks), axis=(1, 3))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ds = sum_ds / count_ds
        
    return np.where(count_ds == 0, invalid_val, mean_ds).astype(np.float32)

def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

def calculate_binary_scores(y_true, y_pred, thresh=0.5):
    """Calculates binary stats at a specific threshold."""
    if len(y_true) == 0: return {'acc':0,'prec':0,'rec':0,'f1':0}
    
    y_true_bin = (y_true > (thresh * 100)).astype(int)
    y_pred_bin = (y_pred > (thresh * 100)).astype(int)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

def compute_metrics_package(y_true, y_pred, pixel_area):
    """Computes ALL metrics (Global, Snow-Only, Binary, Coverage)."""
    if len(y_true) == 0: return None
    
    # 1. Global Continuous Metrics
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    bias_global = np.mean(diff) 
    
    # 2. Snow-Only Metrics (Union Mask > 1%)
    mask_snow = (y_true >= 1.0) | (y_pred >= 1.0)
    if mask_snow.any():
        diff_snow = y_pred[mask_snow] - y_true[mask_snow]
        mae_snow = np.mean(np.abs(diff_snow))
        rmse_snow = np.sqrt(np.mean(diff_snow**2))
        bias_snow = np.mean(diff_snow) 
    else:
        mae_snow, rmse_snow, bias_snow = 0.0, 0.0, 0.0
    
    # 3. Binary Metrics
    stats_50 = calculate_binary_scores(y_true, y_pred, thresh=0.5)
    stats_01 = calculate_binary_scores(y_true, y_pred, thresh=0.01)
    
    # 4. Coverage Metrics
    total_area_m2 = (np.sum(y_pred) / 100.0) * pixel_area
    
    # Pixel Counts for Detection Rate Table
    count_gt_50   = (y_true > 50).sum()
    count_pred_50 = (y_pred > 50).sum()
    count_gt_01   = (y_true >= 1.0).sum()
    count_pred_01 = (y_pred >= 1.0).sum()
    
    return {
        'count': len(y_true),
        'mae': mae, 'rmse': rmse, 'bias': bias_global,
        'mae_snow': mae_snow, 'rmse_snow': rmse_snow, 'bias_snow': bias_snow,
        'area_m2': total_area_m2,
        'count_gt_50': count_gt_50, 'count_pred_50': count_pred_50,
        'count_gt_01': count_gt_01, 'count_pred_01': count_pred_01,
        'std': stats_50,  # >50% stats
        'sens': stats_01  # >1% stats
    }

def main():
    args = get_args()

    print("--- LOADING FILE LIST ---")
    split_csv_path = os.path.join(args.data_root, args.split_file)
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"Split file not found: {split_csv_path}")

    df = pd.read_csv(split_csv_path)
    path_col = 'relative_path' if 'relative_path' in df.columns else 'filepath'
    split_col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    
    test_df = df[df[split_col] == 'Test']
    test_files = test_df[path_col].tolist()
    
    # Storage for TIER 1 (10m)
    data_10m = {'gt': [], 'spun': [], 'ndsi': [], 'wang': []}
    
    # Storage for TIER 2 (20m) - Now split into Clear and All Sky
    data_20m_clear = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []}
    data_20m_all   = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []}

    print(f"Comparing {len(test_files)} tiles across multi-resolution outputs...")
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            
            spun_path = os.path.join(args.spun_dir, f"{date_id}_{tile_id}_SPUN.zarr")
            lis_path  = os.path.join(args.lis_dir, f"{date_id}_{tile_id}_LIS.zarr")
            ndsi_path = os.path.join(args.ndsi_dir, f"{date_id}_{tile_id}_NDSI.zarr")
            wang_path_v1 = os.path.join(args.wang_dir, f"{date_id}_{tile_id}_Wang2022.zarr")
            wang_path_v2 = os.path.join(args.wang_dir, f"{date_id}_{tile_id}_SPUN.zarr")
            wang_path = wang_path_v1 if os.path.exists(wang_path_v1) else (wang_path_v2 if os.path.exists(wang_path_v2) else None)
            
            if not (os.path.exists(spun_path) and os.path.exists(lis_path) and os.path.exists(ndsi_path) and wang_path):
                continue

            # --- LOAD 10m DATA ---
            z_spun = zarr.open(spun_path, mode='r')
            spun_10m = z_spun['spun_scf'][:]
            gt_10m   = z_spun['label'][:] 
            z_ndsi = zarr.open(ndsi_path, mode='r')
            ndsi_mask_10m = (z_ndsi['ndsi'][:] > NDSI_THRESH).astype(np.float32) * 100.0
            wang_10m = zarr.open(wang_path, mode='r')['wang_scf'][:]

            # --- STORE TIER 1 (10m - All Sky) ---
            mask_10m = (gt_10m != LABEL_CLOUD)
            if mask_10m.any():
                data_10m['gt'].append(gt_10m[mask_10m])
                data_10m['spun'].append(spun_10m[mask_10m])
                data_10m['ndsi'].append(ndsi_mask_10m[mask_10m])
                data_10m['wang'].append(wang_10m[mask_10m])

            # --- DOWNSCALE 10m -> 20m ---
            spun_20 = downscale_label_masked(spun_10m)
            gt_20   = downscale_label_masked(gt_10m, factor=2, invalid_val=LABEL_CLOUD)
            ndsi_20 = downscale_label_masked(ndsi_mask_10m, factor=2, invalid_val=-999)
            wang_20 = downscale_label_masked(wang_10m, factor=2, invalid_val=-999)
            
            lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:]
            if spun_20.shape != lis_arr.shape: continue
            
            # FLATTEN FOR FILTERING
            y_gt   = gt_20.flatten()
            y_spun = spun_20.flatten()
            y_lis  = lis_arr.flatten()
            y_ndsi = ndsi_20.flatten()
            y_wang = wang_20.flatten()

            # --- STORE TIER 2: CLEAR SKY CONTROL ---
            mask_clear = (y_gt != LABEL_CLOUD) & (y_lis != LIS_CLOUD) & (y_lis != LIS_NODATA) & (~np.isnan(y_lis))
            if mask_clear.any():
                data_20m_clear['gt'].append(y_gt[mask_clear])
                data_20m_clear['spun'].append(y_spun[mask_clear])
                data_20m_clear['lis'].append(y_lis[mask_clear])
                data_20m_clear['ndsi'].append(y_ndsi[mask_clear])
                data_20m_clear['wang'].append(y_wang[mask_clear])

            # --- STORE TIER 2: ALL SKY REALITY ---
            mask_all = (y_gt != LABEL_CLOUD)
            if mask_all.any():
                lis_filled = np.nan_to_num(np.where(y_lis >= 200, 0, y_lis), nan=0.0)
                
                data_20m_all['gt'].append(y_gt[mask_all])
                data_20m_all['spun'].append(y_spun[mask_all])
                data_20m_all['lis'].append(lis_filled[mask_all])
                data_20m_all['ndsi'].append(y_ndsi[mask_all])
                data_20m_all['wang'].append(y_wang[mask_all])

        except Exception: 
            continue

    # --- AGGREGATE ---
    def aggregate_data(d):
        return {k: np.concatenate(v) for k, v in d.items() if v}
    
    d10 = aggregate_data(data_10m)
    d20_cl = aggregate_data(data_20m_clear)
    d20_al = aggregate_data(data_20m_all)

    # --- REPORTING ---
    out_dir = os.path.dirname(args.output_report)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output_report, "w") as f:
        def log(text):
            print(text)
            f.write(text + "\n")
            
        row_fmt = "{:<10} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7}"


        def print_full_tier(title, dataset, pixel_area):
            if not dataset.get('gt') is not None: return
            log(f"\n{title}")
            log("="*120)
            
            # 1. Main Metrics Table
            log(row_fmt.format("Model", "Rec", "Prec", "F1", "MAE", "RMSE", "Bias", "MAE(S)", "RMSE(S)", "Bias(S)"))
            log("-" * 120)
            
            gt_all = dataset['gt']
            gt_area = (np.sum(gt_all)/100.0) * pixel_area
            
            keys = ['spun', 'lis', 'ndsi', 'wang'] if 'lis' in dataset else ['spun', 'ndsi', 'wang']
            metrics_store = {} 
            
            for k in keys:
                if k not in dataset: continue
                m = compute_metrics_package(gt_all, dataset[k], pixel_area)
                metrics_store[k] = m
                
                log(row_fmt.format(
                    k.upper(),
                    f"{m['std']['rec']:.3f}", f"{m['std']['prec']:.3f}", f"{m['std']['f1']:.3f}",
                    f"{m['mae']:.3f}", f"{m['rmse']:.3f}", f"{m['bias']:+.3f}",
                    f"{m['mae_snow']:.2f}", f"{m['rmse_snow']:.2f}", f"{m['bias_snow']:.2f}"
                ))
            
            log("-" * 120)

            # 2. Coverage Analysis Table 
            log("\n  OPERATIONAL COVERAGE & SENSITIVITY")
            log("  " + "-" * 115)
            cov_fmt = "  {:<15} | {:<12} | {:<10} | {:<15} | {:<15} | {:<15} | {:<15}"
            log(cov_fmt.format("Method", "Area(km2)", "Dev(%)", "Count(>50%)", "Rate(>50%)", "Count(>1%)", "Rate(>1%)"))
            log("  " + "-" * 115)
            
            gt_km2 = gt_area / 1_000_000.0
            gt_c50 = metrics_store['spun']['count_gt_50']
            gt_c01 = metrics_store['spun']['count_gt_01']
            
            log(cov_fmt.format("Ground Truth", f"{gt_km2:.2f}", "Ref", f"{gt_c50:,}", "100%", f"{gt_c01:,}", "100%"))
            
            for k in keys:
                m = metrics_store[k]
                
                pred_km2 = m['area_m2'] / 1_000_000.0
                dev = ((pred_km2 - gt_km2) / gt_km2 * 100.0) if gt_km2 > 0 else 0
                
                c50 = m['count_pred_50']
                r50 = (c50 / gt_c50 * 100.0) if gt_c50 > 0 else 0
                c01 = m['count_pred_01']
                r01 = (c01 / gt_c01 * 100.0) if gt_c01 > 0 else 0
                
                log(cov_fmt.format(k.upper(), f"{pred_km2:.2f}", f"{dev:+.1f}%", 
                                   f"{c50:,}", f"{r50:.1f}%", 
                                   f"{c01:,}", f"{r01:.1f}%"))
            log("  " + "-" * 115)

        # --- RUN REPORTS ---
        print_full_tier("TIER 1: INTRINSIC PERFORMANCE (Native 10m - All Sky)", d10, PIXEL_AREA_10M)
        print_full_tier("TIER 2a: OPERATIONAL - CLEAR SKY CONTROL (20m)", d20_cl, PIXEL_AREA_20M)
        print_full_tier("TIER 2b: OPERATIONAL - ALL SKY REALITY (20m)", d20_al, PIXEL_AREA_20M)

    print(f"\nSaved full report to: {os.path.abspath(args.output_report)}")

if __name__ == "__main__":
    main()
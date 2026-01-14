import pandas as pd
import numpy as np
import os
import zarr
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --- CONFIGURATION ---
SPLIT_CSV_PATH      = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
LIS_PATCHES_DIR     = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
REPORT_OUTPUT       = 'final_comparison_report.txt'

# --- CODES & PARAMETERS ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255
DICE_SMOOTH = 1.0 
FLIP_LIS    = False 

# --- HELPER FUNCTIONS ---

def downscale_label_masked(arr, factor=2, invalid_val=200):
    """Downscales 10m -> 20m (100x100 -> 50x50)."""
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

# --- METRIC FUNCTIONS ---

def calculate_binary_scores(y_true, y_pred, thresh=0.5):
    """Calculates binary stats for a specific threshold."""
    if len(y_true) == 0:
        return {'acc': 0.0, 'prec': 0.0, 'rec': 0.0, 'f1': 0.0, 'dice': 0.0}
    
    y_true_bin = (y_true > (thresh * 100)).astype(int)
    y_pred_bin = (y_pred > (thresh * 100)).astype(int)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    inter = (y_true_bin * y_pred_bin).sum()
    union = y_true_bin.sum() + y_pred_bin.sum()
    dice = (2. * inter + DICE_SMOOTH) / (union + DICE_SMOOTH)
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'dice': dice}

def compute_metrics_package(y_true, y_pred):
    """
    Computes Standard (50%), Sensitive (1%), and Snow-Only metrics.
    """
    if len(y_true) == 0:
        return None
    
    # 1. Global Continuous Metrics (All Valid Pixels)
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    
    # 2. Snow-Only Metrics (Where Ground Truth > 0)
    mask_snow = y_true > 0
    if mask_snow.any():
        diff_snow = y_pred[mask_snow] - y_true[mask_snow]
        mae_snow = np.mean(np.abs(diff_snow))
        rmse_snow = np.sqrt(np.mean(diff_snow**2))
    else:
        mae_snow, rmse_snow = 0.0, 0.0
    
    # 3. Standard Binary Metrics (Threshold > 50%)
    stats_50 = calculate_binary_scores(y_true, y_pred, thresh=0.5)
    
    # 4. Sensitive Binary Metrics (Threshold > 1%)
    stats_01 = calculate_binary_scores(y_true, y_pred, thresh=0.01)
    
    return {
        'count': len(y_true),
        'mae': mae,
        'rmse': rmse,
        'mae_snow': mae_snow,   # New Metric
        'rmse_snow': rmse_snow, # New Metric
        'std': stats_50, 
        'sens': stats_01 
    }

def main():
    print("--- LOADING FILE LIST ---")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    data_clear = {'gt': [], 'spun': [], 'lis': []} 
    data_all   = {'gt': [], 'spun': [], 'lis': []} 

    print(f"Comparing {len(test_files)} tiles...")
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            
            # 1. Load Data
            spun_path = os.path.join(SPUN_PREDICTIONS_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
            lis_path = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
            
            if not os.path.exists(spun_path) or not os.path.exists(lis_path):
                continue

            # LIS (20m)
            lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:]
            
            # SPUN (10m)
            z_spun = zarr.open(spun_path, mode='r')
            spun_10m = z_spun['spun_scf'][:]
            gt_10m = z_spun['label'][:]

            # 2. Downscale 10m -> 20m
            spun_20 = downscale_label_masked(spun_10m, factor=2, invalid_val=-999)
            gt_20 = downscale_label_masked(gt_10m, factor=2, invalid_val=LABEL_CLOUD)

            if spun_20.shape != lis_arr.shape:
                continue

            # 4. Flatten & Filter
            y_gt = gt_20.flatten()
            y_spun = spun_20.flatten()
            y_lis = lis_arr.flatten()

            # Scenario 1: Clear Sky
            mask_clear = (y_gt != LABEL_CLOUD) & (y_lis != LIS_CLOUD) & (y_lis != LIS_NODATA) & (~np.isnan(y_lis))
            if mask_clear.any():
                data_clear['gt'].append(y_gt[mask_clear])
                data_clear['spun'].append(y_spun[mask_clear])
                data_clear['lis'].append(y_lis[mask_clear])

            # Scenario 2: All Sky
            mask_all = (y_gt != LABEL_CLOUD)
            if mask_all.any():
                lis_filled = np.where(y_lis >= 200, 0, y_lis) 
                lis_filled = np.nan_to_num(lis_filled, nan=0.0)
                data_all['gt'].append(y_gt[mask_all])
                data_all['spun'].append(y_spun[mask_all])
                data_all['lis'].append(lis_filled[mask_all])

        except Exception as e:
            continue

    # --- REPORTING ---
    print("\nComputing Metrics...")
    
    def get_m(d):
        if not d['gt']: return None, None
        gt, sp, li = np.concatenate(d['gt']), np.concatenate(d['spun']), np.concatenate(d['lis'])
        return compute_metrics_package(gt, sp), compute_metrics_package(gt, li)

    m_sp_cl, m_li_cl = get_m(data_clear)
    m_sp_al, m_li_al = get_m(data_all)

    with open(REPORT_OUTPUT, "w") as f:
        def log_print(text):
            print(text)
            f.write(text + "\n")

        # Expanded Table Width
        row_fmt = "{:<15} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<10}"
        
        log_print("\n" + "="*120)
        log_print("FINAL COMPARISON REPORT: SPUN vs LIS")
        log_print("="*120)
        
        def print_model_rows(name, m):
            if not m: return
            # Row 1: Standard Metrics (>50%) + Snow MAE/RMSE
            log_print(row_fmt.format(f"{name} (>50%)", 
                                     f"{m['std']['rec']:.3f}", f"{m['std']['prec']:.3f}", 
                                     f"{m['std']['f1']:.3f}", f"{m['std']['dice']:.3f}", 
                                     f"{m['mae']:.3f}", f"{m['rmse']:.3f}", 
                                     f"{m['mae_snow']:.3f}", f"{m['rmse_snow']:.3f}", # Added Cols
                                     f"{m['count']:,}"))
            
            # Row 2: Sensitivity Metrics (>1%) - MAE/RMSE repeated or blank (since they don't depend on thresh)
            log_print(row_fmt.format(f"{name} (>1%)", 
                                     f"{m['sens']['rec']:.3f}", f"{m['sens']['prec']:.3f}", 
                                     f"{m['sens']['f1']:.3f}", f"{m['sens']['dice']:.3f}", 
                                     "-", "-", "-", "-", "-"))

        if m_sp_cl:
            log_print("\nSCENARIO 1: CLEAR-SKY")
            log_print("-" * 120)
            log_print(row_fmt.format("Model", "Recall", "Precis", "F1", "Dice", "MAE", "RMSE", "MAE(S)", "RMSE(S)", "Pixels"))
            log_print("-" * 120)
            print_model_rows("SPUN", m_sp_cl)
            log_print("-" * 120)
            print_model_rows("LIS", m_li_cl)
            
        if m_sp_al:
            log_print("\n\nSCENARIO 2: ALL-SKY")
            log_print("-" * 120)
            log_print(row_fmt.format("Model", "Recall", "Precis", "F1", "Dice", "MAE", "RMSE", "MAE(S)", "RMSE(S)", "Pixels"))
            log_print("-" * 120)
            print_model_rows("SPUN", m_sp_al)
            log_print("-" * 120)
            print_model_rows("LIS", m_li_al)
            
            # Coverage Gap
            gt_arr = np.concatenate(data_all['gt'])
            lis_arr = np.concatenate(data_all['lis'])
            spun_arr = np.concatenate(data_all['spun'])
            
            snow_gt = (gt_arr > 50).sum()
            snow_lis = (lis_arr > 50).sum()
            snow_spun = (spun_arr > 50).sum()
            
            log_print("\n\nSCENARIO 3: COVERAGE GAP ANALYSIS (Threshold > 50%)")
            log_print(f"Total Snow Pixels (GT): {snow_gt:,}")
            log_print(f"Detected by LIS:        {snow_lis:,} ({snow_lis/snow_gt*100:.1f}%)")
            log_print(f"Detected by SPUN:       {snow_spun:,} ({snow_spun/snow_gt*100:.1f}%)")
            log_print("="*120)

    print(f"\nReport saved to: {os.path.abspath(REPORT_OUTPUT)}")

if __name__ == "__main__":
    main()
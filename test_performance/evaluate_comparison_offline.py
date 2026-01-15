import pandas as pd
import numpy as np
import os
import zarr
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --- CONFIGURATION ---
SPLIT_CSV_PATH       = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# DATA DIRECTORIES
# 1. LIS (Reference 20m)
LIS_PATCHES_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
# 2. SPUN Model Predictions (10m)
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
# 3. NDSI Baseline Data (10m)
NDSI_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_ndsi'
# 4. Wang2022 Predictions (10m)
WANG_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_wang2022'

REPORT_OUTPUT        = 'final_comparison_report.txt'

# --- CODES & PARAMETERS ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255
DICE_SMOOTH = 1.0 
NDSI_THRESH = 0.4  # Standard threshold for NDSI

# Constants for Area Calculation
PIXEL_SIZE_M = 20  # 20m resolution (after downscaling)
AREA_PER_PIXEL_M2 = PIXEL_SIZE_M * PIXEL_SIZE_M  # 400 m^2

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

# --- METRIC FUNCTIONS ---

def calculate_binary_scores(y_true, y_pred, thresh=0.5):
    """Calculates binary classification metrics at a specific threshold."""
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
    """Computes comprehensive metrics: Continuous, Snow-Only, Binary (Std/Sens), and Area."""
    if len(y_true) == 0:
        return None
    
    # 1. Global Continuous Metrics
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    bias = np.mean(diff) 
    
    # 2. Snow-Only Metrics (Errors calculated only on confirmed snow pixels)
    mask_snow = y_true > 0
    if mask_snow.any():
        diff_snow = y_pred[mask_snow] - y_true[mask_snow]
        mae_snow = np.mean(np.abs(diff_snow))
        rmse_snow = np.sqrt(np.mean(diff_snow**2))
    else:
        mae_snow, rmse_snow = 0.0, 0.0
    
    # 3. Binary Metrics
    stats_50 = calculate_binary_scores(y_true, y_pred, thresh=0.5)
    stats_01 = calculate_binary_scores(y_true, y_pred, thresh=0.01)
    
    # 4. Total Area Calculation (m^2)
    total_area_m2 = (np.sum(y_pred) / 100.0) * AREA_PER_PIXEL_M2
    
    return {
        'count': len(y_true),
        'mae': mae,
        'rmse': rmse,
        'bias': bias,           
        'mae_snow': mae_snow,
        'rmse_snow': rmse_snow,
        'area_m2': total_area_m2, 
        'std': stats_50, 
        'sens': stats_01 
    }

def main():
    print("--- LOADING FILE LIST ---")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    # Storage for all models including NDSI and Wang
    data_clear = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []} 
    data_all   = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []} 

    print(f"Comparing {len(test_files)} tiles...")
    
    missing_count = 0
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            
            # Construct Paths
            spun_path = os.path.join(SPUN_PREDICTIONS_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
            lis_path  = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
            # FIXED: Now looks for _SPUN.zarr (assuming typical generator output)
            ndsi_path = os.path.join(NDSI_DIR, f"{date_id}_{tile_id}_NDSI.zarr")
            wang_path = os.path.join(WANG_DIR, f"{date_id}_{tile_id}_Wang2022.zarr")
            
            # Check existence
            if not (os.path.exists(spun_path) and os.path.exists(lis_path) and 
                    os.path.exists(ndsi_path) and os.path.exists(wang_path)):
                missing_count += 1
                continue

            # --- LOAD DATA ---
            # 1. LIS (20m)
            lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:]
            
            # 2. SPUN (10m)
            z_spun = zarr.open(spun_path, mode='r')
            spun_10m = z_spun['spun_scf'][:]
            gt_10m   = z_spun['label'][:] 
            
            # 3. NDSI (10m)
            z_ndsi = zarr.open(ndsi_path, mode='r')
            ndsi_raw_10m = z_ndsi['ndsi'][:] 
            
            # 4. Wang (10m)
            z_wang = zarr.open(wang_path, mode='r')
            wang_10m = z_wang['wang_scf'][:]

            # --- PROCESS NDSI ---
            # Threshold raw NDSI to create binary mask (0 or 100)
            ndsi_mask_10m = (ndsi_raw_10m > NDSI_THRESH).astype(np.float32) * 100.0
            
            # --- DOWNSCALE (10m -> 20m) ---
            spun_20 = downscale_label_masked(spun_10m, factor=2, invalid_val=-999)
            gt_20   = downscale_label_masked(gt_10m, factor=2, invalid_val=LABEL_CLOUD)
            ndsi_20 = downscale_label_masked(ndsi_mask_10m, factor=2, invalid_val=-999)
            wang_20 = downscale_label_masked(wang_10m, factor=2, invalid_val=-999)

            # Shape Check
            if spun_20.shape != lis_arr.shape: continue

            # --- FLATTEN ---
            y_gt   = gt_20.flatten()
            y_spun = spun_20.flatten()
            y_lis  = lis_arr.flatten()
            y_ndsi = ndsi_20.flatten()
            y_wang = wang_20.flatten()

            # --- SCENARIO 1: CLEAR SKY ---
            mask_clear = (y_gt != LABEL_CLOUD) & (y_lis != LIS_CLOUD) & (y_lis != LIS_NODATA) & (~np.isnan(y_lis))
            if mask_clear.any():
                data_clear['gt'].append(y_gt[mask_clear])
                data_clear['spun'].append(y_spun[mask_clear])
                data_clear['lis'].append(y_lis[mask_clear])
                data_clear['ndsi'].append(y_ndsi[mask_clear])
                data_clear['wang'].append(y_wang[mask_clear])

            # --- SCENARIO 2: ALL SKY ---
            mask_all = (y_gt != LABEL_CLOUD)
            if mask_all.any():
                lis_filled = np.where(y_lis >= 200, 0, y_lis) 
                lis_filled = np.nan_to_num(lis_filled, nan=0.0)
                
                data_all['gt'].append(y_gt[mask_all])
                data_all['spun'].append(y_spun[mask_all])
                data_all['lis'].append(lis_filled[mask_all])
                data_all['ndsi'].append(y_ndsi[mask_all])
                data_all['wang'].append(y_wang[mask_all])

        except Exception as e:
            continue

    if missing_count > 0:
        print(f"Skipped {missing_count} tiles due to missing files (check paths!).")

    # --- REPORTING ---
    print("\nComputing Metrics...")
    
    def get_aggregated_metrics(d):
        if not d['gt']: return None, None, None, None, 0
        gt = np.concatenate(d['gt'])
        sp = np.concatenate(d['spun'])
        li = np.concatenate(d['lis'])
        nd = np.concatenate(d['ndsi'])
        wa = np.concatenate(d['wang'])
        
        gt_area = (np.sum(gt) / 100.0) * AREA_PER_PIXEL_M2
        
        return (compute_metrics_package(gt, sp), 
                compute_metrics_package(gt, li), 
                compute_metrics_package(gt, nd), 
                compute_metrics_package(gt, wa),
                gt_area)

    m_sp_cl, m_li_cl, m_nd_cl, m_wa_cl, gt_area_cl = get_aggregated_metrics(data_clear)
    m_sp_al, m_li_al, m_nd_al, m_wa_al, gt_area_al = get_aggregated_metrics(data_all)

    with open(REPORT_OUTPUT, "w") as f:
        def log_print(text):
            print(text)
            f.write(text + "\n")

        row_fmt = "{:<15} | {:<7} | {:<7} | {:<7} | {:<7} | {:<8} | {:<8} | {:<8} | {:<8}"
        
        log_print("\n" + "="*120)
        log_print(f"FINAL COMPARISON REPORT: SPUN vs LIS vs NDSI vs WANG")
        log_print("="*120)
        
        def print_model_rows(name, m):
            if not m: return
            log_print(row_fmt.format(f"{name} (>50%)", 
                                     f"{m['std']['rec']:.3f}", f"{m['std']['prec']:.3f}", 
                                     f"{m['std']['f1']:.3f}", f"{m['std']['dice']:.3f}", 
                                     f"{m['mae']:.3f}", f"{m['rmse']:.3f}", 
                                     f"{m['mae_snow']:.3f}", f"{m['rmse_snow']:.3f}"))
            
            log_print(row_fmt.format(f"{name} (>1%)", 
                                     f"{m['sens']['rec']:.3f}", f"{m['sens']['prec']:.3f}", 
                                     f"{m['sens']['f1']:.3f}", f"{m['sens']['dice']:.3f}", 
                                     f"Bias:{m['bias']:.2f}", "-", "-", "-"))

        def print_scenario_block(title, m_sp, m_li, m_nd, m_wa, gt_area):
             if not m_sp: return
             log_print(f"\n{title}")
             log_print("-" * 120)
             log_print(row_fmt.format("Model", "Recall", "Precis", "F1", "Dice", "MAE", "RMSE", "MAE(S)", "RMSE(S)"))
             log_print("-" * 120)
             print_model_rows("SPUN", m_sp)
             log_print("-" * 120)
             print_model_rows("LIS", m_li)
             log_print("-" * 120)
             print_model_rows("NDSI", m_nd)
             log_print("-" * 120)
             print_model_rows("WANG", m_wa)
             
             log_print("-" * 120)
             log_print(f"Total Snow Area (GT):   {gt_area:,.2f} m2")
             log_print(f"Total Snow Area (SPUN): {m_sp['area_m2']:,.2f} m2 (Diff: {m_sp['area_m2']-gt_area:+,.2f} m2)")
             log_print(f"Total Snow Area (LIS):  {m_li['area_m2']:,.2f} m2 (Diff: {m_li['area_m2']-gt_area:+,.2f} m2)")
             log_print(f"Total Snow Area (NDSI): {m_nd['area_m2']:,.2f} m2 (Diff: {m_nd['area_m2']-gt_area:+,.2f} m2)")
             log_print(f"Total Snow Area (WANG): {m_wa['area_m2']:,.2f} m2 (Diff: {m_wa['area_m2']-gt_area:+,.2f} m2)")

        # --- OUTPUT ---
        print_scenario_block("SCENARIO 1: CLEAR-SKY", m_sp_cl, m_li_cl, m_nd_cl, m_wa_cl, gt_area_cl)
        print_scenario_block("\nSCENARIO 2: ALL-SKY", m_sp_al, m_li_al, m_nd_al, m_wa_al, gt_area_al)
        
        # --- COVERAGE GAP ---
        if m_sp_al:
            gt_arr = np.concatenate(data_all['gt'])
            lis_arr = np.concatenate(data_all['lis'])
            spun_arr = np.concatenate(data_all['spun'])
            ndsi_arr = np.concatenate(data_all['ndsi'])
            wang_arr = np.concatenate(data_all['wang'])
            
            snow_gt = (gt_arr > 50).sum()
            snow_lis = (lis_arr > 50).sum()
            snow_spun = (spun_arr > 50).sum()
            snow_ndsi = (ndsi_arr > 50).sum()
            snow_wang = (wang_arr > 50).sum()
            
            log_print("\n\nSCENARIO 3: COVERAGE GAP ANALYSIS (Binary > 50%)")
            log_print(f"Total Snow Pixels (GT): {snow_gt:,}")
            log_print(f"Detected by LIS:        {snow_lis:,} ({snow_lis/snow_gt*100:.1f}%)")
            log_print(f"Detected by SPUN:       {snow_spun:,} ({snow_spun/snow_gt*100:.1f}%)")
            log_print(f"Detected by NDSI:       {snow_ndsi:,} ({snow_ndsi/snow_gt*100:.1f}%)")
            log_print(f"Detected by WANG:       {snow_wang:,} ({snow_wang/snow_gt*100:.1f}%)")
            log_print("="*120)

    print(f"\nReport saved to: {os.path.abspath(REPORT_OUTPUT)}")

if __name__ == "__main__":
    main()
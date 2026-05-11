import pandas as pd
import numpy as np
import os
import zarr
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# --- CONFIGURATION ---
SPLIT_CSV_PATH       = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# DATA DIRECTORIES
LIS_PATCHES_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
NDSI_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_ndsi'
WANG_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_wang2022'

REPORT_OUTPUT        = 'final_comparison_report.txt'

# --- CODES & PARAMETERS ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255
DICE_SMOOTH = 1.0 
NDSI_THRESH = 0.4 
PIXEL_SIZE_M = 20  
AREA_PER_PIXEL_M2 = PIXEL_SIZE_M * PIXEL_SIZE_M 

# --- HELPER FUNCTIONS ---
def downscale_label_masked(arr, factor=2, invalid_val=200):
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
    if len(y_true) == 0: return {'acc':0,'prec':0,'rec':0,'f1':0,'dice':0}
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
    if len(y_true) == 0: return None
    
    # Continuous
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    bias_global = np.mean(diff) 
    
    # Snow-Only (Union Mask > 1%)
    mask_snow = (y_true >= 1.0) | (y_pred >= 1.0)
    if mask_snow.any():
        diff_snow = y_pred[mask_snow] - y_true[mask_snow]
        mae_snow = np.mean(np.abs(diff_snow))
        rmse_snow = np.sqrt(np.mean(diff_snow**2))
        bias_snow = np.mean(diff_snow) 
    else:
        mae_snow, rmse_snow, bias_snow = 0.0, 0.0, 0.0
    
    # Binary Stats
    stats_50 = calculate_binary_scores(y_true, y_pred, thresh=0.5)
    stats_01 = calculate_binary_scores(y_true, y_pred, thresh=0.01)
    
    # Coverage Data
    total_area_m2 = (np.sum(y_pred) / 100.0) * AREA_PER_PIXEL_M2
    
    # Pixel Counts for BOTH thresholds
    count_gt_50   = (y_true > 50).sum()
    count_pred_50 = (y_pred > 50).sum()
    count_gt_01   = (y_true >= 1.0).sum()
    count_pred_01 = (y_pred >= 1.0).sum()
    
    return {
        'count': len(y_true),
        'mae': mae, 'rmse': rmse, 'bias_global': bias_global,           
        'mae_snow': mae_snow, 'rmse_snow': rmse_snow, 'bias_snow': bias_snow,
        'area_m2': total_area_m2,
        'count_gt_50': count_gt_50, 'count_pred_50': count_pred_50,
        'count_gt_01': count_gt_01, 'count_pred_01': count_pred_01,
        'std': stats_50, 'sens': stats_01 
    }

def main():
    print("--- LOADING FILE LIST ---")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    data_clear = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []} 
    data_all   = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []} 

    print(f"Comparing {len(test_files)} tiles...")
    missing_count = 0
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            spun_path = os.path.join(SPUN_PREDICTIONS_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
            lis_path  = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
            ndsi_path = os.path.join(NDSI_DIR, f"{date_id}_{tile_id}_NDSI.zarr")
            wang_path_v1 = os.path.join(WANG_DIR, f"{date_id}_{tile_id}_Wang2022.zarr")
            wang_path_v2 = os.path.join(WANG_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
            wang_path = wang_path_v1 if os.path.exists(wang_path_v1) else (wang_path_v2 if os.path.exists(wang_path_v2) else None)
            
            if not (os.path.exists(spun_path) and os.path.exists(lis_path) and os.path.exists(ndsi_path) and wang_path):
                missing_count += 1
                continue

            lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:]
            z_spun = zarr.open(spun_path, mode='r')
            spun_10m = z_spun['spun_scf'][:]
            gt_10m   = z_spun['label'][:] 
            z_ndsi = zarr.open(ndsi_path, mode='r')
            ndsi_mask_10m = (z_ndsi['ndsi'][:] > NDSI_THRESH).astype(np.float32) * 100.0
            wang_10m = zarr.open(wang_path, mode='r')['wang_scf'][:]

            spun_20 = downscale_label_masked(spun_10m)
            gt_20   = downscale_label_masked(gt_10m, factor=2, invalid_val=LABEL_CLOUD)
            ndsi_20 = downscale_label_masked(ndsi_mask_10m, factor=2, invalid_val=-999)
            wang_20 = downscale_label_masked(wang_10m, factor=2, invalid_val=-999)

            if spun_20.shape != lis_arr.shape: continue

            y_gt, y_spun, y_lis = gt_20.flatten(), spun_20.flatten(), lis_arr.flatten()
            y_ndsi, y_wang = ndsi_20.flatten(), wang_20.flatten()

            mask_clear = (y_gt != LABEL_CLOUD) & (y_lis != LIS_CLOUD) & (y_lis != LIS_NODATA) & (~np.isnan(y_lis))
            if mask_clear.any():
                for k, v in zip(['gt','spun','lis','ndsi','wang'], [y_gt, y_spun, y_lis, y_ndsi, y_wang]):
                    data_clear[k].append(v[mask_clear])

            mask_all = (y_gt != LABEL_CLOUD)
            if mask_all.any():
                lis_filled = np.nan_to_num(np.where(y_lis >= 200, 0, y_lis), nan=0.0)
                for k, v in zip(['gt','spun','lis','ndsi','wang'], [y_gt, y_spun, lis_filled, y_ndsi, y_wang]):
                    data_all[k].append(v[mask_all])
        except: continue

    def get_aggregated_metrics(d):
        if not d['gt']: return None, None, None, None, 0
        gt = np.concatenate(d['gt'])
        gt_area = (np.sum(gt) / 100.0) * AREA_PER_PIXEL_M2
        return (compute_metrics_package(gt, np.concatenate(d['spun'])), 
                compute_metrics_package(gt, np.concatenate(d['lis'])), 
                compute_metrics_package(gt, np.concatenate(d['ndsi'])), 
                compute_metrics_package(gt, np.concatenate(d['wang'])), gt_area)

    m_sp_cl, m_li_cl, m_nd_cl, m_wa_cl, gt_area_cl = get_aggregated_metrics(data_clear)
    m_sp_al, m_li_al, m_nd_al, m_wa_al, gt_area_al = get_aggregated_metrics(data_all)

    with open(REPORT_OUTPUT, "w") as f:
        def log_print(t): print(t); f.write(t + "\n")
        row_fmt = "{:<15} | {:<7} | {:<7} | {:<7} | {:<7} | {:<8} | {:<8} | {:<8} | {:<8}"
        
        def print_scenario(title, m_sp, m_li, m_nd, m_wa, gt_area):
            if not m_sp: return
            log_print(f"\n{title}\n" + "-"*120)
            log_print(row_fmt.format("Model", "Recall", "Precis", "F1", "Dice", "MAE", "RMSE", "MAE(S)", "RMSE(S)"))
            log_print("-" * 120)
            
            for name, m in [("SPUN", m_sp), ("LIS", m_li), ("NDSI", m_nd), ("WANG", m_wa)]:
                log_print(row_fmt.format(f"{name} (>50%)", 
                    f"{m['std']['rec']:.3f}", f"{m['std']['prec']:.3f}", f"{m['std']['f1']:.3f}", f"{m['std']['dice']:.3f}", 
                    f"{m['mae']:.3f}", f"{m['rmse']:.3f}", f"{m['mae_snow']:.3f}", f"{m['rmse_snow']:.3f}"))
                log_print(row_fmt.format(f"{name} (>1%)", 
                    f"{m['sens']['rec']:.3f}", f"{m['sens']['prec']:.3f}", f"{m['sens']['f1']:.3f}", f"{m['sens']['dice']:.3f}", 
                    f"Bias:{m['bias_snow']:.2f}", "-", "-", "-"))
            
            log_print("\n  OPERATIONAL COVERAGE ANALYSIS (DUAL THRESHOLD)")
            log_print("  " + "-" * 115)
            # Expanded Table Header
            cov_fmt = "  {:<15} | {:<12} | {:<12} | {:<15} | {:<15} | {:<15} | {:<15}"
            log_print(cov_fmt.format("Method", "Area (km2)", "Dev (%)", "Count (>50%)", "Rate (>50%)", "Count (>1%)", "Rate (>1%)"))
            log_print("  " + "-" * 115)
            
            gt_km2 = gt_area / 1_000_000.0
            gt_c50 = m_sp['count_gt_50']
            gt_c01 = m_sp['count_gt_01']
            
            log_print(cov_fmt.format("Ground Truth", f"{gt_km2:.2f}", "Ref", f"{gt_c50:,}", "100%", f"{gt_c01:,}", "100%"))
            
            for name, m in [("LIS", m_li), ("Wang22", m_wa), ("NDSI", m_nd), ("SPUN", m_sp)]:
                pred_km2 = m['area_m2'] / 1_000_000.0
                dev = ((pred_km2 - gt_km2) / gt_km2 * 100.0) if gt_km2 > 0 else 0
                
                c50 = m['count_pred_50']
                r50 = (c50 / gt_c50 * 100.0) if gt_c50 > 0 else 0
                
                c01 = m['count_pred_01']
                r01 = (c01 / gt_c01 * 100.0) if gt_c01 > 0 else 0
                
                log_print(cov_fmt.format(name, f"{pred_km2:.2f}", f"{dev:+.1f}%", f"{c50:,}", f"{r50:.1f}%", f"{c01:,}", f"{r01:.1f}%"))
            log_print("  " + "-" * 115)

        print_scenario("SCENARIO 1: CLEAR-SKY", m_sp_cl, m_li_cl, m_nd_cl, m_wa_cl, gt_area_cl)
        print_scenario("\nSCENARIO 2: ALL-SKY", m_sp_al, m_li_al, m_nd_al, m_wa_al, gt_area_al)

    print(f"\nReport saved to: {os.path.abspath(REPORT_OUTPUT)}")

if __name__ == "__main__":
    main()
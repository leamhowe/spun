import pandas as pd
import numpy as np
import os
import zarr
from tqdm import tqdm

# --- CONFIGURATION ---
SPLIT_CSV_PATH       = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
LIS_PATCHES_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
NDSI_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_ndsi'
WANG_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_wang2022'

# --- PARAMETERS ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255
NDSI_THRESH = 0.4 

# --- HELPERS ---
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

def calculate_sensitivity(y_true, y_pred, threshold_val):
    """
    Calculates verified Recall (Sensitivity) at a specific snow fraction threshold.
    GT must be > threshold. Prediction is a 'Hit' if it is also > threshold at that specific location.
    """
    # 1. Identify all pixels where Snow SHOULD be detected (The Denominator)
    mask_true_snow = (y_true > threshold_val)
    total_snow_pixels = np.sum(mask_true_snow)
    
    if total_snow_pixels == 0:
        return 0, 0, 0.0
        
    # 2. Check if Model found them (The Numerator)
    # We check y_pred at the SAME locations where mask_true_snow is True
    # A "Hit" is if y_pred > 0 (any detection) or should we match threshold?
    # Standard practice for sensitivity: Did it detect snow? 
    # Strict practice: Did it detect > threshold snow?
    # Let's use Strict: Model must also see > threshold to count as "Correct classification of Deep Snow"
    
    hits = np.sum(y_pred[mask_true_snow] > threshold_val)
    
    recall = hits / total_snow_pixels
    
    return total_snow_pixels, hits, recall

def main():
    print("Calculating Verified Sensitivity (Recall)...")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    # Storage for Clear-Sky pixels only (The Control)
    data_clear = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []}

    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            spun_path = os.path.join(SPUN_PREDICTIONS_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
            lis_path  = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
            ndsi_path = os.path.join(NDSI_DIR, f"{date_id}_{tile_id}_NDSI.zarr")
            wang_path = os.path.join(WANG_DIR, f"{date_id}_{tile_id}_Wang2022.zarr")
            if not os.path.exists(wang_path): 
                 wang_path = os.path.join(WANG_DIR, f"{date_id}_{tile_id}_SPUN.zarr")

            if not (os.path.exists(spun_path) and os.path.exists(lis_path) and os.path.exists(ndsi_path) and os.path.exists(wang_path)):
                continue

            # Load & Downscale
            z_spun = zarr.open(spun_path, mode='r')
            spun_20 = downscale_label_masked(z_spun['spun_scf'][:])
            gt_20   = downscale_label_masked(z_spun['label'][:], invalid_val=LABEL_CLOUD)
            
            z_ndsi = zarr.open(ndsi_path, mode='r')
            ndsi_20 = downscale_label_masked((z_ndsi['ndsi'][:] > NDSI_THRESH).astype(np.float32) * 100.0, invalid_val=-999)
            
            wang_20 = downscale_label_masked(zarr.open(wang_path, mode='r')['wang_scf'][:], invalid_val=-999)
            
            lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:]
            if spun_20.shape != lis_arr.shape: continue

            # Flatten
            y_gt, y_spun, y_lis = gt_20.flatten(), spun_20.flatten(), lis_arr.flatten()
            y_ndsi, y_wang = ndsi_20.flatten(), wang_20.flatten()

            # Filter Clear Sky
            m_cl = (y_gt != LABEL_CLOUD) & (y_lis != LIS_CLOUD) & (y_lis != LIS_NODATA) & (~np.isnan(y_lis))
            
            if m_cl.any():
                data_clear['gt'].append(y_gt[m_cl])
                data_clear['spun'].append(y_spun[m_cl])
                data_clear['lis'].append(y_lis[m_cl])
                data_clear['ndsi'].append(y_ndsi[m_cl])
                data_clear['wang'].append(y_wang[m_cl])

        except Exception: continue

    # Concatenate
    if not data_clear['gt']: return
    d = {k: np.concatenate(v) for k, v in data_clear.items()}
    
    # Calculate Sensitivity
    print("\n--- SENSITIVITY ANALYSIS (Verified Recall) ---")
    print(f"{'Model':<10} | {'>50% Recall':<12} | {'>1% Recall':<12} | {'Drop':<8}")
    print("-" * 50)
    
    gt = d['gt']
    
    # Thresholds (Percentages)
    t_deep = 50.0
    t_trace = 1.0
    
    # Get total pixels for denominators
    n_deep = np.sum(gt > t_deep)
    n_trace = np.sum(gt > t_trace)
    
    print(f"GT Pixels  | N={n_deep:<10} | N={n_trace:<10} |")
    print("-" * 50)

    for k in ['spun', 'lis', 'ndsi', 'wang']:
        pred = d[k]
        
        # Calculate Deep Recall
        # (Of the pixels that were truly deep snow, how many did model predict as deep snow?)
        hits_deep = np.sum((pred > t_deep) & (gt > t_deep))
        rec_deep = hits_deep / n_deep if n_deep > 0 else 0
        
        # Calculate Trace Recall
        # (Of the pixels that were truly trace snow, how many did model predict as trace snow?)
        hits_trace = np.sum((pred > t_trace) & (gt > t_trace))
        rec_trace = hits_trace / n_trace if n_trace > 0 else 0
        
        drop = rec_deep - rec_trace
        
        print(f"{k.upper():<10} | {rec_deep:.1%}      | {rec_trace:.1%}      | -{drop:.1%}")

if __name__ == "__main__":
    main()
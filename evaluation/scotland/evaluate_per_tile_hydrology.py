import pandas as pd
import numpy as np
import os
import zarr
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error

# --- CONFIGURATION ---
SPLIT_CSV_PATH       = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# DATA DIRECTORIES
LIS_PATCHES_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
NDSI_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_ndsi'
WANG_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_wang2022'

REPORT_OUTPUT        = 'final_hydrology_report.txt'

# --- PARAMETERS ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255
NDSI_THRESH = 0.4 
# Each tile is 1x1km approx (50x50 at 20m), but let's calculate SCA in km2
PIXEL_AREA_KM2 = (20 * 20) / 1_000_000.0  # 0.0004 km2

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

def main():
    print("--- LOADING FILE LIST ---")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    # Store per-tile SCA values (Area in km2)
    # Only for ALL-SKY (Reality) and CLEAR-SKY (Control)
    results = {
        'clear': {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []},
        'all':   {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []}
    }

    print(f"Comparing {len(test_files)} basins (Per-Tile Analysis)...")
    
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
                continue

            # Load
            z_spun = zarr.open(spun_path, mode='r')
            spun_10 = z_spun['spun_scf'][:]
            gt_10   = z_spun['label'][:] 
            ndsi_10 = (zarr.open(ndsi_path, mode='r')['ndsi'][:] > NDSI_THRESH).astype(np.float32) * 100.0
            wang_10 = zarr.open(wang_path, mode='r')['wang_scf'][:]

            # Downscale everything to 20m for Operational Comparison
            spun_20 = downscale_label_masked(spun_10)
            gt_20   = downscale_label_masked(gt_10, invalid_val=LABEL_CLOUD)
            ndsi_20 = downscale_label_masked(ndsi_10, invalid_val=-999)
            wang_20 = downscale_label_masked(wang_10, invalid_val=-999)
            lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:]
            
            if spun_20.shape != lis_arr.shape: continue

            # --- CALCULATE SCA PER TILE (km2) ---
            
            # 1. Clear Sky Condition (Tile must be fully valid in LIS)
            # Relaxing strictness: Allow tile if >90% pixels are valid? 
            # For now, let's just mask pixels and sum valid ones.
            
            y_gt   = gt_20.flatten()
            y_lis  = lis_arr.flatten()
            
            # --- SCENARIO 1: CLEAR SKY (Pixel-wise Mask) ---
            # We calculate total SCA *only on clear pixels*
            m_cl = (y_gt != LABEL_CLOUD) & (y_lis != LIS_CLOUD) & (y_lis != LIS_NODATA) & (~np.isnan(y_lis))
            
            if m_cl.sum() > 0: # If at least 1 valid pixel
                # Calculate Sum of Snow Fractions (0-100) / 100 * Area
                # This gives "Snow Water Equivalent Area" (if density was 1)
                
                sca_gt   = (y_gt[m_cl].sum() / 100.0) * PIXEL_AREA_KM2
                sca_spun = (spun_20.flatten()[m_cl].sum() / 100.0) * PIXEL_AREA_KM2
                sca_lis  = (y_lis[m_cl].sum() / 100.0) * PIXEL_AREA_KM2
                sca_ndsi = (ndsi_20.flatten()[m_cl].sum() / 100.0) * PIXEL_AREA_KM2
                sca_wang = (wang_20.flatten()[m_cl].sum() / 100.0) * PIXEL_AREA_KM2
                
                # Only add if GT has SOME snow? Or include zeros for True Negatives?
                # Including zeros is important for "False Positive" analysis
                results['clear']['gt'].append(sca_gt)
                results['clear']['spun'].append(sca_spun)
                results['clear']['lis'].append(sca_lis)
                results['clear']['ndsi'].append(sca_ndsi)
                results['clear']['wang'].append(sca_wang)

            # --- SCENARIO 2: ALL SKY (Pixel-wise Mask) ---
            m_al = (y_gt != LABEL_CLOUD)
            if m_al.sum() > 0:
                lis_filled = np.nan_to_num(np.where(y_lis >= 200, 0, y_lis), nan=0.0)
                
                sca_gt   = (y_gt[m_al].sum() / 100.0) * PIXEL_AREA_KM2
                sca_spun = (spun_20.flatten()[m_al].sum() / 100.0) * PIXEL_AREA_KM2
                sca_lis  = (lis_filled[m_al].sum() / 100.0) * PIXEL_AREA_KM2
                sca_ndsi = (ndsi_20.flatten()[m_al].sum() / 100.0) * PIXEL_AREA_KM2
                sca_wang = (wang_20.flatten()[m_al].sum() / 100.0) * PIXEL_AREA_KM2
                
                results['all']['gt'].append(sca_gt)
                results['all']['spun'].append(sca_spun)
                results['all']['lis'].append(sca_lis)
                results['all']['ndsi'].append(sca_ndsi)
                results['all']['wang'].append(sca_wang)

        except Exception: continue

    with open(REPORT_OUTPUT, "w") as f:
        def log(t): print(t); f.write(t + "\n")
        
        row_fmt = "{:<12} | {:<10} | {:<12} | {:<12}"
        
        def print_table(title, data):
            log(f"\n{title}")
            log("="*60)
            log(row_fmt.format("Model", "R2 Score", "SCA Bias(km2)", "SCA MAE(km2)"))
            log("-" * 60)
            
            gt = np.array(data['gt'])
            if len(gt) == 0: return

            for k in ['spun', 'lis', 'ndsi', 'wang']:
                pred = np.array(data[k])
                
                # Metrics
                r2 = r2_score(gt, pred)
                bias = np.mean(pred - gt)
                mae = mean_absolute_error(gt, pred)
                
                log(row_fmt.format(k.upper(), f"{r2:.3f}", f"{bias:+.4f}", f"{mae:.4f}"))
            log("-" * 60)
            log(f"Average Basin Snow Area (GT): {gt.mean():.4f} km2")
            log(f"Number of Basins Evaluated:   {len(gt)}")

        print_table("TIER 2a: PER-BASIN HYDROLOGY (CLEAR SKY)", results['clear'])
        print_table("TIER 2b: PER-BASIN HYDROLOGY (ALL SKY)", results['all'])

    print(f"\nReport saved to: {os.path.abspath(REPORT_OUTPUT)}")

if __name__ == "__main__":
    main()
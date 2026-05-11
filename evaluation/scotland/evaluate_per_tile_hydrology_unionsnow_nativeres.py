import pandas as pd
import numpy as np
import os
import zarr
from tqdm import tqdm
from sklearn.metrics import r2_score

# --- CONFIGURATION ---
SPLIT_CSV_PATH       = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# DIRECTORIES
LIS_PATCHES_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
NDSI_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_ndsi'
WANG_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_wang2022'

REPORT_OUTPUT        = 'final_hydrology_native_report.txt'

# --- PARAMETERS ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255
NDSI_THRESH = 0.4 

# Area Factors (km2)
AREA_10M = (10 * 10) / 1_000_000.0  # 0.0001 km2
AREA_20M = (20 * 20) / 1_000_000.0  # 0.0004 km2

# --- HELPERS ---
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
    
    # Storage (Lists of Areas in km2)
    # Clear Sky
    res_cl = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []}
    # All Sky
    res_al = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []}

    print(f"Analyzing {len(test_files)} basins at NATIVE RESOLUTIONS...")
    
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

            # 1. Load NATIVE 10m Data
            z_spun = zarr.open(spun_path, mode='r')
            spun_10 = z_spun['spun_scf'][:]
            gt_10   = z_spun['label'][:] 
            ndsi_10 = (zarr.open(ndsi_path, mode='r')['ndsi'][:] > NDSI_THRESH).astype(np.float32) * 100.0
            wang_10 = zarr.open(wang_path, mode='r')['wang_scf'][:]
            
            # 2. Load NATIVE 20m Data (LIS)
            lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:]

            # --- FILTER MASKS ---
            y_gt_10 = gt_10.flatten()
            y_lis   = lis_arr.flatten()
            
            # Check LIS validity (percentage of valid pixels)
            # We can't do pixel-wise matching between 10m and 20m easily without resizing.
            # STRATEGY: 
            # 1. Calculate Area for LIS on valid LIS pixels.
            # 2. Calculate Area for GT/SPUN on corresponding valid area (implies resizing GT to check mask).
            
            # To be strictly fair, we must use the LIS mask to exclude clouds from GT/SPUN calculation
            # Upscale LIS Mask to 10m
            lis_upscaled = np.repeat(np.repeat(lis_arr, 2, axis=0), 2, axis=1)
            
            # --- SCENARIO 1: CLEAR SKY ---
            # Valid if GT is not Cloud AND LIS is not Cloud
            m_cl = (gt_10 != LABEL_CLOUD) & (lis_upscaled != LIS_CLOUD) & (lis_upscaled != LIS_NODATA) & (~np.isnan(lis_upscaled))
            
            if m_cl.sum() > 0:
                # Calculate Areas (Sum of fractions * Pixel Area)
                # SPUN/GT/NDSI/Wang use 10m Area
                a_gt   = (gt_10[m_cl].sum()   / 100.0) * AREA_10M
                a_spun = (spun_10[m_cl].sum() / 100.0) * AREA_10M
                a_ndsi = (ndsi_10[m_cl].sum() / 100.0) * AREA_10M
                a_wang = (wang_10[m_cl].sum() / 100.0) * AREA_10M
                
                # LIS uses 20m Area (but we used upscaled mask, so essentially we treat upscaled LIS as 10m for sum)
                # Ideally: Sum(LIS_Original) where valid. 
                # Simpler: Sum(LIS_Upscaled[m_cl]) * AREA_10M. It is mathematically equivalent.
                a_lis = (lis_upscaled[m_cl].sum() / 100.0) * AREA_10M
                
                res_cl['gt'].append(a_gt)
                res_cl['spun'].append(a_spun)
                res_cl['lis'].append(a_lis)
                res_cl['ndsi'].append(a_ndsi)
                res_cl['wang'].append(a_wang)

            # --- SCENARIO 2: ALL SKY ---
            # Valid if GT is not Cloud (LIS clouds become 0)
            m_al = (gt_10 != LABEL_CLOUD)
            
            if m_al.sum() > 0:
                # Fill LIS clouds with 0
                lis_filled = np.where(lis_upscaled >= 200, 0, lis_upscaled)
                lis_filled = np.nan_to_num(lis_filled, nan=0.0)
                
                a_gt   = (gt_10[m_al].sum()   / 100.0) * AREA_10M
                a_spun = (spun_10[m_al].sum() / 100.0) * AREA_10M
                a_ndsi = (ndsi_10[m_al].sum() / 100.0) * AREA_10M
                a_wang = (wang_10[m_al].sum() / 100.0) * AREA_10M
                a_lis  = (lis_filled[m_al].sum() / 100.0) * AREA_10M # LIS treating clouds as 0
                
                res_al['gt'].append(a_gt)
                res_al['spun'].append(a_spun)
                res_al['lis'].append(a_lis)
                res_al['ndsi'].append(a_ndsi)
                res_al['wang'].append(a_wang)

        except Exception: continue

    with open(REPORT_OUTPUT, "w") as f:
        def log(t): print(t); f.write(t + "\n")
        
        row_fmt = "{:<12} | {:<10} | {:<15} | {:<15}"
        
        def print_table(title, data):
            gt = np.array(data['gt'])
            if len(gt) == 0: return

            # --- FILTER: SNOW-RELEVANT (UNION > 0.0001 km2) ---
            # Roughly 1 pixel of snow
            threshold = 0.0001
            mask_active = (gt > threshold)
            for k in ['spun', 'lis', 'ndsi', 'wang']:
                mask_active |= (np.array(data[k]) > threshold)
            
            gt_filt = gt[mask_active]
            
            if len(gt_filt) > 0:
                log(f"\n{title}")
                log(f"Snow-Relevant Basins: {len(gt_filt)}")
                log("="*65)
                log(row_fmt.format("Model", "R2 Score", "Total Vol(km2)", "Vol Deviation"))
                log("-" * 65)
                
                total_gt_vol = gt_filt.sum()
                log(f"{'Ground Truth':<12} | {'Ref':<10} | {total_gt_vol:<15.4f} | {'Ref':<15}")
                log("-" * 65)
                
                for k in ['spun', 'lis', 'ndsi', 'wang']:
                    pred = np.array(data[k])[mask_active]
                    
                    # 1. Correlation (Per-Basin fit)
                    r2 = r2_score(gt_filt, pred)
                    
                    # 2. Total Volume (Aggregate)
                    total_pred_vol = pred.sum()
                    
                    # 3. Deviation %
                    dev_pct = ((total_pred_vol - total_gt_vol) / total_gt_vol) * 100.0
                    
                    log(row_fmt.format(k.upper(), f"{r2:.3f}", f"{total_pred_vol:.4f}", f"{dev_pct:+.1f}%"))
                log("-" * 65)

        print_table("HYDROLOGY: CLEAR-SKY CONTROL (Native Res)", res_cl)
        print_table("HYDROLOGY: ALL-SKY REALITY (Native Res)", res_al)

    print(f"\nReport saved to: {os.path.abspath(REPORT_OUTPUT)}")

if __name__ == "__main__":
    main()
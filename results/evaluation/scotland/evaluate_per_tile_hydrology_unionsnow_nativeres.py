import pandas as pd
import numpy as np
import os
import argparse
import zarr
from tqdm import tqdm
from sklearn.metrics import r2_score

# --- PARAMETERS ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255
NDSI_THRESH = 0.4 

# Area Factors (km2)
AREA_10M = (10 * 10) / 1_000_000.0  # 0.0001 km2

def get_args():
    parser = argparse.ArgumentParser(description="Generate hydrology evaluation report (basin-level areas).")
    
    # Dataset Metadata
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the Zenodo dataset')
    parser.add_argument('--split_file', type=str, default='dataset_metadata.csv', help='Name of the metadata CSV')
    
    # Predictions Directories
    parser.add_argument('--spun_dir', type=str, required=True, help='Directory containing SPUN prediction Zarrs')
    parser.add_argument('--lis_dir', type=str, required=True, help='Directory containing LIS patches')
    parser.add_argument('--ndsi_dir', type=str, required=True, help='Directory containing NDSI Zarrs')
    parser.add_argument('--wang_dir', type=str, required=True, help='Directory containing Wang2022 Zarrs')
    
    # Output
    parser.add_argument('--output_report', type=str, default='final_hydrology_native_report.txt', help='Path to save the output text report')
    
    return parser.parse_args()

# --- HELPERS ---
def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

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
    
    # Storage (Lists of Areas in km2)
    res_cl = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []}
    res_al = {'gt': [], 'spun': [], 'lis': [], 'ndsi': [], 'wang': []}

    print(f"Analyzing {len(test_files)} chips at NATIVE RESOLUTIONS...")
    
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

            # 1. Load NATIVE 10m Data
            z_spun = zarr.open(spun_path, mode='r')
            spun_10 = z_spun['spun_scf'][:]
            gt_10   = z_spun['label'][:] 
            ndsi_10 = (zarr.open(ndsi_path, mode='r')['ndsi'][:] > NDSI_THRESH).astype(np.float32) * 100.0
            wang_10 = zarr.open(wang_path, mode='r')['wang_scf'][:]
            
            # 2. Load NATIVE 20m Data (LIS)
            lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:]

            # --- FILTER MASKS ---
            lis_upscaled = np.repeat(np.repeat(lis_arr, 2, axis=0), 2, axis=1)
            
            # --- SCENARIO 1: CLEAR SKY ---
            m_cl = (gt_10 != LABEL_CLOUD) & (lis_upscaled != LIS_CLOUD) & (lis_upscaled != LIS_NODATA) & (~np.isnan(lis_upscaled))
            
            if m_cl.sum() > 0:
                a_gt   = (gt_10[m_cl].sum()   / 100.0) * AREA_10M
                a_spun = (spun_10[m_cl].sum() / 100.0) * AREA_10M
                a_ndsi = (ndsi_10[m_cl].sum() / 100.0) * AREA_10M
                a_wang = (wang_10[m_cl].sum() / 100.0) * AREA_10M
                a_lis = (lis_upscaled[m_cl].sum() / 100.0) * AREA_10M
                
                res_cl['gt'].append(a_gt)
                res_cl['spun'].append(a_spun)
                res_cl['lis'].append(a_lis)
                res_cl['ndsi'].append(a_ndsi)
                res_cl['wang'].append(a_wang)

            # --- SCENARIO 2: ALL SKY ---
            m_al = (gt_10 != LABEL_CLOUD)
            
            if m_al.sum() > 0:
                lis_filled = np.where(lis_upscaled >= 200, 0, lis_upscaled)
                lis_filled = np.nan_to_num(lis_filled, nan=0.0)
                
                a_gt   = (gt_10[m_al].sum()   / 100.0) * AREA_10M
                a_spun = (spun_10[m_al].sum() / 100.0) * AREA_10M
                a_ndsi = (ndsi_10[m_al].sum() / 100.0) * AREA_10M
                a_wang = (wang_10[m_al].sum() / 100.0) * AREA_10M
                a_lis  = (lis_filled[m_al].sum() / 100.0) * AREA_10M 
                
                res_al['gt'].append(a_gt)
                res_al['spun'].append(a_spun)
                res_al['lis'].append(a_lis)
                res_al['ndsi'].append(a_ndsi)
                res_al['wang'].append(a_wang)

        except Exception: continue

    out_dir = os.path.dirname(args.output_report)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output_report, "w") as f:
        def log(t): print(t); f.write(t + "\n")
        
        row_fmt = "{:<12} | {:<10} | {:<15} | {:<15}"
        
        def print_table(title, data):
            gt = np.array(data['gt'])
            if len(gt) == 0: return

            threshold = 0.0001
            mask_active = (gt > threshold)
            for k in ['spun', 'lis', 'ndsi', 'wang']:
                mask_active |= (np.array(data[k]) > threshold)
            
            gt_filt = gt[mask_active]
            
            if len(gt_filt) > 0:
                log(f"\n{title}")
                log(f"Snow-Relevant Chips: {len(gt_filt)}")
                log("="*65)
                log(row_fmt.format("Model", "R2 Score", "Total Vol(km2)", "Vol Deviation"))
                log("-" * 65)
                
                total_gt_vol = gt_filt.sum()
                log(f"{'Ground Truth':<12} | {'Ref':<10} | {total_gt_vol:<15.4f} | {'Ref':<15}")
                log("-" * 65)
                
                for k in ['spun', 'lis', 'ndsi', 'wang']:
                    pred = np.array(data[k])[mask_active]
                    r2 = r2_score(gt_filt, pred)
                    total_pred_vol = pred.sum()
                    dev_pct = ((total_pred_vol - total_gt_vol) / total_gt_vol) * 100.0
                    log(row_fmt.format(k.upper(), f"{r2:.3f}", f"{total_pred_vol:.4f}", f"{dev_pct:+.1f}%"))
                log("-" * 65)

        print_table("HYDROLOGY: CLEAR-SKY CONTROL (Native Res)", res_cl)
        print_table("HYDROLOGY: ALL-SKY REALITY (Native Res)", res_al)

    print(f"\nReport saved to: {os.path.abspath(args.output_report)}")

if __name__ == "__main__":
    main()
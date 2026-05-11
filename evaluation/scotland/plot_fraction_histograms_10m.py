import pandas as pd
import numpy as np
import os
import zarr
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
SPLIT_CSV_PATH       = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# DIRECTORIES
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
LIS_PATCHES_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'

OUTPUT_PLOT_PATH     = 'snow_fraction_distribution_10m_native.png'

# --- PARAMETERS ---
LABEL_CLOUD = 200

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
    
    # Storage for Native 10m Data
    data_10m = {'gt': [], 'spun': []}

    print(f"Scanning {len(test_files)} tiles at NATIVE 10m resolution...")
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            
            spun_path = os.path.join(SPUN_PREDICTIONS_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
            lis_path  = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
            
            if not (os.path.exists(spun_path) and os.path.exists(lis_path)):
                continue

            # --- LOAD 10m DATA (No Downscaling) ---
            z_spun   = zarr.open(spun_path, mode='r')
            spun_10m = z_spun['spun_scf'][:]
            gt_10m   = z_spun['label'][:] 

            # Flatten
            y_gt   = gt_10m.flatten()
            y_spun = spun_10m.flatten()

            # --- FILTER: ALL SKY ---
            mask_valid = (y_gt != LABEL_CLOUD)
            
            if mask_valid.any():
                data_10m['gt'].append(y_gt[mask_valid])
                data_10m['spun'].append(y_spun[mask_valid])

        except Exception:
            continue

    print("Generating 10m Histogram...")
    
    if not data_10m['gt']:
        print("No valid data found!")
        return

    gt_all   = np.concatenate(data_10m['gt'])
    spun_all = np.concatenate(data_10m['spun'])

    # --- FILTER: Only look at "Snow Presence" (> 1%) ---
    gt_snow   = gt_all[gt_all >= 1.0]
    spun_snow = spun_all[spun_all >= 1.0]

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    
    bins = 50
    plot_range = (1, 100)
    
    plt.hist(gt_snow, bins=bins, range=plot_range, density=True, 
             alpha=0.5, color='black', label=f'Ground Truth (Native 10m)\nN={len(gt_snow):,}')
    
    plt.hist(spun_snow, bins=bins, range=plot_range, density=True, 
             alpha=0.5, color='red', label=f'SPUN Prediction (Native 10m)\nN={len(spun_snow):,}')

    plt.title("Intrinsic Snow Fraction Distribution (10 m Native Resolution)", fontsize=14, fontweight='bold')
    plt.xlabel("Snow Fraction (%)")
    plt.ylabel("Probability Density")
    
    # 1. Force Legend to Upper Right
    plt.legend(loc='upper center')
    
    plt.grid(True, alpha=0.3)
    
    # 2. Move Stats Text to Upper Left (Away from Legend)
    stats_text = (f"GT Mean: {gt_snow.mean():.1f}%\n"
                  f"SPUN Mean: {spun_snow.mean():.1f}%")
    
    plt.gca().text(0.05, 0.90, stats_text, transform=plt.gca().transAxes, 
                   ha='left', va='top', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
    print(f"Histogram saved to: {os.path.abspath(OUTPUT_PLOT_PATH)}")
    plt.close()

if __name__ == "__main__":
    main()
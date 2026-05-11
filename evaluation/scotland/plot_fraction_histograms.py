import pandas as pd
import numpy as np
import os
import zarr
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
SPLIT_CSV_PATH       = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# DATA DIRECTORIES
LIS_PATCHES_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'

OUTPUT_PLOT_PATH     = 'snow_fraction_distribution.png'

# --- PARAMETERS ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255

# --- HELPER FUNCTIONS ---

def downscale_label_masked(arr, factor=2, invalid_val=200):
    """Downscales 10m -> 20m."""
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
    
    # Storage
    clear_sky_data = {'gt': [], 'spun': []}
    all_sky_data   = {'gt': [], 'spun': []}

    print(f"Scanning {len(test_files)} tiles...")
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            
            spun_path = os.path.join(SPUN_PREDICTIONS_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
            lis_path  = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
            
            if not (os.path.exists(spun_path) and os.path.exists(lis_path)):
                continue

            # Load Data
            lis_arr  = zarr.open(lis_path, mode='r')['lis_snow'][:]
            z_spun   = zarr.open(spun_path, mode='r')
            spun_10m = z_spun['spun_scf'][:]
            gt_10m   = z_spun['label'][:] 

            # Downscale
            spun_20 = downscale_label_masked(spun_10m, factor=2, invalid_val=-999)
            gt_20   = downscale_label_masked(gt_10m, factor=2, invalid_val=LABEL_CLOUD)

            if spun_20.shape != lis_arr.shape: continue

            # Flatten
            y_gt   = gt_20.flatten()
            y_spun = spun_20.flatten()
            y_lis  = lis_arr.flatten()

            # --- 1. CLEAR SKY FILTER ---
            # Valid GT + Valid LIS (No Cloud/NoData)
            mask_clear = (y_gt != LABEL_CLOUD) & (y_lis != LIS_CLOUD) & (y_lis != LIS_NODATA) & (~np.isnan(y_lis))
            if mask_clear.any():
                clear_sky_data['gt'].append(y_gt[mask_clear])
                clear_sky_data['spun'].append(y_spun[mask_clear])

            # --- 2. ALL SKY FILTER ---
            # Valid GT only (Cloudy LIS pixels included)
            mask_all = (y_gt != LABEL_CLOUD)
            if mask_all.any():
                all_sky_data['gt'].append(y_gt[mask_all])
                all_sky_data['spun'].append(y_spun[mask_all])

        except Exception:
            continue

    print("Generating Plots...")
    
    # Setup Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    def plot_histogram(ax, data_dict, title):
        if not data_dict['gt']:
            ax.text(0.5, 0.5, "No Data", ha='center')
            return

        # Concatenate all tiles
        gt_all   = np.concatenate(data_dict['gt'])
        spun_all = np.concatenate(data_dict['spun'])

        # FILTER: Only look at pixels where Snow > 1%
        # We want to compare the distribution of DETECTED snow
        gt_snow   = gt_all[gt_all >= 1.0]
        spun_snow = spun_all[spun_all >= 1.0]

        # Plot Histograms
        # Density=True normalizes the area to 1 so we can compare shapes despite count differences
        ax.hist(gt_snow, bins=50, range=(1, 100), density=True, 
                alpha=0.6, color='black', label=f'Ground Truth (N={len(gt_snow):,})')
        
        ax.hist(spun_snow, bins=50, range=(1, 100), density=True, 
                alpha=0.6, color='red', label=f'SPUN Prediction (N={len(spun_snow):,})')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Snow Fraction (%)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate stats for annotation
        ax.text(0.5, 0.9, f"GT Mean: {gt_snow.mean():.1f}%\nSPUN Mean: {spun_snow.mean():.1f}%", 
                transform=ax.transAxes, ha='center', bbox=dict(facecolor='white', alpha=0.8))

    # Plot 1: Clear Sky
    plot_histogram(axes[0], clear_sky_data, "Scenario 1: Clear-Sky Distribution")
    
    # Plot 2: All Sky
    plot_histogram(axes[1], all_sky_data, "Scenario 2: All-Sky Distribution")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=150)
    print(f"Histogram saved to: {os.path.abspath(OUTPUT_PLOT_PATH)}")
    plt.close()

if __name__ == "__main__":
    main()
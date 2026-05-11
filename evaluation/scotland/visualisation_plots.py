import pandas as pd
import numpy as np
import os
import zarr
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

# --- CONFIGURATION ---
SPLIT_CSV_PATH       = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# INPUT DIRECTORIES
LIS_PATCHES_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
NDSI_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_ndsi'
WANG_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_wang2022'

# OUTPUT DIRECTORY
OUTPUT_DIR           = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/comparison_visualisations'

# PARAMETERS
NDSI_THRESH = 0.4
ALPHA = 0.5  # Transparency for both snow and clouds

# --- HELPERS ---

def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

def normalize_rgb(arr):
    """Normalizes RGB values to 0-1 range for plotting."""
    if arr.max() > 0:
        return np.clip(arr / 5000, 0, 1) # 3000 is standard for S2 visualization
    return arr

def load_rgb(zarr_path):
    """
    Loads RGB and flips vertically to match the geometric correction 
    applied to the predictions/snowmaps.
    """
    try:
        with xr.open_zarr(zarr_path) as ds:
            r = ds['B4'].values
            g = ds['B3'].values
            b = ds['B2'].values
            rgb = np.stack([r, g, b], axis=-1)
            
            # Vertical Flip to align with predictions
            rgb = rgb[::-1, :, :] 
            
            return normalize_rgb(rgb)
    except Exception as e:
        return np.zeros((100, 100, 3))

def upsample_lis(lis_arr):
    """Simple nearest-neighbor upsampling for LIS (50x50 -> 100x100)."""
    return np.repeat(np.repeat(lis_arr, 2, axis=0), 2, axis=1)

def plot_heatmap_overlay(ax, rgb, data, cloud_mask, title):
    """
    Plots RGB image with:
    1. Viridis Heatmap for Snow (0=Purple, 100=Yellow)
    2. Grey Overlay for Clouds
    """
    # 1. Plot Background (RGB)
    ax.imshow(rgb)
    
    # 2. Plot Snow (Viridis)
    if data is not None:
        # Normalize 0-100 -> 0-1
        norm_data = np.clip(data / 100.0, 0, 1)
        cmap = plt.cm.viridis
        overlay_rgba = cmap(norm_data)
        
        # Set global alpha
        overlay_rgba[..., 3] = ALPHA
        
        # --- LOGIC CHANGE: Masking ---
        # If we have a cloud mask, make the SNOW layer transparent there
        # so we don't blend Purple (Snow=0) + Grey (Cloud)
        if cloud_mask is not None:
            is_cloud = cloud_mask > 0
            overlay_rgba[is_cloud, 3] = 0.0
            
        # Also mask NaNs if present
        is_nan = np.isnan(data)
        overlay_rgba[is_nan, 3] = 0.0
        
        ax.imshow(overlay_rgba)

    # 3. Plot Clouds (Grey)
    if cloud_mask is not None:
        # Create a grey RGBA array
        # Grey color: (0.7, 0.7, 0.7)
        grey_rgba = np.zeros((*cloud_mask.shape, 4))
        cloud_pixels = cloud_mask > 0
        
        grey_rgba[cloud_pixels, 0:3] = 0.7 # Light Grey
        grey_rgba[cloud_pixels, 3]   = ALPHA # Same transparency as snow
        
        ax.imshow(grey_rgba)
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Saving visualisations to: {OUTPUT_DIR}")

    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    print(f"Scanning {len(test_files)} tiles for snow...")
    
    count = 0
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            
            # 1. Construct Paths
            spun_path = os.path.join(SPUN_PREDICTIONS_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
            lis_path  = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
            ndsi_path = os.path.join(NDSI_DIR, f"{date_id}_{tile_id}_NDSI.zarr")
            wang_path = os.path.join(WANG_DIR, f"{date_id}_{tile_id}_Wang2022.zarr")

            if not (os.path.exists(spun_path) and os.path.exists(lis_path)):
                continue

            # 2. Load Data
            rgb_img = load_rgb(label_path)
            
            # Ground Truth
            z_spun = zarr.open(spun_path, mode='r')
            gt_raw = z_spun['label'][:]
            
            # Extract GT Clouds (>100 usually means cloud/nodata in this dataset)
            gt_cloud_mask = (gt_raw > 100).astype(int)
            # Clean GT Snow: Set cloud values to 0 (Purple) for the snow layer
            # (Note: They will be masked out by the cloud layer anyway due to logic above)
            gt_snow = np.where(gt_raw > 100, 0, gt_raw)

            # SPUN
            spun_snow = z_spun['spun_scf'][:]
            
            # LIS
            z_lis = zarr.open(lis_path, mode='r')
            lis_raw_20m = z_lis['lis_snow'][:]
            
            # Upsample LIS (50x50 -> 100x100)
            lis_raw = upsample_lis(lis_raw_20m)
            
            # Extract LIS Clouds (205=Cloud, 255=NoData)
            lis_cloud_mask = (lis_raw >= 200).astype(int)
            # Clean LIS Snow -> 0
            lis_snow = np.where(lis_raw >= 200, 0, lis_raw)

            # NDSI
            if os.path.exists(ndsi_path):
                z_ndsi = zarr.open(ndsi_path, mode='r')
                ndsi_raw = z_ndsi['ndsi'][:]
                ndsi_snow = (ndsi_raw > NDSI_THRESH).astype(float) * 100
            else:
                ndsi_snow = np.zeros_like(spun_snow)

            # Wang
            if os.path.exists(wang_path):
                z_wang = zarr.open(wang_path, mode='r')
                wang_snow = z_wang['wang_scf'][:]
            else:
                wang_snow = np.zeros_like(spun_snow)

            # 3. Filter Empty Tiles
            # Check if any snow is present (ignoring clouds)
            has_snow = (gt_snow.max() > 0) or (spun_snow.mean() > 5) or (lis_snow.max() > 0)
            if not has_snow:
                continue

            # 4. Plotting (2x3 Grid)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1
            axes[0,0].imshow(rgb_img)
            axes[0,0].set_title(f"RGB (True Color)\n{date_id}_{tile_id}", fontsize=10)
            axes[0,0].axis('off')
            
            # Plot with explicit Cloud Masks for GT and LIS
            plot_heatmap_overlay(axes[0,1], rgb_img, gt_snow, gt_cloud_mask, "Ground Truth")
            plot_heatmap_overlay(axes[0,2], rgb_img, spun_snow, None, "SPUN (Ours)")
            
            # Row 2
            plot_heatmap_overlay(axes[1,0], rgb_img, lis_snow, lis_cloud_mask, "LIS (Operational)")
            plot_heatmap_overlay(axes[1,1], rgb_img, ndsi_snow, None, f"NDSI > {NDSI_THRESH}")
            plot_heatmap_overlay(axes[1,2], rgb_img, wang_snow, None, "Wang2022")
            
            plt.tight_layout()
            
            # 5. Save
            out_name = f"{date_id}_{tile_id}_compare.png"
            plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=150)
            plt.close(fig)
            
            count += 1
            # if count >= 50: break 

        except Exception as e:
            # print(f"Error on {label_path}: {e}")
            continue

    print(f"Generated {count} visualisations in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import os
import zarr
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
SPLIT_CSV_PATH       = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# INPUT DIRECTORIES
LIS_PATCHES_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
NDSI_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_ndsi'
WANG_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_wang2022'

# OUTPUT DIRECTORY
OUTPUT_DIR           = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/comparison_visualisations_strips'

# PARAMETERS
NDSI_THRESH = 0.4
ALPHA = 0.5 

# --- HELPERS ---

def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

def normalize_rgb(arr):
    if arr.max() > 0:
        return np.clip(arr / 3000, 0, 1) 
    return arr

def load_rgb(zarr_path):
    try:
        with xr.open_zarr(zarr_path) as ds:
            r = ds['B4'].values
            g = ds['B3'].values
            b = ds['B2'].values
            rgb = np.stack([r, g, b], axis=-1)
            rgb = rgb[::-1, :, :] # Flip
            return normalize_rgb(rgb)
    except Exception as e:
        return np.zeros((100, 100, 3))

def upsample_lis(lis_arr):
    return np.repeat(np.repeat(lis_arr, 2, axis=0), 2, axis=1)

def plot_heatmap_overlay(ax, rgb, data, cloud_mask, title):
    """Plots RGB with Viridis Snow + Grey Cloud overlay."""
    ax.imshow(rgb)
    
    # 1. Snow (Viridis)
    if data is not None:
        norm_data = np.clip(data / 100.0, 0, 1)
        cmap = plt.cm.viridis
        overlay_rgba = cmap(norm_data)
        overlay_rgba[..., 3] = ALPHA 
        
        # Mask snow if cloud exists
        if cloud_mask is not None:
            is_cloud = cloud_mask > 0
            overlay_rgba[is_cloud, 3] = 0.0
            
        # Mask NaNs
        is_nan = np.isnan(data)
        overlay_rgba[is_nan, 3] = 0.0
        
        ax.imshow(overlay_rgba)

    # 2. Clouds (Grey)
    if cloud_mask is not None:
        grey_rgba = np.zeros((*cloud_mask.shape, 4))
        cloud_pixels = cloud_mask > 0
        grey_rgba[cloud_pixels, 0:3] = 0.7 # Grey
        grey_rgba[cloud_pixels, 3]   = ALPHA 
        ax.imshow(grey_rgba)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Saving strips to: {OUTPUT_DIR}")

    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    print(f"Scanning {len(test_files)} tiles...")
    
    count = 0
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            
            spun_path = os.path.join(SPUN_PREDICTIONS_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
            lis_path  = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
            ndsi_path = os.path.join(NDSI_DIR, f"{date_id}_{tile_id}_NDSI.zarr")
            wang_path = os.path.join(WANG_DIR, f"{date_id}_{tile_id}_SPUN.zarr")

            if not (os.path.exists(spun_path) and os.path.exists(lis_path)): continue

            # --- LOAD DATA ---
            rgb_img = load_rgb(label_path)
            
            # Ground Truth
            z_spun = zarr.open(spun_path, mode='r')
            gt_raw = z_spun['label'][:]
            gt_cloud_mask = (gt_raw > 100).astype(int)
            gt_snow = np.where(gt_raw > 100, 0, gt_raw)

            # SPUN
            spun_snow = z_spun['spun_scf'][:]
            
            # LIS
            z_lis = zarr.open(lis_path, mode='r')
            lis_raw = upsample_lis(z_lis['lis_snow'][:])
            lis_cloud_mask = (lis_raw >= 200).astype(int)
            lis_snow = np.where(lis_raw >= 200, 0, lis_raw)

            # NDSI
            if os.path.exists(ndsi_path):
                ndsi_raw = zarr.open(ndsi_path, mode='r')['ndsi'][:]
                ndsi_snow = (ndsi_raw > NDSI_THRESH).astype(float) * 100
            else:
                ndsi_snow = np.zeros_like(spun_snow)

            # Wang
            wang_cloud = None
            if os.path.exists(wang_path):
                z_wang = zarr.open(wang_path, mode='r')
                wang_snow = z_wang['wang_scf'][:]
                if 'wang_cloud' in z_wang:
                    wang_cloud = z_wang['wang_cloud'][:]
            else:
                wang_snow = np.zeros_like(spun_snow)

            # Filter: Skip if image is empty of snow (optional)
            has_snow = (gt_snow.max() > 0) or (spun_snow.mean() > 5) or (lis_snow.max() > 0)
            if not has_snow: continue

            # --- PLOTTING (1x6 ROW) ---
            # figsize=(width, height). 24, 4 gives a nice aspect ratio for 6 squares.
            fig, axes = plt.subplots(1, 6, figsize=(24, 4))
            
            # 1. RGB
            axes[0].imshow(rgb_img)
            axes[0].set_title(f"RGB\n{date_id}_{tile_id}", fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # 2. Ground Truth
            plot_heatmap_overlay(axes[1], rgb_img, gt_snow, gt_cloud_mask, "Ground Truth")
            
            # 3. SPUN
            plot_heatmap_overlay(axes[2], rgb_img, spun_snow, None, "SPUN (Ours)")
            
            # 4. LIS
            plot_heatmap_overlay(axes[3], rgb_img, lis_snow, lis_cloud_mask, "LIS")
            
            # 5. NDSI
            plot_heatmap_overlay(axes[4], rgb_img, ndsi_snow, None, "NDSI")
            
            # 6. Wang
            plot_heatmap_overlay(axes[5], rgb_img, wang_snow, wang_cloud, "Wang2022")
            
            plt.tight_layout()
            
            # Save
            out_name = f"{date_id}_{tile_id}_strip.png"
            plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            count += 1
            # Remove limit if you want to generate all
            # if count >= 50: break 

        except Exception as e:
            continue

    print(f"Generated {count} strip visualisations in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
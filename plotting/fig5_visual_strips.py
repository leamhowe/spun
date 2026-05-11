import pandas as pd
import numpy as np
import os
import zarr
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# --- CONFIGURATION ---
SPLIT_CSV_PATH       = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# INPUT DIRECTORIES
LIS_PATCHES_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
NDSI_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_ndsi'
WANG_DIR             = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_wang2022'
OUTPUT_DIR           = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/comparison_visualisations_strips'

NDSI_THRESH = 0.4
ALPHA = 0.5 

# --- TARGET SCENARIOS FOR PUBLICATION ---
# List of (date_id, tile_id, Row_Label)
TARGET_TILES = [
    ("20170508", "NN3884", "1. Well Illuminated Snow"),
    ("20210621", "NO0098", "2. Patchy Snow, Clouds, Lake"),
    ("20200530", "NN4690", "3. Mild Atmospheric Distortion"),
    ("20200530", "NN3885", "4. Cloud Shadow Detection"),
    ("20210621", "NN9999", "5. Heavy Cloud, Patchy Views"),
    ("20170508", "NN4087", "6. Training Set Error - Identified"),
    ("20180625", "NN4577", "7. Rock Classified Correctly") 
    # Note: Row 6 and 7 use the same tile. If these were manually cropped in your 
    # previous version, you may need to add cropping logic to this script.
]

# --- HELPERS ---
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
        print(f"Failed to load RGB for {zarr_path}")
        return np.zeros((100, 100, 3))

def upsample_lis(lis_arr):
    return np.repeat(np.repeat(lis_arr, 2, axis=0), 2, axis=1)

def plot_heatmap_overlay(ax, rgb, data, cloud_mask):
    """Plots RGB with Viridis Snow + Grey Cloud overlay."""
    ax.imshow(rgb)
    
    # 1. Snow (Viridis)
    if data is not None:
        norm_data = np.clip(data / 100.0, 0, 1)
        cmap = plt.cm.viridis
        overlay_rgba = cmap(norm_data)
        
        # Apply global alpha
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

    # Remove axis ticks for cleanliness
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False) 

    ax.set_axis_off()

# --- MAIN PLOTTING SCRIPT ---
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load CSV to look up raw RGB paths
    df = pd.read_csv(SPLIT_CSV_PATH)
    
    nrows = len(TARGET_TILES)
    ncols = 6
    
    # Create the massive figure (adjust figsize to tune aspect ratio)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3 * nrows))
    
    # Column Headers
    headers = ["Sentinel-2 RGB", "Label", "SPUN (Ours)", "LIS", "NDSI > 0.4", "Wang et al. (2022)"]
    for j, header in enumerate(headers):
        axes[0, j].set_title(header, fontsize=18, fontweight='bold', pad=15)

    for i, (date_id, tile_id, row_label) in enumerate(TARGET_TILES):
        print(f"Processing row {i+1}/{nrows}: {date_id}_{tile_id}...")
        
        # Lookup original RGB path
        matched_rows = df[df['filepath'].str.contains(f"{date_id}_{tile_id}", na=False)]
        label_path = matched_rows.iloc[0]['filepath'] if not matched_rows.empty else None
        
        spun_path = os.path.join(SPUN_PREDICTIONS_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
        lis_path  = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
        ndsi_path = os.path.join(NDSI_DIR, f"{date_id}_{tile_id}_NDSI.zarr")
        
        wang_path_v1 = os.path.join(WANG_DIR, f"{date_id}_{tile_id}_Wang2022.zarr")
        wang_path_v2 = os.path.join(WANG_DIR, f"{date_id}_{tile_id}_SPUN.zarr")
        wang_path = wang_path_v1 if os.path.exists(wang_path_v1) else (wang_path_v2 if os.path.exists(wang_path_v2) else None)

        # --- LOAD DATA ---
        rgb_img = load_rgb(label_path) if label_path else np.zeros((100, 100, 3))
        
        # Label
        if os.path.exists(spun_path):
            z_spun = zarr.open(spun_path, mode='r')
            gt_raw = z_spun['label'][:]
            gt_cloud_mask = (gt_raw > 100).astype(int)
            gt_snow = np.where(gt_raw > 100, 0, gt_raw)
            spun_snow = z_spun['spun_scf'][:]
        else:
            gt_snow, gt_cloud_mask, spun_snow = None, None, None

        # LIS
        if os.path.exists(lis_path):
            z_lis = zarr.open(lis_path, mode='r')
            lis_raw = upsample_lis(z_lis['lis_snow'][:])
            lis_cloud_mask = (lis_raw >= 200).astype(int)
            lis_snow = np.where(lis_raw >= 200, 0, lis_raw)
        else:
            lis_snow, lis_cloud_mask = None, None

        # NDSI
        if os.path.exists(ndsi_path):
            ndsi_raw = zarr.open(ndsi_path, mode='r')['ndsi'][:]
            ndsi_snow = (ndsi_raw > NDSI_THRESH).astype(float) * 100
        else:
            ndsi_snow = np.zeros((100, 100))

        # Wang
        wang_cloud = None
        if wang_path and os.path.exists(wang_path):
            z_wang = zarr.open(wang_path, mode='r')
            wang_snow = z_wang['wang_scf'][:]
            if 'wang_cloud' in z_wang:
                wang_cloud = z_wang['wang_cloud'][:]
        else:
            wang_snow = np.zeros((100, 100))

        # --- PLOTTING ROW ---
        # 1. RGB
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        for spine in axes[i, 0].spines.values():
            spine.set_visible(False)
        
        # Add the row label to the left of the RGB image
        # axes[i, 0].set_ylabel(f"{row_label}\n({date_id})", fontsize=14, fontweight='bold', rotation=90, labelpad=15)
        axes[i, 0].set_ylabel(f"{date_id}", fontsize=18, fontweight='bold', rotation=90, labelpad=15)
        
        # 2-6. Models
        plot_heatmap_overlay(axes[i, 1], rgb_img, gt_snow, gt_cloud_mask)
        plot_heatmap_overlay(axes[i, 2], rgb_img, spun_snow, None)
        plot_heatmap_overlay(axes[i, 3], rgb_img, lis_snow, lis_cloud_mask)
        plot_heatmap_overlay(axes[i, 4], rgb_img, ndsi_snow, None)
        plot_heatmap_overlay(axes[i, 5], rgb_img, wang_snow, wang_cloud)

    # Tighten up the spacing between images so it looks like a cohesive grid
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.15)
    
    # --- ADD SHARED COLORBAR ---
    # Create an axes specifically for the colorbar at the bottom center
    cbar_ax = fig.add_axes([0.3, 0.1, 0.4, 0.02]) # [left, bottom, width, height]
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                      cax=cbar_ax, orientation='horizontal')
    
    cb.set_label('Snow Cover Fraction (%)', fontsize=16, fontweight='bold', labelpad=10)
    cb.ax.tick_params(labelsize=12)
    
    # --- PROPER CUSTOM LEGEND ---
    # 1. Create a custom patch object
    cloud_patch = mpatches.Patch(color='grey', label='Cloud')
    
    # 2. Attach the legend directly to the colorbar axis
    # bbox_to_anchor=(1.02, 0.5) puts it just slightly to the right of the colorbar
    cbar_ax.legend(handles=[cloud_patch], 
                   loc='center left', 
                   bbox_to_anchor=(1.02, 0.5), 
                   frameon=False, 
                   prop={'size': 18, 'weight': 'bold'},
                   handlelength=1.5,  # Makes the box wider
                   handleheight=1.5)  # Makes the box taller)

    # --- SAVE ---
    out_name = os.path.join(OUTPUT_DIR, "Fig5_Visual_Strips.png")
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Success! Saved high-res figure to {out_name}")

if __name__ == "__main__":
    main()
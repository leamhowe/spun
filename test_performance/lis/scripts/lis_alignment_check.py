import pandas as pd
import numpy as np
import os
import zarr
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

# --- CONFIGURATION ---
SPLIT_CSV_PATH  = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# LIS Data Directory (20m resolution patches)
LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'

# Output Directory for Visualisations
OUTPUT_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/lis_alignment_check'

# Visualization Parameters
ALPHA = 0.6  # Transparency of the overlay

# --- HELPERS ---

def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

def normalize_rgb(arr):
    """Normalizes RGB values (fixed division by 3000 for consistency)."""
    if arr.max() > 0:
        return np.clip(arr / 3000, 0, 1) 
    return arr

def load_rgb(zarr_path):
    """
    Loads RGB bands (B4, B3, B2) and flips them vertically to match 
    the geometric correction applied to LIS/Prediction data.
    """
    try:
        with xr.open_zarr(zarr_path) as ds:
            r = ds['B4'].values
            g = ds['B3'].values
            b = ds['B2'].values
            rgb = np.stack([r, g, b], axis=-1)
            
            # --- CRITICAL: Vertical Flip ---
            rgb = rgb[::-1, :, :] 
            
            return normalize_rgb(rgb)
    except Exception as e:
        print(f"Failed to load RGB for {zarr_path}: {e}")
        return np.zeros((100, 100, 3))

def upsample_lis(lis_arr):
    """
    Upsamples LIS data (50x50 -> 100x100) using nearest neighbor 
    to match the Sentinel-2 10m grid.
    """
    # Repeat rows, then repeat columns
    return np.repeat(np.repeat(lis_arr, 2, axis=0), 2, axis=1)

def plot_alignment(save_path, rgb, lis_raw, title):
    """
    Plots the RGB background with LIS data overlaid.
    - Snow (0-100): Viridis colormap (Purple -> Yellow)
    - Cloud/NoData (>100): Grey
    - 0% Snow: Transparent
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Plot Background (RGB)
    ax.imshow(rgb)
    
    # 2. Process LIS Data
    # Separate Snow and Clouds
    lis_snow = np.where(lis_raw > 100, 0, lis_raw) # Set clouds to 0 for snow layer
    lis_cloud_mask = lis_raw > 100                 # Binary mask for clouds
    
    # 3. Plot Snow Layer (Viridis)
    if lis_snow.max() > 0:
        norm_data = np.clip(lis_snow / 100.0, 0, 1)
        cmap = plt.cm.viridis
        overlay_rgba = cmap(norm_data)
        
        # Transparency Logic
        overlay_rgba[..., 3] = ALPHA  # Set global alpha
        
        # Mask out 0% snow (make transparent)
        mask_nodata = (lis_snow <= 0)
        overlay_rgba[mask_nodata, 3] = 0.0
        
        # Mask out clouds (so they don't blend colors)
        overlay_rgba[lis_cloud_mask, 3] = 0.0
        
        ax.imshow(overlay_rgba)
        
    # 4. Plot Cloud Layer (Grey)
    if lis_cloud_mask.any():
        grey_rgba = np.zeros((*lis_cloud_mask.shape, 4))
        
        grey_rgba[lis_cloud_mask, 0:3] = 0.7  # Light Grey
        grey_rgba[lis_cloud_mask, 3]   = ALPHA # Same transparency
        
        ax.imshow(grey_rgba)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100) # dpi=100 is enough for a quick check
    plt.close(fig)

def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}")
    else:
        print(f"Saving to existing directory: {OUTPUT_DIR}")

    print("--- LOADING FILE LIST ---")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    print(f"Processing {len(test_files)} tiles...")
    
    count = 0
    skipped = 0
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            
            # Construct LIS Path
            lis_path = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
            
            if not os.path.exists(lis_path):
                skipped += 1
                continue

            # Load Data
            rgb_img = load_rgb(label_path)
            
            z_lis = zarr.open(lis_path, mode='r')
            lis_raw_20m = z_lis['lis_snow'][:]
            
            # Upsample LIS to match RGB
            lis_raw_100m = upsample_lis(lis_raw_20m)
            
            # Plot
            out_name = f"{date_id}_{tile_id}_lis_check.png"
            save_path = os.path.join(OUTPUT_DIR, out_name)
            
            plot_alignment(save_path, rgb_img, lis_raw_100m, f"LIS Alignment: {date_id}_{tile_id}")
            
            count += 1
            
        except Exception as e:
            print(f"Error on {label_path}: {e}")
            continue

    print(f"\nProcessing Complete.")
    print(f"Generated: {count} images")
    print(f"Skipped:   {skipped} (LIS file missing)")
    print(f"Output Dir: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
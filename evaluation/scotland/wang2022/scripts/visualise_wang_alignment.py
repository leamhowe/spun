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

# Wang Predictions Directory
WANG_DIR        = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_wang2022'

# Output Directory
OUTPUT_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/wang_alignment_check'

# Visualization Parameters
ALPHA = 0.6 

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
    """Loads RGB and flips vertically."""
    try:
        with xr.open_zarr(zarr_path) as ds:
            r = ds['B4'].values
            g = ds['B3'].values
            b = ds['B2'].values
            rgb = np.stack([r, g, b], axis=-1)
            # Vertical Flip matches prediction geometry
            rgb = rgb[::-1, :, :] 
            return normalize_rgb(rgb)
    except Exception as e:
        return np.zeros((100, 100, 3))

def plot_alignment(save_path, rgb, snow, cloud, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 1. Background
    ax.imshow(rgb)
    
    # 2. Snow (Viridis)
    if snow is not None and snow.max() > 0:
        norm_data = np.clip(snow / 100.0, 0, 1)
        cmap = plt.cm.viridis
        overlay_rgba = cmap(norm_data)
        
        overlay_rgba[..., 3] = ALPHA 
        
        # Mask 0% snow
        mask_nodata = (snow <= 0)
        overlay_rgba[mask_nodata, 3] = 0.0
        
        # Mask where clouds exist (if cloud mask is provided)
        if cloud is not None:
            is_cloud = cloud > 0
            overlay_rgba[is_cloud, 3] = 0.0
            
        ax.imshow(overlay_rgba)
        
    # 3. Clouds (Grey)
    if cloud is not None and cloud.max() > 0:
        grey_rgba = np.zeros((*cloud.shape, 4))
        is_cloud = cloud > 0
        
        grey_rgba[is_cloud, 0:3] = 0.7 # Grey
        grey_rgba[is_cloud, 3]   = ALPHA 
        
        ax.imshow(grey_rgba)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}")

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
            
            # Construct Wang Path
            wang_path = os.path.join(WANG_DIR, f"{date_id}_{tile_id}_Wang2022.zarr")
            
            if not os.path.exists(wang_path):
                skipped += 1
                continue

            # Load RGB
            rgb_img = load_rgb(label_path)
            
            # Load Wang Data
            z_wang = zarr.open(wang_path, mode='r')
            wang_scf = z_wang['wang_scf'][:]
            
            # Load Cloud (if exists)
            wang_cloud = None
            if 'wang_cloud' in z_wang:
                wang_cloud = z_wang['wang_cloud'][:]

            # Plot
            out_name = f"{date_id}_{tile_id}_wang_check.png"
            save_path = os.path.join(OUTPUT_DIR, out_name)
            
            plot_alignment(save_path, rgb_img, wang_scf, wang_cloud, f"Wang2022 Alignment: {date_id}_{tile_id}")
            
            count += 1
            
        except Exception as e:
            continue

    print(f"\nProcessing Complete.")
    print(f"Generated: {count} images")
    print(f"Skipped:   {skipped} (Wang file missing)")
    print(f"Output Dir: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
import os
import glob
import math
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & ARGUMENTS
# ==========================================

COLS = 7

# --- BRIGHTNESS SETTINGS ---
GAMMA = 0.65  # Lower value (e.g. 0.4-0.6) makes shadows brighter
SCALE_FACTOR = 1.0 # Keep 1.0 since your data is already 0-1. Increase if whole image is too dim.

def get_args():
    parser = argparse.ArgumentParser(description="Generate a grid plot of all Sentinel-2 RGB images")
    parser.add_argument('--s2_dir', type=str, required=True, help='Directory containing the _s2.tif files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the grid plot')
    parser.add_argument('--output_name', type=str, default='all_s2_images_grid.png', help='Filename for the output plot')
    return parser.parse_args()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_geotiff_as_array(path):
    with rasterio.open(path) as src:
        array = src.read()
    return array

def get_brightened_rgb(path):
    """
    Loads RGB (Bands 3,2,1) and applies Gamma Correction to brighten shadows.
    """
    try:
        # 1. Load data
        image_array = load_geotiff_as_array(path)
        image_array = image_array.transpose(1, 2, 0)
        
        if image_array.shape[2] >= 4:
            # 2. Extract RGB (Indices 3, 2, 1)
            rgb = np.stack([
                image_array[:, :, 3], 
                image_array[:, :, 2], 
                image_array[:, :, 1]
            ], axis=2)
            
            # 3. Clip to ensure 0-1 range (Safety step)
            rgb = np.clip(rgb * SCALE_FACTOR, 0, 1)
            
            # 4. Apply Gamma Correction
            # This boosts the dark pixels (shadows) significantly
            rgb_bright = np.power(rgb, GAMMA)
            
            return rgb_bright, False
        else:
            return image_array[:, :, 0], True
            
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return np.zeros((100, 100, 3)), False

def extract_date_from_filename(filepath):
    filename = os.path.basename(filepath)
    try:
        parts = filename.split('_')
        for part in parts:
            if 'T' in part and len(part) >= 8 and part[0].isdigit():
                return part.split('T')[0]
        return filename[:15]
    except IndexError:
        return filename[:15]

def plot_s2_grid(file_paths, save_path=None):
    n_images = len(file_paths)
    if n_images == 0:
        print("No images found!")
        return

    rows = math.ceil(n_images / COLS)
    
    fig, axs = plt.subplots(rows, COLS, figsize=(COLS * 2, rows * 2.2))
    axs = axs.flatten()

    print(f"Generating grid for {n_images} images with Gamma {GAMMA}...")

    for i, ax in enumerate(axs):
        if i < n_images:
            path = file_paths[i]
            
            # Load Brightened RGB
            img_data, is_gray = get_brightened_rgb(path)
            
            if is_gray:
                ax.imshow(img_data, cmap='gray')
            else:
                ax.imshow(img_data)
            
            title = extract_date_from_filename(path)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, wspace=0.05, hspace=0.2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid to {save_path}")
    
    plt.close()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    args = get_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")
    
    print(f"Searching for S2 files in: {args.s2_dir}")
    image_paths_pred = sorted(glob.glob(os.path.join(args.s2_dir, 'pred_*_s2.tif')))
    
    print(f"Found {len(image_paths_pred)} files.")
    
    out_path = os.path.join(args.output_dir, args.output_name)
    plot_s2_grid(image_paths_pred, save_path=out_path)
import os
import glob
import math
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Path to the directory containing the _s2.tif files (Your SPUN Model folder)
# INPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/Final_T15_dino_none_scale10000_best_mosaicked'
INPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/Final_T15_dino_none_scale10000_best'

# Search pattern for the S2 inputs
FILE_PATTERN = '*_s2.tif' 

OUTPUT_FILENAME = 'all_s2_inputs_grid.png'
GRID_COLS = 5

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def normalize_for_plot(img_array):
    """
    Normalizes a satellite image array (H, W, C) for visualization.
    Uses a 2%-98% percentile stretch to handle outliers and dark images.
    """
    img_array = np.nan_to_num(img_array)
    
    # Calculate robust min and max
    p2, p98 = np.percentile(img_array, (2, 98))
    
    if p98 == p2: return img_array
        
    img_norm = (img_array - p2) / (p98 - p2)
    img_norm = np.clip(img_norm, 0, 1)
    
    return img_norm

def load_s2_image(path):
    with rasterio.open(path) as src:
        # Check if it has enough bands (RGB)
        if src.count >= 3:
            # Read Bands 1, 2, 3 (Assuming RGB order)
            data = src.read([1, 2, 3])
            # Transpose to (H, W, 3)
            data = np.transpose(data, (1, 2, 0))
            return normalize_for_plot(data), False
        else:
            # Grayscale fallback
            data = src.read(1)
            return normalize_for_plot(data), True

def extract_date_title(filename):
    # Attempts to grab the date part '20200101' from filenames like 'pred_20200101_..._s2.tif'
    parts = filename.split('_')
    for part in parts:
        if part.isdigit() and len(part) == 8: # e.g. 20200101
            return part
    # Fallback: return first 15 chars
    return filename[:15]

# ==========================================
# 3. MAIN SCRIPT
# ==========================================

if __name__ == "__main__":
    
    # 1. Find files
    print(f"Searching for '{FILE_PATTERN}' in:\n{INPUT_DIR}")
    files = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN)))
    
    n_files = len(files)
    print(f"Found {n_files} images.")
    
    if n_files == 0:
        print("No files found. Check the directory and pattern.")
        exit()

    # 2. Setup Grid
    n_rows = math.ceil(n_files / GRID_COLS)
    fig, axs = plt.subplots(n_rows, GRID_COLS, figsize=(GRID_COLS * 3, n_rows * 3))
    
    # Flatten for iteration
    if n_files > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    # 3. Plot Loop
    for i, ax in enumerate(axs):
        if i < n_files:
            file_path = files[i]
            filename = os.path.basename(file_path)
            title = extract_date_title(filename)
            
            try:
                img_data, is_gray = load_s2_image(file_path)
                
                if is_gray:
                    ax.imshow(img_data, cmap='gray')
                else:
                    ax.imshow(img_data)
                    
                ax.set_title(title, fontsize=8, fontweight='bold')
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                ax.text(0.5, 0.5, "Error", ha='center', color='red')
            
            ax.axis('off')
        else:
            ax.axis('off')

    # 4. Save
    print("Saving grid...")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=200, bbox_inches='tight')
    print(f"Done! Saved to {OUTPUT_FILENAME}")
import os
import glob
import math
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================

RAW_IMAGES_DIR = '/home/users/leamhowe/phd/1_DETECT/Wang2022/datasets/raw_images'
OUTPUT_FILENAME = 'all_raw_images_grid.png'
GRID_COLS = 5  # Number of columns in the grid

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def normalize_for_plot(img_array):
    """
    Normalizes a satellite image array (H, W, C) for visualization.
    Uses a 2%-98% percentile stretch to handle outliers and dark images.
    """
    # Handle NaNs
    img_array = np.nan_to_num(img_array)
    
    # Calculate robust min and max
    p2, p98 = np.percentile(img_array, (2, 98))
    
    # Avoid division by zero
    if p98 == p2:
        return img_array
        
    # Stretch and clip
    img_norm = (img_array - p2) / (p98 - p2)
    img_norm = np.clip(img_norm, 0, 1)
    
    return img_norm

def load_image(path):
    """Loads a geotiff and prepares it for matplotlib."""
    with rasterio.open(path) as src:
        # Check how many bands we have
        if src.count >= 3:
            # Read RGB (Bands 1, 2, 3)
            # Note: Adjust order to [3, 2, 1] if your TIFs are BGR
            data = src.read([1, 2, 3])
            
            # Transpose from (Channels, Height, Width) to (Height, Width, Channels)
            data = np.transpose(data, (1, 2, 0))
            return normalize_for_plot(data), False # False = not grayscale
        else:
            # Read Single Band (Grayscale)
            data = src.read(1)
            return normalize_for_plot(data), True # True = grayscale

# ==========================================
# 3. MAIN SCRIPT
# ==========================================

if __name__ == "__main__":
    
    # 1. Find all TIFF files
    print(f"Searching in: {RAW_IMAGES_DIR}")
    files = sorted(glob.glob(os.path.join(RAW_IMAGES_DIR, '*.tif')))
    
    n_files = len(files)
    print(f"Found {n_files} images.")
    
    if n_files == 0:
        print("No images found. Check the directory path.")
        exit()

    # 2. Setup Grid
    n_rows = math.ceil(n_files / GRID_COLS)
    
    # Create a large figure size to accommodate many images
    fig, axs = plt.subplots(n_rows, GRID_COLS, figsize=(GRID_COLS * 4, n_rows * 4))
    
    # Flatten axs for easy iteration
    axs = axs.flatten()

    # 3. Plot Loop
    for i, ax in enumerate(axs):
        if i < n_files:
            file_path = files[i]
            filename = os.path.basename(file_path)
            
            # Extract a shorter title (e.g., just the date)
            title = filename.split('_')[0] 
            
            try:
                img_data, is_gray = load_image(file_path)
                
                if is_gray:
                    ax.imshow(img_data, cmap='gray')
                else:
                    ax.imshow(img_data)
                    
                ax.set_title(title, fontsize=9, fontweight='bold')
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                ax.text(0.5, 0.5, "Load Error", ha='center', va='center', color='red')
            
            ax.axis('off')
        else:
            # Hide empty subplots at the end
            ax.axis('off')

    # 4. Save
    print("Saving grid...")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=150, bbox_inches='tight')
    print(f"Done! Saved to {OUTPUT_FILENAME}")
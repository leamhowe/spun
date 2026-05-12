import os
import glob
import math
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ==========================================
# 1. CONFIGURATION & ARGUMENTS
# ==========================================

# Plotting Parameters
SNOW_THRESHOLD = 50.0  # Threshold for Snow vs Background
CLOUD_VALUE = 200      # Value for Cloud/NoData
COLS = 7               # 7 Columns for the grid

def get_args():
    parser = argparse.ArgumentParser(description="Generate a grid plot of all SPUN predictions")
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing the prediction TIFFs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the grid plot')
    parser.add_argument('--output_name', type=str, default='all_predictions_grid.png', help='Filename for the output plot')
    return parser.parse_args()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_geotiff(path):
    with rasterio.open(path) as src:
        return src.read(1) # Read the first band

def extract_date_from_filename(filepath):
    """Extracts YYYYMMDD from filenames like 'pred_20190110T...'"""
    filename = os.path.basename(filepath)
    try:
        # Split by '_' to get '20190110T...' then split by 'T'
        parts = filename.split('_')
        for part in parts:
            if 'T' in part and len(part) >= 8 and part[0].isdigit():
                return part.split('T')[0]
        return filename[:8]
    except IndexError:
        return filename[:8] 

def plot_prediction_grid(file_paths, save_path=None):
    n_images = len(file_paths)
    if n_images == 0:
        print("No images found to plot!")
        return

    rows = math.ceil(n_images / COLS)
    
    # Calculate figure size
    fig, axs = plt.subplots(rows, COLS, figsize=(COLS * 2, rows * 2.2))
    axs = axs.flatten() 

    # --- Define Custom Color Map ---
    # 0: Background (Black) -> val <= 50
    # 1: Snow (Cyan)       -> val > 50
    # 2: Cloud (Red)       -> val == 200
    cmap_colors = ['black', 'cyan', 'red']
    cmap = mcolors.ListedColormap(cmap_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5] 
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    print(f"Generating {COLS}x{rows} grid for {n_images} images...")

    for i, ax in enumerate(axs):
        if i < n_images:
            path = file_paths[i]
            try:
                data = load_geotiff(path)
                
                # --- Categorize Data ---
                categorical_map = np.zeros_like(data, dtype=np.int8)
                
                # 1. Background is already 0
                # 2. Set Snow (1)
                categorical_map[data > SNOW_THRESHOLD] = 1
                # 3. Set Clouds (2) - strict equality check
                categorical_map[data == CLOUD_VALUE] = 2
                
                # Plot
                ax.imshow(categorical_map, cmap=cmap, norm=norm, interpolation='nearest')
                
                # Styling
                title = extract_date_from_filename(path)
                ax.set_title(title, fontsize=9, fontweight='bold')
                ax.axis('off')
                ax.set_aspect('equal')
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
                ax.axis('off')
        else:
            # Hide empty slots
            ax.axis('off')

    # --- Legend ---
    legend_patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='cyan', label='Snow'),
        mpatches.Patch(color='red', label='Cloud')
    ]
    
    # Adjust legend position
    fig.legend(handles=legend_patches, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, wspace=0.05, hspace=0.2) # Adjust spacing

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
    
    print(f"Searching for prediction files in: {args.pred_dir}")
    
    # Find all 'pred_*.tif' files but exclude the '_s2' source files
    image_paths_s2 = glob.glob(os.path.join(args.pred_dir, 'pred_*_s2.tif')) 
    spun_paths = sorted([f.replace('_s2.tif', '.tif') for f in image_paths_s2])
    
    # Fallback search if the _s2 replacement trick doesn't work
    if not spun_paths:
        print("Fallback search...")
        candidates = glob.glob(os.path.join(args.pred_dir, 'pred_*.tif'))
        spun_paths = sorted([c for c in candidates if '_s2' not in c])

    print(f"Found {len(spun_paths)} files.")
    
    # Run the plot
    out_path = os.path.join(args.output_dir, args.output_name)
    plot_prediction_grid(spun_paths, save_path=out_path)
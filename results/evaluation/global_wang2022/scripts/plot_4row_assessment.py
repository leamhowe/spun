import os
import glob
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================

TARGET_FILES = [
    "20191123T111259_20191123T112151_T29RNQ_-8-3_31-22.tif",
    "20200713T103031_20200713T103026_T33VWH_15-31_60-46.tif",
    "20200804T223709_20200804T223712_T59GLM_169-47_-44-02.tif",
    "20200805T001109_20200805T001647_T55GCN_145-04_-42-94.tif",
    "20201126T041121_20201126T041842_T47SNT_100-09_33-93.tif",
    "20210714T081609_20210714T083805_T34HCJ_19-16_-33-13.tif"
]

OUTPUT_FILENAME = 'SPUN_4Row_Assessment_Final.png'
SCF_THRESHOLD = 50.0
NODATA_VAL = 200
GAMMA_VALUE = 0.65  # User selected optimal value

def get_args():
    parser = argparse.ArgumentParser(description="Generate 4-Row Visual Assessment Plot")
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing the Ground Truth mask TIFFs')
    parser.add_argument('--spun_dir', type=str, required=True, help='Directory containing SPUN prediction TIFFs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the plot')
    return parser.parse_args()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_geotiff_as_array(path):
    with rasterio.open(path) as src:
        array = src.read()  # Read all bands (Channels, Height, Width)
    return array

def get_optimized_s2_rgb(path, gamma=GAMMA_VALUE):
    """
    Loads S2 image using correct RGB bands [3, 2, 1] and applies Gamma 0.65.
    """
    try:
        # 1. Load data
        image_array = load_geotiff_as_array(path)
        
        # 2. Transpose to (Height, Width, Channels) for manipulation
        image_array = image_array.transpose(1, 2, 0)
        
        # 3. Check for at least 4 bands (Indices 0, 1, 2, 3)
        if image_array.shape[2] >= 4:
            # Extract RGB: Red=Index 3, Green=Index 2, Blue=Index 1
            rgb = np.stack([
                image_array[:, :, 3], 
                image_array[:, :, 2], 
                image_array[:, :, 1]
            ], axis=2)
            
            # 4. Clip to 0-1 range (Safety step)
            rgb = np.clip(rgb, 0, 1)
            
            # 5. Apply Gamma Correction (Brighten Shadows)
            rgb_bright = np.power(rgb, gamma)
            
            return rgb_bright, False
        else:
            # Fallback for grayscale/single band
            print(f"Warning: {os.path.basename(path)} has {image_array.shape[2]} bands. Using Band 1.")
            return image_array[:, :, 0], True
            
    except Exception as e:
        print(f"Error loading S2 {path}: {e}")
        return np.zeros((100, 100, 3)), False

def get_file_paths(gt_filename, base_dir):
    pred_path = os.path.join(base_dir, f"pred_{gt_filename}")
    if not os.path.exists(pred_path):
        unique_id = gt_filename.split('.')[0]
        candidates = glob.glob(os.path.join(base_dir, f"pred_*{unique_id}*.tif"))
        candidates = [c for c in candidates if '_s2' not in c]
        pred_path = candidates[0] if candidates else None

    s2_path = None
    if pred_path:
        candidate_s2 = pred_path.replace('.tif', '_s2.tif')
        if os.path.exists(candidate_s2):
             s2_path = candidate_s2

    return pred_path, s2_path

# ==========================================
# 3. MAIN PLOTTING SCRIPT
# ==========================================

def generate_4row_plot(gt_dir, spun_dir, output_dir):
    num_cols = len(TARGET_FILES)
    num_rows = 4 
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, 13))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # --- Colormaps ---
    cmap_scf = plt.get_cmap('viridis')
    cmap_scf.set_bad(color='red')

    # 0=Black(BG), 1=Cyan(Snow), 2=Red(Cloud)
    cmap_bin = mcolors.ListedColormap(['black', 'cyan', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm_bin = mcolors.BoundaryNorm(bounds, cmap_bin.N)

    print(f"Generating plot for {num_cols} tiles with Gamma {GAMMA_VALUE}...")

    for i, gt_filename in enumerate(TARGET_FILES):
        pred_path, s2_path = get_file_paths(gt_filename, spun_dir)
        gt_full_path = os.path.join(gt_dir, gt_filename)
        
        # =================================
        # ROW 1: Sentinel-2 (Optimized)
        # =================================
        ax_s2 = axs[0, i]
        if s2_path:
            img_data, is_gray = get_optimized_s2_rgb(s2_path)
            
            if is_gray:
                ax_s2.imshow(img_data, cmap='gray')
            else:
                ax_s2.imshow(img_data)
        else:
            ax_s2.text(0.5, 0.5, "Missing S2", ha='center')
        
        ax_s2.set_title(gt_filename.split('_')[0], fontsize=14, fontweight='bold')
        ax_s2.axis('off')

        # Load Prediction Data
        scf_data = None
        if pred_path and os.path.exists(pred_path):
            with rasterio.open(pred_path) as src:
                scf_data = src.read(1)

        # =================================
        # ROW 2: Predicted SCF
        # =================================
        ax_scf = axs[1, i]
        if scf_data is not None:
            is_cloud = (scf_data == NODATA_VAL) | np.isnan(scf_data)
            scf_masked = np.ma.masked_where(is_cloud, scf_data)
            ax_scf.imshow(scf_masked, cmap=cmap_scf, vmin=0, vmax=100, interpolation='nearest')
        else:
            ax_scf.text(0.5, 0.5, "Missing", ha='center')
        ax_scf.axis('off')

        # =================================
        # ROW 3: Predicted Binary
        # =================================
        ax_bin = axs[2, i]
        if scf_data is not None:
            binary_map = np.zeros_like(scf_data, dtype=np.int8)
            is_nodata = (scf_data == NODATA_VAL) | np.isnan(scf_data)
            is_snow = (~is_nodata) & (scf_data > SCF_THRESHOLD)
            
            binary_map[is_snow] = 1 
            binary_map[is_nodata] = 2 
            
            ax_bin.imshow(binary_map, cmap=cmap_bin, norm=norm_bin, interpolation='nearest')
        else:
            ax_bin.text(0.5, 0.5, "Missing", ha='center')
        ax_bin.axis('off')

        # =================================
        # ROW 4: Ground Truth
        # =================================
        ax_gt = axs[3, i]
        if os.path.exists(gt_full_path):
            try:
                gt_img = Image.open(gt_full_path).convert('L')
                gt_arr = np.array(gt_img)
                
                # Remap GT (1=BG, 3=Snow, 2=Cloud) -> Viz (0=BG, 1=Snow, 2=Cloud)
                gt_viz = np.zeros_like(gt_arr, dtype=np.int8)
                gt_viz[gt_arr == 1] = 0
                gt_viz[gt_arr == 3] = 1
                gt_viz[gt_arr == 2] = 2
                
                ax_gt.imshow(gt_viz, cmap=cmap_bin, norm=norm_bin, interpolation='nearest')
            except Exception as e:
                ax_gt.text(0.5, 0.5, "Error", ha='center')
        else:
            ax_gt.text(0.5, 0.5, "Missing GT", ha='center')
        ax_gt.axis('off')

    # --- Labels & Legends ---
    rows = ["Sentinel-2", "Predicted SCF", "Predicted Binary", "Label"]
    for idx, label in enumerate(rows):
        axs[idx, 0].text(-0.1, 0.5, label, transform=axs[idx, 0].transAxes, 
                         rotation=90, va='center', ha='right', fontsize=16, fontweight='bold')

    # Colorbar (SCF)
    cax = fig.add_axes([0.91, 0.51, 0.015, 0.17]) # [left, bottom, width, height] in figure coordinates
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 100), cmap=cmap_scf), cax=cax)
    cbar.set_label('Snow Fraction (%)', rotation=270, labelpad=15, fontsize=12)

    # Legend (Binary)
    patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='cyan', label='Snow'),
        mpatches.Patch(color='red', label='Cloud')
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.05))

    out_path = os.path.join(output_dir, OUTPUT_FILENAME)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")
    plt.close() # Close instead of show() to safely run in headless server environments

if __name__ == "__main__":
    args = get_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    generate_4row_plot(args.gt_dir, args.spun_dir, args.output_dir)
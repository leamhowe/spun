import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ==========================================
# 1. CONFIGURATION
# ==========================================

SPUN_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/Final_T15_dino_none_scale10000_best_mosaicked'

TARGET_FILES = [
    "20191123T111259_20191123T112151_T29RNQ_-8-3_31-22.tif",
    "20200713T103031_20200713T103026_T33VWH_15-31_60-46.tif",
    "20200804T223709_20200804T223712_T59GLM_169-47_-44-02.tif",
    "20200805T001109_20200805T001647_T55GCN_145-04_-42-94.tif",
    "20201126T041121_20201126T041842_T47SNT_100-09_33-93.tif",
    "20210714T081609_20210714T083805_T34HCJ_19-16_-33-13.tif"
]

OUTPUT_FILENAME = 'SPUN_3Row_Assessment.png'
SCF_THRESHOLD = 50.0
NODATA_VAL = 200

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def create_glacial_colormap():
    """
    Creates a custom colormap for Snow Fraction.
    0%   -> Black/Dark Grey (Bare Ground)
    50%  -> Cyan (Thin Snow)
    100% -> White (Deep Snow)
    """
    colors = [
        (0.0,  '#000000'), # 0% - Black
        (0.2,  '#0b1d36'), # 20% - Dark Blue
        (0.5,  '#008080'), # 50% - Teal/Cyan mix
        (0.8,  '#00FFFF'), # 80% - Bright Cyan
        (1.0,  '#FFFFFF')  # 100% - White
    ]
    return mcolors.LinearSegmentedColormap.from_list('glacial', colors)

def normalize_s2_for_plot(img_array):
    img_array = np.nan_to_num(img_array)
    p2, p98 = np.percentile(img_array, (2, 98))
    img_norm = np.clip((img_array - p2) / (p98 - p2), 0, 1)
    return img_norm

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

def generate_3row_plot():
    num_cols = len(TARGET_FILES)
    num_rows = 3
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # --- Initialize Custom Colormap ---
    # cmap_scf = create_glacial_colormap()
    cmap_scf = plt.get_cmap('viridis')
    # use the mako seaborn colormap as it goes from dark to light
    # cmap_scf = plt.get_cmap('mako')
    cmap_scf.set_bad(color='red') # Set cloud mask color

    print(f"Generating plot for {num_cols} tiles...")

    for i, gt_filename in enumerate(TARGET_FILES):
        pred_path, s2_path = get_file_paths(gt_filename, SPUN_DIR)
        
        # --- ROW 1: Sentinel-2 ---
        ax_s2 = axs[0, i]
        if s2_path:
            try:
                with rasterio.open(s2_path) as src:
                    if src.count >= 3:
                        s2_data = src.read([1, 2, 3])
                        s2_plot = np.transpose(s2_data, (1, 2, 0))
                        s2_plot = normalize_s2_for_plot(s2_plot)
                        ax_s2.imshow(s2_plot)
                    else:
                        ax_s2.imshow(src.read(1), cmap='gray')
            except Exception as e:
                ax_s2.text(0.5, 0.5, "Error", ha='center')
        else:
            ax_s2.text(0.5, 0.5, "Missing S2", ha='center')
        
        ax_s2.set_title(gt_filename.split('_')[0], fontsize=10, fontweight='bold')
        ax_s2.axis('off')

        # --- Load Prediction Data ---
        scf_data = None
        if pred_path and os.path.exists(pred_path):
            with rasterio.open(pred_path) as src:
                scf_data = src.read(1)

        # --- ROW 2: Predicted SCF (Glacial + Red Clouds) ---
        ax_scf = axs[1, i]
        if scf_data is not None:
            # Mask clouds so they appear Red (via set_bad)
            is_cloud = (scf_data == NODATA_VAL) | np.isnan(scf_data)
            scf_masked = np.ma.masked_where(is_cloud, scf_data)
            
            im_scf = ax_scf.imshow(scf_masked, cmap=cmap_scf, vmin=0, vmax=100, interpolation='nearest')
        else:
            ax_scf.text(0.5, 0.5, "Missing", ha='center')
        ax_scf.axis('off')

        # --- ROW 3: Binary Snow ---
        ax_bin = axs[2, i]
        if scf_data is not None:
            binary_map = np.zeros_like(scf_data, dtype=np.int8)
            is_nodata = (scf_data == NODATA_VAL) | np.isnan(scf_data)
            is_snow = (~is_nodata) & (scf_data > SCF_THRESHOLD)
            
            binary_map[is_snow] = 1 
            binary_map[is_nodata] = 2 

            cmap_bin = mcolors.ListedColormap(['black', 'cyan', 'red'])
            bounds = [-0.5, 0.5, 1.5, 2.5]
            norm_bin = mcolors.BoundaryNorm(bounds, cmap_bin.N)
            
            ax_bin.imshow(binary_map, cmap=cmap_bin, norm=norm_bin, interpolation='nearest')
        else:
            ax_bin.text(0.5, 0.5, "Missing", ha='center')
        ax_bin.axis('off')

    # --- Labels & Legends ---
    rows = ["Sentinel-2", "Predicted SCF", "Binary Snow"]
    for idx, label in enumerate(rows):
        axs[idx, 0].text(-0.1, 0.5, label, transform=axs[idx, 0].transAxes, 
                         rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')

    # Colorbar (SCF)
    cax = fig.add_axes([0.91, 0.40, 0.015, 0.2])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 100), cmap=cmap_scf), cax=cax)
    cbar.set_label('Snow Fraction (%)', rotation=270, labelpad=15)

    # Legend (Binary)
    patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='cyan', label='Snow'),
        mpatches.Patch(color='red', label='Cloud')
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.05))

    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    generate_3row_plot()
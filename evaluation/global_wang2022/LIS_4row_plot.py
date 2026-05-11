import os
import glob
import re
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image

# ==========================================
# 1. CONFIGURATION
# ==========================================

SPUN_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/Final_T15_dino_none_scale10000_best_mosaicked'
GT_MASK_DIR = '/home/users/leamhowe/phd/1_DETECT/Wang2022/datasets/masks'
LIS_BASE_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc/wang_scenes/'

TARGET_FILES = [
    "20200713T103031_20200713T103026_T33VWH_15-31_60-46.tif", # <-- New Tile Added Here
    "20200804T223709_20200804T223712_T59GLM_169-47_-44-02.tif",
    "20200805T001109_20200805T001647_T55GCN_145-04_-42-94.tif",
    "20210714T081609_20210714T083805_T34HCJ_19-16_-33-13.tif",
]

OUTPUT_FILENAME = 'LIS_4Row_Assessment_Fixed_NoData.png'
LIS_THRESHOLD = 50.0  

# --- SEPARATED VALUES ---
LIS_VAL_CLOUD = 205
LIS_VAL_NODATA = 255
GAMMA_VALUE = 0.65  

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_geotiff_as_array(path):
    with rasterio.open(path) as src:
        array = src.read()  
    return array

def get_optimized_s2_rgb(path, gamma=GAMMA_VALUE):
    try:
        image_array = load_geotiff_as_array(path).transpose(1, 2, 0)
        if image_array.shape[2] >= 4:
            rgb = np.stack([image_array[:, :, 3], image_array[:, :, 2], image_array[:, :, 1]], axis=2)
            rgb = np.clip(rgb, 0, 1)
            rgb_bright = np.power(rgb, gamma)
            return rgb_bright, False
        else:
            return image_array[:, :, 0], True
    except Exception as e:
        print(f"Error loading S2 {path}: {e}")
        return np.zeros((100, 100, 3)), False

def get_s2_path(gt_filename, base_dir):
    unique_id = gt_filename.split('.')[0]
    candidates = glob.glob(os.path.join(base_dir, f"pred_*{unique_id}*_s2.tif"))
    return candidates[0] if candidates else None

def get_lis_path(gt_filename, lis_base_dir):
    date_match = re.search(r'\d{8}', gt_filename)
    tile_match = re.search(r'T\d{2}[A-Z]{3}', gt_filename)
    
    if not date_match or not tile_match: return None
    date_str, tile_str = date_match.group(), tile_match.group()

    folders = glob.glob(os.path.join(lis_base_dir, f"*{date_str}*{tile_str}*"))
    if not folders: return None
    
    lis_files = glob.glob(os.path.join(folders[0], 'LIS_S2-SNOW-FSC-TOC_*.tif'))
    return lis_files[0] if lis_files else None

def crop_and_align_lis(lis_path, gt_path):
    with rasterio.open(gt_path) as tmp:
        template_meta = tmp.meta.copy()
        
    with rasterio.open(lis_path) as src:
        out_arr = np.zeros((template_meta['height'], template_meta['width']), dtype=src.meta['dtype'])
        reproject(
            source=rasterio.band(src, 1),
            destination=out_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_meta['transform'],
            dst_crs=template_meta['crs'],
            resampling=Resampling.nearest 
        )
    return out_arr

# ==========================================
# 3. MAIN PLOTTING SCRIPT
# ==========================================

def generate_4row_plot():
    num_cols = len(TARGET_FILES)
    num_rows = 4 
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    # --- Colormaps ---
    cmap_scf = plt.get_cmap('viridis')
    # Set default "bad" data (NaNs) to White for NoData
    cmap_scf.set_bad(color='white')

    # Updated Categorical Colormap: 0=Black(BG), 1=Cyan(Snow), 2=Red(Cloud), 3=White(NoData)
    cmap_bin = mcolors.ListedColormap(['black', 'cyan', 'red', 'white'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm_bin = mcolors.BoundaryNorm(bounds, cmap_bin.N)

    print(f"Generating LIS plot for {num_cols} tiles...")

    for i, gt_filename in enumerate(TARGET_FILES):
        s2_path = get_s2_path(gt_filename, SPUN_DIR)
        lis_path = get_lis_path(gt_filename, LIS_BASE_DIR)
        gt_full_path = os.path.join(GT_MASK_DIR, gt_filename)
        
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
        
        ax_s2.set_title(gt_filename.split('_')[0], fontsize=16, fontweight='bold')
        ax_s2.axis('off')

        lis_cropped = None
        if lis_path and os.path.exists(gt_full_path):
            try:
                lis_cropped = crop_and_align_lis(lis_path, gt_full_path)
            except Exception as e:
                print(f"Error cropping LIS {lis_path}: {e}")

        # Define Masks
        if lis_cropped is not None:
            is_cloud = (lis_cropped == LIS_VAL_CLOUD)
            is_nodata = (lis_cropped == LIS_VAL_NODATA) | np.isnan(lis_cropped)
            is_snow = (~is_cloud) & (~is_nodata) & (lis_cropped > LIS_THRESHOLD)

        # =================================
        # ROW 2: Predicted SCF (LIS)
        # =================================
        ax_scf = axs[1, i]
        if lis_cropped is not None:
            # 1. Plot SCF (Masking out both clouds and nodata so they don't skew the 0-100 scale)
            # cmap_scf.set_bad('white') will make NoData areas white.
            scf_masked = np.ma.masked_where(is_cloud | is_nodata, lis_cropped)
            ax_scf.imshow(scf_masked, cmap=cmap_scf, vmin=0, vmax=100, interpolation='nearest')
            
            # 2. Overlay Clouds strictly in Red
            cloud_overlay = np.ma.masked_where(~is_cloud, np.ones_like(lis_cropped))
            ax_scf.imshow(cloud_overlay, cmap=mcolors.ListedColormap(['red']), interpolation='nearest')
        else:
            ax_scf.text(0.5, 0.5, "Missing LIS", ha='center')
        ax_scf.axis('off')

        # =================================
        # ROW 3: Predicted Binary (LIS)
        # =================================
        ax_bin = axs[2, i]
        if lis_cropped is not None:
            # Create Viz Map: 0=BG, 1=Snow, 2=Cloud, 3=NoData
            binary_map = np.zeros_like(lis_cropped, dtype=np.int8)
            
            binary_map[is_snow] = 1 
            binary_map[is_cloud] = 2 
            binary_map[is_nodata] = 3
            
            ax_bin.imshow(binary_map, cmap=cmap_bin, norm=norm_bin, interpolation='nearest')
        else:
            ax_bin.text(0.5, 0.5, "Missing LIS", ha='center')
        ax_bin.axis('off')

        # =================================
        # ROW 4: Ground Truth
        # =================================
        ax_gt = axs[3, i]
        if os.path.exists(gt_full_path):
            try:
                gt_img = Image.open(gt_full_path).convert('L')
                gt_arr = np.array(gt_img)
                
                # Default entirely to NoData (3=White)
                gt_viz = np.full_like(gt_arr, 3, dtype=np.int8)
                
                # Remap known GT values: 1=BG, 3=Snow, 2=Cloud
                gt_viz[gt_arr == 1] = 0  # Background (Black)
                gt_viz[gt_arr == 3] = 1  # Snow (Cyan)
                gt_viz[gt_arr == 2] = 2  # Cloud (Red)
                
                ax_gt.imshow(gt_viz, cmap=cmap_bin, norm=norm_bin, interpolation='nearest')
            except Exception as e:
                ax_gt.text(0.5, 0.5, "Error", ha='center')
        else:
            ax_gt.text(0.5, 0.5, "Missing GT", ha='center')
        ax_gt.axis('off')

    # --- Labels & Legends ---
    rows = ["Sentinel-2", "LIS SCF", "LIS Binary", "Label"]
    for idx, label in enumerate(rows):
        axs[idx, 0].text(-0.1, 0.5, label, transform=axs[idx, 0].transAxes, 
                         rotation=90, va='center', ha='right', fontsize=18, fontweight='bold')

    # Colorbar (SCF)
    cax = fig.add_axes([0.91, 0.51, 0.02, 0.17]) 
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 100), cmap=cmap_scf), cax=cax)
    cbar.set_label('Snow Fraction (%)', rotation=270, labelpad=15, fontsize=14)

    # Legend (Binary) - Added No Data
    patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='cyan', label='Snow'),
        mpatches.Patch(color='red', label='Cloud'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='No Data') # Added black edge so it's visible against white background
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, frameon=False, fontsize=16, bbox_to_anchor=(0.5, 0.06))

    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    generate_4row_plot()
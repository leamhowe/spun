import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = 'LIS_Assessment'

# Directories
LIS_BASE_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc/wang_scenes/'
GT_MASK_DIR = '/home/users/leamhowe/phd/1_DETECT/Wang2022/datasets/masks'

# The three specific LIS folders we want to process
LIS_FOLDERS = [
    'SENTINEL2B_20200805-001852-484_L2A_T55GCN_C_V1-0',
    'SENTINEL2B_20200804-223828-903_L2A_T59GLM_C_V1-0',
    'SENTINEL2B_20210714-084937-758_L2A_T34HCJ_C_V1-0'
]

# --- LIS Product Specifics ---
# LIS FSC typically uses 0-100 for snow fraction.
# It uses 205 for clouds and 255 for NoData. 
LIS_THRESHOLD = 50.0  # Equivalent to 0.5 if data is 0-100. Change to 0.5 if data is 0-1.
LIS_CLOUD_NODATA_VALS = [205, 255] 

# --- Metric Standard Values (Ground Truth) ---
# 1=Background, 2=Cloud, 3=Snow
STD_VAL_BG = 1
STD_VAL_CLOUD = 2
STD_VAL_SNOW = 3

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def crop_and_align(big_raster_path, small_template_path):
    """
    Acts like a cookie cutter. Reads the geospatial bounds and shape of the 
    small template (GT) and crops out the exact matching area from the big raster (LIS).
    """
    with rasterio.open(small_template_path) as tmp:
        template_meta = tmp.meta.copy()
        
    with rasterio.open(big_raster_path) as src:
        # Initialize empty array matching the GT mask's shape
        out_arr = np.zeros((template_meta['height'], template_meta['width']), dtype=src.meta['dtype'])
        
        # Reproject exactly onto the GT grid
        reproject(
            source=rasterio.band(src, 1),
            destination=out_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_meta['transform'],
            dst_crs=template_meta['crs'],
            resampling=Resampling.nearest # Nearest neighbor is best for thematic/categorical data
        )
    return out_arr

def plot_confusion_matrix(cm, labels, save_path=None):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks=ticks, labels=labels)
    plt.yticks(ticks=ticks, labels=labels)
    plt.xlabel('Predicted (LIS)')
    plt.ylabel('Actual (Ground Truth)')
    plt.title('Total Confusion Matrix: LIS vs GT')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:,}"
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, txt, ha='center', va='center', color=color, fontsize=9)
            
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison_grid(dates, gt_arrays, lis_arrays, save_path=None):
    """Plots the GT masks next to the LIS predictions for visual sanity check."""
    n_images = len(dates)
    fig, axs = plt.subplots(n_images, 2, figsize=(8, n_images * 4))
    
    cmap = mcolors.ListedColormap(['black', 'red', 'cyan']) # 1=BG, 2=Cloud, 3=Snow
    bounds = [0.5, 1.5, 2.5, 3.5] 
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for i in range(n_images):
        # Ground Truth
        ax_gt = axs[i, 0] if n_images > 1 else axs[0]
        ax_gt.imshow(gt_arrays[i], cmap=cmap, norm=norm, interpolation='nearest')
        ax_gt.set_title(f"GT: {dates[i]}", fontsize=10, fontweight='bold')
        ax_gt.axis('off')

        # LIS Prediction
        ax_lis = axs[i, 1] if n_images > 1 else axs[1]
        ax_lis.imshow(lis_arrays[i], cmap=cmap, norm=norm, interpolation='nearest')
        ax_lis.set_title(f"LIS Binary: {dates[i]}", fontsize=10, fontweight='bold')
        ax_lis.axis('off')

    legend_patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='red', label='Cloud'),
        mpatches.Patch(color='cyan', label='Snow')
    ]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
    plt.subplots_adjust(bottom=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison grid to {save_path}")
    plt.close(fig)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    all_gt_flat = []
    all_pred_flat = []
    
    vis_dates = []
    vis_gt = []
    vis_lis = []

    print("--- Matching LIS Files to Ground Truth ---")
    
    # Get all potential GT masks
    all_gt_files = glob.glob(os.path.join(GT_MASK_DIR, '*.tif'))

    for folder in LIS_FOLDERS:
        # Extract Date and Tile from LIS folder name (e.g., 20200805 and T55GCN)
        parts = folder.split('_')
        date_str = parts[1].split('-')[0] # '20200805'
        tile_str = parts[3]               # 'T55GCN'
        
        # 1. Find matching GT file
        matching_gt = [f for f in all_gt_files if date_str in f and tile_str in f]
        
        if not matching_gt:
            print(f"Warning: Could not find GT mask for Date: {date_str}, Tile: {tile_str}")
            continue
        gt_file = matching_gt[0]

        # 2. Find LIS FSC TIF inside the folder
        folder_path = os.path.join(LIS_BASE_DIR, folder)
        lis_files = glob.glob(os.path.join(folder_path, 'LIS_S2-SNOW-FSC-TOC_*.tif'))
        
        if not lis_files:
            print(f"Warning: No FSC TIF found in {folder_path}")
            continue
        lis_file = lis_files[0]

        print(f"\nProcessing {date_str}_{tile_str}...")
        
        try:
            # --- STEP 1: Load GT and Crop LIS ---
            gt_img = Image.open(gt_file).convert('L')
            gt_arr = np.array(gt_img)
            
            lis_raw_cropped = crop_and_align(lis_file, gt_file)
            
            # --- STEP 2: Threshold LIS to Binary ---
            # Output container filled with Background (1)
            standardized_lis = np.ones_like(lis_raw_cropped, dtype=np.uint8) * STD_VAL_BG
            
            # Identify Clouds/NoData
            is_cloud = np.isin(lis_raw_cropped, LIS_CLOUD_NODATA_VALS) | np.isnan(lis_raw_cropped)
            
            # Identify Snow
            is_snow = (~is_cloud) & (lis_raw_cropped > LIS_THRESHOLD)
            
            # Apply assignments
            standardized_lis[is_cloud] = STD_VAL_CLOUD
            standardized_lis[is_snow] = STD_VAL_SNOW

            # --- STORE FOR METRICS ---
            all_gt_flat.append(gt_arr.ravel())
            all_pred_flat.append(standardized_lis.ravel())
            
            # --- STORE FOR VISUALIZATION ---
            vis_dates.append(f"{date_str}_{tile_str}")
            vis_gt.append(gt_arr)
            vis_lis.append(standardized_lis)

        except Exception as e:
            print(f"Error processing pair: {e}")

    # --- STEP 3: VISUALIZATION & METRICS ---
    if all_gt_flat:
        print("\n--- Generating Visual Comparison ---")
        plot_comparison_grid(vis_dates, vis_gt, vis_lis, save_path=f"{MODEL_NAME}_comparison_grid.png")

        print("\n--- Calculating Global Metrics ---")
        Y_true = np.concatenate(all_gt_flat)
        Y_pred = np.concatenate(all_pred_flat)
        
        class_ids = [STD_VAL_BG, STD_VAL_CLOUD, STD_VAL_SNOW]
        class_names = ['BG', 'Cloud', 'Snow']

        p = precision_score(Y_true, Y_pred, labels=class_ids, average=None, zero_division=0)
        r = recall_score(Y_true, Y_pred, labels=class_ids, average=None, zero_division=0)
        f = f1_score(Y_true, Y_pred, labels=class_ids, average=None, zero_division=0)
        iou = jaccard_score(Y_true, Y_pred, labels=class_ids, average=None, zero_division=0)

        print(f"\n{'Metric':<10} | {'BG':<10} | {'Cloud':<10} | {'Snow':<10}")
        print("-" * 50)
        print(f"{'Precision':<10} | {p[0]:.4f}     | {p[1]:.4f}     | {p[2]:.4f}")
        print(f"{'Recall':<10} | {r[0]:.4f}     | {r[1]:.4f}     | {r[2]:.4f}")
        print(f"{'F1':<10} | {f[0]:.4f}     | {f[1]:.4f}     | {f[2]:.4f}")
        print(f"{'IoU':<10} | {iou[0]:.4f}     | {iou[1]:.4f}     | {iou[2]:.4f}")
        
        print("\n--- Macro-Averages ---")
        print(f"Macro Precision: {np.mean(p):.4f}")
        print(f"Macro Recall:    {np.mean(r):.4f}")
        print(f"Macro F1:        {np.mean(f):.4f}")
        print(f"Macro IoU:       {np.mean(iou):.4f}")

        cm = confusion_matrix(Y_true, Y_pred, labels=class_ids)
        plot_confusion_matrix(cm, class_names, save_path=f'{MODEL_NAME}_confusion_matrix.png')
    else:
        print("No data was successfully processed.")
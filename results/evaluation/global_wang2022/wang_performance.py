import glob
import os
import math
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = 'Wang_Predictions_Assessment'
BASE_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/wang_predictions'
GT_MASK_DIR = '/home/users/leamhowe/phd/1_DETECT/Wang2022/datasets/masks'

DATES_TO_PROCESS = [20191123, 20200713, 20200804, 20200805, 20201126, 20210714]

# --- Value Mapping Configuration ---
# 1. Wang Input Values (What is in the TIF files?)
WANG_VAL_BG = 0
WANG_VAL_CLOUD = 1
WANG_VAL_SNOW = 2

# 2. Metric Standard Values (What does your Ground Truth use?)
# Based on your previous scripts: 1=BG, 2=Cloud, 3=Snow
STD_VAL_BG = 1
STD_VAL_CLOUD = 2
STD_VAL_SNOW = 3

GRID_COLS = 6

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def filter_files_by_date(file_list, dates_to_find):
    """Filters file paths to keep only those matching specific date strings."""
    date_strings = [str(date) for date in dates_to_find]
    found_files = [
        path for path in file_list
        if any(date_str in path for date_str in date_strings)
    ]
    return sorted(found_files)

def load_geotiff(path):
    with rasterio.open(path) as src:
        return src.read(1)

def extract_date_from_filename(filepath):
    filename = os.path.basename(filepath)
    import re
    match = re.search(r'\d{8}', filename)
    if match: return match.group(0)
    return filename[:8]

def plot_prediction_grid(file_paths, save_path=None):
    n_images = len(file_paths)
    if n_images == 0: return

    rows = math.ceil(n_images / GRID_COLS)
    fig, axs = plt.subplots(rows, GRID_COLS, figsize=(GRID_COLS * 3, rows * 3))
    if n_images == 1: axs = [axs]
    else: axs = axs.flatten()

    # Define Visualization Colors: 0=Black, 1=Cyan, 2=Red
    cmap = mcolors.ListedColormap(['black', 'cyan', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5] 
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    print(f"Generating visual grid for {n_images} images...")

    for i, ax in enumerate(axs):
        if i < n_images:
            path = file_paths[i]
            try:
                data = load_geotiff(path)
                
                # Create a temporary array for visualization 
                # Target: 0=BG(Black), 1=Snow(Cyan), 2=Cloud(Red)
                viz_data = np.zeros_like(data, dtype=np.int8)
                
                # Remap Wang values to Viz values
                viz_data[data == WANG_VAL_BG] = 0    # Black
                viz_data[data == WANG_VAL_SNOW] = 1  # Cyan
                viz_data[data == WANG_VAL_CLOUD] = 2 # Red

                ax.imshow(viz_data, cmap=cmap, norm=norm, interpolation='nearest')
                ax.set_title(extract_date_from_filename(path), fontsize=10, fontweight='bold')
                ax.axis('off')
            except Exception as e:
                ax.axis('off')
        else:
            ax.axis('off')

    legend_patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='cyan', label='Snow'),
        mpatches.Patch(color='red', label='Cloud')
    ]
    fig.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
    plt.subplots_adjust(bottom=0.2, wspace=0.0, hspace=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid to {save_path}")
    plt.close(fig)

def plot_confusion_matrix(cm, labels, save_path=None):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks=ticks, labels=labels); plt.yticks(ticks=ticks, labels=labels)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Total Confusion Matrix')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:,}"
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, txt, ha='center', va='center', color=color, fontsize=9)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. File Discovery
    print(f"Searching in: {BASE_DIR}")
    candidate_files = glob.glob(os.path.join(BASE_DIR, '*.tif'))
    gt_mask_paths_all = glob.glob(os.path.join(GT_MASK_DIR, '*.tif'))

    # 2. Filter by Date
    test_pred_paths = filter_files_by_date(candidate_files, DATES_TO_PROCESS)
    test_gt_paths = filter_files_by_date(gt_mask_paths_all, DATES_TO_PROCESS)

    print(f"Found {len(test_pred_paths)} prediction files.")
    print(f"Found {len(test_gt_paths)} ground truth files.")

    # 3. Visual Check
    plot_prediction_grid(test_pred_paths, save_path=f'{MODEL_NAME}_visual_grid.png')

    # 4. Metrics Calculation
    all_gt_flat = []
    all_pred_flat = []
    
    print("Processing image pairs...")
    for pred_file in test_pred_paths:
        date_key = extract_date_from_filename(pred_file)
        matching_gts = [g for g in test_gt_paths if date_key in os.path.basename(g)]
        
        if not matching_gts:
            print(f"Warning: No GT found for date {date_key}")
            continue
        gt_file = matching_gts[0]

        try:
            with rasterio.open(pred_file) as src: pred_arr = src.read(1)
            gt_img = Image.open(gt_file).convert('L')
            gt_arr = np.array(gt_img)
            
            if pred_arr.shape != gt_arr.shape:
                print(f"Shape mismatch ({date_key}): {pred_arr.shape} vs {gt_arr.shape} - Skipping")
                continue

            # --- VITAL STEP: REMAP WANG VALUES TO GT STANDARD VALUES ---
            # Input: 0=BG, 1=Cloud, 2=Snow
            # Output: 1=BG, 2=Cloud, 3=Snow
            
            standardized_pred = np.zeros_like(pred_arr, dtype=np.uint8)
            
            # Map Background (0 -> 1)
            standardized_pred[pred_arr == WANG_VAL_BG] = STD_VAL_BG
            
            # Map Cloud (1 -> 2)
            standardized_pred[pred_arr == WANG_VAL_CLOUD] = STD_VAL_CLOUD
            
            # Map Snow (2 -> 3)
            standardized_pred[pred_arr == WANG_VAL_SNOW] = STD_VAL_SNOW

            all_gt_flat.append(gt_arr.ravel())
            all_pred_flat.append(standardized_pred.ravel())
            
        except Exception as e:
            print(f"Error on {date_key}: {e}")

    # 5. Calculate Global Stats
    if all_gt_flat:
        print("\nCalculating metrics...")
        Y_true = np.concatenate(all_gt_flat)
        Y_pred = np.concatenate(all_pred_flat)
        
        # We calculate metrics based on the STANDARD values (1, 2, 3)
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
        
        cm = confusion_matrix(Y_true, Y_pred, labels=class_ids)
        plot_confusion_matrix(cm, class_names, save_path=f'{MODEL_NAME}_cm.png')
    else:
        print("No valid data processed.")
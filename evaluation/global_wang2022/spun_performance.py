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
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================

# Model and Paths
MODEL_NAME = 'Final_T15_dino_none_scale10000_best_mosaicked'
BASE_DIR = f'/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/{MODEL_NAME}'
GT_MASK_DIR = '/home/users/leamhowe/phd/1_DETECT/Wang2022/datasets/masks'

# Dates to process
DATES_TO_PROCESS = [20191123, 20200713, 20200804, 20200805, 20201126, 20210714]

# Thresholds & Values
# Note: Providing two separate thresholds as used in your snippets (Raw vs Probability)
SNOW_THRESHOLD_RAW = 50.0       # For visualization (raw pixel value)
PREDICTION_THRESHOLD = 0.5      # For metrics (probability 0.0-1.0)
CLOUD_VALUE_VIS = 200           # Value for clouds in visualization
PREDICTION_NODATA_VALUE = 200   # Value representing nodata/clouds in metrics

# Plotting Settings
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
    
    # WARNING: Hardcoded index removal from your original snippet.
    # Remove this if running on a different dataset where index 4 isn't invalid.
    if len(found_files) > 4:
        found_files.pop(4) 
        
    return sorted(found_files)

def load_geotiff(path):
    with rasterio.open(path) as src:
        return src.read(1)

def extract_date_from_filename(filepath):
    filename = os.path.basename(filepath)
    try:
        parts = filename.split('_')
        for part in parts:
            if 'T' in part and len(part) >= 8:
                return part.split('T')[0]
        return filename[:8]
    except IndexError:
        return filename[:8]

def plot_prediction_grid(file_paths, save_path=None):
    """Generates a grid of prediction images."""
    n_images = len(file_paths)
    if n_images == 0:
        print("No images to plot.")
        return

    rows = math.ceil(n_images / GRID_COLS)
    fig, axs = plt.subplots(rows, GRID_COLS, figsize=(GRID_COLS * 3, rows * 3))
    
    # Ensure axs is iterable even if only 1 plot
    if n_images == 1: axs = [axs]
    else: axs = axs.flatten()

    # Custom Color Map: 0=Black(BG), 1=Cyan(Snow), 2=Red(Cloud)
    cmap = mcolors.ListedColormap(['black', 'cyan', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5] 
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    print(f"Generating {GRID_COLS}x{rows} grid for {n_images} images...")

    for i, ax in enumerate(axs):
        if i < n_images:
            path = file_paths[i]
            try:
                data = load_geotiff(path)
                
                # Categorize Data for Visualization
                categorical_map = np.zeros_like(data, dtype=np.int8) # Default 0 (BG)
                categorical_map[data > SNOW_THRESHOLD_RAW] = 1       # Snow
                categorical_map[data == CLOUD_VALUE_VIS] = 2         # Cloud
                
                ax.imshow(categorical_map, cmap=cmap, norm=norm, interpolation='nearest')
                ax.set_title(extract_date_from_filename(path), fontsize=10, fontweight='bold')
                ax.axis('off')
            except Exception as e:
                print(f"Error loading {path}: {e}")
                ax.axis('off')
        else:
            ax.axis('off')

    # Legend
    legend_patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='cyan', label='Snow'),
        mpatches.Patch(color='red', label='Cloud')
    ]
    fig.legend(handles=legend_patches, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=12)

    plt.subplots_adjust(bottom=0.2, wspace=0.0, hspace=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid to {save_path}")
    
    plt.close(fig) # Close to free memory

def plot_confusion_matrix(cm, labels, save_path=None):
    """Plots and saves the total confusion matrix."""
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    ticks = np.arange(len(labels))
    plt.xticks(ticks=ticks, labels=labels)
    plt.yticks(ticks=ticks, labels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Total Confusion Matrix (pixel counts)')

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:,}"
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, txt, ha='center', va='center', color=color, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Confusion Matrix to {save_path}")
    plt.close()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # --- A. File Discovery ---
    print("--- Searching for files ---")
    image_paths_s2 = glob.glob(os.path.join(BASE_DIR, 'pred_*_s2.tif')) 
    
    # Construct corresponding spun paths and mask paths
    # Note: Logic assumes spun files are in the same dir but without '_s2'
    spun_paths = [f.replace('_s2.tif', '.tif') for f in image_paths_s2]
    gt_mask_paths_all = glob.glob(os.path.join(GT_MASK_DIR, '*.tif'))

    # --- B. Filter by Date ---
    test_spun_paths = filter_files_by_date(spun_paths, DATES_TO_PROCESS)
    test_gt_mask_paths = filter_files_by_date(gt_mask_paths_all, DATES_TO_PROCESS)

    print(f"Found {len(test_spun_paths)} matching spun files.")
    print(f"Found {len(test_gt_mask_paths)} matching ground truth mask files.")

    # Validation check
    if len(test_spun_paths) != len(test_gt_mask_paths):
        print("CRITICAL WARNING: Number of prediction files and ground truth files do not match.")
        print("Please check your file names and date filtering logic.")

    # --- C. Visual Sanity Check ---
    print("\n--- Generating Visual Grid ---")
    plot_prediction_grid(
        test_spun_paths, 
        save_path=f'{MODEL_NAME}_pred_grid_mosaic_test_images.png'
    )

    # --- D. Metric Calculation (Global & Confusion Matrix) ---
    print("\n--- Calculating Metrics ---")
    
    all_gt_flat = []
    all_pred_flat = []
    
    for pred_file, gt_file in zip(test_spun_paths, test_gt_mask_paths):
        if not os.path.exists(pred_file) or not os.path.exists(gt_file):
            print(f"Skipping missing pair: {pred_file}")
            continue
            
        try:
            # Load Data
            with rasterio.open(pred_file) as src:
                pred_arr = src.read(1)
            gt_img = Image.open(gt_file).convert('L')
            gt_arr = np.array(gt_img)
            
            # Shape Check
            if pred_arr.shape != gt_arr.shape:
                print(f"Skipping mismatch: Pred {pred_arr.shape} vs GT {gt_arr.shape}")
                continue

            # Classify Predictions
            # 1=Background, 2=Cloud, 3=Snow
            pred_hard_labels = np.zeros_like(pred_arr, dtype=np.uint8)
            
            # Identify Clouds/Nodata
            is_nodata_cloud = np.isnan(pred_arr) | (pred_arr == PREDICTION_NODATA_VALUE)
            
            # Identify Snow vs Background
            pred_probs = pred_arr / 100.0
            is_snow = (~is_nodata_cloud) & (pred_probs > PREDICTION_THRESHOLD)
            is_background = (~is_nodata_cloud) & (pred_probs <= PREDICTION_THRESHOLD)
            
            pred_hard_labels[is_background] = 1
            pred_hard_labels[is_nodata_cloud] = 2
            pred_hard_labels[is_snow] = 3

            # Accumulate flat arrays for global metrics
            all_gt_flat.append(gt_arr.ravel())
            all_pred_flat.append(pred_hard_labels.ravel())
            
        except Exception as e:
            print(f"Error processing pair {os.path.basename(pred_file)}: {e}")

    # --- E. Final Statistics ---
    if all_gt_flat:
        print("\nConcatenating all pixels for global stats...")
        Y_true_global = np.concatenate(all_gt_flat)
        Y_pred_global = np.concatenate(all_pred_flat)
        
        class_ids = [1, 2, 3]
        class_names = ['BG', 'Cloud', 'Snow']

        # 1. Calculate Standard Metrics
        g_prec = precision_score(Y_true_global, Y_pred_global, labels=class_ids, average=None, zero_division=0)
        g_rec = recall_score(Y_true_global, Y_pred_global, labels=class_ids, average=None, zero_division=0)
        g_f1 = f1_score(Y_true_global, Y_pred_global, labels=class_ids, average=None, zero_division=0)
        g_iou = jaccard_score(Y_true_global, Y_pred_global, labels=class_ids, average=None, zero_division=0)

        # 2. Print Table
        print(f"\n{'Metric':<12} | {'Background (1)':<14} | {'Cloud (2)':<14} | {'Snow (3)':<14}")
        print("-" * 65)
        print(f"{'Precision':<12} | {g_prec[0]:.4f}         | {g_prec[1]:.4f}         | {g_prec[2]:.4f}")
        print(f"{'Recall':<12} | {g_rec[0]:.4f}         | {g_rec[1]:.4f}         | {g_rec[2]:.4f}")
        print(f"{'F1 (Dice)':<12} | {g_f1[0]:.4f}         | {g_f1[1]:.4f}         | {g_f1[2]:.4f}")
        print(f"{'IoU':<12} | {g_iou[0]:.4f}         | {g_iou[1]:.4f}         | {g_iou[2]:.4f}")
        
        print("\n--- Macro-Averages ---")
        print(f"Macro Precision: {np.mean(g_prec):.4f}")
        print(f"Macro Recall:    {np.mean(g_rec):.4f}")
        print(f"Macro F1:        {np.mean(g_f1):.4f}")
        print(f"Macro IoU:       {np.mean(g_iou):.4f}")

        # 3. Calculate and Plot Confusion Matrix
        print("\nCalculating Total Confusion Matrix...")
        total_cm = confusion_matrix(Y_true_global, Y_pred_global, labels=class_ids)
        
        # Print textual CM
        print("\nPredicted ->   1 (BG)        2 (Cloud)     3 (Snow)")
        for i, row in enumerate(total_cm):
            print(f"Actual {i+1} ({class_names[i]}) : {row[0]:<13} {row[1]:<13} {row[2]:<13}")

        # Save visual CM
        plot_confusion_matrix(
            total_cm, 
            class_names, 
            save_path=f'{MODEL_NAME}_total_confusion_matrix.png'
        )
        
    else:
        print("No valid data found to calculate metrics.")
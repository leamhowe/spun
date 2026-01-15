import os
import glob
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

# --- Directories ---
GT_MASK_DIR = '/home/users/leamhowe/phd/1_DETECT/Wang2022/datasets/masks'

# 1. SPUN Model Config
SPUN_NAME = 'SPUN_Model'
SPUN_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/Final_T15_dino_none_scale10000_best_mosaicked'
SPUN_IS_CATEGORICAL = False # Inputs are probabilities (0-100) or raw logits
SPUN_NODATA_VAL = 200

# 2. Wang Model Config
WANG_NAME = 'Wang_Method'
WANG_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/wang_predictions'
WANG_IS_CATEGORICAL = True  # Inputs are classes (0, 1, 2)
WANG_MAPPING = {0: 1, 1: 2, 2: 3} # Map Wang(0=BG,1=Cloud,2=Snow) -> GT(1=BG,2=Cloud,3=Snow)

# --- The 6 Specific Test Files (Basenames) ---
# We extract the filenames from the full paths you provided to ensure we match EXACTLY these tiles.
TARGET_FILES = [
    "20191123T111259_20191123T112151_T29RNQ_-8-3_31-22.tif",
    "20200713T103031_20200713T103026_T33VWH_15-31_60-46.tif",
    "20200804T223709_20200804T223712_T59GLM_169-47_-44-02.tif",
    "20200805T001109_20200805T001647_T55GCN_145-04_-42-94.tif",
    "20201126T041121_20201126T041842_T47SNT_100-09_33-93.tif",
    "20210714T081609_20210714T083805_T34HCJ_19-16_-33-13.tif"
]

# Standard Ground Truth Values: 1=BG, 2=Cloud, 3=Snow
CLASS_LABELS = [1, 2, 3]
CLASS_NAMES = ['Background', 'Cloud', 'Snow']

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def find_matching_prediction(gt_filename, prediction_dir):
    """
    Finds a prediction file that matches the GT filename.
    Handles 'pred_' prefixes often found in ML outputs.
    """
    # 1. Try exact match
    exact_path = os.path.join(prediction_dir, gt_filename)
    if os.path.exists(exact_path):
        return exact_path
    
    # 2. Try 'pred_' prefix (Common in your SPUN outputs)
    pred_path = os.path.join(prediction_dir, f"pred_{gt_filename}")
    if os.path.exists(pred_path):
        return pred_path

    # 3. Fuzzy search if needed (fallback)
    # Search for the unique timestamp/tile ID in the directory
    unique_id = gt_filename.split('.')[0] # remove extension
    candidates = glob.glob(os.path.join(prediction_dir, f"*{unique_id}*"))
    
    # Filter out files that might be 's2' inputs if mixed in same folder
    candidates = [c for c in candidates if '_s2' not in c]
    
    if candidates:
        return candidates[0]
    
    return None

def load_geotiff(path):
    with rasterio.open(path) as src:
        return src.read(1)

def plot_visual_grid(images, titles, save_path):
    """Generates a visual grid of the classified outputs."""
    n_images = len(images)
    cols = 3
    rows = math.ceil(n_images / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axs = axs.flatten() if n_images > 1 else [axs]

    # Colors: 0=Black(BG), 1=Cyan(Snow), 2=Red(Cloud)
    # Note: We visualize standardized classes (1,2,3) mapped to 0,1,2 for colorbar
    cmap = mcolors.ListedColormap(['black', 'red', 'cyan']) # 1=BG(Black), 2=Cloud(Red), 3=Snow(Cyan)
    # We need to map data 1,2,3 to color indices 0,1,2
    # Easier way: Just make colormap match values 1, 2, 3 directly
    cmap = mcolors.ListedColormap(['white', 'black', 'red', 'cyan']) 
    # Index 0=White(Unused), 1=BG(Black), 2=Cloud(Red), 3=Snow(Cyan)
    bounds = [0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for i, ax in enumerate(axs):
        if i < n_images:
            ax.imshow(images[i], cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_title(titles[i], fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')
    
    # Legend
    patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='red', label='Cloud'),
        mpatches.Patch(color='cyan', label='Snow')
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cm(cm, classes, save_path):
    """Plots confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def standardize_prediction(pred_arr, is_categorical, mapping=None, nodata_val=None, prob_thresh=0.5):
    """
    Converts raw prediction (prob map or foreign class map) 
    to Standard Map: 1=BG, 2=Cloud, 3=Snow
    """
    output = np.zeros_like(pred_arr, dtype=np.uint8)
    
    if is_categorical:
        # Map Discrete Values (e.g. Wang 0->1)
        for src_val, dst_val in mapping.items():
            output[pred_arr == src_val] = dst_val
    else:
        # Map Probabilities (e.g. SPUN)
        # Assumes: pred_arr is 0-100 or 0-1
        is_nodata = np.isnan(pred_arr) | (pred_arr == nodata_val)
        
        # Normalize to 0-1 if needed
        probs = pred_arr if pred_arr.max() <= 1.0 else pred_arr / 100.0
        
        is_snow = (~is_nodata) & (probs > prob_thresh)
        is_bg = (~is_nodata) & (probs <= prob_thresh)
        
        output[is_bg] = 1
        output[is_nodata] = 2
        output[is_snow] = 3
        
    return output

# ==========================================
# 3. CORE EVALUATION LOGIC
# ==========================================

def evaluate_model(model_name, pred_dir, is_categorical, mapping=None, nodata_val=None):
    print(f"\n--- Evaluating {model_name} ---")
    
    all_gt = []
    all_pred = []
    vis_images = []
    vis_titles = []
    
    for gt_filename in TARGET_FILES:
        gt_path = os.path.join(GT_MASK_DIR, gt_filename)
        
        # 1. Find Prediction
        pred_path = find_matching_prediction(gt_filename, pred_dir)
        
        if not pred_path:
            print(f"MISSING PREDICTION for: {gt_filename}")
            continue
            
        try:
            # 2. Load Data
            with rasterio.open(pred_path) as src: 
                pred_raw = src.read(1)
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)
            
            if pred_raw.shape != gt_arr.shape:
                print(f"Shape Mismatch: {gt_filename}")
                continue
                
            # 3. Standardize
            pred_std = standardize_prediction(
                pred_raw, 
                is_categorical, 
                mapping=mapping, 
                nodata_val=nodata_val
            )
            
            # 4. Store for Stats
            all_gt.append(gt_arr.ravel())
            all_pred.append(pred_std.ravel())
            
            # 5. Store for Vis (Short title)
            vis_images.append(pred_std)
            vis_titles.append(gt_filename.split('_')[0]) # Date only
            
        except Exception as e:
            print(f"Error processing {gt_filename}: {e}")

    # --- Generate Outputs ---
    if not all_gt:
        return "No data processed.", {}

    # 1. Plot Grid
    plot_visual_grid(vis_images, vis_titles, f"{model_name}_visual_grid.png")
    
    # 2. Global Metrics
    Y_true = np.concatenate(all_gt)
    Y_pred = np.concatenate(all_pred)
    
    precision = precision_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    recall = recall_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    f1 = f1_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    iou = jaccard_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    
    # 3. Plot Confusion Matrix
    cm = confusion_matrix(Y_true, Y_pred, labels=CLASS_LABELS)
    plot_cm(cm, CLASS_NAMES, f"{model_name}_confusion_matrix.png")
    
    # 4. Create Report String
    report = f"\n=== {model_name} RESULTS ===\n"
    report += f"Processed {len(vis_images)} files.\n\n"
    
    # Table Header
    report += f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'IoU':<10}\n"
    report += "-" * 65 + "\n"
    
    # Table Rows
    for i, label in enumerate(CLASS_NAMES):
        report += f"{label:<12} | {precision[i]:.4f}     | {recall[i]:.4f}     | {f1[i]:.4f}     | {iou[i]:.4f}\n"
        
    report += "-" * 65 + "\n"
    report += f"{'MACRO AVG':<12} | {np.mean(precision):.4f}     | {np.mean(recall):.4f}     | {np.mean(f1):.4f}     | {np.mean(iou):.4f}\n"
    
    return report

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    final_report = "COMPARISON REPORT: SPUN vs WANG\n"
    final_report += "===============================\n"
    
    # --- Run SPUN Evaluation ---
    spun_text = evaluate_model(
        SPUN_NAME, 
        SPUN_DIR, 
        is_categorical=SPUN_IS_CATEGORICAL, 
        nodata_val=SPUN_NODATA_VAL
    )
    final_report += spun_text
    
    # --- Run Wang Evaluation ---
    wang_text = evaluate_model(
        WANG_NAME, 
        WANG_DIR, 
        is_categorical=WANG_IS_CATEGORICAL, 
        mapping=WANG_MAPPING
    )
    final_report += wang_text
    
    # --- Save Report ---
    with open("comparison_metrics_report.txt", "w") as f:
        f.write(final_report)
        
    print("\n------------------------------------------------")
    print("Done! Check 'comparison_metrics_report.txt' and png plots.")
    print("------------------------------------------------")
import os
import glob
import math
import re
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

# --- Directories ---
GT_MASK_DIR = '/home/users/leamhowe/phd/1_DETECT/Wang2022/datasets/masks'

# 1. SPUN Model Config
SPUN_NAME = 'SPUN_Model'
SPUN_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/Final_T15_dino_none_scale10000_best_mosaicked'
SPUN_IS_CATEGORICAL = False 
SPUN_NODATA_VAL = 200

# 2. Wang Model Config
WANG_NAME = 'Wang_Method'
WANG_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/wang_predictions'
WANG_IS_CATEGORICAL = True  
WANG_MAPPING = {0: 1, 1: 2, 2: 3} 

# 3. LIS Product Config
LIS_NAME = 'LIS_Product'
LIS_BASE_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc/wang_scenes/'
LIS_THRESHOLD = 50.0  
LIS_VAL_CLOUD = 205
LIS_VAL_NODATA = 255

# --- TARGET FILES ---
# The full 6 test files
ALL_TARGET_FILES = [
    "20191123T111259_20191123T112151_T29RNQ_-8-3_31-22.tif",
    "20200713T103031_20200713T103026_T33VWH_15-31_60-46.tif",
    "20200804T223709_20200804T223712_T59GLM_169-47_-44-02.tif",
    "20200805T001109_20200805T001647_T55GCN_145-04_-42-94.tif",
    "20201126T041121_20201126T041842_T47SNT_100-09_33-93.tif",
    "20210714T081609_20210714T083805_T34HCJ_19-16_-33-13.tif"
]

# The exact 3 test files where LIS data is available
SUBSET_TARGET_FILES = [
    "20200804T223709_20200804T223712_T59GLM_169-47_-44-02.tif",
    "20200805T001109_20200805T001647_T55GCN_145-04_-42-94.tif",
    "20210714T081609_20210714T083805_T34HCJ_19-16_-33-13.tif",
]

# Standard Ground Truth Values: 1=BG, 2=Cloud, 3=Snow
CLASS_LABELS = [1, 2, 3]
CLASS_NAMES = ['Background', 'Cloud', 'Snow']

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def find_matching_prediction(gt_filename, prediction_dir):
    exact_path = os.path.join(prediction_dir, gt_filename)
    if os.path.exists(exact_path): return exact_path
    
    pred_path = os.path.join(prediction_dir, f"pred_{gt_filename}")
    if os.path.exists(pred_path): return pred_path

    unique_id = gt_filename.split('.')[0] 
    candidates = glob.glob(os.path.join(prediction_dir, f"*{unique_id}*"))
    candidates = [c for c in candidates if '_s2' not in c]
    
    if candidates: return candidates[0]
    return None

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

def plot_visual_grid(images, titles, save_path):
    n_images = len(images)
    if n_images == 0: return
    
    cols = 3
    rows = math.ceil(n_images / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axs = axs.flatten() if n_images > 1 else [axs]

    # Precise 1:1 boundaries for the 4 colors
    cmap = mcolors.ListedColormap(['white', 'black', 'red', 'cyan']) 
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for i, ax in enumerate(axs):
        if i < n_images:
            ax.imshow(images[i], cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_title(titles[i], fontsize=10, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    patches = [
        mpatches.Patch(color='black', label='Background'),
        mpatches.Patch(color='red', label='Cloud'),
        mpatches.Patch(color='cyan', label='Snow'),
        mpatches.Patch(facecolor='white', edgecolor='black', label='No Data')
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, frameon=False, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cm(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
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
    output = np.zeros_like(pred_arr, dtype=np.uint8)
    if is_categorical:
        for src_val, dst_val in mapping.items():
            output[pred_arr == src_val] = dst_val
    else:
        is_nodata = np.isnan(pred_arr) | (pred_arr == nodata_val)
        probs = pred_arr if pred_arr.max() <= 1.0 else pred_arr / 100.0
        is_snow = (~is_nodata) & (probs > prob_thresh)
        is_bg = (~is_nodata) & (probs <= prob_thresh)
        output[is_bg] = 1
        output[is_nodata] = 2  # SPUN maps nodata directly to "Cloud" class
        output[is_snow] = 3
    return output

# ==========================================
# 3. CORE EVALUATION LOGIC
# ==========================================

def build_report(model_name, vis_images, precision, recall, f1, iou, note=""):
    report = f"\n=== {model_name} RESULTS ===\n"
    if note:
        report += f"*** NOTE: {note} ***\n"
    report += f"Processed {len(vis_images)} files.\n\n"
    
    report += f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'IoU':<10}\n"
    report += "-" * 65 + "\n"
    
    for i, label in enumerate(CLASS_NAMES):
        report += f"{label:<12} | {precision[i]:.4f}     | {recall[i]:.4f}     | {f1[i]:.4f}     | {iou[i]:.4f}\n"
        
    report += "-" * 65 + "\n"
    report += f"{'MACRO AVG':<12} | {np.mean(precision):.4f}     | {np.mean(recall):.4f}     | {np.mean(f1):.4f}     | {np.mean(iou):.4f}\n"
    return report

def evaluate_model(model_name, pred_dir, target_files, is_categorical, mapping=None, nodata_val=None, plot_suffix=""):
    print(f"\n--- Evaluating {model_name} ({len(target_files)} tiles) ---")
    all_gt, all_pred, vis_images, vis_titles = [], [], [], []
    
    for gt_filename in target_files:
        gt_path = os.path.join(GT_MASK_DIR, gt_filename)
        pred_path = find_matching_prediction(gt_filename, pred_dir)
        
        if not pred_path:
            continue
            
        try:
            with rasterio.open(pred_path) as src: pred_raw = src.read(1)
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)
            
            if pred_raw.shape != gt_arr.shape: continue
                
            pred_std = standardize_prediction(
                pred_raw, is_categorical, mapping=mapping, nodata_val=nodata_val)
            
            all_gt.append(gt_arr.ravel())
            all_pred.append(pred_std.ravel())
            vis_images.append(pred_std)
            vis_titles.append(gt_filename.split('_')[0]) 
            
        except Exception as e:
            print(f"Error processing {gt_filename}: {e}")

    if not all_gt: return f"\n=== {model_name} ===\nNo data processed.\n"

    plot_visual_grid(vis_images, vis_titles, f"{model_name}_{plot_suffix}visual_grid.png")
    
    Y_true = np.concatenate(all_gt)
    Y_pred = np.concatenate(all_pred)
    
    precision = precision_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    recall = recall_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    f1 = f1_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    iou = jaccard_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    
    cm = confusion_matrix(Y_true, Y_pred, labels=CLASS_LABELS)
    plot_cm(cm, CLASS_NAMES, f"{model_name}_{plot_suffix}confusion_matrix.png")
    
    return build_report(model_name, vis_images, precision, recall, f1, iou)


def evaluate_lis(model_name, lis_base_dir, target_files, plot_suffix=""):
    print(f"\n--- Evaluating {model_name} ({len(target_files)} tiles) ---")
    all_gt, all_pred, vis_images, vis_titles = [], [], [], []
    
    for gt_filename in target_files:
        gt_path = os.path.join(GT_MASK_DIR, gt_filename)
        lis_path = get_lis_path(gt_filename, lis_base_dir)
        
        if not lis_path:
            continue
            
        try:
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)
            lis_cropped = crop_and_align_lis(lis_path, gt_path)
            
            # Standardize LIS directly (0=NoData, 1=BG, 2=Cloud, 3=Snow)
            pred_std = np.ones_like(lis_cropped, dtype=np.uint8) * 1  # Default BG
            
            is_cloud = (lis_cropped == LIS_VAL_CLOUD)
            is_nodata = (lis_cropped == LIS_VAL_NODATA) | np.isnan(lis_cropped)
            is_snow = (~is_cloud) & (~is_nodata) & (lis_cropped > LIS_THRESHOLD)
            
            pred_std[is_nodata] = 0 # 0 = White
            pred_std[is_cloud] = 2  # 2 = Red
            pred_std[is_snow] = 3   # 3 = Cyan

            all_gt.append(gt_arr.ravel())
            all_pred.append(pred_std.ravel())
            vis_images.append(pred_std)
            vis_titles.append(gt_filename.split('_')[0])
            
        except Exception as e:
            print(f"Error processing LIS for {gt_filename}: {e}")

    if not all_gt: return f"\n=== {model_name} ===\nNo data processed.\n"

    plot_visual_grid(vis_images, vis_titles, f"{model_name}_{plot_suffix}visual_grid.png")
    
    Y_true = np.concatenate(all_gt)
    Y_pred = np.concatenate(all_pred)
    
    # Notice that `labels=CLASS_LABELS` is [1, 2, 3].
    # This automatically IGNORES any pixel marked as 0 (NoData) in metrics!
    precision = precision_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    recall = recall_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    f1 = f1_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    iou = jaccard_score(Y_true, Y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    
    cm = confusion_matrix(Y_true, Y_pred, labels=CLASS_LABELS)
    plot_cm(cm, CLASS_NAMES, f"{model_name}_{plot_suffix}confusion_matrix.png")
    
    return build_report(model_name, vis_images, precision, recall, f1, iou, 
                        note="NoData pixels (0) from LIS are excluded from metrics calculation.")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    final_report = "COMPARISON REPORT: SPUN vs WANG vs LIS\n"
    final_report += "========================================\n\n"
    
    # ========================================================
    # PART 1: MAXIMUM AVAILABLE DATASET (6 TILES)
    # ========================================================
    final_report += "PART 1: MAXIMUM AVAILABLE DATASET (6 TILES)\n"
    final_report += "-------------------------------------------\n"
    final_report += "Evaluating SPUN and Wang on all 6 tiles.\n"
    
    spun_text_full = evaluate_model(
        SPUN_NAME, SPUN_DIR, ALL_TARGET_FILES, 
        is_categorical=SPUN_IS_CATEGORICAL, nodata_val=SPUN_NODATA_VAL, plot_suffix="6tile_"
    )
    final_report += spun_text_full
    
    wang_text_full = evaluate_model(
        WANG_NAME, WANG_DIR, ALL_TARGET_FILES, 
        is_categorical=WANG_IS_CATEGORICAL, mapping=WANG_MAPPING, plot_suffix="6tile_"
    )
    final_report += wang_text_full
    
    # ========================================================
    # PART 2: DIRECT 3-TILE SUBSET (APPLES-TO-APPLES)
    # ========================================================
    final_report += "\n\nPART 2: DIRECT 3-TILE SUBSET (APPLES-TO-APPLES)\n"
    final_report += "-----------------------------------------------\n"
    final_report += "Evaluating ALL methods strictly on the 3 tiles where LIS is available.\n"
    
    spun_text_subset = evaluate_model(
        SPUN_NAME, SPUN_DIR, SUBSET_TARGET_FILES, 
        is_categorical=SPUN_IS_CATEGORICAL, nodata_val=SPUN_NODATA_VAL, plot_suffix="3tile_subset_"
    )
    final_report += spun_text_subset
    
    wang_text_subset = evaluate_model(
        WANG_NAME, WANG_DIR, SUBSET_TARGET_FILES, 
        is_categorical=WANG_IS_CATEGORICAL, mapping=WANG_MAPPING, plot_suffix="3tile_subset_"
    )
    final_report += wang_text_subset
    
    lis_text_subset = evaluate_lis(
        LIS_NAME, LIS_BASE_DIR, SUBSET_TARGET_FILES, plot_suffix="3tile_subset_"
    )
    final_report += lis_text_subset
    
    # --- Save Report ---
    report_path = "comparison_metrics_report_complete.txt"
    with open(report_path, "w") as f:
        f.write(final_report)
        
    print("\n------------------------------------------------")
    print(f"Done! Check '{report_path}' and png plots.")
    print("Plots are saved with '6tile_' and '3tile_subset_' prefixes.")
    print("------------------------------------------------")
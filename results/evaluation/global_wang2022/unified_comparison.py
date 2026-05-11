import os
import glob
import math
import re
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
# import matplotlib.subplots
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

# The exact 4 test files where LIS data is available (UPDATED)
SUBSET_TARGET_FILES = [
    "20200713T103031_20200713T103026_T33VWH_15-31_60-46.tif", # NEW
    "20200804T223709_20200804T223712_T59GLM_169-47_-44-02.tif",
    "20200805T001109_20200805T001647_T55GCN_145-04_-42-94.tif",
    "20210714T081609_20210714T083805_T34HCJ_19-16_-33-13.tif"
]

# Standard Ground Truth Values: 1=BG, 2=Cloud, 3=Snow
CLASS_LABELS = [1, 2, 3]
CLASS_NAMES = ['Background', 'Cloud', 'Snow']
SNOW_VAL = 3

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

def plot_cm(cm, classes, title, save_path, cmap=plt.cm.Blues):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True Label', xlabel='Predicted Label')
    
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
        output[is_nodata] = 2  
        output[is_snow] = 3
    return output

# ==========================================
# 3. CORE EVALUATION LOGIC
# ==========================================

def calculate_all_metrics(Y_true, Y_pred, model_name, num_files, plot_suffix, note=""):
    """Calculates both 3-Class and Snow-Union metrics in one go."""
    
    # --- Filter out NoData (0) globally ---
    valid_mask = (Y_true != 0) & (Y_pred != 0)
    Y_true_valid = Y_true[valid_mask]
    Y_pred_valid = Y_pred[valid_mask]
    
    report = f"\n======================================================\n"
    report += f"=== {model_name.upper()} RESULTS ({num_files} TILES) ===\n"
    report += f"======================================================\n"
    if note: report += f"*** NOTE: {note} ***\n"
    
    # ----------------------------------------------------
    # A. 3-CLASS MACRO EVALUATION
    # ----------------------------------------------------
    precision = precision_score(Y_true_valid, Y_pred_valid, labels=CLASS_LABELS, average=None, zero_division=0)
    recall = recall_score(Y_true_valid, Y_pred_valid, labels=CLASS_LABELS, average=None, zero_division=0)
    f1 = f1_score(Y_true_valid, Y_pred_valid, labels=CLASS_LABELS, average=None, zero_division=0)
    iou = jaccard_score(Y_true_valid, Y_pred_valid, labels=CLASS_LABELS, average=None, zero_division=0)
    
    cm_3class = confusion_matrix(Y_true_valid, Y_pred_valid, labels=CLASS_LABELS)
    plot_cm(cm_3class, CLASS_NAMES, f'3-Class Confusion Matrix ({model_name})', 
            f"{model_name}_{plot_suffix}3class_cm.png", cmap=plt.cm.Blues)
    
    report += f"\n[ 1. GLOBAL 3-CLASS EVALUATION ]\n"
    report += f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'IoU':<10}\n"
    report += "-" * 65 + "\n"
    for i, label in enumerate(CLASS_NAMES):
        report += f"{label:<12} | {precision[i]:.4f}     | {recall[i]:.4f}     | {f1[i]:.4f}     | {iou[i]:.4f}\n"
    report += "-" * 65 + "\n"
    report += f"{'MACRO AVG':<12} | {np.mean(precision):.4f}     | {np.mean(recall):.4f}     | {np.mean(f1):.4f}     | {np.mean(iou):.4f}\n\n"

    # ----------------------------------------------------
    # B. SNOW-UNION EVALUATION
    # ----------------------------------------------------
    union_mask = (Y_true_valid == SNOW_VAL) | (Y_pred_valid == SNOW_VAL)
    Y_true_union = Y_true_valid[union_mask]
    Y_pred_union = Y_pred_valid[union_mask]
    
    TP = np.sum((Y_true_union == SNOW_VAL) & (Y_pred_union == SNOW_VAL))
    FP = np.sum((Y_true_union != SNOW_VAL) & (Y_pred_union == SNOW_VAL))
    FN = np.sum((Y_true_union == SNOW_VAL) & (Y_pred_union != SNOW_VAL))
    
    prec_u = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec_u = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_u = (2 * prec_u * rec_u) / (prec_u + rec_u) if (prec_u + rec_u) > 0 else 0
    iou_u = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    
    cm_union = confusion_matrix(Y_true_union, Y_pred_union, labels=CLASS_LABELS)
    plot_cm(cm_union, CLASS_NAMES, f'Snow-Union Confusion Matrix ({model_name})', 
            f"{model_name}_{plot_suffix}snow_union_cm.png", cmap=plt.cm.Reds)
    
    report += f"[ 2. SNOW-UNION ONLY EVALUATION ]\n"
    report += f"Total pixels in Snow Union: {np.sum(union_mask):,}\n"
    report += f"True Positives: {TP:,} | False Positives: {FP:,} | False Negatives: {FN:,}\n\n"
    report += f"{'Metric':<12} | {'Score':<10}\n"
    report += "-" * 25 + "\n"
    report += f"{'Precision':<12} | {prec_u:.4f}\n"
    report += f"{'Recall':<12} | {rec_u:.4f}\n"
    report += f"{'F1-Score':<12} | {f1_u:.4f}\n"
    report += f"{'IoU':<12} | {iou_u:.4f}\n"
    report += "-" * 25 + "\n"
    
    return report

def run_evaluation(model_name, pred_dir, target_files, is_categorical, mapping=None, nodata_val=None, plot_suffix=""):
    print(f"\nEvaluating {model_name} on {len(target_files)} tiles...")
    all_gt, all_pred = [], []
    
    for gt_filename in target_files:
        gt_path = os.path.join(GT_MASK_DIR, gt_filename)
        pred_path = find_matching_prediction(gt_filename, pred_dir)
        
        if not pred_path: continue
        try:
            with rasterio.open(pred_path) as src: pred_raw = src.read(1)
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)
            
            if pred_raw.shape != gt_arr.shape: continue
                
            pred_std = standardize_prediction(pred_raw, is_categorical, mapping=mapping, nodata_val=nodata_val)
            all_gt.append(gt_arr.ravel())
            all_pred.append(pred_std.ravel())
        except Exception as e:
            print(f"Error processing {gt_filename}: {e}")

    if not all_gt: return f"\n=== {model_name} ===\nNo data processed.\n"
    return calculate_all_metrics(np.concatenate(all_gt), np.concatenate(all_pred), model_name, len(target_files), plot_suffix)

def run_lis_evaluation(model_name, lis_base_dir, target_files, plot_suffix=""):
    print(f"\nEvaluating {model_name} on {len(target_files)} tiles...")
    all_gt, all_pred = [], []
    
    for gt_filename in target_files:
        gt_path = os.path.join(GT_MASK_DIR, gt_filename)
        lis_path = get_lis_path(gt_filename, lis_base_dir)
        
        if not lis_path: continue
        try:
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)
            lis_cropped = crop_and_align_lis(lis_path, gt_path)
            
            pred_std = np.ones_like(lis_cropped, dtype=np.uint8) * 1  
            is_cloud = (lis_cropped == LIS_VAL_CLOUD)
            is_nodata = (lis_cropped == LIS_VAL_NODATA) | np.isnan(lis_cropped)
            is_snow = (~is_cloud) & (~is_nodata) & (lis_cropped > LIS_THRESHOLD)
            
            pred_std[is_nodata] = 0 # Mark 0 for filtering later
            pred_std[is_cloud] = 2  
            pred_std[is_snow] = 3   

            all_gt.append(gt_arr.ravel())
            all_pred.append(pred_std.ravel())
        except Exception as e:
            print(f"Error processing LIS for {gt_filename}: {e}")

    if not all_gt: return f"\n=== {model_name} ===\nNo data processed.\n"
    
    note = "NoData pixels from LIS are excluded from all metric calculations."
    return calculate_all_metrics(np.concatenate(all_gt), np.concatenate(all_pred), model_name, len(target_files), plot_suffix, note=note)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    final_report = "UNIFIED COMPARISON REPORT: SPUN vs WANG vs LIS\n"
    final_report += "================================================\n\n"
    
    # ----------------------------------------------------
    # PART 1: MAXIMUM AVAILABLE DATASET (6 TILES)
    # ----------------------------------------------------
    final_report += "################################################\n"
    final_report += "PART 1: MAXIMUM AVAILABLE DATASET (6 TILES)\n"
    final_report += "################################################\n"
    
    final_report += run_evaluation(SPUN_NAME, SPUN_DIR, ALL_TARGET_FILES, SPUN_IS_CATEGORICAL, nodata_val=SPUN_NODATA_VAL, plot_suffix="6tile_")
    final_report += run_evaluation(WANG_NAME, WANG_DIR, ALL_TARGET_FILES, WANG_IS_CATEGORICAL, mapping=WANG_MAPPING, plot_suffix="6tile_")
    
    # ----------------------------------------------------
    # PART 2: DIRECT SUBSET COMPARISON (4 TILES)
    # ----------------------------------------------------
    final_report += "\n\n################################################\n"
    final_report += "PART 2: DIRECT SUBSET COMPARISON (4 TILES)\n"
    final_report += "################################################\n"
    
    final_report += run_evaluation(SPUN_NAME, SPUN_DIR, SUBSET_TARGET_FILES, SPUN_IS_CATEGORICAL, nodata_val=SPUN_NODATA_VAL, plot_suffix="4tile_")
    final_report += run_evaluation(WANG_NAME, WANG_DIR, SUBSET_TARGET_FILES, WANG_IS_CATEGORICAL, mapping=WANG_MAPPING, plot_suffix="4tile_")
    final_report += run_lis_evaluation(LIS_NAME, LIS_BASE_DIR, SUBSET_TARGET_FILES, plot_suffix="4tile_")
    
    # --- Save Report ---
    report_path = "unified_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(final_report)
        
    print("\n------------------------------------------------")
    print(f"Done! Check '{report_path}'.")
    print("Confusion matrices saved with '6tile_' and '4tile_' prefixes.")
    print("------------------------------------------------")
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
from sklearn.metrics import confusion_matrix

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

# --- The 6 Specific Test Files ---
TARGET_FILES = [
    "20191123T111259_20191123T112151_T29RNQ_-8-3_31-22.tif",
    "20200713T103031_20200713T103026_T33VWH_15-31_60-46.tif",
    "20200804T223709_20200804T223712_T59GLM_169-47_-44-02.tif",
    "20200805T001109_20200805T001647_T55GCN_145-04_-42-94.tif",
    "20201126T041121_20201126T041842_T47SNT_100-09_33-93.tif",
    "20210714T081609_20210714T083805_T34HCJ_19-16_-33-13.tif"
]

CLASS_LABELS = [1, 2, 3]
CLASS_NAMES = ['Background', 'Cloud', 'Snow']
SNOW_CLASS_VAL = 3

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

def plot_cm(cm, classes, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds) # Changed to Reds to highlight errors
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Snow-Union Confusion Matrix',
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
        output[is_nodata] = 2  
        output[is_snow] = 3
    return output

# ==========================================
# 3. CORE EVALUATION LOGIC
# ==========================================

def evaluate_snow_union(Y_true, Y_pred, model_name, note=""):
    # 1. Identify where Snow exists in either Ground Truth OR Prediction
    union_mask = (Y_true == SNOW_CLASS_VAL) | (Y_pred == SNOW_CLASS_VAL)
    
    # 2. Extract only those pixels
    Y_true_union = Y_true[union_mask]
    Y_pred_union = Y_pred[union_mask]
    
    # 3. Calculate Binary True Positives, False Positives, False Negatives
    # Within this union, TNs are completely irrelevant/zero.
    TP = np.sum((Y_true_union == SNOW_CLASS_VAL) & (Y_pred_union == SNOW_CLASS_VAL))
    FP = np.sum((Y_true_union != SNOW_CLASS_VAL) & (Y_pred_union == SNOW_CLASS_VAL))
    FN = np.sum((Y_true_union == SNOW_CLASS_VAL) & (Y_pred_union != SNOW_CLASS_VAL))
    
    # Avoid division by zero
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    
    # 4. Generate the targeted Confusion Matrix
    # This matrix isolates WHAT the snow was confused with (e.g. was False Positive snow actually Cloud?)
    cm = confusion_matrix(Y_true_union, Y_pred_union, labels=CLASS_LABELS)
    plot_cm(cm, CLASS_NAMES, f"{model_name}_snow_union_cm.png")
    
    # 5. Build Report
    report = f"\n=== {model_name} RESULTS (SNOW-UNION FOCUS) ===\n"
    if note:
        report += f"*** NOTE: {note} ***\n"
    report += f"Total pixels in Snow Union (Pred OR Truth = Snow): {np.sum(union_mask):,}\n\n"
    
    report += f"--- SNOW Binary Metrics ---\n"
    report += f"True Positives (Match):       {TP:,}\n"
    report += f"False Positives (Over-pred):  {FP:,}\n"
    report += f"False Negatives (Missed):     {FN:,}\n\n"
    
    report += f"{'Metric':<12} | {'Score':<10}\n"
    report += "-" * 25 + "\n"
    report += f"{'Precision':<12} | {precision:.4f}\n"
    report += f"{'Recall':<12} | {recall:.4f}\n"
    report += f"{'F1-Score':<12} | {f1:.4f}\n"
    report += f"{'IoU':<12} | {iou:.4f}\n"
    report += "-" * 25 + "\n"
    
    return report

def evaluate_model(model_name, pred_dir, is_categorical, mapping=None, nodata_val=None):
    print(f"\n--- Evaluating {model_name} ---")
    all_gt, all_pred = [], []
    
    for gt_filename in TARGET_FILES:
        gt_path = os.path.join(GT_MASK_DIR, gt_filename)
        pred_path = find_matching_prediction(gt_filename, pred_dir)
        
        if not pred_path: continue
            
        try:
            with rasterio.open(pred_path) as src: pred_raw = src.read(1)
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)
            
            if pred_raw.shape != gt_arr.shape: continue
                
            pred_std = standardize_prediction(
                pred_raw, is_categorical, mapping=mapping, nodata_val=nodata_val)
            
            all_gt.append(gt_arr.ravel())
            all_pred.append(pred_std.ravel())
            
        except Exception as e:
            print(f"Error processing {gt_filename}: {e}")

    if not all_gt: return f"\n=== {model_name} ===\nNo data processed.\n"
    
    Y_true = np.concatenate(all_gt)
    Y_pred = np.concatenate(all_pred)
    
    return evaluate_snow_union(Y_true, Y_pred, model_name)

def evaluate_lis(model_name, lis_base_dir):
    print(f"\n--- Evaluating {model_name} ---")
    all_gt, all_pred = [], []
    
    for gt_filename in TARGET_FILES:
        gt_path = os.path.join(GT_MASK_DIR, gt_filename)
        lis_path = get_lis_path(gt_filename, lis_base_dir)
        
        if not lis_path: continue
            
        try:
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)
            lis_cropped = crop_and_align_lis(lis_path, gt_path)
            
            # Standardize LIS directly (0=NoData, 1=BG, 2=Cloud, 3=Snow)
            pred_std = np.ones_like(lis_cropped, dtype=np.uint8) * 1  
            
            is_cloud = (lis_cropped == LIS_VAL_CLOUD)
            is_nodata = (lis_cropped == LIS_VAL_NODATA) | np.isnan(lis_cropped)
            is_snow = (~is_cloud) & (~is_nodata) & (lis_cropped > LIS_THRESHOLD)
            
            pred_std[is_nodata] = 0 
            pred_std[is_cloud] = 2  
            pred_std[is_snow] = 3   

            all_gt.append(gt_arr.ravel())
            all_pred.append(pred_std.ravel())
            
        except Exception as e:
            print(f"Error processing LIS for {gt_filename}: {e}")

    if not all_gt: return f"\n=== {model_name} ===\nNo data processed.\n"
    
    Y_true = np.concatenate(all_gt)
    Y_pred = np.concatenate(all_pred)
    
    # We must explicitly ignore the NoData (0) values that LIS might output before unioning
    valid_mask = (Y_pred != 0)
    Y_true = Y_true[valid_mask]
    Y_pred = Y_pred[valid_mask]
    
    note = "This product was evaluated for 3 out of the 6 test tiles. NoData pixels were excluded before forming the union."
    return evaluate_snow_union(Y_true, Y_pred, model_name, note=note)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    final_report = "SNOW-UNION COMPARISON REPORT: SPUN vs WANG vs LIS\n"
    final_report += "====================================================\n"
    final_report += "Metrics are calculated exclusively on pixels where either the Prediction OR the Ground Truth contains Snow.\n\n"
    
    # --- Run SPUN Evaluation ---
    spun_text = evaluate_model(
        SPUN_NAME, SPUN_DIR, 
        is_categorical=SPUN_IS_CATEGORICAL, nodata_val=SPUN_NODATA_VAL
    )
    final_report += spun_text
    
    # --- Run Wang Evaluation ---
    wang_text = evaluate_model(
        WANG_NAME, WANG_DIR, 
        is_categorical=WANG_IS_CATEGORICAL, mapping=WANG_MAPPING
    )
    final_report += wang_text
    
    # --- Run LIS Evaluation ---
    lis_text = evaluate_lis(
        LIS_NAME, LIS_BASE_DIR
    )
    final_report += lis_text
    
    # --- Save Report ---
    with open("comparison_snow_union_report.txt", "w") as f:
        f.write(final_report)
        
    print("\n------------------------------------------------")
    print("Done! Check 'comparison_snow_union_report.txt' and '*_snow_union_cm.png'.")
    print("------------------------------------------------")
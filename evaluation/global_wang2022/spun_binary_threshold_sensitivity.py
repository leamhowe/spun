import glob
import os
import numpy as np
import rasterio
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================

GT_MASK_DIR = '/home/users/leamhowe/phd/1_DETECT/Wang2022/datasets/masks'

DIRS_TO_COMPARE = {
    "No Mask (Raw)": '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/Final_T15_dino_none_scale10000_best_nomask',
    "With Cloud Mask":     '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/Final_T15_dino_none_scale10000_best_mosaicked'
}

# Define the Thresholds to Test (0.01 = 1%, 0.50 = 50%)
THRESHOLDS_TO_TEST = [0.01, 0.50]

NODATA_VALUES = [200, 255]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_gt_path_generic(pred_path, gt_pool):
    fname = os.path.basename(pred_path)
    target_name = fname[5:] if fname.startswith('pred_') else fname
    return next((g for g in gt_pool if os.path.basename(g) == target_name), None)

def calculate_metrics_at_threshold(file_list, gt_pool, threshold):
    """Calculates global metrics for a specific SCF threshold."""
    y_true_all = []
    y_pred_all = []

    for pred_path in file_list:
        gt_path = get_gt_path_generic(pred_path, gt_pool)
        if not gt_path: continue

        try:
            with rasterio.open(pred_path) as src: pred_arr = src.read(1)
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)

            if pred_arr.shape != gt_arr.shape: continue

            # --- GT Logic: Clouds(2) -> 0 ---
            gt_binary = np.zeros_like(gt_arr, dtype=np.uint8)
            gt_binary[gt_arr == 3] = 1

            # --- Pred Logic: Thresholding ---
            pred_binary = np.zeros_like(pred_arr, dtype=np.uint8)
            
            # Identify Valid Pixels
            is_valid = ~np.isnan(pred_arr)
            for nd_val in NODATA_VALUES:
                is_valid = is_valid & (pred_arr != nd_val)

            # Apply Dynamic Threshold
            pred_probs = pred_arr / 100.0
            is_snow = (pred_probs > threshold) & is_valid
            
            pred_binary[is_snow] = 1
            
            y_true_all.append(gt_binary.flatten())
            y_pred_all.append(pred_binary.flatten())

        except Exception:
            continue

    if not y_true_all: return None

    y_true_global = np.concatenate(y_true_all)
    y_pred_global = np.concatenate(y_pred_all)
    
    prec = precision_score(y_true_global, y_pred_global, zero_division=0)
    rec  = recall_score(y_true_global, y_pred_global, zero_division=0)
    f1   = f1_score(y_true_global, y_pred_global, zero_division=0)
    
    cm = confusion_matrix(y_true_global, y_pred_global)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "Threshold": f"> {int(threshold*100)}%",
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "FP Rate": fpr
    }

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    gt_files_all = glob.glob(os.path.join(GT_MASK_DIR, '*.tif'))
    results_list = []

    for model_name, dir_path in DIRS_TO_COMPARE.items():
        print(f"--- Scanning {model_name} ---")
        
        # Get valid files (Full Dataset N=40)
        all_tifs = sorted(glob.glob(os.path.join(dir_path, '*.tif')))
        valid_files = [f for f in all_tifs if not f.endswith('_s2.tif') and get_gt_path_generic(f, gt_files_all)]
        
        print(f"   Processing {len(valid_files)} files...")

        # Run for each threshold
        for thresh in THRESHOLDS_TO_TEST:
            res = calculate_metrics_at_threshold(valid_files, gt_files_all, thresh)
            if res:
                res["Model"] = model_name
                results_list.append(res)

    if results_list:
        df = pd.DataFrame(results_list)
        cols = ["Model", "Threshold", "Precision", "Recall", "F1 Score", "FP Rate"]
        
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS: Impact of SCF Threshold on Global Performance")
        print("Dataset: Full Global Set (N=40)")
        print("Metric: Clouds in GT are treated as 'No Snow' (0).")
        print("="*80)
        print(df[cols].round(4).to_string(index=False))
        print("="*80)
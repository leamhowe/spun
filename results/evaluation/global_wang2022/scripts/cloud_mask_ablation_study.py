import glob
import os
import argparse
import numpy as np
import rasterio
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. CONFIGURATION
# ==========================================

# THE 6 SPECIFIC TEST IMAGES (Subset)
TARGET_SUBSET_IDS = [
    "20191123T111259_20191123T112151_T29RNQ_-8-3_31-22",
    "20200713T103031_20200713T103026_T33VWH_15-31_60-46",
    "20200804T223709_20200804T223712_T59GLM_169-47_-44-02",
    "20200805T001109_20200805T001647_T55GCN_145-04_-42-94",
    "20201126T041121_20201126T041842_T47SNT_100-09_33-93",
    "20210714T081609_20210714T083805_T34HCJ_19-16_-33-13"
]

# Thresholds & NoData Handling
PREDICTION_THRESHOLD = 0.5  # 50% SCF cut-off for binary classification
NODATA_VALUES = [200, 255]  # Values in the TIF that represent Cloud/NoData

def get_args():
    parser = argparse.ArgumentParser(description="Cloud Confusion Analysis: Masked vs Unmasked SPUN")
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing the Ground Truth mask TIFFs')
    parser.add_argument('--nomask_dir', type=str, required=True, help='Directory containing Unmasked SPUN predictions')
    parser.add_argument('--masked_dir', type=str, required=True, help='Directory containing Cloud-Masked SPUN predictions')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the text report')
    return parser.parse_args()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_gt_path_generic(pred_path, gt_pool):
    """Finds GT file by matching filename (ignoring 'pred_' prefix)."""
    fname = os.path.basename(pred_path)
    if fname.startswith('pred_'):
        target_name = fname[5:] # Remove 'pred_'
    else:
        target_name = fname

    # Find exact match in GT pool
    match = next((g for g in gt_pool if os.path.basename(g) == target_name), None)
    return match

def calculate_metrics_for_files(file_list, gt_pool, dataset_name):
    """Core function to calculate metrics for a given list of files."""
    print(f"   -> Computing metrics for {dataset_name} (n={len(file_list)})...")
    
    y_true_all = []
    y_pred_all = []

    for pred_path in file_list:
        gt_path = get_gt_path_generic(pred_path, gt_pool)
        
        if not gt_path: 
            continue # Should be handled before, but safety check

        try:
            # 1. Load Prediction
            with rasterio.open(pred_path) as src:
                pred_arr = src.read(1)
            
            # 2. Load GT
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)

            if pred_arr.shape != gt_arr.shape:
                continue

            # --- Logic: GT Clouds (2) -> Background (0) ---
            gt_binary = np.zeros_like(gt_arr, dtype=np.uint8)
            gt_binary[gt_arr == 3] = 1  # Only Class 3 is Snow

            # --- Logic: Prediction Processing ---
            pred_binary = np.zeros_like(pred_arr, dtype=np.uint8)
            
            # Valid = Not NaN, Not 200, Not 255
            is_valid = ~np.isnan(pred_arr)
            for nd_val in NODATA_VALUES:
                is_valid = is_valid & (pred_arr != nd_val)

            pred_probs = pred_arr / 100.0
            is_snow = (pred_probs > PREDICTION_THRESHOLD) & is_valid
            
            pred_binary[is_snow] = 1
            
            y_true_all.append(gt_binary.flatten())
            y_pred_all.append(pred_binary.flatten())

        except Exception as e:
            print(f"Error: {e}")

    if not y_true_all:
        return None

    y_true_global = np.concatenate(y_true_all)
    y_pred_global = np.concatenate(y_pred_all)
    
    # Metrics
    prec = precision_score(y_true_global, y_pred_global, zero_division=0)
    rec  = recall_score(y_true_global, y_pred_global, zero_division=0)
    f1   = f1_score(y_true_global, y_pred_global, zero_division=0)
    
    cm = confusion_matrix(y_true_global, y_pred_global)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "Dataset": dataset_name,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "False Positives": fp,
        "FP Rate": fpr,
        "Count": len(file_list)
    }

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    args = get_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    # Map the command-line arguments to the comparison dictionary
    DIRS_TO_COMPARE = {
        "No Mask (Raw Model)": args.nomask_dir,
        "With Cloud Mask":     args.masked_dir
    }
    
    # 1. Index all Ground Truth Files
    gt_files_all = glob.glob(os.path.join(args.gt_dir, '*.tif'))
    print(f"Total GT Files indexed: {len(gt_files_all)}")

    results_list = []

    for model_name, dir_path in DIRS_TO_COMPARE.items():
        print(f"\n--- Scanning: {model_name} ---")
        
        # A. Find ALL valid prediction files in this directory
        all_tifs = glob.glob(os.path.join(dir_path, '*.tif'))
        # Filter out _s2.tif and ensure matching GT exists
        valid_full_set = []
        for f in all_tifs:
            if f.endswith('_s2.tif'): continue
            if get_gt_path_generic(f, gt_files_all):
                valid_full_set.append(f)
        
        valid_full_set = sorted(valid_full_set)
        
        # B. Identify the SUBSET from the full list
        subset_files = []
        for f in valid_full_set:
            fname = os.path.basename(f)
            if any(tid in fname for tid in TARGET_SUBSET_IDS):
                subset_files.append(f)
        
        # C. Run Metrics for Full Set
        res_full = calculate_metrics_for_files(valid_full_set, gt_files_all, "Full Dataset")
        if res_full:
            res_full["Model"] = model_name
            results_list.append(res_full)

        # D. Run Metrics for Subset
        res_sub = calculate_metrics_for_files(subset_files, gt_files_all, "Test Subset")
        if res_sub:
            res_sub["Model"] = model_name
            results_list.append(res_sub)

    # 4. Display & Save Results
    if results_list:
        df = pd.DataFrame(results_list)
        
        # Reorder columns
        cols = ["Model", "Dataset", "Count", "Precision", "Recall", "F1 Score", "FP Rate"]
        
        report_text = "="*90 + "\n"
        report_text += "CLOUD CONFUSION ANALYSIS: Full Dataset (N=40) vs Subset (N=6)\n"
        report_text += "Metric: Clouds in GT are treated as 'No Snow' (0).\n"
        report_text += "="*90 + "\n"
        report_text += df[cols].round(4).to_string(index=False) + "\n"
        report_text += "="*90 + "\n"
        
        # Print to console
        print(f"\n{report_text}")
        
        # Save to output directory
        report_path = os.path.join(args.output_dir, "cloud_confusion_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        print(f"Report successfully saved to: {report_path}")
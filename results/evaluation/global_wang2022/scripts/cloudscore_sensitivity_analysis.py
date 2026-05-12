import glob
import os
import argparse
import numpy as np
import rasterio
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

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

# --- SENSITIVITY CONFIGURATION ---
STATIC_SNOW_THRESHOLD = 0.5  # Locked SPUN snow threshold (50%)
CLOUD_THRESHOLDS_TO_TEST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  

def get_args():
    parser = argparse.ArgumentParser(description="CloudScore+ Sensitivity Analysis for SPUN")
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing the Ground Truth mask TIFFs')
    parser.add_argument('--spun_dir', type=str, required=True, help='Directory containing SPUN prediction TIFFs')
    parser.add_argument('--cloud_dir', type=str, required=True, help='Directory containing CloudScore+ probability TIFFs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the report and plots')
    return parser.parse_args()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_matching_gt(snow_path, gt_dir):
    """Finds Ground Truth matching the prediction."""
    base_name = os.path.basename(snow_path).replace('pred_', '')
    candidates = glob.glob(os.path.join(gt_dir, base_name))
    return candidates[0] if candidates else None

def get_matching_cloud(snow_path, cloud_dir):
    """Finds the continuous Cloud Score+ TIF matching the prediction."""
    base_name = os.path.basename(snow_path).replace('pred_', '')
    candidates = glob.glob(os.path.join(cloud_dir, f"cs_prob_{base_name}"))
    return candidates[0] if candidates else None

def calculate_metrics_dynamic_cloud(snow_files, dataset_name, clear_thresh, gt_dir, cloud_dir):
    """Calculates metrics by dynamically applying a 'Clear' threshold."""
    y_true_all = []
    y_pred_all = []
    processed_count = 0

    for snow_path in snow_files:
        gt_path = get_matching_gt(snow_path, gt_dir)
        cloud_path = get_matching_cloud(snow_path, cloud_dir)
        
        if not gt_path or not cloud_path: 
            continue 

        try:
            with rasterio.open(snow_path) as src: snow_arr = src.read(1)
            with rasterio.open(cloud_path) as src: cloud_arr = src.read(1)
            gt_img = Image.open(gt_path).convert('L')
            gt_arr = np.array(gt_img)

            # Robust Shape Alignment: Find minimum intersection
            h = min(snow_arr.shape[0], cloud_arr.shape[0], gt_arr.shape[0])
            w = min(snow_arr.shape[1], cloud_arr.shape[1], gt_arr.shape[1])
            
            # Crop all to the same size
            snow_arr = snow_arr[:h, :w]
            cloud_arr = cloud_arr[:h, :w]
            gt_arr = gt_arr[:h, :w]

            # Ground Truth: 3 is Snow
            gt_binary = (gt_arr == 3).astype(np.uint8)

            # --- THE LOGIC FIX ---
            # cs_cdf is CLEAR probability. 
            # If clear probability < thresh, it is CLOUDY (mask it).
            is_cloud = (cloud_arr < clear_thresh)
            
            snow_probs = snow_arr / 100.0
            # Valid if Snow > 0.5 AND NOT cloudy
            is_snow_pred = (snow_probs >= STATIC_SNOW_THRESHOLD) & (~is_cloud)
            
            pred_binary = is_snow_pred.astype(np.uint8)
            
            y_true_all.append(gt_binary.flatten())
            y_pred_all.append(pred_binary.flatten())
            processed_count += 1

        except Exception as e:
            print(f"Error processing {os.path.basename(snow_path)}: {e}")

    if not y_true_all:
        return None

    # Print how many files actually contributed to this result
    if clear_thresh == CLOUD_THRESHOLDS_TO_TEST[0]:
        print(f"   -> Successfully processed {processed_count} files for {dataset_name}")

    y_true_global = np.concatenate(y_true_all)
    y_pred_global = np.concatenate(y_pred_all)
    
    # Calculate Metrics
    prec = precision_score(y_true_global, y_pred_global, zero_division=0)
    rec  = recall_score(y_true_global, y_pred_global, zero_division=0)
    f1   = f1_score(y_true_global, y_pred_global, zero_division=0)
    
    cm = confusion_matrix(y_true_global, y_pred_global)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "Dataset": dataset_name,
        "Clear Threshold": clear_thresh,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "FP Rate": fpr,
        "Count": processed_count
    }

def plot_cloud_sensitivity(df, output_dir, output_filename="cloudscore_threshold_sensitivity.png"):
    """Generates a line plot showing how metrics change as the CLEAR threshold increases."""
    datasets = df['Dataset'].unique()
    
    fig, axs = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    
    for i, dataset in enumerate(datasets):
        ax = axs[0, i]
        subset = df[df['Dataset'] == dataset].sort_values(by="Clear Threshold")
        
        if subset.empty: continue
        
        ax.plot(subset['Clear Threshold'], subset['F1 Score'], marker='o', label='F1 Score', color='black', linewidth=2)
        ax.plot(subset['Clear Threshold'], subset['Precision'], marker='s', label='Precision', color='blue', linestyle='--')
        ax.plot(subset['Clear Threshold'], subset['Recall'], marker='^', label='Recall', color='green', linestyle='--')
        
        ax.set_title(f"CloudScore+ Sensitivity\n({dataset})", fontweight='bold')
        ax.set_xlabel('Clear Probability Threshold (Strict \u2192 Lenient)')
        ax.set_ylabel('Score')
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='lower right')
        
        # Highlight max F1 score
        max_f1_idx = subset['F1 Score'].idxmax()
        best_thresh = subset.loc[max_f1_idx, 'Clear Threshold']
        best_f1 = subset.loc[max_f1_idx, 'F1 Score']
        ax.annotate(f'Max F1: {best_f1:.3f}\n@ thresh {best_thresh}', 
                    xy=(best_thresh, best_f1), xytext=(best_thresh-0.1, best_f1-0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    plt.tight_layout()
    out_path = os.path.join(output_dir, output_filename)
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved successfully to {out_path}")
    plt.close()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    args = get_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    print(f"Scanning Snow Predictions in: {args.spun_dir}")
    all_snow_tifs = glob.glob(os.path.join(args.spun_dir, '*.tif'))
    valid_full_set = [f for f in all_snow_tifs if not f.endswith('_s2.tif')]
    valid_full_set = sorted(valid_full_set)
    
    # Identify the SUBSET
    subset_files = []
    for f in valid_full_set:
        fname = os.path.basename(f)
        if any(tid in fname for tid in TARGET_SUBSET_IDS):
            subset_files.append(f)
            
    print(f"Found {len(valid_full_set)} Full Dataset files.")
    print(f"Found {len(subset_files)} Subset files.")

    results_list = []

    for thresh in CLOUD_THRESHOLDS_TO_TEST:
        print(f"\n--- Testing Cloud Threshold: {thresh} ---")
        
        # Full Set
        res_full = calculate_metrics_dynamic_cloud(valid_full_set, "Full Dataset", thresh, args.gt_dir, args.cloud_dir)
        if res_full: results_list.append(res_full)

        # Subset
        res_sub = calculate_metrics_dynamic_cloud(subset_files, "Test Subset (N=6)", thresh, args.gt_dir, args.cloud_dir)
        if res_sub: results_list.append(res_sub)

    # 4. Display Results
    if results_list:
        df = pd.DataFrame(results_list)
        
        cols = ["Dataset", "Clear Threshold", "Precision", "Recall", "F1 Score", "FP Rate"]
        
        print("\n" + "="*80)
        print("CLOUDSCORE+ SENSITIVITY ANALYSIS (Static Snow Threshold = 0.5)")
        print("NOTE: 'Clear Threshold' is the minimum cs_cdf required for a pixel to NOT be masked.")
        print("="*80)
        
        report_text = "CLOUDSCORE+ SENSITIVITY ANALYSIS\n================================\n\n"
        
        for dataset in df['Dataset'].unique():
            print(f"\n>>> {dataset} <<<")
            sub_df = df[df['Dataset'] == dataset][cols].sort_values(by="Clear Threshold")
            output_str = sub_df.round(4).to_string(index=False)
            print(output_str)
            report_text += f"\n>>> {dataset} <<<\n{output_str}\n"
            
        print("="*80)
        
        # Save results to output directory
        report_path = os.path.join(args.output_dir, "cloud_sensitivity_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        print(f"Text report saved to {report_path}")
        
        # 5. Generate Visual Curves
        plot_cloud_sensitivity(df, args.output_dir)
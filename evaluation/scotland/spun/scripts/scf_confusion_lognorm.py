import os
import glob
import zarr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LogNorm  # <--- NEW IMPORT

# ==========================================
# 1. CONFIGURATION
# ==========================================

INPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
OUTPUT_FILENAME = 'scf_binned_confusion_matrix_log.png'

NODATA_VAL = 200 
BIN_LABELS = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
              '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

# ==========================================
# 2. MAIN EXECUTION
# ==========================================

def main():
    print(f"Searching for Zarr stores in: {INPUT_DIR}")
    zarr_files = glob.glob(os.path.join(INPUT_DIR, '*.zarr'))
    
    if not zarr_files:
        print("No Zarr files found.")
        return

    total_cm = np.zeros((10, 10), dtype=np.int64)

    for zarr_path in zarr_files:
        try:
            root = zarr.open(zarr_path, mode='r')
            
            pred_scf = root['spun_scf'][:].flatten()
            gt_scf = root['label'][:].flatten()
            
            if np.nanmax(gt_scf) <= 1.0 and np.nanmax(gt_scf) > 0:
                gt_scf = gt_scf * 100.0
                
            valid_mask = (
                (~np.isnan(pred_scf)) & (~np.isnan(gt_scf)) & 
                (pred_scf <= 100) & (gt_scf <= 100) & 
                (pred_scf >= 0) & (gt_scf >= 0)
            )
            
            pred_valid = pred_scf[valid_mask]
            gt_valid = gt_scf[valid_mask]

            if len(pred_valid) == 0: continue

            pred_bins = (pred_valid // 10).astype(int)
            gt_bins = (gt_valid // 10).astype(int)

            pred_bins[pred_bins == 10] = 9
            gt_bins[gt_bins == 10] = 9

            cm = confusion_matrix(gt_bins, pred_bins, labels=range(10))
            total_cm += cm

        except Exception as e:
            print(f"Error processing {os.path.basename(zarr_path)}: {e}")

    # ==========================================
    # 3. NORMALIZATION & PLOTTING
    # ==========================================
    
    if np.sum(total_cm) == 0:
        print("No valid pixels processed.")
        return

    # Normalize by row (True Labels)
    row_sums = total_cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1 
    cm_normalized = total_cm.astype('float') / row_sums
    cm_percentages = cm_normalized * 100

    # Create Plot
    plt.figure(figsize=(10, 8))
    
    # --- THE FIX: Apply LogNorm ---
    # We add a tiny value (0.1) to vmin so log(0) doesn't break the colormap.
    # This will make a 1% cell visible even next to a 99% cell.
    sns.heatmap(
        cm_percentages, 
        annot=True,               
        fmt=".1f",                
        cmap="Blues",             
        norm=LogNorm(vmin=0.1, vmax=100), # <-- Logarithmic color scale
        xticklabels=BIN_LABELS, 
        yticklabels=BIN_LABELS,
        cbar_kws={'label': 'Percentage of True Class (%) (Log Scale)'}
    )

    plt.title('Normalized Snow Cover Fraction Confusion Matrix\n(Logarithmic Color Scale)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted SCF', fontsize=12, fontweight='bold')
    plt.ylabel('True SCF', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"\nMatrix saved to {OUTPUT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    main()
import glob
import os

# ==========================================
# CONFIGURATION
# ==========================================
GT_MASK_DIR = '/home/users/leamhowe/phd/1_DETECT/Wang2022/datasets/masks'

DIRS_TO_COMPARE = {
    "No Mask": '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/Final_T15_dino_none_scale10000_best_nomask',
    "With Mask": '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/wang_comparison/Final_T15_dino_none_scale10000_best_mosaicked'
}

DATES_TO_PROCESS = [20191123, 20200713, 20200804, 20200805, 20201126, 20210714]

# ==========================================
# DEBUG LOGIC
# ==========================================

def debug_directory(name, dir_path):
    print(f"\n--- DEBUGGING: {name} ---")
    print(f"Path: {dir_path}")
    
    # 1. Check if directory exists
    if not os.path.exists(dir_path):
        print("ERROR: Directory does not exist.")
        return

    # 2. List all TIF files
    all_files = glob.glob(os.path.join(dir_path, '*.tif'))
    print(f"Total TIF files found: {len(all_files)}")
    
    if len(all_files) == 0:
        print("ERROR: No .tif files found. Check extension (is it .tiff?)")
        return

    print(f"First 3 files found: {[os.path.basename(f) for f in all_files[:3]]}")

    # 3. Check Date Filtering
    print(f"\nAttempting to filter for dates: {DATES_TO_PROCESS}")
    date_strings = [str(d) for d in DATES_TO_PROCESS]
    
    filtered_files = [f for f in all_files if any(d in os.path.basename(f) for d in date_strings)]
    print(f"Files remaining after date filter: {len(filtered_files)}")
    
    if len(filtered_files) == 0:
        print("FAILURE: Date filtering removed all files. Check filename format vs DATES_TO_PROCESS.")
        return

    # 4. Attempt Matching with Ground Truth
    print("\nAttempting to match with Ground Truth...")
    gt_files = glob.glob(os.path.join(GT_MASK_DIR, '*.tif'))
    print(f"Total GT files available: {len(gt_files)}")
    print(f"Sample GT filename: {os.path.basename(gt_files[0]) if gt_files else 'NONE'}")
    
    sample_pred = filtered_files[0]
    sample_pred_name = os.path.basename(sample_pred)
    
    # Extraction Logic Debug
    # Extract date (assuming YYYYMMDD format exists in filename)
    date_part = next((d for d in sample_pred_name.split('_') if d.isdigit() and len(d)==8), None)
    print(f"Extracted Date from '{sample_pred_name}': {date_part}")
    
    if date_part:
        match = next((g for g in gt_files if date_part in os.path.basename(g)), None)
        if match:
            print(f"SUCCESS: Matched with GT file: {os.path.basename(match)}")
        else:
            print(f"FAILURE: Could not find GT file containing string '{date_part}'")
    else:
        print("FAILURE: Could not extract an 8-digit date from filename.")

if __name__ == "__main__":
    for name, path in DIRS_TO_COMPARE.items():
        debug_directory(name, path)
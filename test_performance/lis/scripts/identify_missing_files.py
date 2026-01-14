import pandas as pd
import os

# --- CONFIGURATION (Matches your evaluate_lis.py) ---
SPLIT_CSV_PATH = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches_DEBUG_old2'

def extract_ids_from_path(filepath):
    """
    Extracts date and tile ID from the label filename.
    Example: '.../20190908_NJ1605.zarr' -> ('20190908', 'NJ1605')
    """
    fname = os.path.basename(filepath)
    # Split logic matches your evaluation script
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

def find_missing_files():
    # 1. Load the CSV
    print(f"Loading CSV: {SPLIT_CSV_PATH}")
    df = pd.read_csv(SPLIT_CSV_PATH)
    
    # 2. Filter for Test set (since your log showed 934 tiles)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_df = df[df[col] == 'Test']
    
    print(f"Found {len(test_df)} tiles in the '{col} == Test' set.")
    
    missing_files = []
    
    # 3. Check for existence of corresponding LIS files
    print(f"Checking against directory: {LIS_PATCHES_DIR} ...")
    
    for idx, row in test_df.iterrows():
        label_path = row['filepath']
        
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
        except Exception as e:
            print(f"Skipping malformed label path at row {idx}: {label_path}")
            continue
            
        # Construct the expected LIS filename
        expected_lis_name = f"{date_id}_{tile_id}_LIS.zarr"
        expected_lis_path = os.path.join(LIS_PATCHES_DIR, expected_lis_name)
        
        # Check if it exists
        if not os.path.exists(expected_lis_path):
            missing_files.append(expected_lis_name)
    
    # 4. Report Results
    print("="*40)
    print(f"RESULTS")
    print("="*40)
    print(f"Total Expected: {len(test_df)}")
    print(f"Total Missing:  {len(missing_files)}")
    print(f"Total Found:    {len(test_df) - len(missing_files)}")
    print("-" * 30)
    
    if len(missing_files) > 0:
        print("First 10 missing files:")
        for f in missing_files[:10]:
            print(f" - {f}")
            
        # Save full list to file
        out_file = "missing_lis_files.txt"
        with open(out_file, "w") as f:
            for item in missing_files:
                f.write(f"{item}\n")
        print("-" * 30)
        print(f"Full list of missing files saved to: {out_file}")
    else:
        print("All LIS files are present!")

if __name__ == "__main__":
    find_missing_files()
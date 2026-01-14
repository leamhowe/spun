import pandas as pd
import zarr
import os
import sys

# --- CONFIG ---
SPLIT_CSV_PATH = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

print(f"Python Executable: {sys.executable}")
print(f"Zarr Version: {zarr.__version__}")

# 1. Get the first file
df = pd.read_csv(SPLIT_CSV_PATH)
col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
test_files = df[df[col] == 'Test']['filepath'].tolist()
target_file = test_files[0]

print(f"\nAttempting to open: {target_file}")

# 2. Try to open it exactly how the script does
try:
    # Explicitly check if file exists
    if not os.path.exists(target_file):
        print(">> FATAL: File path does not exist!")
    
    # Try opening
    z = zarr.open(target_file, mode='r')
    print(">> SUCCESS: Group opened.")
    
    # Try reading data
    data = z['label'][:]
    print(f">> SUCCESS: Data read. Shape: {data.shape}")

except Exception as e:
    print("\n>> CRASH REPORT <<")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {e}")
    import traceback
    traceback.print_exc()
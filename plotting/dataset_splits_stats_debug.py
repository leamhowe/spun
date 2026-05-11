import os
import zarr
import numpy as np

# Grab the very first filepath from your dataframe
test_file = merged_gdf_cleaned['filepath'].iloc[0]

print(f"1. Testing file path: {test_file}")
print(f"2. Does this file actually exist on disk? {os.path.exists(test_file)}")

if os.path.exists(test_file):
    try:
        store = zarr.open(test_file, mode='r')
        
        # Check what keys are actually inside this Zarr store
        # Is it 'label', 'mask', 'labels'?
        # For a flat zarr array, this might throw an AttributeError, which is also good to know!
        try:
            keys = list(store.keys())
            print(f"3. Keys in Zarr store: {keys}")
        except AttributeError:
            print("3. Zarr store does not have keys (it might be a direct array).")

        # Try to load the label
        gt = store['label'][:]
        print(f"4. Label array shape: {gt.shape}")
        
        # See what values are actually inside
        unique_vals = np.unique(gt)
        print(f"5. Unique values in this label array: {unique_vals}")
        
    except Exception as e:
        print(f"-> ERROR while reading Zarr: {e}")
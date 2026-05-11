import pandas as pd
import numpy as np
import os
import zarr
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- IMPORTS ---
# We ONLY need the dataset loader, not the model
sys.path.append('/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts') 
from src.snow_s2_dataset_13bands import Snows2Dataset

# --- CONFIGURATION ---
SPLIT_CSV_PATH  = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
OUTPUT_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_ndsi'

# Sentinel-2 Bands (Index mapping)
BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
IDX_GREEN = BANDS.index('B3')  # Index 2
IDX_SWIR  = BANDS.index('B11') # Index 11

# Target Output Size (Matches LIS 50x50 @ 20m -> 100x100 @ 10m)
TARGET_SHAPE = (100, 100) 

def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

def center_crop(arr, target_h=100, target_w=100):
    """Crops the center of the array to target size."""
    h, w = arr.shape
    if h == target_h and w == target_w: return arr
    
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    if start_h < 0 or start_w < 0: return arr 
    return arr[start_h:start_h+target_h, start_w:start_w+target_w]

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print("--- LOADING CONFIGURATION ---")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    # Init Dataset (No model needed)
    print("Initializing Dataset...")
    ds = Snows2Dataset(test_files, bands=BANDS, train=False, target_type='scf', train_with_clouds=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    print(f"Generating NDSI Baselines for {len(test_files)} tiles...")

    for i, batch in enumerate(tqdm(dl)):
        try:
            # 1. Get Bands (Batch 0)
            # image shape: [Batch, Channel, H, W]
            b_green = batch['image'][0, IDX_GREEN, :, :].numpy()
            b_swir  = batch['image'][0, IDX_SWIR, :, :].numpy()
            
            # 2. Calculate NDSI
            # NDSI = (Green - SWIR) / (Green + SWIR)
            denom = b_green + b_swir
            denom[denom == 0] = 1e-6 # Avoid div by zero
            ndsi_raw = (b_green - b_swir) / denom
            
            # 3. Get Ground Truth
            gt_raw = batch['mask'][0].numpy()
            
            # 4. Geometric Corrections (Flip & Crop)
            # Essential to match LIS orientation and size
            ndsi_flipped = ndsi_raw[::-1, :] 
            gt_flipped   = gt_raw[::-1, :]
            
            ndsi_final = center_crop(ndsi_flipped, TARGET_SHAPE[0], TARGET_SHAPE[1])
            gt_final   = center_crop(gt_flipped, TARGET_SHAPE[0], TARGET_SHAPE[1])

            # 5. Save to Zarr
            label_path = test_files[i]
            date_id, tile_id = extract_ids_from_path(label_path)
            
            # We append to the existing SPUN file if it exists, or create new if not
            save_name = f"{date_id}_{tile_id}_NDSI.zarr"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            # Open in 'a' (append) mode if exists, else 'w'
            mode = 'r+' if os.path.exists(save_path) else 'w'
            
            root = zarr.open(save_path, mode=mode)
            
            # Save NDSI (Overwrite if exists)
            # We save raw float values. Evaluation script will handle thresholding > 0.4
            root.create_dataset('ndsi', data=ndsi_final, shape=ndsi_final.shape, chunks=(100, 100), dtype='float32', overwrite=True)
            
            # If label doesn't exist (e.g. running this script first), save label too
            if 'label' not in root:
                root.create_dataset('label', data=gt_final, shape=gt_final.shape, chunks=(100, 100), dtype='float32', overwrite=True)
            
        except Exception as e:
            print(f"Failed on index {i}: {e}")
            continue
                
    print(f"\nNDSI Baselines saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
import os
import sys
import argparse
import pandas as pd
import numpy as np
import zarr
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- PATH HACK ---
# Guarantees Python can find the 'src' module regardless of where the script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))) 

from src.snow_s2_dataset_13bands import Snows2Dataset

# --- PARAMETERS ---
BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
IDX_GREEN = BANDS.index('B3')  # Index 2
IDX_SWIR  = BANDS.index('B11') # Index 11

# Target Output Size (Matches LIS 50x50 @ 20m -> 100x100 @ 10m)
TARGET_SHAPE = (100, 100) 

def get_args():
    parser = argparse.ArgumentParser(description="Generate NDSI Baselines from the Test Set.")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the Zenodo dataset')
    parser.add_argument('--split_file', type=str, default='dataset_metadata.csv', help='Name of the metadata CSV')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save NDSI prediction Zarrs')
    return parser.parse_args()

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
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print("--- LOADING CONFIGURATION ---")
    split_csv_path = os.path.join(args.data_root, args.split_file)
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"Split file not found: {split_csv_path}")

    df = pd.read_csv(split_csv_path)
    path_col = 'relative_path' if 'relative_path' in df.columns else 'filepath'
    split_col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    
    test_df = df[df[split_col] == 'Test']
    test_files = [os.path.join(args.data_root, p) for p in test_df[path_col].tolist()]
    
    # Init Dataset (No model needed)
    print(f"Initializing Dataset for {len(test_files)} test files...")
    ds = Snows2Dataset(test_files, bands=BANDS, train=False, target_type='scf', train_with_clouds=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    print("Generating NDSI Baselines...")

    success_count = 0
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
            
            save_name = f"{date_id}_{tile_id}_NDSI.zarr"
            save_path = os.path.join(args.output_dir, save_name)

            # Open in 'a' (append) mode if exists, else 'w'
            mode = 'a' if os.path.exists(save_path) else 'w'
            root = zarr.open(save_path, mode=mode)
            
            # Save NDSI (Overwrite if exists using new Zarr API without shape/dtype)
            root.create_array('ndsi', data=ndsi_final, chunks=(100, 100), overwrite=True)
            
            # If label doesn't exist, save label too
            if 'label' not in root:
                root.create_array('label', data=gt_final, chunks=(100, 100), overwrite=True)
            
            success_count += 1
            
        except Exception as e:
            # print(f"Failed on index {i}: {e}")
            continue
                
    print(f"\nSuccessfully generated {success_count} NDSI Baselines. Saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
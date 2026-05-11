# PYTHONPATH=. python results/evaluation/scotland/spun/scripts/predict_test_set.py \
#     --data_root /path/to/SPUN_Dataset \
#     --model_path /path/to/SPUN_Model.pt \
#     --output_dir /path/to/output/dir/test_set_predictions

import os
import argparse
import pandas as pd
import numpy as np
import zarr
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- IMPORTS ---
# Assuming you run this from the project root!
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel
from src.snow_s2_dataset_13bands import Snows2Dataset

# Target Output Size (100px * 10m = 1000m)
TARGET_SHAPE = (100, 100) 
BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(description="Run inference on the test set and save predictions.")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the Zenodo dataset')
    parser.add_argument('--split_file', type=str, default='dataset_metadata.csv', help='Name of the metadata CSV')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pt model weights')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save prediction Zarrs')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone used during training')
    return parser.parse_args()

def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

def center_crop(arr, target_h=100, target_w=100):
    """Crops the center of the array to target size."""
    h, w = arr.shape
    if h == target_h and w == target_w:
        return arr
    
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    if start_h < 0 or start_w < 0:
        return arr 

    return arr[start_h:start_h+target_h, start_w:start_w+target_w]

def main():
    args = get_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print("--- LOADING CONFIGURATION ---")
    
    # Handle the CSV split loading
    split_csv_path = os.path.join(args.data_root, args.split_file)
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"Split file not found: {split_csv_path}")
        
    df = pd.read_csv(split_csv_path)
    
    path_col = 'relative_path' if 'relative_path' in df.columns else 'filepath'
    split_col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    
    # Filter for Test set and build absolute paths
    test_df = df[df[split_col] == 'Test']
    test_files = [os.path.join(args.data_root, p) for p in test_df[path_col].tolist()]
    
    print(f"Found {len(test_files)} test images in {split_csv_path}")
    
    print(f"Loading Model: {os.path.basename(args.model_path)}")
    model = SnowModel(
        bands=BANDS, train_files=[], val_files=[], mask_clouds=False, 
        use_augmentations=False, train_with_clouds=True, 
        hparams={'backbone': args.backbone}, in_channels=len(BANDS)
    )
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=False))
    model.to(DEVICE).eval()
    
    ds = Snows2Dataset(test_files, bands=BANDS, train=False, target_type='scf', train_with_clouds=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    print(f"Running Inference & Saving {TARGET_SHAPE} Crops...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dl)):
            try:
                # 1. Inference
                images = batch['image'].to(DEVICE)
                out = model(images).squeeze(1).cpu().numpy() 
                
                # [1, H, W] -> [H, W] and Clip 0-100
                spun_raw = (out * 100.0).clip(0, 100)[0]
                gt_raw = batch['mask'].cpu().numpy()[0]
                
                # 2. FLIP (Correct Orientation)
                spun_flipped = spun_raw[::-1, :]
                gt_flipped = gt_raw[::-1, :]
                
                # 3. CROP (Force 100x100)
                spun_final = center_crop(spun_flipped, TARGET_SHAPE[0], TARGET_SHAPE[1])
                gt_final = center_crop(gt_flipped, TARGET_SHAPE[0], TARGET_SHAPE[1])

                # 4. Save to Zarr 
                label_path = test_files[i]
                date_id, tile_id = extract_ids_from_path(label_path)
                save_name = f"{date_id}_{tile_id}_SPUN.zarr"
                save_path = os.path.join(args.output_dir, save_name)

                # mode='w' creates a fresh directory/group, avoiding overwrite conflicts
                root = zarr.open(save_path, mode='w')
                
                root.create_dataset('spun_scf', data=spun_final, shape=spun_final.shape, chunks=(100, 100), dtype='float32')
                root.create_dataset('label', data=gt_final, shape=gt_final.shape, chunks=(100, 100), dtype='float32')
                
            except Exception as e:
                print(f"\nFailed on index {i} ({test_files[i]}): {e}")
                continue
                
    print(f"\nAll predictions successfully saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
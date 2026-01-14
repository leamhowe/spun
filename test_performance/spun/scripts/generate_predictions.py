import pandas as pd
import numpy as np
import os
import zarr
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- IMPORTS ---
sys.path.append('/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts') 
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel
from src.snow_s2_dataset_13bands import Snows2Dataset

# --- CONFIGURATION ---
SPLIT_CSV_PATH  = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
MODEL_PATH      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/models/Final_T15_dino_none_scale10000_best.pt'
OUTPUT_DIR      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'

BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
BACKBONE = 'resnet50'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Target Output Size (100px * 10m = 1000m)
TARGET_SHAPE = (100, 100) 

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
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print("--- LOADING CONFIGURATION ---")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    print(f"Loading Model: {os.path.basename(MODEL_PATH)}")
    model = SnowModel(bands=BANDS, train_files=[], val_files=[], mask_clouds=False, use_augmentations=False, train_with_clouds=True, hparams={'backbone': BACKBONE}, in_channels=len(BANDS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
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

                # 4. Save to Zarr (Minimal Arguments for Stability)
                label_path = test_files[i]
                date_id, tile_id = extract_ids_from_path(label_path)
                save_name = f"{date_id}_{tile_id}_SPUN.zarr"
                save_path = os.path.join(OUTPUT_DIR, save_name)

                # mode='w' creates a fresh directory/group, avoiding overwrite conflicts
                root = zarr.open(save_path, mode='w')
                
                # We DROP compression args to let Zarr use safe defaults for your version
                root.create_dataset('spun_scf', data=spun_final, shape=spun_final.shape, chunks=(100, 100), dtype='float32')
                root.create_dataset('label', data=gt_final, shape=gt_final.shape, chunks=(100, 100), dtype='float32')
                
            except Exception as e:
                print(f"Failed on index {i}: {e}")
                continue
                
    print(f"\nAll predictions saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
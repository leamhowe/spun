import pandas as pd
import numpy as np
import os
import zarr
import sys
import torch
import xarray as xr
from tqdm import tqdm

# --- CONFIGURATION ---
SPLIT_CSV_PATH    = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
MODEL_PATH        = '/home/users/leamhowe/phd/1_DETECT/Wang2022/models/unet_4bands.pth'
MODEL_DIR         = '/home/users/leamhowe/phd/1_DETECT/Wang2022' # Directory containing u_net.py
OUTPUT_DIR        = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_wang2022'

# Wang Model Specifics
BANDS = ['B2', 'B11', 'B4', 'B9']
INPUT_SIZE = 256 # Model expects 256x256
ORIG_SIZE = 100  # Patches are 100x100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- IMPORTS (Dynamic for Model) ---
# We need to add the directory containing u_net.py to path so torch.load can find the class definition
sys.path.append(MODEL_DIR)
try:
    from u_net import UNet # Check if we can import it
except ImportError:
    print(f"WARNING: Could not import UNet from {MODEL_DIR}. 'torch.load' might fail.")

def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

def center_crop(arr, target_h=100, target_w=100):
    """Crops the center of the array to target size (for geometric alignment)."""
    h, w = arr.shape
    if h == target_h and w == target_w: return arr
    
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    if start_h < 0 or start_w < 0: return arr 
    return arr[start_h:start_h+target_h, start_w:start_w+target_w]

def load_and_preprocess(zarr_path, bands):
    """Loads bands, normalizes, and pads to 256x256."""
    try:
        with xr.open_zarr(zarr_path) as ds:
            bands_data = []
            for b in bands:
                bands_data.append(ds[b].values)
            
            # Stack: (C, H, W) -> (4, 100, 100)
            img = np.stack(bands_data).astype(np.float32)
            
            # Normalize (Wang method: / 20000, clip 0-1)
            img = np.clip(img / 20000.0, 0, 1)
            
            # Pad to 256x256 (Top-Left alignment)
            padded = np.zeros((len(bands), INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
            padded[:, :ORIG_SIZE, :ORIG_SIZE] = img
            
            # Get Label (GT) for saving
            if 'label' in ds:
                gt = ds['label'].values
            else:
                # Fallback if label is named differently or missing
                gt = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.float32)

            return padded, gt
    except Exception as e:
        # print(f"Error reading {zarr_path}: {e}")
        return None, None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    print(f"--- LOADING MODEL ---")
    print(f"Path: {MODEL_PATH}")
    # Load full model (architecture + weights)
    # weights_only=False is required because we are loading a full pickled model object
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.eval()
    model = model.to(DEVICE).double() # Wang model expects Double precision
    print("Model loaded successfully.")

    print("--- PROCESSING TILES ---")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()

    
    print(f"Processing {len(test_files)} files...")

    for l1c_path in tqdm(test_files):
        try:
            # 1. Load Data
            l2a_path = l1c_path.replace('s2_training_scf', 's2l2a_training_scf')
            padded_img, gt_raw = load_and_preprocess(l2a_path, BANDS)
            if padded_img is None: continue

            # 2. Prepare Tensor
            # (C, H, W) -> (1, C, H, W)
            input_tensor = torch.from_numpy(padded_img).unsqueeze(0).to(DEVICE).double()

            # 3. Inference
            with torch.no_grad():
                out = model(input_tensor) # (1, NClass, 256, 256)
                out = out.cpu().numpy()

            # 4. Post-Process
            # Crop back to 100x100 (Top-Left)
            pred_crop = out[0, :, :ORIG_SIZE, :ORIG_SIZE]
            
            # Argmax to get class (0=Bg, 1=Cloud, 2=Snow)
            pred_mask = np.argmax(pred_crop, axis=0)
            
            # Convert to Binary SCF (0 or 100)
            # We treat Class 2 (Snow) as 100%, everything else as 0%
            wang_scf = (pred_mask == 2).astype(np.float32) * 100.0

            # 5. Geometric Correction (Flip & Center Crop)
            # Essential to match LIS grid (same as SPUN/NDSI generation)
            wang_flipped = wang_scf[::-1, :]
            gt_flipped   = gt_raw[::-1, :]
            
            wang_final = center_crop(wang_flipped)
            gt_final   = center_crop(gt_flipped)

            # 6. Save to Zarr
            date_id, tile_id = extract_ids_from_path(l2a_path)
            
            # Naming convention: {date}_{tile}_SPUN.zarr 
            # (Keeping suffix _SPUN.zarr allows the comparison script to read it easily 
            # if we point it to this directory, though _WANG.zarr is clearer human-readability.
            # Let's use _SPUN.zarr to maximize compatibility with the existing eval script logic).
            save_name = f"{date_id}_{tile_id}_Wang2022.zarr"
            save_path = os.path.join(OUTPUT_DIR, save_name)

            root = zarr.open(save_path, mode='w')
            
            # Save Dataset
            # We call it 'wang_scf' to distinguish, or 'spun_scf' to trick the eval script?
            # Better to use 'wang_scf' and update the eval script one last time to look for it.
            # BUT: To make it "plug and play" with the minimal changes, let's save it as 
            # a dataset named 'wang_scf' and 'label'.
            root.create_dataset('wang_scf', data=wang_final, shape=wang_final.shape, chunks=(100, 100), dtype='float32')
            root.create_dataset('label',    data=gt_final,   shape=gt_final.shape,   chunks=(100, 100), dtype='float32')
            
        except Exception as e:
            # print(f"Failed on {label_path}: {e}")
            continue

    print(f"\nWang2022 predictions saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
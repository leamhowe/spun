import os
import argparse
import sys
import pandas as pd
import numpy as np
import zarr
import torch
import xarray as xr
from tqdm import tqdm

# Wang Model Specifics
BANDS = ['B2', 'B11', 'B4', 'B9']
INPUT_SIZE = 256 # Model strictly expects 256x256
ORIG_SIZE = 100  # Patches are natively 100x100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(description="Run inference using the Wang 2022 UNet baseline with reflection padding.")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the L2A dataset')
    parser.add_argument('--split_file', type=str, default='dataset_metadata.csv', help='Name of the metadata CSV')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing the u_net.py architecture file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained Wang UNet .pth weights file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save prediction Zarrs')
    return parser.parse_args()

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
    """Loads bands, normalizes, and applies reflection padding to 256x256."""
    try:
        # Check if folder is empty (catches bad copies)
        if not os.path.exists(zarr_path) or len(os.listdir(zarr_path)) == 0:
            return None, None
            
        with xr.open_zarr(zarr_path) as ds:
            bands_data = [ds[b].values for b in bands]
            
            # Stack: (C, H, W) -> (4, 100, 100)
            img = np.stack(bands_data).astype(np.float32)
            
            # Normalize (Wang method: / 20000, clip 0-1)
            img = np.clip(img / 20000.0, 0, 1)
            
            # Pad to 256x256 using Reflection Padding (Top-Left alignment)
            pad_amount = INPUT_SIZE - ORIG_SIZE
            padded = np.pad(img, pad_width=((0, 0), (0, pad_amount), (0, pad_amount)), mode='reflect')
            
            # Get Label (GT) for saving
            if 'label' in ds:
                gt = ds['label'].values
            else:
                gt = np.zeros((ORIG_SIZE, ORIG_SIZE), dtype=np.float32)

            return padded, gt
    except Exception as e:
        return None, None

def main():
    args = get_args()

    # Add the directory containing u_net.py to path so torch.load can find the class definition
    sys.path.append(os.path.abspath(args.model_dir))
    try:
        from u_net import UNet
    except ImportError:
        print(f"WARNING: Could not import UNet from {args.model_dir}. 'torch.load' will likely fail.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    print(f"--- LOADING MODEL ---")
    print(f"Path: {args.model_path}")
    model = torch.load(args.model_path, map_location=DEVICE, weights_only=False)
    model.eval()
    model = model.to(DEVICE).double() # Wang model expects Double precision
    print("Model loaded successfully.")

    print("\n--- LOADING TEST SET ---")
    split_csv_path = os.path.join(args.data_root, args.split_file)
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"Split file not found: {split_csv_path}")

    df = pd.read_csv(split_csv_path)
    path_col = 'relative_path' if 'relative_path' in df.columns else 'filepath'
    split_col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    
    test_df = df[df[split_col] == 'Test']
    test_files = [os.path.join(args.data_root, p) for p in test_df[path_col].tolist()]
    
    print(f"Processing {len(test_files)} files...")

    success_count = 0
    for file_path in tqdm(test_files):
        try:
            # 1. Load Data
            padded_img, gt_raw = load_and_preprocess(file_path, BANDS)
            if padded_img is None: continue

            # 2. Prepare Tensor: (C, H, W) -> (1, C, H, W)
            input_tensor = torch.from_numpy(padded_img).unsqueeze(0).to(DEVICE).double()

            # 3. Inference
            with torch.no_grad():
                out = model(input_tensor) # (1, NClass, 256, 256)
                out = out.cpu().numpy()

            # 4. Post-Process: Crop back to 100x100 (Top-Left, because padding was applied to bottom/right)
            pred_crop = out[0, :, :ORIG_SIZE, :ORIG_SIZE]
            
            # Argmax to get class (0=Bg, 1=Cloud, 2=Snow)
            pred_mask = np.argmax(pred_crop, axis=0)
            
            # Convert to Binary SCF (0 or 100)
            wang_scf = (pred_mask == 2).astype(np.float32) * 100.0

            # 5. Geometric Correction (Flip & Center Crop)
            wang_flipped = wang_scf[::-1, :]
            gt_flipped   = gt_raw[::-1, :]
            
            wang_final = center_crop(wang_flipped)
            gt_final   = center_crop(gt_flipped)

            # 6. Save to Zarr
            date_id, tile_id = extract_ids_from_path(file_path)
            save_name = f"{date_id}_{tile_id}_Wang2022.zarr"
            save_path = os.path.join(args.output_dir, save_name)

            root = zarr.open(save_path, mode='w')
            
            # Using the new Zarr API rules (no shape, no dtype)
            root.create_array('wang_scf', data=wang_final, chunks=(100, 100))
            root.create_array('label',    data=gt_final,   chunks=(100, 100))
            
            success_count += 1
            
        except Exception as e:
            continue

    print(f"\nSuccessfully generated {success_count} Wang2022 predictions. Saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
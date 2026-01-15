import pandas as pd
import numpy as np
import os
import zarr
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- CONFIGURATION ---
SPLIT_CSV_PATH    = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
OUTPUT_DIR        = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/visualisations/pipeline_sanity_check'

# Wang Model Bands (Indices for reference: 0=B2, 1=B11, 2=B4, 3=B9)
BANDS = ['B2', 'B11', 'B4', 'B9']
INPUT_SIZE = 256
ORIG_SIZE = 100

def center_crop(arr, target_h=100, target_w=100):
    start_h = (arr.shape[0] - target_h) // 2
    start_w = (arr.shape[1] - target_w) // 2
    return arr[start_h:start_h+target_h, start_w:start_w+target_w]

def load_l2a_bands(zarr_path):
    try:
        with xr.open_zarr(zarr_path) as ds:
            stack = []
            for b in BANDS:
                if b in ds: stack.append(ds[b].values)
                else: return None
            # Stack (C, H, W)
            img = np.stack(stack).astype(np.float32)
            # Normalize like the model does
            img = np.clip(img / 20000.0, 0, 1)
            return img
    except Exception:
        return None

def process_tile_dummy(l2a_path):
    """
    Simulates the EXACT pipeline but replaces the Model with a simple threshold.
    """
    # 1. Load Data
    img = load_l2a_bands(l2a_path)
    if img is None: return None
    
    # 2. Pad to 256x256 (Same as generation script)
    padded = np.zeros((4, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    padded[:, :ORIG_SIZE, :ORIG_SIZE] = img
    
    # 3. "FAKE MODEL" Inference
    # Instead of U-Net, we use simple physics on the padded array.
    # Snow Rule: Blue (Band 0) is Bright (> 0.2) AND SWIR (Band 1) is Dark (< 0.1)
    # Note: These indices correspond to ['B2', 'B11', 'B4', 'B9']
    blue_band = padded[0]
    swir_band = padded[1]
    
    # Create a binary mask (1=Snow, 0=Background)
    dummy_pred = ((blue_band > 0.15) & (swir_band < 0.1)).astype(np.float32)
    
    # 4. Post-Processing (Crop back to 100x100)
    # We crop the 256x256 mask back to 100x100
    pred_crop = dummy_pred[:ORIG_SIZE, :ORIG_SIZE]
    
    # 5. Geometric Correction (Flip)
    final_result = pred_crop[::-1, :]
    
    # 6. Center Crop (Final step in generation script)
    final_result = center_crop(final_result)
    
    return final_result

def load_rgb_l1c(zarr_path):
    """Loads L1C RGB for comparison background."""
    try:
        with xr.open_zarr(zarr_path) as ds:
            rgb = np.stack([ds['B4'], ds['B3'], ds['B2']], axis=-1)
            # Apply same flip as visualizer
            rgb = rgb[::-1, :, :]
            return np.clip(rgb / 3000.0, 0, 1)
    except:
        return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Saving sanity checks to: {OUTPUT_DIR}")

    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    
    count = 0
    print("Running pipeline sanity check on first 20 valid tiles...")
    
    for l1c_path in tqdm(test_files):
        l2a_path = l1c_path.replace('s2_training_scf', 's2l2a_training_scf')
        if not os.path.exists(l2a_path): continue

        # 1. Run the "Dummy" Pipeline
        dummy_mask = process_tile_dummy(l2a_path)
        if dummy_mask is None: continue
        
        # 2. Load RGB (Ground Truth Geometry)
        rgb = load_rgb_l1c(l1c_path)
        if rgb is None: continue
        
        # 3. Plot Overlay
        if dummy_mask.max() > 0: # Only plot if our simple rule found snow
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(rgb)
            
            # Plot Dummy Mask in Bright Green
            green_rgba = np.zeros((*dummy_mask.shape, 4))
            green_rgba[dummy_mask > 0, 1] = 1.0 # Green channel
            green_rgba[dummy_mask > 0, 3] = 0.6 # Alpha
            
            ax.imshow(green_rgba)
            ax.set_title(f"Pipeline Sanity Check\nGreen = Simple Threshold (Blue-SWIR) processed via pipeline", fontsize=9)
            ax.axis('off')
            
            date_id = os.path.basename(l1c_path).split('_')[0]
            tile_id = os.path.basename(l1c_path).split('_')[-1].split('.')[0]
            
            plt.savefig(os.path.join(OUTPUT_DIR, f"{date_id}_{tile_id}_sanity.png"))
            plt.close()
            
            count += 1
            if count >= 20: break
            
    print(f"Done. Check images in {OUTPUT_DIR}")
    print("If GREEN aligns with WHITE SNOW in the image -> Pipeline is perfect, Model is bad.")
    print("If GREEN is shifted relative to WHITE SNOW -> L2A data is shifted.")

if __name__ == "__main__":
    main()
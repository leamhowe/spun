import pandas as pd
import numpy as np
import os
import zarr
import traceback
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# --- CONFIGURATION ---
SPLIT_CSV_PATH = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
# LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches_DEBUG_old2'

# Metric Parameters
DICE_THRESH = 0.5
FLIP_LABELS = True 
LABEL_CLOUD = 200
LIS_CLOUD = 205
LIS_NODATA = 255

# --- HELPER FUNCTIONS ---
def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

def downscale_label_masked(arr_100, factor=2, invalid_val=200):
    # (Same function as before - omitted for brevity, assume it's here)
    arr_masked = np.where(arr_100 == invalid_val, np.nan, arr_100)
    h, w = arr_masked.shape
    blocks = arr_masked.reshape(h // factor, factor, w // factor, factor)
    sum_ds = np.nansum(blocks, axis=(1, 3))
    count_ds = np.count_nonzero(~np.isnan(blocks), axis=(1, 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ds = sum_ds / count_ds
    arr_final = np.where(count_ds == 0, invalid_val, mean_ds).astype(np.float32)
    return arr_final

# --- MAIN EXECUTION ---
print("Loading Test Set List...")
df = pd.read_csv(SPLIT_CSV_PATH)
col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
test_files = df[df[col] == 'Test']['filepath'].tolist()

gts_all = []
preds_all = []

print(f"Starting Evaluation on {len(test_files)} tiles...")

error_count = 0

for label_path in tqdm(test_files):
    if error_count > 5:
        print("\nStopping early due to too many errors. Check the debug output above.")
        break

    try:
        date_id, tile_id = extract_ids_from_path(label_path)
    except: continue
        
    lis_zarr_name = f"{date_id}_{tile_id}_LIS.zarr"
    lis_path = os.path.join(LIS_PATCHES_DIR, lis_zarr_name)
    
    # DEBUG 1: Existence Check
    if not os.path.exists(lis_path):
        continue # Skip quietly if we just don't have this LIS tile
    if not os.path.exists(label_path):
        print(f"Warning: Label file missing: {label_path}")
        continue

    try:
        # DEBUG 2: Inspect LIS Zarr Structure
        # Check if it's a valid group or array
        try:
            lis_ds = zarr.open(lis_path, mode='r')
        except Exception as e:
            print(f"\n[ERROR] Corrupt LIS File: {lis_zarr_name}")
            print(f"  Path: {lis_path}")
            print(f"  Zarr Error: {e}")
            print(f"  Contains .zgroup? {os.path.exists(os.path.join(lis_path, '.zgroup'))}")
            print(f"  Contains .zarray? {os.path.exists(os.path.join(lis_path, '.zarray'))}")
            error_count += 1
            continue

        # Check for keys
        if isinstance(lis_ds, zarr.hierarchy.Group):
            keys = list(lis_ds.keys())
            if 'lis_snow' not in keys:
                print(f"\n[ERROR] Missing 'lis_snow' key in {lis_zarr_name}")
                print(f"  Found keys: {keys}")
                error_count += 1
                continue
            lis_arr = lis_ds['lis_snow'][:]
        elif isinstance(lis_ds, zarr.core.Array):
            # Maybe it was saved as a root array?
            lis_arr = lis_ds[:]
        else:
            print(f"\n[ERROR] Unknown Zarr object type: {type(lis_ds)}")
            continue

        # DEBUG 3: Inspect Label Zarr Structure
        try:
            lbl_ds = zarr.open(label_path, mode='r')
        except Exception as e:
            print(f"\n[ERROR] Corrupt Label File: {os.path.basename(label_path)}")
            error_count += 1
            continue

        # Logic to handle if Label is Group vs Array
        if isinstance(lbl_ds, zarr.hierarchy.Group):
            if 'label' in lbl_ds:
                lbl_arr_100 = lbl_ds['label'][:]
            else:
                print(f"\n[ERROR] Label file {tile_id} has no 'label' key. Keys: {list(lbl_ds.keys())}")
                continue
        elif isinstance(lbl_ds, zarr.core.Array):
            lbl_arr_100 = lbl_ds[:] # It's a root array
        
        # --- PREPROCESSING (Same as before) ---
        if FLIP_LABELS:
            lbl_arr_100 = lbl_arr_100[::-1, :]
        
        lbl_arr_50 = downscale_label_masked(lbl_arr_100, factor=2, invalid_val=LABEL_CLOUD)
        
        y_hat_raw = lis_arr.flatten()
        y_gt_raw = lbl_arr_50.flatten()
        
        valid_mask = (
            (y_gt_raw != LABEL_CLOUD) & 
            (y_hat_raw != LIS_CLOUD) & 
            (y_hat_raw != LIS_NODATA) & 
            (~np.isnan(y_hat_raw))
        )
        
        if valid_mask.any():
            preds_valid = np.clip(y_hat_raw[valid_mask], 0, 100)
            gts_valid = np.clip(y_gt_raw[valid_mask], 0, 100)
            preds_all.append(preds_valid)
            gts_all.append(gts_valid)
            
    except Exception as e:
        print(f"\n[CRITICAL] Unexpected error on {tile_id}")
        traceback.print_exc()
        error_count += 1
        continue

# --- METRIC CALCULATION ---

print("\nComputing Metrics...")

if len(gts_all) > 0:
    preds = np.concatenate(preds_all) # LIS
    gts = np.concatenate(gts_all)     # Labels
    
    # 1. Continuous Metrics (MAE, RMSE, Bias)
    diff = preds - gts
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    mean_error = np.mean(diff)
    
    # Snow-Specific Metrics (Only where Ground Truth > 0)
    snow_mask = gts > 0
    if snow_mask.any():
        diff_snow = preds[snow_mask] - gts[snow_mask]
        mae_snow = np.mean(np.abs(diff_snow))
        rmse_snow = np.sqrt(np.mean(diff_snow ** 2))
    else:
        mae_snow, rmse_snow = float('nan'), float('nan')
        
    # 2. Binary Metrics Preparation
    # Normalize to 0-1
    y_hat_norm = preds / 100.0
    y_gt_norm = gts / 100.0
    
    # Threshold to create binary arrays (0 or 1)
    y_hat_bin = (y_hat_norm > DICE_THRESH).astype(int)
    y_bin = (y_gt_norm > DICE_THRESH).astype(int)
    
    # 3. Calculate Dice (Manual calculation to match previous logic)
    intersection = (y_hat_bin * y_bin).sum()
    union = y_hat_bin.sum() + y_bin.sum()
    dice = (2. * intersection + DICE_SMOOTH) / (union + DICE_SMOOTH) if union > 0 else float('nan')

    # 4. Calculate Standard Classification Metrics (using sklearn)
    # These operate on the binarized arrays calculated above
    precision = precision_score(y_bin, y_hat_bin, zero_division=0)
    recall = recall_score(y_bin, y_hat_bin, zero_division=0)
    f1 = f1_score(y_bin, y_hat_bin, zero_division=0)
    accuracy = accuracy_score(y_bin, y_hat_bin)
    
    # Optional: Confusion Matrix components
    tn, fp, fn, tp = confusion_matrix(y_bin, y_hat_bin).ravel()
    
    # Specificity (True Negative Rate) - useful for snow detection
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    # --- OUTPUT ---
    print("\n" + "="*50)
    print("LIS PERFORMANCE REPORT")
    print("="*50)
    print(f"Total Valid Pixels: {len(gts):,}")
    print(f"Binary Threshold:   {DICE_THRESH*100}%")
    print("-" * 30)
    print("CONTINUOUS METRICS (0-100 scale):")
    print(f"  MAE:       {mae:.3f}")
    print(f"  RMSE:      {rmse:.3f}")
    print(f"  Bias:      {mean_error:.3f}")
    print(f"  MAE_snow:  {mae_snow:.3f}")
    print(f"  RMSE_snow: {rmse_snow:.3f}")
    print("-" * 30)
    print("CLASSIFICATION METRICS (Binary):")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  Dice:      {dice:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    print("-" * 30)
    print("CONFUSION MATRIX COUNTS:")
    print(f"  TP: {tp:,} | FP: {fp:,}")
    print(f"  FN: {fn:,} | TN: {tn:,}")
    print("-" * 30)
    
    if mean_error > 0.5:
        print(">> LIS is OVERESTIMATING snow cover.")
    elif mean_error < -0.5:
        print(">> LIS is UNDERESTIMATING snow cover.")
    else:
        print(">> LIS has no significant average bias.")
        
else:
    print("No valid pixels found to evaluate.")
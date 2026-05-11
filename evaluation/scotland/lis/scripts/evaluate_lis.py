import pandas as pd
import numpy as np
import os
import zarr
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# --- CONFIGURATION ---
SPLIT_CSV_PATH = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
# LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches_DEBUG'
# LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches_DEBUG_old2'
# LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches_DEBUG_old'
# LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches_old'

# --- CLASSIFICATION CODES ---
# Label Codes
LABEL_SNOW    = 100
LABEL_NOSNOW  = 0
LABEL_CLOUD   = 200  # Invalid for comparison

# LIS Codes
LIS_SNOW      = 100
LIS_NOSNOW    = 0
LIS_CLOUD     = 205  # Invalid for comparison
LIS_NODATA    = 255  # Invalid for comparison

# Metric Parameters
DICE_THRESH = 0.5  # 50% threshold for binary metrics
DICE_SMOOTH = 1.0
FLIP_LABELS = True 

# --- HELPER FUNCTIONS ---

def downscale_label_masked(arr_100, factor=2, invalid_val=200):
    """
    Downscales label 100x100 -> 50x50 ignoring the invalid_val (200).
    The result is a fractional coverage (0-100) of the valid pixels.
    """
    # Convert invalid to NaN so they don't count towards the mean
    arr_masked = np.where(arr_100 == invalid_val, np.nan, arr_100)
    
    h, w = arr_masked.shape
    blocks = arr_masked.reshape(h // factor, factor, w // factor, factor)
    
    # Calculate Mean of valid pixels
    sum_ds = np.nansum(blocks, axis=(1, 3))
    count_ds = np.count_nonzero(~np.isnan(blocks), axis=(1, 3))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ds = sum_ds / count_ds
        
    # If a block was entirely invalid (count=0), set it back to invalid_val
    # Otherwise, we keep the fraction (0.0 - 100.0)
    arr_final = np.where(count_ds == 0, invalid_val, mean_ds).astype(np.float32)
    
    return arr_final

def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

# --- MAIN EXECUTION ---

print("Loading Test Set List...")
df = pd.read_csv(SPLIT_CSV_PATH)
col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
test_files = df[df[col] == 'Test']['filepath'].tolist()

gts_all = []
preds_all = []

print(f"Starting Evaluation on {len(test_files)} tiles...")

for label_path in tqdm(test_files):
    
    try:
        date_id, tile_id = extract_ids_from_path(label_path)
    except: continue
        
    lis_zarr_name = f"{date_id}_{tile_id}_LIS.zarr"
    lis_path = os.path.join(LIS_PATCHES_DIR, lis_zarr_name)
    
    if not os.path.exists(lis_path) or not os.path.exists(label_path):
        continue

    try:
        # 1. Load Data
        lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:] 
        lbl_arr_100 = zarr.open(label_path, mode='r')['label'][:] 
        
        # 2. Preprocess Label
        if FLIP_LABELS:
            lbl_arr_100 = lbl_arr_100[::-1, :]
        
        # Downscale Label (10m -> 20m)
        # This converts 0/100 binary labels into 0-100 continuous fractions
        lbl_arr_50 = downscale_label_masked(lbl_arr_100, factor=2, invalid_val=LABEL_CLOUD)
        
        # 3. Flatten
        y_hat_raw = lis_arr.flatten() # LIS (contains 0-100, 205, 255)
        y_gt_raw = lbl_arr_50.flatten()  # Labels (contains 0-100, 200)
        
        # 4. Create MASK
        valid_mask = (
            (y_gt_raw != LABEL_CLOUD) & 
            (y_hat_raw != LIS_CLOUD) & 
            (y_hat_raw != LIS_NODATA) & 
            (~np.isnan(y_hat_raw))
        )
        
        if valid_mask.any():
            # Extract valid pixels
            preds_valid = y_hat_raw[valid_mask]
            gts_valid = y_gt_raw[valid_mask]
            
            # 5. Sanity Clip
            preds_valid = np.clip(preds_valid, 0, 100)
            gts_valid = np.clip(gts_valid, 0, 100)

            preds_all.append(preds_valid)
            gts_all.append(gts_valid)
            
    except Exception as e:
        print(f"Error on {tile_id}: {e}")
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
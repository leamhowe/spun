import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Add scripts directory to path to find custom src modules
sys.path.append('../../../../..')  # adjust path as necessary to point to the src directory
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel
from src.snow_s2_dataset_13bands import Snows2Dataset

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '../../SPUN.pt' # download from Zenodo: https://zenodo.org/records/20055817
SPLIT_CSV = '<path/to/SPUN_dataset>/dataset_metadata.csv' # download from Zenodo: https://zenodo.org/records/20055817
BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
BACKBONE = 'resnet50'
BATCH_SIZE = 4

# --- METRIC PARAMETERS ---
DICE_THRESH = 0.5 
DICE_SMOOTH = 1.0
LABEL_CLOUD = 200  
DOWNSCALE_FACTOR = 2  # 10m -> 20m

# --- DOWNSCALING FUNCTION (Matching LIS Script) ---

def downscale_label_masked(arr, factor=2, invalid_val=200):
    """
    Downscales a 2D array by a factor, ignoring invalid_val.
    Returns fractional coverage (0-100).
    """
    # Convert invalid to NaN
    arr_masked = np.where(arr == invalid_val, np.nan, arr)
    
    h, w = arr_masked.shape
    # Ensure the array is divisible by the factor
    h_new, w_new = h // factor, w // factor
    arr_masked = arr_masked[:h_new*factor, :w_new*factor]
    
    blocks = arr_masked.reshape(h_new, factor, w_new, factor)
    
    # Calculate Mean of valid pixels in each block
    sum_ds = np.nansum(blocks, axis=(1, 3))
    count_ds = np.count_nonzero(~np.isnan(blocks), axis=(1, 3))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ds = sum_ds / count_ds
        
    # If a block was entirely invalid, set it back to invalid_val
    return np.where(count_ds == 0, invalid_val, mean_ds).astype(np.float32)

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # 1. Load Data
    df = pd.read_csv(SPLIT_CSV)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()

    ds = Snows2Dataset(test_files, bands=BANDS, train=False, target_type='scf', train_with_clouds=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Load Model
    model = SnowModel(bands=BANDS, train_files=[], val_files=[], mask_clouds=False, use_augmentations=False, train_with_clouds=True, hparams={'backbone': BACKBONE}, in_channels=len(BANDS))
    # model = SnowModel(bands=BANDS, train_files=[], val_files=[], hparams={'backbone': BACKBONE}, in_channels=len(BANDS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()

    preds_all, gts_all = [], []
    
    print(f"Running Inference & Downscaling to 20m (Factor {DOWNSCALE_FACTOR})...")
    
    with torch.no_grad():
        for batch in tqdm(dl):
            images = batch['image'].to(DEVICE)
            masks_10m = batch['mask'].cpu().numpy() # [B, H, W]
            
            # Predict 10m
            outputs = model(images).squeeze(1) # [B, H, W]
            preds_10m = (outputs * 100.0).clamp(0, 100).cpu().numpy()

            # Process each image in batch for downscaling
            for i in range(masks_10m.shape[0]):
                # Downscale Ground Truth to 20m
                gt_20m = downscale_label_masked(masks_10m[i], factor=DOWNSCALE_FACTOR)
                # Downscale Prediction to 20m
                pred_20m = downscale_label_masked(preds_10m[i], factor=DOWNSCALE_FACTOR)
                
                # Flatten and Mask
                gt_flat = gt_20m.flatten()
                pred_flat = pred_20m.flatten()
                
                valid = (gt_flat != LABEL_CLOUD)
                if valid.any():
                    preds_all.append(pred_flat[valid])
                    gts_all.append(gt_flat[valid])

    # 3. Compute Metrics
    if not gts_all:
        print("No valid pixels found.")
        return

    preds = np.concatenate(preds_all)
    gts = np.concatenate(gts_all)

    # Continuous
    diff = preds - gts
    mae, rmse, bias = np.mean(np.abs(diff)), np.sqrt(np.mean(diff**2)), np.mean(diff)
    
    # Snow-Specific
    snow_mask = gts > 0
    mae_snow = np.mean(np.abs(diff[snow_mask])) if snow_mask.any() else np.nan
    rmse_snow = np.sqrt(np.mean(diff[snow_mask]**2)) if snow_mask.any() else np.nan

    # Binary
    y_hat_bin = (preds > (DICE_THRESH * 100)).astype(int)
    y_bin = (gts > (DICE_THRESH * 100)).astype(int)
    
    acc = accuracy_score(y_bin, y_hat_bin)
    prec = precision_score(y_bin, y_hat_bin, zero_division=0)
    rec = recall_score(y_bin, y_hat_bin, zero_division=0)
    f1 = f1_score(y_bin, y_hat_bin, zero_division=0)
    
    inter = (y_hat_bin * y_bin).sum()
    dice = (2. * inter + DICE_SMOOTH) / (y_hat_bin.sum() + y_bin.sum() + DICE_SMOOTH)
    
    tn, fp, fn, tp = confusion_matrix(y_bin, y_hat_bin).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # 4. REPORT
    print("\n" + "="*50)
    print("MODEL PERFORMANCE REPORT (20m RESOLUTION)")
    print("="*50)
    print(f"Total Valid Pixels (20m): {len(gts):,}")
    print(f"Binary Threshold:         {DICE_THRESH*100}%")
    print("-" * 30)
    print("CONTINUOUS METRICS (0-100 scale):")
    print(f"  MAE:       {mae:.3f}")
    print(f"  RMSE:      {rmse:.3f}")
    print(f"  Bias:      {bias:.3f}")
    print(f"  MAE_snow:  {mae_snow:.3f}")
    print(f"  RMSE_snow: {rmse_snow:.3f}")
    print("-" * 30)
    print("CLASSIFICATION METRICS (Binary):")
    print(f"  Accuracy:    {acc:.3f}")
    print(f"  Precision:   {prec:.3f}")
    print(f"  Recall:      {rec:.3f}")
    print(f"  F1 Score:    {f1:.3f}")
    print(f"  Dice:        {dice:.3f}")
    print(f"  Specificity: {spec:.3f}")
    print("-" * 30)
    print("CONFUSION MATRIX COUNTS:")
    print(f"  TP: {tp:,} | FP: {fp:,}")
    print(f"  FN: {fn:,} | TN: {tn:,}")
    print("-" * 30)

if __name__ == "__main__":
    main()
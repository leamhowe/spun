import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Add scripts directory to path to find custom src modules
sys.path.append('/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts') 
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel
from src.snow_s2_dataset_13bands import Snows2Dataset

# --- CONFIGURATION (From your Notebook) ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/models/Final_T15_dino_none_scale10000_best.pt'
SPLIT_CSV = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
BACKBONE = 'resnet50'
BATCH_SIZE = 4

# --- METRIC PARAMETERS (From your Template) ---
DICE_THRESH = 0.5  # 50% threshold for binary metrics
DICE_SMOOTH = 1.0
LABEL_CLOUD = 200  # Value to ignore in ground truth

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # 1. Load Test Set List
    print("Loading Test Set List...")
    df = pd.read_csv(SPLIT_CSV)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()

    # 2. Setup Dataset and Model
    ds = Snows2Dataset(test_files, bands=BANDS, train=False, target_type='scf', train_with_clouds=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    in_channels = len(BANDS)
    hparams = {'backbone': BACKBONE, 'weights': None, 'batch_size': BATCH_SIZE}
    model = SnowModel(bands=BANDS, train_files=[], val_files=[], mask_clouds=False, use_augmentations=False, train_with_clouds=True, hparams=hparams, in_channels=in_channels)
    
    print(f"Loading Model: {os.path.basename(MODEL_PATH)}")
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    # 3. Inference Loop
    gts_all = []
    preds_all = []
    
    print(f"Starting Evaluation on {len(test_files)} tiles...")
    with torch.no_grad():
        for batch in tqdm(dl):
            x = batch['image'].to(DEVICE)
            y_raw = batch['mask'].to(DEVICE).float() # Ground truth 0-100, 200=cloud
            
            # Get model prediction (assumed 0.0 to 1.0)
            y_hat = model(x).squeeze(1) 
            y_hat_pct = (y_hat * 100.0).clamp(0, 100) # Convert to 0-100 scale
            
            # Mask out invalid pixels (Cloud value 200)
            valid_mask = (y_raw != LABEL_CLOUD)
            
            if valid_mask.any():
                preds_valid = y_hat_pct[valid_mask].cpu().numpy()
                gts_valid = y_raw[valid_mask].cpu().numpy()
                
                preds_all.append(preds_valid)
                gts_all.append(gts_valid)

    # 4. Metric Calculation
    print("\nComputing Metrics...")
    if len(gts_all) > 0:
        preds = np.concatenate(preds_all)
        gts = np.concatenate(gts_all)
        
        # Continuous Metrics
        diff = preds - gts
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        mean_error = np.mean(diff)
        
        # Snow-Specific Metrics (GT > 0)
        snow_mask = gts > 0
        if snow_mask.any():
            diff_snow = preds[snow_mask] - gts[snow_mask]
            mae_snow = np.mean(np.abs(diff_snow))
            rmse_snow = np.sqrt(np.mean(diff_snow ** 2))
        else:
            mae_snow, rmse_snow = float('nan'), float('nan')
            
        # Binary Metrics (Threshold at 50%)
        y_hat_bin = (preds > (DICE_THRESH * 100)).astype(int)
        y_bin = (gts > (DICE_THRESH * 100)).astype(int)
        
        precision = precision_score(y_bin, y_hat_bin, zero_division=0)
        recall = recall_score(y_bin, y_hat_bin, zero_division=0)
        f1 = f1_score(y_bin, y_hat_bin, zero_division=0)
        accuracy = accuracy_score(y_bin, y_hat_bin)
        
        # Dice calculation
        intersection = (y_hat_bin * y_bin).sum()
        union = y_hat_bin.sum() + y_bin.sum()
        dice = (2. * intersection + DICE_SMOOTH) / (union + DICE_SMOOTH)
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_bin, y_hat_bin).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

        # 5. FINAL OUTPUT REPORT (Same format as your template)
        print("\n" + "="*50)
        print("MODEL PERFORMANCE REPORT")
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
            print(">> Model is OVERESTIMATING snow cover.")
        elif mean_error < -0.5:
            print(">> Model is UNDERESTIMATING snow cover.")
        else:
            print(">> Model has no significant average bias.")
            
    else:
        print("No valid pixels found to evaluate.")

if __name__ == "__main__":
    main()
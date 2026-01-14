import os
import sys
import torch
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Add scripts directory to path to find custom src modules
sys.path.append('/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts') 
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel
from src.snow_s2_dataset_13bands import Snows2Dataset

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/models/Final_T15_dino_none_scale10000_best.pt'
SPLIT_CSV = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']

# --- CODES & PARAMS ---
LABEL_CLOUD = 200
LIS_CLOUD = 205
LIS_NODATA = 255
DICE_THRESH = 0.5 
DICE_SMOOTH = 1.0

# --- HELPER FUNCTIONS ---

def downscale_label_masked(arr, factor=2, invalid_val=200):
    arr_masked = np.where(arr == invalid_val, np.nan, arr)
    h, w = arr_masked.shape
    h_new, w_new = h // factor, w // factor
    blocks = arr_masked[:h_new*factor, :w_new*factor].reshape(h_new, factor, w_new, factor)
    sum_ds = np.nansum(blocks, axis=(1, 3))
    count_ds = np.count_nonzero(~np.isnan(blocks), axis=(1, 3))
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ds = sum_ds / count_ds
    return np.where(count_ds == 0, invalid_val, mean_ds).astype(np.float32)

def extract_ids_from_path(filepath):
    """
    Adjust this logic if your filenames differ. 
    Expected: DATE_STUFF_TILE.zarr -> returns DATE, TILE
    """
    fname = os.path.basename(filepath)
    parts = fname.split('_')
    date_part = parts[0]
    tile_part = parts[-1].split('.')[0]
    return date_part, tile_part

def calculate_metrics(y_true, y_pred, name="Model"):
    y_true_bin = (y_true > (DICE_THRESH * 100)).astype(int)
    y_pred_bin = (y_pred > (DICE_THRESH * 100)).astype(int)
    
    # Check if we have both classes for confusion matrix
    unique_classes = np.unique(np.concatenate([y_true_bin, y_pred_bin]))
    
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    
    return {
        "name": name, "pixels": len(y_true), 
        "mae": np.mean(np.abs(y_pred - y_true)),
        "rmse": np.sqrt(np.mean((y_pred - y_true)**2)),
        "acc": accuracy_score(y_true_bin, y_pred_bin),
        "prec": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "rec": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def main():
    df = pd.read_csv(SPLIT_CSV)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()

    if not test_files:
        print(f"Error: No files found in {SPLIT_CSV} with split 'Test'")
        return

    # Load Model (Added weights_only=True to suppress warning)
    # model = SnowModel(bands=BANDS, train_files=[], val_files=[], hparams={'backbone': 'resnet50'}, in_channels=len(BANDS))
    model = SnowModel(bands=BANDS, train_files=[], val_files=[], mask_clouds=False, use_augmentations=False, train_with_clouds=True, hparams={'backbone': 'resnet50'}, in_channels=len(BANDS))

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()

    strict_truth, strict_model, strict_lis = [], [], []
    inter_truth, inter_model, inter_lis = [], [], []

    skipped_no_lis = 0
    skipped_all_clouds = 0
    errors = 0

    print(f"Comparing UNet vs LIS on {len(test_files)} potential tiles...")
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            lis_name = f"{date_id}_{tile_id}_LIS.zarr"
            lis_path = os.path.join(LIS_PATCHES_DIR, lis_name)
            
            if not os.path.exists(lis_path):
                skipped_no_lis += 1
                continue
                
            # Load LIS and Model Data
            lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:] 
            temp_ds = Snows2Dataset([label_path], bands=BANDS, train=False, target_type='scf')
            sample = temp_ds[0]
            img_tensor = sample['image'].unsqueeze(0).to(DEVICE)
            mask_10m = sample['mask'].numpy()
            
            with torch.no_grad():
                out = model(img_tensor).squeeze().cpu().numpy()
                pred_10m = np.clip(out * 100.0, 0, 100)

            pred_20m = downscale_label_masked(pred_10m, factor=2)
            gt_20m = downscale_label_masked(mask_10m, factor=2)

            # Mask A: Strict
            mask_a = (gt_20m != LABEL_CLOUD)
            if mask_a.any():
                strict_truth.append(gt_20m[mask_a])
                strict_model.append(pred_20m[mask_a])
                lis_strict = lis_arr[mask_a].copy()
                lis_strict[lis_strict > 100] = 0
                strict_lis.append(lis_strict)
            else:
                skipped_all_clouds += 1

            # Mask B: Intersection
            mask_b = (gt_20m != LABEL_CLOUD) & (lis_arr <= 100)
            if mask_b.any():
                inter_truth.append(gt_20m[mask_b])
                inter_model.append(pred_20m[mask_b])
                inter_lis.append(lis_arr[mask_b])
                
        except Exception as e:
            errors += 1
            continue

    # Diagnostic Summary
    print(f"\n--- Processing Summary ---")
    print(f"Total files in Test set: {len(test_files)}")
    print(f"Skipped (LIS file not found): {skipped_no_lis}")
    print(f"Skipped (Entirely cloudy):    {skipped_all_clouds}")
    print(f"Processing Errors:            {errors}")

    if not strict_truth:
        print("\nERROR: No valid data found for comparison. Please check your LIS_PATCHES_DIR.")
        print(f"Last attempted LIS path: {lis_path}")
        return

    # Final Compilation
    print("\n" + "="*60)
    print("JUSTIFIABLE PERFORMANCE REPORT (20m RESOLUTION)")
    print("="*60)

    # Strategy A
    y_t = np.concatenate(strict_truth)
    y_m = np.concatenate(strict_model)
    y_l = np.concatenate(strict_lis)
    
    print("\nSTRATEGY 1: STRICT COMPARISON (Ground Truth Mask)")
    for res in [calculate_metrics(y_t, y_m, "Your UNet"), calculate_metrics(y_t, y_l, "LIS Product")]:
        print(f"\n[{res['name']}] Pixels: {res['pixels']:,}")
        print(f"MAE: {res['mae']:.3f} | F1: {res['f1']:.3f} | Recall: {res['rec']:.3f}")

    # Strategy B
    if inter_truth:
        y_t_i = np.concatenate(inter_truth)
        y_m_i = np.concatenate(inter_model)
        y_l_i = np.concatenate(inter_lis)
        print("\nSTRATEGY 2: INTERSECTION COMPARISON (Clear-Sky Only)")
        for res in [calculate_metrics(y_t_i, y_m_i, "Your UNet"), calculate_metrics(y_t_i, y_l_i, "LIS Product")]:
            print(f"\n[{res['name']}] Pixels: {res['pixels']:,}")
            print(f"MAE: {res['mae']:.3f} | F1: {res['f1']:.3f} | Recall: {res['rec']:.3f}")

if __name__ == "__main__":
    main()
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
DICE_THRESH = 0.5 
DICE_SMOOTH = 1.0

# --- HELPER FUNCTIONS ---

def crop_center_or_topleft(img, target_shape):
    """Crops the larger image to match the smaller target_shape."""
    h, w = img.shape
    th, tw = target_shape
    
    # Simple top-left crop (usually safe for grid-aligned data)
    # You can change to center crop if alignment is off by margins
    return img[:th, :tw]

def downscale_label_masked(arr, factor=2, invalid_val=200):
    """Downscales 10m -> 20m ignoring invalid_val (clouds)."""
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
    fname = os.path.basename(filepath)
    parts = fname.split('_')
    date_part = parts[0]
    tile_part = parts[-1].split('.')[0]
    return date_part, tile_part

def calculate_metrics(y_true, y_pred, name="Model"):
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    bias = np.mean(diff)
    
    y_true_bin = (y_true > (DICE_THRESH * 100)).astype(int)
    y_pred_bin = (y_pred > (DICE_THRESH * 100)).astype(int)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    return {
        "name": name, "pixels": len(y_true), "mae": mae, "rmse": rmse, "bias": bias,
        "acc": acc, "rec": rec, "f1": f1
    }

def main():
    df = pd.read_csv(SPLIT_CSV)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()

    # model = SnowModel(bands=BANDS, train_files=[], val_files=[], hparams={'backbone': 'resnet50'}, in_channels=len(BANDS))
    model = SnowModel(bands=BANDS, train_files=[], val_files=[], mask_clouds=False, use_augmentations=False, train_with_clouds=True, hparams={'backbone': 'resnet50'}, in_channels=len(BANDS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()

    strict_truth, strict_model, strict_lis = [], [], []
    inter_truth, inter_model, inter_lis = [], [], []

    print(f"Comparing UNet vs LIS on {len(test_files)} potential tiles...")
    
    errors = 0
    first_error = True
    
    for label_path in tqdm(test_files):
        try:
            date_id, tile_id = extract_ids_from_path(label_path)
            lis_path = os.path.join(LIS_PATCHES_DIR, f"{date_id}_{tile_id}_LIS.zarr")
            
            if not os.path.exists(lis_path): continue
                
            # 1. Load Data
            lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:] 
            
            # 2. Model Prediction
            temp_ds = Snows2Dataset([label_path], bands=BANDS, train=False, target_type='scf')
            sample = temp_ds[0]
            img_tensor = sample['image'].unsqueeze(0).to(DEVICE)
            mask_10m = sample['mask'].numpy()
            
            with torch.no_grad():
                out = model(img_tensor).squeeze().cpu().numpy()
                pred_10m = np.clip(out * 100.0, 0, 100)

            # 3. Downscale
            pred_20m = downscale_label_masked(pred_10m, factor=2)
            gt_20m = downscale_label_masked(mask_10m, factor=2)

            # 4. SHAPE MATCHING (Fixing the likely crash)
            if gt_20m.shape != lis_arr.shape:
                # Find min dims
                h = min(gt_20m.shape[0], lis_arr.shape[0])
                w = min(gt_20m.shape[1], lis_arr.shape[1])
                
                # Crop all to min dims
                gt_20m = gt_20m[:h, :w]
                pred_20m = pred_20m[:h, :w]
                lis_arr = lis_arr[:h, :w]

            # 5. Masking Logic
            # Mask A: Strict
            mask_a = (gt_20m != LABEL_CLOUD)
            if mask_a.any():
                strict_truth.append(gt_20m[mask_a])
                strict_model.append(pred_20m[mask_a])
                
                lis_strict = lis_arr[mask_a].copy()
                lis_strict[lis_strict > 100] = 0
                strict_lis.append(lis_strict)

            # Mask B: Intersection
            mask_b = (gt_20m != LABEL_CLOUD) & (lis_arr <= 100)
            if mask_b.any():
                inter_truth.append(gt_20m[mask_b])
                inter_model.append(pred_20m[mask_b])
                inter_lis.append(lis_arr[mask_b])
                
        except Exception as e:
            if first_error:
                print(f"\n[DIAGNOSTIC] FAILED on {tile_id}: {e}")
                print(f"Expected LIS Path: {lis_path}")
                first_error = False
            errors += 1
            continue

    print(f"\nProcessing Errors: {errors}")

    if not strict_truth:
        print("No valid data found.")
        return

    # Final Compilation
    print("\n" + "="*60)
    print("JUSTIFIABLE PERFORMANCE REPORT (20m RESOLUTION)")
    print("="*60)

    y_t, y_m, y_l = np.concatenate(strict_truth), np.concatenate(strict_model), np.concatenate(strict_lis)
    print("\nSTRATEGY 1: STRICT COMPARISON (Ground Truth Mask)")
    for res in [calculate_metrics(y_t, y_m, "Your UNet"), calculate_metrics(y_t, y_l, "LIS Product")]:
        print(f"\n[{res['name']}] Pixels: {res['pixels']:,}")
        print(f"MAE: {res['mae']:.3f} | F1: {res['f1']:.3f} | Recall: {res['rec']:.3f}")

    if inter_truth:
        y_t_i, y_m_i, y_l_i = np.concatenate(inter_truth), np.concatenate(inter_model), np.concatenate(inter_lis)
        print("\n\nSTRATEGY 2: INTERSECTION COMPARISON (Clear-Sky Only)")
        for res in [calculate_metrics(y_t_i, y_m_i, "Your UNet"), calculate_metrics(y_t_i, y_l_i, "LIS Product")]:
            print(f"\n[{res['name']}] Pixels: {res['pixels']:,}")
            print(f"MAE: {res['mae']:.3f} | F1: {res['f1']:.3f} | Recall: {res['rec']:.3f}")

if __name__ == "__main__":
    main()
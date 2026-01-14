import pandas as pd
import numpy as np
import os
import zarr
import sys
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# --- IMPORTS ---
sys.path.append('/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/scripts') 
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel
from src.snow_s2_dataset_13bands import Snows2Dataset

# --- CONFIGURATION ---
SPLIT_CSV_PATH  = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
LIS_PATCHES_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
MODEL_PATH      = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/models/Final_T15_dino_none_scale10000_best.pt'
REPORT_OUTPUT   = 'comparison_report_final.txt'

BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
BACKBONE = 'resnet50'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- PARAMETERS ---
LABEL_CLOUD = 200
LIS_CLOUD   = 205
LIS_NODATA  = 255
DICE_SMOOTH = 1.0 
DICE_THRESH = 0.5
FLIP_LIS    = False 

# --- HELPER FUNCTIONS ---

def downscale_label_masked(arr, factor=2, invalid_val=200):
    """Downscales array, handling invalid values."""
    arr_masked = np.where(arr == invalid_val, np.nan, arr)
    h, w = arr_masked.shape
    h_new, w_new = h // factor, w // factor
    
    # Trim to factor multiple if needed (before reshaping)
    arr_masked = arr_masked[:h_new*factor, :w_new*factor]
    
    blocks = arr_masked.reshape(h_new, factor, w_new, factor)
    
    sum_ds = np.nansum(blocks, axis=(1, 3))
    count_ds = np.count_nonzero(~np.isnan(blocks), axis=(1, 3))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ds = sum_ds / count_ds
        
    return np.where(count_ds == 0, invalid_val, mean_ds).astype(np.float32)

def center_crop_to_match(arr, match_shape):
    """
    Crops 'arr' to the size of 'match_shape' from the center.
    Used to remove the padding added by the DataLoader.
    """
    h_in, w_in = arr.shape
    h_tgt, w_tgt = match_shape
    
    if h_in <= h_tgt and w_in <= w_tgt:
        return arr # No cropping needed
        
    diff_h = h_in - h_tgt
    diff_w = w_in - w_tgt
    
    # Integer division automatically handles the offset
    top = diff_h // 2
    left = diff_w // 2
    
    return arr[top : top+h_tgt, left : left+w_tgt]

def extract_ids_from_path(filepath):
    fname = os.path.basename(filepath)
    date_part = fname.split('_')[0]
    tile_part = fname.split('_')[-1].split('.')[0]
    return date_part, tile_part

def compute_metrics(y_true, y_pred, thresh=0.5):
    if len(y_true) == 0:
        return {k: 0.0 for k in ['acc', 'prec', 'rec', 'f1', 'dice', 'mae', 'rmse', 'count']}
    
    y_true_bin = (y_true > (thresh * 100)).astype(int)
    y_pred_bin = (y_pred > (thresh * 100)).astype(int)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    inter = (y_true_bin * y_pred_bin).sum()
    union = y_true_bin.sum() + y_pred_bin.sum()
    dice = (2. * inter + DICE_SMOOTH) / (union + DICE_SMOOTH)
    
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2)) # Added RMSE calculation
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'dice': dice, 'mae': mae, 'rmse': rmse, 'count': len(y_true)}

# --- MAIN EXECUTION ---

def main():
    print("--- LOADING CONFIGURATION ---")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files_list = df[df[col] == 'Test']['filepath'].tolist()
    
    print(f"Loading Model: {os.path.basename(MODEL_PATH)}")
    model = SnowModel(bands=BANDS, train_files=[], val_files=[], mask_clouds=False, use_augmentations=False, train_with_clouds=True, hparams={'backbone': BACKBONE}, in_channels=len(BANDS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.to(DEVICE).eval()
    
    # Init Dataset
    ds = Snows2Dataset(test_files_list, bands=BANDS, train=False, target_type='scf', train_with_clouds=True)
    
    data_clear = {'gt': [], 'spun': [], 'lis': []} 
    data_all   = {'gt': [], 'spun': [], 'lis': []} 

    print(f"\nProcessing {len(test_files_list)} tiles...")
    
    with torch.no_grad():
        for i, label_path in enumerate(tqdm(test_files_list)):
            try:
                # 1. LIS Data
                date_id, tile_id = extract_ids_from_path(label_path)
                lis_zarr_name = f"{date_id}_{tile_id}_LIS.zarr"
                lis_path = os.path.join(LIS_PATCHES_DIR, lis_zarr_name)
                
                if not os.path.exists(lis_path): continue
                
                lis_arr = zarr.open(lis_path, mode='r')['lis_snow'][:] 
                if FLIP_LIS: lis_arr = lis_arr[::-1, :]

                # 2. SPUN Inference
                sample = ds[i] 
                image = sample['image'].unsqueeze(0).to(DEVICE)
                mask = sample['mask'].numpy() 
                
                out = model(image).squeeze(1).cpu().numpy()
                spun_arr_100 = (out * 100.0).clip(0, 100)[0][::-1, :]
                lbl_arr_100 = mask[::-1, :]
                
                # 3. Downscale
                gt_20 = downscale_label_masked(lbl_arr_100, factor=2)
                spun_20 = downscale_label_masked(spun_arr_100, factor=2, invalid_val=-999)
                
                # 4. FIX: Center Crop GT/SPUN to match LIS
                # LIS is the "Truth" size. GT is padded. We remove padding here.
                gt_20 = center_crop_to_match(gt_20, lis_arr.shape)
                spun_20 = center_crop_to_match(spun_20, lis_arr.shape)

                # Final Shape Check
                if lis_arr.shape != gt_20.shape:
                    # If shapes still don't match, LIS might be bigger or something odd is happening
                    print(f"Skipping {tile_id}: Shape mismatch after crop {lis_arr.shape} vs {gt_20.shape}")
                    continue

                y_gt = gt_20.flatten()
                y_spun = spun_20.flatten()
                y_lis = lis_arr.flatten()

                # 5. Populate Metrics Lists (Same logic as before)
                mask_clear = (y_gt != LABEL_CLOUD) & (y_lis != LIS_CLOUD) & (y_lis != LIS_NODATA) & (~np.isnan(y_lis))
                if mask_clear.any():
                    data_clear['gt'].append(y_gt[mask_clear])
                    data_clear['spun'].append(y_spun[mask_clear])
                    data_clear['lis'].append(y_lis[mask_clear])

                mask_all = (y_gt != LABEL_CLOUD)
                if mask_all.any():
                    lis_filled = np.where(y_lis >= 200, 0, y_lis) 
                    lis_filled = np.nan_to_num(lis_filled, nan=0.0)
                    data_all['gt'].append(y_gt[mask_all])
                    data_all['spun'].append(y_spun[mask_all])
                    data_all['lis'].append(lis_filled[mask_all])

            except Exception:
                continue

    # --- REPORT ---
    print("\nComputing Metrics...")
    
    def get_m(d):
        if not d['gt']: return None, None
        gt, sp, li = np.concatenate(d['gt']), np.concatenate(d['spun']), np.concatenate(d['lis'])
        return compute_metrics(gt, sp), compute_metrics(gt, li)

    m_sp_cl, m_li_cl = get_m(data_clear)
    m_sp_al, m_li_al = get_m(data_all)

    with open(REPORT_OUTPUT, "w") as f:
        def log_print(text):
            print(text)
            f.write(text + "\n")

        # Added extra column space for RMSE
        row_fmt = "{:<12} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<12}"
        
        log_print("\n" + "="*98)
        log_print("FINAL COMPARISON REPORT: SPUN vs LIS")
        log_print("="*98)
        
        if m_sp_cl:
            log_print("\nSCENARIO 1: CLEAR-SKY")
            log_print("-" * 98)
            # Added RMSE to header and row formatting
            log_print(row_fmt.format("Model", "Recall", "Precis", "F1", "Dice", "MAE", "RMSE", "Pixels"))
            log_print("-" * 98)
            log_print(row_fmt.format("SPUN", f"{m_sp_cl['rec']:.3f}", f"{m_sp_cl['prec']:.3f}", f"{m_sp_cl['f1']:.3f}", f"{m_sp_cl['dice']:.3f}", f"{m_sp_cl['mae']:.3f}", f"{m_sp_cl['rmse']:.3f}", f"{m_sp_cl['count']:,}"))
            log_print(row_fmt.format("LIS", f"{m_li_cl['rec']:.3f}", f"{m_li_cl['prec']:.3f}", f"{m_li_cl['f1']:.3f}", f"{m_li_cl['dice']:.3f}", f"{m_li_cl['mae']:.3f}", f"{m_li_cl['rmse']:.3f}", f"{m_li_cl['count']:,}"))
            log_print("-" * 98)
            
        if m_sp_al:
            log_print("\nSCENARIO 2: ALL-SKY")
            log_print("-" * 98)
            log_print(row_fmt.format("Model", "Recall", "Precis", "F1", "Dice", "MAE", "RMSE", "Pixels"))
            log_print("-" * 98)
            log_print(row_fmt.format("SPUN", f"{m_sp_al['rec']:.3f}", f"{m_sp_al['prec']:.3f}", f"{m_sp_al['f1']:.3f}", f"{m_sp_al['dice']:.3f}", f"{m_sp_al['mae']:.3f}", f"{m_sp_al['rmse']:.3f}", f"{m_sp_al['count']:,}"))
            log_print(row_fmt.format("LIS", f"{m_li_al['rec']:.3f}", f"{m_li_al['prec']:.3f}", f"{m_li_al['f1']:.3f}", f"{m_li_al['dice']:.3f}", f"{m_li_al['mae']:.3f}", f"{m_li_al['rmse']:.3f}", f"{m_li_al['count']:,}"))
            log_print("-" * 98)

    print(f"\nReport saved to: {os.path.abspath(REPORT_OUTPUT)}")

if __name__ == "__main__":
    main()
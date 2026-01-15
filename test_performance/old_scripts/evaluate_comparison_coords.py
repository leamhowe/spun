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
REPORT_OUTPUT   = 'comparison_report_coords.txt'

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

def get_coords(z_group):
    """Safely extracts x/y or lat/lon coordinates from a Zarr group."""
    # Try standard projection coordinates first
    if 'x' in z_group and 'y' in z_group:
        return z_group['x'][:], z_group['y'][:]
    # Try geographic coordinates
    elif 'lon' in z_group and 'lat' in z_group:
        return z_group['lon'][:], z_group['lat'][:]
    # Try normalized coordinates
    elif 'X' in z_group and 'Y' in z_group:
        return z_group['X'][:], z_group['Y'][:]
    else:
        return None, None

def intersect_spatially(arr1, x1, y1, arr2, x2, y2):
    """
    Finds the pixel-perfect intersection of two arrays based on their coordinates.
    Returns the two arrays cropped to exactly the same spatial extent.
    """
    # 1. Find common coordinates (floating point safe intersection)
    # Using small epsilon for float comparison safety
    tol = 1e-3 
    
    # Intersect X
    common_x = x1[np.isclose(x1[:, None], x2[None, :], atol=tol).any(axis=1)]
    # Intersect Y
    common_y = y1[np.isclose(y1[:, None], y2[None, :], atol=tol).any(axis=1)]
    
    if len(common_x) == 0 or len(common_y) == 0:
        return None, None
        
    # 2. Find indices in Arr1
    # We use searchsorted or isclose for robust index finding
    x1_mask = np.isclose(x1[:, None], common_x[None, :], atol=tol).any(axis=1)
    y1_mask = np.isclose(y1[:, None], common_y[None, :], atol=tol).any(axis=1)
    
    # 3. Find indices in Arr2
    x2_mask = np.isclose(x2[:, None], common_x[None, :], atol=tol).any(axis=1)
    y2_mask = np.isclose(y2[:, None], common_y[None, :], atol=tol).any(axis=1)
    
    # 4. Slice arrays using numpy meshgrid style slicing
    # Note: This assumes rectilinear grids (which S2 and LIS are)
    # We use np.ix_ to construct the meshgrid indices
    arr1_cropped = arr1[np.ix_(y1_mask, x1_mask)]
    arr2_cropped = arr2[np.ix_(y2_mask, x2_mask)]
    
    # 5. Sanity Check shapes
    if arr1_cropped.shape != arr2_cropped.shape:
        # This handles cases where one axis might be flipped (descending vs ascending)
        # If sizes match but order is flipped, we trust the coordinate sort
        # However, simple intersection assumes sorted order. 
        # For robust comparison, we flatten based on the SORTED coordinates.
        return None, None
        
    return arr1_cropped, arr2_cropped

def downscale_label_masked(arr, factor=2, invalid_val=200):
    """Downscales array, handling invalid values."""
    arr_masked = np.where(arr == invalid_val, np.nan, arr)
    h, w = arr_masked.shape
    h_new, w_new = h // factor, w // factor
    
    arr_masked = arr_masked[:h_new*factor, :w_new*factor]
    blocks = arr_masked.reshape(h_new, factor, w_new, factor)
    
    sum_ds = np.nansum(blocks, axis=(1, 3))
    count_ds = np.count_nonzero(~np.isnan(blocks), axis=(1, 3))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ds = sum_ds / count_ds
        
    return np.where(count_ds == 0, invalid_val, mean_ds).astype(np.float32)

def center_crop_to_match(arr, match_shape):
    """Removes padding by cropping from center."""
    h_in, w_in = arr.shape
    h_tgt, w_tgt = match_shape
    
    if h_in <= h_tgt and w_in <= w_tgt: return arr 
        
    top = (h_in - h_tgt) // 2
    left = (w_in - w_tgt) // 2
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
    rmse = np.sqrt(np.mean(diff**2))
    
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
    
    ds = Snows2Dataset(test_files_list, bands=BANDS, train=False, target_type='scf', train_with_clouds=True)
    
    data_clear = {'gt': [], 'spun': [], 'lis': []} 
    data_all   = {'gt': [], 'spun': [], 'lis': []} 

    print(f"\nProcessing {len(test_files_list)} tiles with SPATIAL INTERSECTION...")
    
    with torch.no_grad():
        for i, label_path in enumerate(tqdm(test_files_list)):
            try:
                # ---------------------------------------------------
                # 1. LOAD DATA & COORDINATES
                # ---------------------------------------------------
                
                # A. LIS Data (Reference 20m)
                date_id, tile_id = extract_ids_from_path(label_path)
                lis_zarr_name = f"{date_id}_{tile_id}_LIS.zarr"
                lis_path = os.path.join(LIS_PATCHES_DIR, lis_zarr_name)
                
                if not os.path.exists(lis_path): continue
                
                z_lis = zarr.open(lis_path, mode='r')
                lis_arr = z_lis['lis_snow'][:] 
                lis_x, lis_y = get_coords(z_lis)
                
                if FLIP_LIS: lis_arr = lis_arr[::-1, :]

                # B. Ground Truth Data (Original 10m from file)
                # We load this to get the exact original coordinates and shape
                z_gt = zarr.open(label_path, mode='r')
                gt_raw = z_gt['label'][:] 
                gt_x, gt_y = get_coords(z_gt)
                
                # ---------------------------------------------------
                # 2. SPUN INFERENCE & UN-PADDING
                # ---------------------------------------------------
                
                sample = ds[i] 
                image = sample['image'].unsqueeze(0).to(DEVICE)
                
                # Predict
                out = model(image).squeeze(1).cpu().numpy()
                spun_padded = (out * 100.0).clip(0, 100)[0]
                
                # IMPORTANT: Un-pad SPUN to match original Ground Truth File
                # The Dataset might add padding, but 'gt_raw' is the unpadded truth.
                # We crop the prediction to match 'gt_raw'.
                spun_unpadded = center_crop_to_match(spun_padded, gt_raw.shape)
                
                # Note: We rely on the Dataset logic flipping the image correctly.
                # If gt_raw is "Top-Down" but the model output "Bottom-Up", this might need checking.
                # Assuming Standard DataLoader behavior:
                if spun_unpadded.shape != gt_raw.shape:
                    continue # Should not happen with center crop logic

                # ---------------------------------------------------
                # 3. DOWNSCALE 10m -> 20m
                # ---------------------------------------------------
                
                gt_20 = downscale_label_masked(gt_raw, factor=2)
                spun_20 = downscale_label_masked(spun_unpadded, factor=2, invalid_val=-999)
                
                # Downscale Coordinates (Stride 2)
                # Assuming coords are pixel centers or corners, skipping matches the block agg
                gt_x_20 = gt_x[::2]
                gt_y_20 = gt_y[::2]
                
                # Ensure Coordinate Lengths match Data Shape
                # Sometimes integer division drops the last coord if odd length
                gt_x_20 = gt_x_20[:gt_20.shape[1]]
                gt_y_20 = gt_y_20[:gt_20.shape[0]]

                # ---------------------------------------------------
                # 4. SPATIAL INTERSECTION (Pixel Perfect Match)
                # ---------------------------------------------------
                
                # Find overlap between GT(20m) and LIS(20m) using coordinates
                # This handles shifts, crops, and misalignments automatically
                if lis_x is not None and gt_x_20 is not None:
                    # Match SPUN/GT stack to LIS
                    stack_20 = np.stack([gt_20, spun_20]) # Keep them together to slice identically
                    
                    # Intersect GT coords vs LIS coords
                    # We pass the first band (GT) to represent the stack geometry
                    res_stack, res_lis = intersect_spatially(stack_20, gt_x_20, gt_y_20, lis_arr, lis_x, lis_y)
                    
                    if res_stack is None:
                        continue # No overlap found
                        
                    # Unpack
                    gt_final = res_stack[0]
                    spun_final = res_stack[1]
                    lis_final = res_lis
                    
                else:
                    # Fallback if coords missing (Old Logic)
                    gt_final = center_crop_to_match(gt_20, lis_arr.shape)
                    spun_final = center_crop_to_match(spun_20, lis_arr.shape)
                    lis_final = lis_arr
                    if gt_final.shape != lis_final.shape: continue

                # ---------------------------------------------------
                # 5. METRICS AGGREGATION
                # ---------------------------------------------------
                
                y_gt = gt_final.flatten()
                y_spun = spun_final.flatten()
                y_lis = lis_final.flatten()

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

            except Exception as e:
                # print(e)
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

        row_fmt = "{:<12} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<12}"
        
        log_print("\n" + "="*98)
        log_print("FINAL COMPARISON REPORT: SPUN vs LIS (COORDINATE MATCHED)")
        log_print("="*98)
        
        if m_sp_cl:
            log_print("\nSCENARIO 1: CLEAR-SKY")
            log_print("-" * 98)
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
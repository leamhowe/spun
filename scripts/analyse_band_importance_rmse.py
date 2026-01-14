import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

# --- Add your scripts folder to path ---
sys.path.append('./scripts') 

# --- Imports from your codebase ---
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel
from src.snow_s2_dataset_13bands import Snows2Dataset

# --- Configuration ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# MODEL_PATH = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/models/Final_T66_Softcon_invlinear_NORM_best.pt'
MODEL_PATH = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/models/Final_T4_softcon_inlinear_scale10000_best.pt'
SPLIT_FILE = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

# Match these to your training settings
BANDS = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
BACKBONE = 'resnet50'
BATCH_SIZE = 8 # Smaller batch size is fine for inference
NUM_WORKERS = 4

def prepare_data_from_split_file(split_csv_path):
    """Helper to get file lists."""
    df = pd.read_csv(split_csv_path)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()
    return test_files

def evaluate_dataset(model, dataloader, device, perturb_band_idx=None):
    """
    Evaluates the model. If perturb_band_idx is set, it shuffles that band's data.
    """
    acc = model._create_accumulator_dict()
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['image'].to(device)
            y_raw = batch['mask'].to(device).float()
            
            # --- PERMUTATION LOGIC ---
            if perturb_band_idx is not None:
                # Get the specific band channel
                band_data = x[:, perturb_band_idx, :, :]
                
                # Shuffle the indices within the batch dimension
                # This breaks the correlation between this band and the specific image/label
                # while preserving the overall distribution of values for this band.
                idx = torch.randperm(band_data.shape[0])
                x[:, perturb_band_idx, :, :] = band_data[idx, :, :]
            # -------------------------

            y_hat = model(x)
            metrics = model._calculate_metrics(y_hat, y_raw)
            model._update_accumulators(acc, y_hat, y_raw, metrics)
    
    # Calculate final RMSE
    n = acc['n']
    rmse = (acc['sq_err'] / n)**0.5
    return rmse

def main():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. Load Data
    test_files = prepare_data_from_split_file(SPLIT_FILE)
    
    # Use a subset for speed if needed (e.g., first 200 files), or use full set for accuracy
    # test_files = test_files[:200] 
    
    ds = Snows2Dataset(
        test_files, 
        bands=BANDS, 
        train=False, 
        mask_clouds=False, 
        use_augmentations=False, 
        target_type='scf', 
        train_with_clouds=True # Important: Match your training setting
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 2. Load Model
    # Re-create hparams to initialize structure (values don't matter for loading weights)
    hparams = {
        'backbone': BACKBONE, 'weights': None, 'loss_fn': 'mse', 'lr': 1e-4,
        'batch_size': BATCH_SIZE, 'patience': 10, 'num_workers': 0,
        'dice_thresh': 0.5, 'dice_loss_weight': 0.5, 'loss_weighting': 'none'
    }
    
    # Determine input channels
    C = len(ds[0]['image'])
    
    model = SnowModel(
        bands=BANDS, train_files=[], val_files=[], mask_clouds=False, 
        use_augmentations=False, train_with_clouds=True,
        hparams=hparams, in_channels=C
    )
    
    # Load weights
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    
    print("Model loaded. Starting Permutation Importance Analysis...")
    
    # 3. Baseline Evaluation
    print("Calculating Baseline RMSE (Original Data)...")
    baseline_rmse = evaluate_dataset(model, dl, DEVICE, perturb_band_idx=None)
    print(f"Baseline RMSE: {baseline_rmse:.5f}")
    
    # 4. Permutation Loop
    results = []
    
    for i, band_name in enumerate(tqdm(BANDS, desc="Permuting Bands")):
        permuted_rmse = evaluate_dataset(model, dl, DEVICE, perturb_band_idx=i)
        
        # Importance is how much the error INCREASED
        importance = permuted_rmse - baseline_rmse
        
        results.append({
            'Band': band_name,
            'Baseline_RMSE': baseline_rmse,
            'Permuted_RMSE': permuted_rmse,
            'Importance_Drop': importance
        })
        print(f"  {band_name}: RMSE = {permuted_rmse:.5f} (Drop: {importance:.5f})")

    # 5. Visualization
    df = pd.DataFrame(results)
    
    # Save raw data
    df.to_csv('band_importance_results.csv', index=False)
    
    # Sort by importance
    df = df.sort_values('Importance_Drop', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Band', y='Importance_Drop', color='skyblue')
    plt.title(f'Sentinel-2 Band Importance (Permutation)\nModel: {os.path.basename(MODEL_PATH)}')
    plt.ylabel('Increase in RMSE when band is shuffled')
    plt.xlabel('Band')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = 'band_importance_plot.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\nAnalysis complete. Plot saved to {plot_path}")
    print("Top 3 Most Important Bands:")
    print(df.head(3)[['Band', 'Importance_Drop']])

if __name__ == "__main__":
    main()
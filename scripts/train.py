import argparse
import os
import torch
import pandas as pd
import numpy as np
import wandb
import random

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        message="The codec `vlen-utf8` is currently not part")

# --- CUSTOM IMPORTS ---
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel 
from src.snow_s2_dataset_13bands import Snows2Dataset
from src.config import MODELS_DIR, METRICS_DIR, WANDB_DIR 

torch.set_float32_matmul_precision('high')

def get_args():
    parser = argparse.ArgumentParser(description="Run UNet model training.")
    
    # Core Model Setup
    parser.add_argument('--model_name', type=str, default='SPUN', help='Model name for saving outputs')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone model for the UNet')
    parser.add_argument('--weights', type=str, default=None, help='Pretrained weights for the backbone')
    parser.add_argument('--bands', type=str, default='all_spectral_bands', help='Specify the bands to use in the model')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to resume training')
    
    # Data & Augmentation
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing training data')
    parser.add_argument('--split_file', type=str, default='dataset_metadata.csv', help='Path to a CSV file defining splits')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test split ratio')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for dataset split')
    parser.add_argument('--use_augmentations', action='store_true', help='Flag to enable image augmentations')
    parser.add_argument('--train_with_clouds', action='store_true', help='Include clouds in training')
    parser.add_argument('--mask_clouds', action='store_true', help='Mask cloud pixels in the images')
    
    # Loss & Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--loss_fn', type=str, default='l1', help="Regression loss: 'l1', 'mse', or 'huber'")
    parser.add_argument('--loss_weighting', type=str, default='none', help="Per-pixel weighting: 'none'|'binary'|'linear'|'inverse_linear'")
    parser.add_argument('--snow_weight', type=float, default=5.0, help='Weight for snow pixels')
    parser.add_argument('--snow_thresh', type=float, default=0.05, help='Snow threshold in [0,1]')
    parser.add_argument('--ignore_value', type=int, default=200, help='Ignore value in labels')
    parser.add_argument('--dice_thresh', type=float, default=0.5, help='Binarization threshold for Dice score.')
    parser.add_argument('--dice_loss_weight', type=float, default=0.5, help='Weight lambda for Dice loss in combined loss.')
    
    return parser.parse_args()


def prepare_data_from_split_file(data_root, split_csv_path):
    """
    Reads the dataset metadata CSV and constructs absolute filepaths 
    by joining the data_root with the relative paths in the CSV.
    """
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"Split file not found: {split_csv_path}")
    
    df = pd.read_csv(split_csv_path)
    
    # Check for the correct column names based on your Zenodo README
    split_col = 'split' if 'split' in df.columns else 'split_cleaned'
    path_col = 'relative_path' if 'relative_path' in df.columns else 'filepath'
    
    # Helper function to construct absolute paths
    def get_abs_paths(split_name):
        rel_paths = df[df[split_col] == split_name][path_col].tolist()
        return [os.path.join(data_root, p) for p in rel_paths]

    train_files = get_abs_paths('Train')
    val_files = get_abs_paths('Validation')
    test_files = get_abs_paths('Test')
    
    print(f"Loaded splits from CSV: {split_csv_path}")
    print(f"Total files: {len(train_files)+len(val_files)+len(test_files)} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    
    return train_files, val_files, test_files


def prepare_data_random(directories, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    all_files = []
    for d in directories:
        if os.path.isdir(d):
            all_files.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith('.zarr')])

    all_files = sorted(list(set(all_files)))
    random.seed(seed)
    random.shuffle(all_files)

    n = len(all_files)
    n_train, n_val = int(train_ratio * n), int(val_ratio * n)
    return all_files[:n_train], all_files[n_train:n_train + n_val], all_files[n_train + n_val:]


def test_model(trainer, snow_model, test_files, bands, args, metrics_save_dir):
    if not test_files:
        print("Test file list is empty. Skipping inference.")
        return

    print(f"\n--- Running inference on {len(test_files)} test files ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snow_model.eval().to(device)

    test_dataset = Snows2Dataset(
        test_files, bands=bands, mask_clouds=args.mask_clouds,
        target_type='scf', train_with_clouds=args.train_with_clouds 
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
    test_acc = snow_model._create_accumulator_dict()

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels_raw = batch.get('mask', batch.get('label')).to(device).float()
            outputs = snow_model(images)
            metrics = snow_model._calculate_metrics(outputs, labels_raw)
            snow_model._update_accumulators(test_acc, outputs, labels_raw, metrics)

    if trainer.global_rank == 0:
        final_metrics = {}
        n, n_snow = test_acc.get('n', 0), test_acc.get('n_snow', 0)
        
        final_metrics['mae'] = test_acc['abs_err'] / n if n > 0 else np.nan
        final_metrics['rmse'] = (test_acc['sq_err'] / n)**0.5 if n > 0 else np.nan
        final_metrics['mae_snow'] = test_acc['abs_err_snow'] / n_snow if n_snow > 0 else np.nan
        final_metrics['rmse_snow'] = (test_acc['sq_err_snow'] / n_snow)**0.5 if n_snow > 0 else np.nan
        
        inter, union = test_acc.get('dice_inter', 0), test_acc.get('dice_union', 0)
        final_metrics['dice'] = (2. * inter + snow_model.dice_smooth) / (union + snow_model.dice_smooth) if union > 0 else np.nan

        print("Test Metrics:", final_metrics)
        df = pd.DataFrame([final_metrics])
        metric_path = os.path.join(metrics_save_dir, f"{args.model_name.split('.')[0]}_test_metrics.csv")
        df.to_csv(metric_path, index=False)
        print(f"Test metrics saved to {metric_path}")


def main():
    args = get_args()

    # Data root and output paths
    data_root = args.data_root
    model_save_dir = str(MODELS_DIR)
    metrics_save_dir = str(METRICS_DIR)
        
    # Setup Weights & Biases Logger
    wandb_logger = WandbLogger(
        project="spun_scf",
        name=args.model_name,
        config=vars(args),
        save_dir=str(WANDB_DIR)
    )

    # --- Prepare Data ---
    
    # If the user provides an absolute path, use it. Otherwise, look inside data_root.
    if os.path.isabs(args.split_file):
        split_file_path = args.split_file
    else:
        split_file_path = os.path.join(data_root, args.split_file)

    if os.path.exists(split_file_path):
        print(f"Loading data splits from: {split_file_path}")
        train_files, val_files, test_files = prepare_data_from_split_file(
            data_root=data_root, 
            split_csv_path=split_file_path
        )
    else:
        # Loud fallback to random splits
        print(f"\n[WARNING] Split file not found at: {split_file_path}")
        print("[WARNING] Falling back to a random dataset split...\n")
        
        data_directories = [data_root, os.path.join(data_root, 'original_training_set')]
        train_files, val_files, test_files = prepare_data_random(
            data_directories, 
            train_ratio=1.0 - args.val_ratio - args.test_ratio,
            val_ratio=args.val_ratio, 
            test_ratio=args.test_ratio, 
            seed=args.split_seed
        )

    # --- Define Bands ---
    band_options = {
        'rgb': ['B4','B3','B2'],
        'all_spectral': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'],
        'all_surface': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'],
        'lis':['B3','B4','B11'],
        'wang': ['B2','B11','B4','B9'],
        'snow_relevant': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
        'snow_and_clouds': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'cs', 'cs_cdf'],
        'all': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60','cs','cs_cdf'],
        'satlas_ms': ['B2','B3','B4','B5','B6','B7','B8','B11','B12']
    }
    bands = band_options.get(args.bands)
    if bands is None:
        raise ValueError(f"Band selection '{args.bands}' is not a valid predefined option.")
    
    model_hyperparams = {k: v for k, v in vars(args).items() if k in [
        "backbone", "weights", "loss_fn", "loss_weighting", "snow_weight", 
        "snow_thresh", "ignore_value", "lr", "batch_size", "patience", 
        "num_workers", "dice_thresh", "dice_loss_weight"
    ]}
    
    snow_model = SnowModel(
        bands=bands, train_files=train_files, val_files=val_files,
        mask_clouds=args.mask_clouds, use_augmentations=args.use_augmentations,
        train_with_clouds=args.train_with_clouds, hparams=model_hyperparams,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_dice_epoch", mode="max", save_top_k=3,
        dirpath=os.path.join(model_save_dir, 'checkpoints'),
        filename=f'{args.model_name}-{{epoch}}-{{val_dice_epoch:.4f}}',
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_dice_epoch', patience=args.patience, verbose=True, mode='max'
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=1000,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
    )

    print("\n--- Starting model training ---")
    trainer.fit(snow_model, ckpt_path=args.checkpoint_path)

    if trainer.global_rank == 0:
        best_model_path = checkpoint_callback.best_model_path
        final_pt_path = os.path.join(model_save_dir, f"{args.model_name}_final_epoch.pt")
        torch.save(snow_model.state_dict(), final_pt_path)
        
        model_to_test = snow_model
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model checkpoint: {best_model_path}")
            model_to_test = SnowModel.load_from_checkpoint(
                best_model_path, bands=bands, train_files=train_files,
                val_files=val_files, mask_clouds=args.mask_clouds,
                use_augmentations=args.use_augmentations, train_with_clouds=args.train_with_clouds
            )
            torch.save(model_to_test.state_dict(), os.path.join(model_save_dir, f"{args.model_name}_best.pt"))

        test_model(trainer, model_to_test, test_files, bands, args, metrics_save_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
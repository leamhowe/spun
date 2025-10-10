import argparse
import os
import torch

import pytorch_lightning as pl
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore", 
                      category=UserWarning,
                      message="The codec `vlen-utf8` is currently not part")

# Define which model to use

# from src.snowpatch_model import SnowModel
from src.unet_torchgeo_lightning_trainer import SnowModel
# from src.unet_torchgeo_PTwang import SnowModel

# Set matrix multiplication precision to 'high'
torch.set_float32_matmul_precision('high')

def get_args():
    parser = argparse.ArgumentParser(description="Run UNet model training.")
    parser.add_argument('--model_name', type=str, default='unet_torchgeo_default',
                        help='Model name for saving outputs')
    parser.add_argument('--use_augmentations', action='store_true', 
                        help='Flag to enable image augmentations')
    parser.add_argument('--bands', type=str, default='all_spectral_bands',
                        help='Specify the bands to use in the model')
    # Hyperparameters
    # parser.add_argument('--backbone', type=str, default='resnet34',
    #                     help='Backbone model for the UNet')
    # parser.add_argument('--weights', type=str, default='imagenet',
    #                     help='Pretrained weights for the backbone')
    parser.add_argument('--loss_fn', type=str, default='l1',
                        help="Regression loss: 'l1' (default), 'mse', or 'huber'")
    parser.add_argument('--loss_weighting', type=str, default='none',
                        help="Per-pixel weighting: 'none'|'binary'|'linear' to emphasize snow")
    parser.add_argument('--snow_weight', type=float, default=5.0,
                        help='Weight for snow pixels (binary) or 1+alpha (linear)')
    parser.add_argument('--snow_thresh', type=float, default=0.05,
                        help='Snow threshold in [0,1] for binary weighting')
    parser.add_argument('--ignore_value', type=float, default=200.0,
                        help='Ignore value in labels (cloud/pad)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone model for the UNet')
    parser.add_argument('--weights', type=str, default=None,
                        help='Pretrained weights for the backbone')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and validation')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader')
    parser.add_argument('--server', type=str, default='jasmin',
                        help='Are you running on Jasmin or Pentland?')
    parser.add_argument('--mask_fill_value', type=int, default=1, help='Fill value for padding and augmenting the label dataset') 
    parser.add_argument('--image_fill_value', type=int, default=0, help='Fill value for padding and augmenting the image dataset')
    parser.add_argument('--mask_clouds', action='store_true', help='Mask cloud pixels in the images and set them to -100')
    parser.add_argument('--use_indices', action='store_true', help='Use indices in the dataset')
    parser.add_argument('--dataset', type=str, default='cs_threshold_50', help='Dataset to use for training and validation. Default is "cs_threshold_50".')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to a checkpoint file to resume training or load model weights')
    # NEW ARGS
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory containing new SCF training data (default set automatically for server)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test split ratio')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for dataset split')
    parser.add_argument('--split_file', type=str, default=None,
                        help='Path to a CSV file defining train/val/test splits')
    return parser.parse_args()

import pandas as pd
import os

def prepare_data_from_split_file(split_csv_path):
    """
    Collects .zarr filepaths based on a pre-defined split file.

    This function reads a CSV file which contains the definitive train, validation,
    and test splits, ensuring a consistent and spatially-aware data setup.

    Args:
        split_csv_path (str): The path to the cleaned split CSV file 
                              (e.g., 'dataset_splits_cleaned_reassigned.csv').

    Returns:
        tuple: A tuple containing three lists of filepaths:
               (train_files, val_files, test_files).
    """
    # --- 1. Validate and Load the Split File ---
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"The specified split file was not found at: {split_csv_path}")

    print(f"Loading data splits from: {split_csv_path}")
    df = pd.read_csv(split_csv_path)

    # Check for the expected column
    if 'split_cleaned' in df.columns:

        # --- 2. Filter Filepaths for Each Split ---
        # Select rows for each category and convert the 'filepath' column to a list
        train_files = df[df['split_cleaned'] == 'Train']['filepath'].tolist()
        val_files = df[df['split_cleaned'] == 'Validation']['filepath'].tolist()
        test_files = df[df['split_cleaned'] == 'Test']['filepath'].tolist()
    elif 'split' in df.columns:
        # If 'split' column exists, use it instead
        train_files = df[df['split'] == 'Train']['filepath'].tolist()
        val_files = df[df['split'] == 'Validation']['filepath'].tolist()
        test_files = df[df['split'] == 'Test']['filepath'].tolist()
    else:
        raise ValueError(f"CSV file must contain a 'split_cleaned' or 'split' column.")

    
    # The 'Excluded' files are ignored and not returned.
    
    # --- 3. Print Summary and Return ---
    total_files = len(train_files) + len(val_files) + len(test_files)
    print(f"Total files loaded: {total_files} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    if not train_files or not val_files or not test_files:
        print("Warning: One or more of the data splits is empty.")
        
    return train_files, val_files, test_files

def prepare_data_random(directories, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Collect all .zarr files from a list of directories and perform a random split.
    """
    import random
    all_files = []
    for d in directories:
        if not os.path.isdir(d):
            print(f"Warning: directory not found -> {d}")
            continue
        all_files.extend(
            [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.zarr')]
        )

    all_files = sorted(list(set(all_files)))  # deduplicate + stable order
    if len(all_files) == 0:
        raise RuntimeError(f"No .zarr files found in: {directories}")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    random.seed(seed)
    random.shuffle(all_files)

    n = len(all_files)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)
    # Remainder goes to test
    n_test  = n - n_train - n_val

    train_files = all_files[:n_train]
    val_files   = all_files[n_train:n_train + n_val]
    test_files  = all_files[n_train + n_val:]

    print(f"Total files: {n} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    return train_files, val_files, test_files


################ MAIN CODE ##############################################################################################################################################################
import wandb
from pytorch_lightning.loggers import WandbLogger
from src.snow_s2_dataset import Snows2Dataset
from torch.utils.data import DataLoader
import numpy as np

def test_model(trainer, snow_model, test_files, bands, args, metrics_save_dir):
    """Run inference on the test set and save metrics."""
    if not test_files:
        print("Test file list is empty. Skipping inference.")
        return

    print(f"\n--- Running inference on {len(test_files)} test files ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snow_model.eval().to(device)

    test_dataset = Snows2Dataset(
        test_files,
        bands=bands,
        mask_clouds=args.mask_clouds,
        use_indices=args.use_indices,
        target_type='scf'
    )
    # Use a larger batch size for inference if possible, and more workers
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)

    predictions_list = []
    labels_list = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch.get('mask', batch.get('label')).to(device).float()
            outputs = snow_model(images)  # (N,1,H,W) in [0,1]
            predictions = (outputs.squeeze(1) * 100.0)  # percent 0..100
            predictions_list.append(predictions.cpu())
            labels_list.append(labels.cpu())

    if trainer.global_rank == 0:
        if len(predictions_list) > 0:
            predictions = torch.cat(predictions_list, dim=0)
            labels = torch.cat(labels_list, dim=0)

            valid = (labels != args.ignore_value)
            y_true = labels[valid]
            y_pred = predictions[valid]

            if y_true.numel() > 0:
                mae = (y_pred - y_true).abs().mean().item()
                rmse = float(torch.sqrt(((y_pred - y_true) ** 2).mean()).item())
                metrics = {'mae': mae, 'rmse': rmse}
            else:
                metrics = {'mae': np.nan, 'rmse': np.nan}
        else:
            metrics = {'mae': np.nan, 'rmse': np.nan}

        print("Test Metrics:", metrics)
        df = pd.DataFrame([metrics])
        metric_path = os.path.join(metrics_save_dir, f"{args.model_name.split('.')[0]}_test_metrics.csv")
        df.to_csv(metric_path, index=False)
        print(f"Test metrics saved to {metric_path}")


def main():
    args = get_args()

    # Setup Weights & Biases Logger
    wandb_logger = WandbLogger(
        project="spun_scf",
        name=args.model_name,
        config=vars(args),
        save_dir="../wandb/"
    )
    
    # --- Setup Paths ---
    if args.server == 'jasmin':
        default_data_root = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/s2_training_scf'
        model_save_dir = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/models/'
        metrics_save_dir = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/metrics/'
    elif args.server == 'pentland':
        default_data_root = '/home/s2112771/r/1_DETECT/data/s2_training_scf'
        model_save_dir = '/home/s2112771/r/1_DETECT/trained_models/unet/'
        metrics_save_dir = '/home/s2112771/r/1_DETECT/trained_models/unet/metrics/'
    else:
        raise ValueError(f"Server '{args.server}' not recognized. Use 'jasmin' or 'pentland'.")

    os.makedirs(os.path.join(model_save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(metrics_save_dir, exist_ok=True)

    # --- Prepare Data ---
    if args.split_file:
        # Use the definitive split file to prepare data
        train_files, val_files, test_files = prepare_data_from_split_file(
            split_csv_path=args.split_file
        )
    else:
        # Perform a random split
        print("No split file provided. Performing random split.")
        data_root = args.data_root or default_data_root
        data_directories = [
            data_root,
            # os.path.join(data_root, 'original_training_set')
            os.path.join(data_root, 'original_training_set_snow_urban_water')
        ]
        train_files, val_files, test_files = prepare_data_random(
            data_directories,
            train_ratio=1.0 - args.val_ratio - args.test_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed
        )

    # --- Define Bands ---
    band_options = {
        'all_bands': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60'],
        'all_spectral_bands': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'],
        'wang_bands': ['B2','B11','B4','B9'],
        'perm_threshold_01': ['B2', 'B3', 'B4', 'B5', 'B8A', 'B11', 'B12'],
        'custom_bands': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60', 'NDSI', 'cs_cdf'],
        'all_spectral': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'],
        'lis': ['B3','B4','B11'],
        'wang': ['B2','B11','B4','B9'],
        'snow_relevant': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
        'all': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60', 'cs_cdf']
    }
    bands = band_options.get(args.bands, [])
    if not bands:
        raise ValueError(f"Band selection '{args.bands}' is not a valid option.")
    
    # --- Configure Model ---
    model_hyperparams = {
        "backbone": args.backbone,
        "weights": args.weights,
        "loss_fn": args.loss_fn,
        "loss_weighting": args.loss_weighting,
        "snow_weight": args.snow_weight,
        "snow_thresh": args.snow_thresh,
        "ignore_value": args.ignore_value,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "num_workers": args.num_workers,
    }
    
    snow_model = SnowModel(
        bands=bands,
        train_files=train_files,
        val_files=val_files,
        mask_clouds=args.mask_clouds,
        use_augmentations=args.use_augmentations,
        use_indices=args.use_indices,
        hparams=model_hyperparams,
    )

    # --- Setup Callbacks ---
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_mae",
        mode="min",
        dirpath=os.path.join(model_save_dir, 'checkpoints'),
        filename=f'{args.model_name}-{{epoch}}-{{val_mae:.4f}}',
        save_top_k=3,
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_mae',
        patience=args.patience,
        verbose=True,
        mode='min'
    )

    # --- Train Model ---
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=1000,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=wandb_logger,
        # strategy="ddp_find_unused_parameters_true" # Use if you get DDP errors
    )

    print("\n--- Starting model training ---")
    trainer.fit(snow_model, ckpt_path=args.checkpoint_path)
    print("--- Model training finished ---")

    # --- Save Final Model ---
    final_ckpt_path = os.path.join(model_save_dir, 'checkpoints', f"{args.model_name}_final.ckpt")
    final_pt_path = os.path.join(model_save_dir, f"{args.model_name}_final.pt")
    trainer.save_checkpoint(final_ckpt_path)
    torch.save(snow_model.state_dict(), final_pt_path)
    print(f"Final model checkpoint saved to: {final_ckpt_path}")
    print(f"Final model state dict saved to: {final_pt_path}")

    # --- Run Inference on Test Data ---
    # Load the best model for testing
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"\nLoading best model for testing from: {best_model_path}")
        # The model is already loaded in snow_model if training just finished,
        # but for clarity/robustness we can load from checkpoint.
        snow_model = SnowModel.load_from_checkpoint(best_model_path)
    else:
        print("\nNo best model checkpoint found. Using the final trained model for testing.")

    test_model(trainer, snow_model, test_files, bands, args, metrics_save_dir)

    wandb.finish()


if __name__ == "__main__":
    main()


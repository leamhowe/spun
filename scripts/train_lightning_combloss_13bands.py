import argparse
import os
import torch
import pandas as pd
import numpy as np
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        message="The codec `vlen-utf8` is currently not part")

# Import your NEW MERGED SnowModel class
# Make sure you save the file above as 'unet_torchgeo_combloss_trainer.py'
# or update this import path to match.
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel 

# Import your dataset class
from src.snow_s2_dataset_13bands import Snows2Dataset

# Set matrix multiplication precision to 'high'
torch.set_float32_matmul_precision('high')

def get_args():
    parser = argparse.ArgumentParser(description="Run UNet model training.")
    parser.add_argument('--model_name', type=str, default='unet_torchgeo_combloss',
                        help='Model name for saving outputs')
    parser.add_argument('--use_augmentations', action='store_true', 
                        help='Flag to enable image augmentations')
    parser.add_argument('--bands', type=str, default='all_spectral_bands',
                        help='Specify the bands to use in the model')
    # Hyperparameters
    parser.add_argument('--loss_fn', type=str, default='l1',
                        help="Regression loss: 'l1' (default), 'mse', or 'huber'")
    parser.add_argument('--loss_weighting', type=str, default='none',
                        help="Per-pixel weighting: 'none'|'binary'|'linear' | 'inverse_linear' to emphasize snow")
    parser.add_argument('--snow_weight', type=float, default=5.0,
                        help='Weight for snow pixels (binary) or 1+alpha (linear)')
    parser.add_argument('--snow_thresh', type=float, default=0.05,
                        help='Snow threshold in [0,1] for binary weighting')
    
    # --- UPDATED/ADDED ARGS ---
    parser.add_argument('--ignore_value', type=int, default=200,
                        help='Ignore value in labels (cloud/pad)')
    parser.add_argument('--train_with_clouds', action='store_true', 
                        help='Include clouds in training (relabel as 0% SCF).')
    # --- END UPDATES ---
    
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone model for the UNet')
    parser.add_argument('--weights', type=str, default=None,
                        help='Pretrained weights for the backbone (e.g., sentinel2_all_moco)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--dice_thresh', type=float, default=0.5,
                        help='Binarization threshold [0,1] for calculating Dice score.')
    parser.add_argument('--dice_loss_weight', type=float, default=0.5,
                        help='Weight lambda for Dice loss in combined loss. L_total = (1-L) * L_reg + L * L_dice.')
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

def prepare_data_from_split_file(split_csv_path):
    """
    Collects .zarr filepaths based on a pre-defined split file.
    """
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"The specified split file was not found at: {split_csv_path}")

    print(f"Loading data splits from: {split_csv_path}")
    df = pd.read_csv(split_csv_path)

    split_col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    if split_col not in df.columns:
        raise ValueError("CSV file must contain a 'split_cleaned' or 'split' column.")

    train_files = df[df[split_col] == 'Train']['filepath'].tolist()
    val_files = df[df[split_col] == 'Validation']['filepath'].tolist()
    test_files = df[df[split_col] == 'Test']['filepath'].tolist()
    
    total_files = len(train_files) + len(val_files) + len(test_files)
    print(f"Total files loaded: {total_files} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    if not train_files or not val_files:
        print("Warning: One or more of the train/validation data splits is empty.")
        
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

    all_files = sorted(list(set(all_files)))
    if len(all_files) == 0:
        raise RuntimeError(f"No .zarr files found in: {directories}")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    random.seed(seed)
    random.shuffle(all_files)

    n = len(all_files)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)

    train_files = all_files[:n_train]
    val_files   = all_files[n_train:n_train + n_val]
    test_files  = all_files[n_train + n_val:]

    print(f"Total files: {n} | Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    return train_files, val_files, test_files


################ MAIN CODE ##############################################################################

def test_model(trainer, snow_model, test_files, bands, args, metrics_save_dir):
    """Run inference on the test set and save metrics by reusing model's internal logic."""
    if not test_files:
        print("Test file list is empty. Skipping inference.")
        return

    print(f"\n--- Running inference on {len(test_files)} test files ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snow_model.eval().to(device)

    # Test dataset should NOT train with clouds, so we don't pass train_with_clouds
    test_dataset = Snows2Dataset(
        test_files,
        bands=bands,
        mask_clouds=args.mask_clouds,
        # use_indices=args.use_indices,
        target_type='scf',
        train_with_clouds=args.train_with_clouds 
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)

    # Use the same accumulator structure as in the model
    test_acc = snow_model._create_accumulator_dict()

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels_raw = batch.get('mask', batch.get('label')).to(device).float()
            outputs = snow_model(images)
            
            # --- REUSE THE MODEL'S METRIC CALCULATION LOGIC ---
            # This will now return loss_reg, loss_dice, etc.
            metrics = snow_model._calculate_metrics(outputs, labels_raw)
            
            # Update accumulators using the model's own helper function
            snow_model._update_accumulators(test_acc, outputs, labels_raw, metrics)

    if trainer.global_rank == 0:
        # --- COMPUTE FINAL METRICS FROM ACCUMULATORS ---
        final_metrics = {}
        n = test_acc.get('n', 0)
        if n > 0:
            final_metrics['mae'] = test_acc['abs_err'] / n
            final_metrics['rmse'] = (test_acc['sq_err'] / n)**0.5
        else:
            final_metrics['mae'], final_metrics['rmse'] = np.nan, np.nan

        n_snow = test_acc.get('n_snow', 0)
        if n_snow > 0:
            final_metrics['mae_snow'] = test_acc['abs_err_snow'] / n_snow
            final_metrics['rmse_snow'] = (test_acc['sq_err_snow'] / n_snow)**0.5
        else:
            final_metrics['mae_snow'], final_metrics['rmse_snow'] = np.nan, np.nan
        
        inter = test_acc.get('dice_inter', 0)
        union = test_acc.get('dice_union', 0)
        if union > 0:
            # Use the model's smoothing constant for consistency
            dice = (2. * inter + snow_model.dice_smooth) / (union + snow_model.dice_smooth)
            final_metrics['dice'] = dice
        else:
            final_metrics['dice'] = np.nan

        print("Test Metrics:", final_metrics)
        df = pd.DataFrame([final_metrics])
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
        train_files, val_files, test_files = prepare_data_from_split_file(split_csv_path=args.split_file)
    else:
        print("No split file provided. Performing random split.")
        data_root = args.data_root or default_data_root
        data_directories = [
            data_root,
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
    
    # --- Configure Model ---
    # This hparams dict now contains all args for both features
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
        "dice_thresh": args.dice_thresh, 
        "dice_loss_weight": args.dice_loss_weight
    }
    
    # --- UPDATED MODEL INSTANTIATION ---
    snow_model = SnowModel(
        bands=bands,
        train_files=train_files,
        val_files=val_files,
        mask_clouds=args.mask_clouds,
        use_augmentations=args.use_augmentations,
        # use_indices=args.use_indices,
        train_with_clouds=args.train_with_clouds, # <-- ADDED THIS
        hparams=model_hyperparams,
    )
    # --- END UPDATE ---

    # --- Setup Callbacks (Updated to monitor Dice score) ---
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_dice_epoch", # This is correct for both models
        mode="max",
        dirpath=os.path.join(model_save_dir, 'checkpoints'),
        filename=f'{args.model_name}-{{epoch}}-{{val_dice_epoch:.4f}}',
        save_top_k=3,
    )

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_dice_epoch', # This is correct for both models
        patience=args.patience,
        verbose=True,
        mode='max'
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

    if trainer.global_rank == 0:
        best_model_path = checkpoint_callback.best_model_path
        
        # Save final model state dict (This is the *last* epoch, which is fine to save)
        final_pt_path = os.path.join(model_save_dir, f"{args.model_name}_final_epoch.pt")
        torch.save(snow_model.state_dict(), final_pt_path)
        print(f"Final (last epoch) model state dict saved to: {final_pt_path}")

        # Default to the model in memory (last epoch) if no best_model_path is found
        model_to_test = snow_model
        
        if best_model_path and os.path.exists(best_model_path):
            print(f"\nBest model checkpoint was saved to: {best_model_path}")
            print("Loading best model checkpoint for testing...")
            # --- Load the BEST model from the checkpoint file ---
            model_to_test = SnowModel.load_from_checkpoint(
                best_model_path,
                # You must pass all __init__ args again that aren't in hparams
                bands=bands,
                train_files=train_files,
                val_files=val_files,
                mask_clouds=args.mask_clouds,
                use_augmentations=args.use_augmentations,
                # use_indices=args.use_indices,
                train_with_clouds=args.train_with_clouds
                # hparams will be loaded from the checkpoint
            )
            
            # --- Save the BEST model's state dict separately ---
            best_pt_path = os.path.join(model_save_dir, f"{args.model_name}_best.pt")
            torch.save(model_to_test.state_dict(), best_pt_path)
            print(f"Best model state dict saved to: {best_pt_path}")

        else:
            print("\nNo best model checkpoint was found. Testing with final (last epoch) model.")

        # --- Pass the potentially reloaded BEST model to test_model ---
        test_model(trainer, model_to_test, test_files, bands, args, metrics_save_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
import argparse
import logging
import os
import torch
import pandas as pd
import pytorch_lightning as pl
import optuna
# Import TrialPruned
from optuna.exceptions import TrialPruned
from optuna.integration import PyTorchLightningPruningCallback
from optuna.storages import JournalStorage, JournalFileStorage
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Import the combined-loss model
from src.unet_torchgeo_combloss_lightning_trainer import SnowModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set matrix multiplication precision to 'high'
torch.set_float32_matmul_precision('high')

# WORKAROUND for version incompatibility
class PyTorchLightningPruningCallbackWithFix(PyTorchLightningPruningCallback, Callback):
    pass

# Helper function to load data splits
def prepare_data_from_split_file(split_csv_path):
    """
    Collects .zarr filepaths based on a pre-defined split file.
    """
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"The specified split file was not found at: {split_csv_path}")
    logging.info(f"Loading data splits from: {split_csv_path}")
    df = pd.read_csv(split_csv_path)
    split_col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    if split_col not in df.columns:
        raise ValueError("CSV file must contain a 'split_cleaned' or 'split' column.")
    train_files = df[df[split_col] == 'Train']['filepath'].tolist()
    val_files = df[df[split_col] == 'Validation']['filepath'].tolist()
    total_files = len(train_files) + len(val_files)
    logging.info(f"Total files loaded for tuning: {total_files} | Train: {len(train_files)} | Val: {len(val_files)}")
    if not train_files or not val_files:
        raise ValueError("Training or validation data split is empty. Check split file.")
    return train_files, val_files

# Helper function to get band lists
def get_band_list(selection_name):
    """
    Returns a list of bands based on a selection name.
    """
    band_options = {
        'all_spectral': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'], # 13 bands
        'all_surface': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'], # 12 bands
        'lis':['B3','B4','B11'], # 3 bands
        'wang': ['B2','B11','B4','B9'], # 4 bands
        'snow_relevant': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'], # 6 bands
        'snow_and_clouds': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'cs', 'cs_cdf'], # 8 bands
        'all': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60','cs','cs_cdf'], # 18 bands (potentially fewer if QA/cs not present)
        'satlas_ms': ['B2','B3','B4','B5','B6','B7','B8','B11','B12'] # 9 bands
    }
    selection = band_options.get(selection_name)
    if selection is None:
        raise ValueError(f"Band selection '{selection_name}' not recognized.")
    return selection

def objective(trial, args, train_files, val_files):
    """
    Objective function suggesting parameters independently then checking compatibility.
    """
    # --- 1. Suggest hyperparameters independently ---

    # --- Suggest backbone from FULL list ---
    backbone_value = trial.suggest_categorical("backbone", ["resnet50"])

    # --- Suggest weights from FULL list ---
    weights_value = trial.suggest_categorical("weights",
        ["imagenet", None,
         "sentinel2_all_moco",
         "sentinel2_all_dino", "sentinel2_all_decur", "sentinel2_softcon",
         ])

    # --- Suggest bands from FULL list ---
    band_selection = trial.suggest_categorical("band_selection", ["all_spectral"])
    bands = get_band_list(band_selection)

    # --- Suggest other parameters ---
    train_with_clouds = args.train_with_clouds # Fixed value from args
    loss_weighting = trial.suggest_categorical("loss_weighting", ["none", "binary", "linear", "inverse_linear"])
    if loss_weighting != "none":
        snow_weight = trial.suggest_float("snow_weight", 1.5, 20.0, log=True)
    else:
        snow_weight = 1.0

    hparams = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "backbone": backbone_value, # Use suggested value
        "weights": weights_value,   # Use suggested value
        "loss_fn": trial.suggest_categorical("loss_fn", ["l1", "mse", "huber"]),
        "num_workers": int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
        "patience": args.patience,
        "loss_weighting": loss_weighting,
        "snow_weight": snow_weight,
        "snow_thresh": trial.suggest_float("snow_thresh", 0.01, 0.5),
        "dice_thresh": 0.5,
        "dice_loss_weight": trial.suggest_float("dice_loss_weight", 0.1, 0.9),
    }


    # --- (Rest of the objective function: logging, model init, trainer, fit, return values) ---
    logging.info(f"--- Starting Trial {trial.number} ---")
    log_params = hparams.copy()
    log_params["use_augmentations"] = args.use_augmentations
    log_params["use_indices"] = args.use_indices # Log fixed value (False)
    log_params["train_with_clouds"] = train_with_clouds # Log fixed value (True)
    log_params["band_selection"] = band_selection
    logging.info(f"  Params: {log_params}")
    logging.info(f"  Bands: {band_selection} ({len(bands)} bands)")

    try:
        model = SnowModel(
            bands=bands, train_files=train_files, val_files=val_files,
            mask_clouds=args.mask_clouds, use_augmentations=args.use_augmentations,
            use_indices=args.use_indices, train_with_clouds=train_with_clouds,
            hparams=hparams
        )
        log_dir = os.path.join(args.output_dir, "logs")
        monitor_metric_stop = 'val_rmse_snow_epoch'
        early_stop_callback = EarlyStopping(monitor=monitor_metric_stop, patience=hparams['patience'], verbose=False, mode='min')
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1, max_epochs=args.epochs,
            logger=TensorBoardLogger(save_dir=log_dir, name=args.study_name, version=f"trial_{trial.number}"),
            callbacks=[early_stop_callback],
            enable_checkpointing=False, enable_progress_bar=False,
        )
        trainer.fit(model)
        final_rmse_snow = trainer.callback_metrics.get(monitor_metric_stop, float('inf'))
        final_dice = trainer.callback_metrics.get('val_dice_epoch', 0.0)
    except TrialPruned as e:
        raise e
    except Exception as e:
        logging.warning(f"Trial {trial.number} failed unexpectedly with error: {e}")
        final_rmse_snow = float('inf'); final_dice = 0.0
    if 'model' in locals(): del model
    if 'trainer' in locals(): del trainer
    torch.cuda.empty_cache()
    return final_rmse_snow, final_dice


def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for SCF U-Net with Combined Loss')
    parser.add_argument('--trials', type=int, default=100, help='Total number of trials goal for study (used for info log)')
    parser.add_argument('--output-dir', type=str, default='/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/combloss_tuning', help='Directory to save tuning results')
    parser.add_argument('--study-name', type=str, default='scf-unet-combloss-tuning-rmse-dice-clouds-cond', help='Name for the Optuna study') # Added '-cond'
    parser.add_argument('--split_file', type=str, required=True, help='Path to a CSV file defining train/val/test splits')
    parser.add_argument('--epochs', type=int, default=75, help='Max epochs per trial') # Increased epochs
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping') # Increased patience
    parser.add_argument('--server', type=str, default='jasmin', help='Running on "jasmin" or "pentland"?')

    # --- Fixed arguments for the study ---
    parser.add_argument('--mask_clouds', action='store_true', help='Mask clouds in the image features (fixed for study)')
    parser.add_argument('--use_augmentations', action='store_true', help='Enable image augmentations (fixed for study)')
    # --- ENSURE use_indices is NOT set if using conditional band logic ---
    parser.add_argument('--use_indices', action='store_false', dest='use_indices', help='Disable indices (fixed for study - DO NOT OVERRIDE)') # Default False
    parser.set_defaults(use_indices=False) # Ensure default is False
    parser.add_argument('--train_with_clouds', action='store_true', dest='train_with_clouds', help='Enable training with clouds (fixed for study)') # Default True
    parser.set_defaults(train_with_clouds=True) # Ensure default is True

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # --- Argument Check for Conditional Logic Assumption ---
    if args.use_indices:
        raise ValueError("This script uses conditional band suggestion assuming --use_indices is FALSE. Do not enable --use_indices with this script version.")
    if not args.train_with_clouds:
         logging.warning("This script fixes train_with_clouds=True in the objective function. The command line flag --no-train_with_clouds will be ignored during tuning.")
         # Force it True here just in case objective is ever bypassed (unlikely)
         args.train_with_clouds = True

    logging.info("Preparing data split from file...")
    train_files, val_files = prepare_data_from_split_file(args.split_file)
    logging.info(f"Data split: {len(train_files)} training files, {len(val_files)} validation files.")
    logging.info(f"Running study with fixed arguments: use_augmentations={args.use_augmentations}, use_indices={args.use_indices}, mask_clouds={args.mask_clouds}, train_with_clouds={args.train_with_clouds}")


    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f'{args.study_name}.log')
    # --- UPDATED Storage to use newer backend class ---
    # storage = JournalStorage(optuna.storages.JournalFileBackend(log_path))
    storage = JournalStorage(optuna.storages.journal.JournalFileBackend(log_path))
    # storage = JournalStorage(JournalFileStorage(log_path)) # Old deprecated way
    logging.info(f"Using JournalStorage: {log_path}")

    # Create study (No changes needed here)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["minimize", "maximize"], # Corresponds to (rmse_snow, dice)
        pruner=optuna.pruners.NopPruner(),  # Disable pruning for multi-objective safety
        load_if_exists=True
    )

    # --- Run Optimization (Using n_trials=1 for SLURM array) ---
    logging.info(f"Starting one Optuna trial for study '{args.study_name}' (Task ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}). Aiming for {args.trials} total trials.")
    study.optimize(
        lambda trial: objective(trial, args, train_files, val_files),
        n_trials=1 # Run one trial per SLURM task
    )
    logging.info(f"--- Optuna trial complete for study '{args.study_name}' (Task ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}) ---")


    # --- Reporting Best (Optional in worker script) ---
    try:
        current_trials = study.trials # Get fresh list
        completed_count = len([t for t in current_trials if t.state == optuna.trial.TrialState.COMPLETE])
        logging.info(f"Study '{args.study_name}' now has {completed_count} completed trials.")
        if study.best_trials:
             logging.info(f"Current Pareto front size after this trial: {len(study.best_trials)}")
    except Exception as e:
        logging.warning(f"Could not report best trials: {e}")
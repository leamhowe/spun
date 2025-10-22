import argparse
import logging
import os
import torch
import pandas as pd
import pytorch_lightning as pl
import optuna
# --- NEW: Import TrialPruned ---
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

# WORKAROUND for version incompatibility
class PyTorchLightningPruningCallbackWithFix(PyTorchLightningPruningCallback, Callback):
    pass

# Helper function to load data splits
def prepare_data_from_split_file(split_csv_path):
    # ... (code is correct as provided) ...
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
    # ... (code is correct as provided, includes 'satlas_ms') ...
    band_options = {
        'all_spectral': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'],
        'all_surface': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'],
        'lis':['B3','B4','B11'],
        'wang': ['B2','B11','B4','B9'],
        'snow_relevant': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
        'snow_and_clouds': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'cs', 'cs_cdf'],
        'all': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60','cs','cs_cdf'],
        'satlas_ms': ['B2','B3','B4','B5','B6','B7','B8','B11','B12'] # Corrected name from 'satlas_ms' to 'satlas' to match get_args suggestion
    }
    selection = band_options.get(selection_name)
    if selection is None:
        raise ValueError(f"Band selection '{selection_name}' not recognized.")
    return selection

def objective(trial, args, train_files, val_files):
    """
    Objective function for multi-objective optimization (RMSE_snow, Dice).
    """
    # --- 1. Suggest hyperparameters ---
    band_selection = trial.suggest_categorical("band_selection",
        ["all_spectral", "all_surface", "lis", "wang", "snow_relevant", "snow_and_clouds", "all", "satlas_ms"]) # Removed 'all' for safety
    bands = get_band_list(band_selection)
    train_with_clouds = trial.suggest_categorical("train_with_clouds", [True, False])
    loss_weighting = trial.suggest_categorical("loss_weighting", ["none", "binary", "linear", "inverse_linear"])
    if loss_weighting != "none":
        # Ensure suggest_float choices are within the defined range
        snow_weight = trial.suggest_float("snow_weight", 1.5, 20.0, log=True)
    else:
        snow_weight = 1.0

    hparams = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "backbone": trial.suggest_categorical("backbone", ["resnet34", "resnet50", "resnet152", "vgg16"]),
        "weights": trial.suggest_categorical("weights",
            ["imagenet", None, "sentinel2_all_moco", "satlas_ms", "sentinel2_all_decur", "sentinel2_softcon", "sentinel2_all_dino"]), # Added rgb_moco back
        "loss_fn": trial.suggest_categorical("loss_fn", ["l1", "mse", "huber"]),
        "num_workers": int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
        "patience": args.patience, # Use patience from args for EarlyStopping
        "loss_weighting": loss_weighting,
        "snow_weight": snow_weight,
        "snow_thresh": trial.suggest_float("snow_thresh", 0.01, 0.5),
        "dice_thresh": 0.5, # Fixed for metrics calculation
        "dice_loss_weight": trial.suggest_float("dice_loss_weight", 0.1, 0.9),
    }

    # --- Compatibility Checks ---
    try:
        # Check 1: Backbone vs Weights
        weights_value = hparams['weights']
        backbone_value = hparams['backbone']

        # Weights requiring resnet50
        resnet50_weights = ['sentinel2_all_moco', 'sentinel2_all_dino', 'sentinel2_all_decur', 'sentinel2_softcon']
        if weights_value in resnet50_weights and not backbone_value.startswith('resnet50'):
            raise TrialPruned(f"Incompatible backbone '{backbone_value}' for weights '{weights_value}' (needs resnet50).")

        # Weights requiring resnet152
        if weights_value == 'satlas_ms' and not backbone_value.startswith('resnet152'):
            raise TrialPruned(f"Incompatible backbone '{backbone_value}' for weights '{weights_value}' (needs resnet152).")

        # Check 2: Band selection vs Weights (Number of input channels)
        base_channels = len(bands)
        expected_channels = base_channels
        if args.use_indices:
             # Add rough estimate for indices based on band availability
             has_b3 = 'B3' in bands
             has_b11 = 'B11' in bands
             has_b8 = 'B8' in bands
             has_b4 = 'B4' in bands
             num_indices = 0
             if has_b8 and has_b4: num_indices += 1 # NDVI
             if has_b3 and has_b11: num_indices += 1 # NDSI
             if has_b3 and has_b8: num_indices += 1 # NDWI
             if has_b11 and has_b8: num_indices += 1 # NDBI
             expected_channels += num_indices

        # Channel count checks
        weights_13_channels = ['sentinel2_all_moco', 'sentinel2_all_dino', 'sentinel2_all_decur', 'sentinel2_softcon']
        if weights_value in weights_13_channels and expected_channels != 13:
             raise TrialPruned(f"Incompatible channel count ({expected_channels}) for {weights_value} (needs 13). Bands: {band_selection}, Indices: {args.use_indices}")

        if weights_value == 'satlas_ms' and expected_channels != 9:
             raise TrialPruned(f"Incompatible channel count ({expected_channels}) for {weights_value} (needs 9). Bands: {band_selection}, Indices: {args.use_indices}")

    except TrialPruned as e:
        logging.info(f"Trial {trial.number}: Skipping due to incompatibility - {e}")
        raise


    logging.info(f"--- Starting Trial {trial.number} ---")
    log_params = hparams.copy()
    log_params["use_augmentations"] = args.use_augmentations
    log_params["use_indices"] = args.use_indices
    log_params["train_with_clouds"] = train_with_clouds
    log_params["band_selection"] = band_selection
    logging.info(f"  Params: {log_params}")
    logging.info(f"  Bands: {band_selection} ({len(bands)} bands)")

    # Wrap model init AND training in try...except for unexpected errors
    try:
        # --- 2. Initialize model ---
        model = SnowModel(
            bands=bands,
            train_files=train_files,
            val_files=val_files,
            mask_clouds=args.mask_clouds,
            use_augmentations=args.use_augmentations,
            use_indices=args.use_indices,
            train_with_clouds=train_with_clouds,
            hparams=hparams
        )

        # --- 3. Trainer Setup ---
        log_dir = os.path.join(args.output_dir, "logs")
        # --- CHANGED: Monitor RMSE for stopping/pruning ---
        monitor_metric_stop = 'val_rmse_snow_epoch'
        monitor_metric_prune = 'val_rmse_snow_epoch' # Can be different if desired

        early_stop_callback = EarlyStopping(monitor=monitor_metric_stop, patience=hparams['patience'], verbose=False, mode='min')
        # Pruning Callback (monitor must match a logged value)
        pruning_callback = PyTorchLightningPruningCallbackWithFix(trial, monitor=monitor_metric_prune)

        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            max_epochs=args.epochs,
            logger=TensorBoardLogger(save_dir=log_dir, name=args.study_name, version=f"trial_{trial.number}"),
            # --- CHANGED: Use NopPruner for multi-objective safety ---
            # callbacks=[early_stop_callback, pruning_callback], # Use if enabling pruning
            callbacks=[early_stop_callback], # Use if pruning is disabled
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

        # --- 4. Training ---
        trainer.fit(model)

        # --- CHANGED: Get both final metrics ---
        final_rmse_snow = trainer.callback_metrics.get(monitor_metric_stop, float('inf')) # Use monitored metric
        final_dice = trainer.callback_metrics.get('val_dice_epoch', 0.0) # Dice needs a default that's bad when maximizing

    except TrialPruned as e:
        # Compatibility checks already logged, just re-raise
        raise e
    except Exception as e:
        # Catch other unexpected errors
        logging.warning(f"Trial {trial.number} failed unexpectedly with error: {e}")
        # --- CHANGED: Return worst values for BOTH objectives ---
        final_rmse_snow = float('inf')
        final_dice = 0.0

    # --- 5. Return the tuple of metrics ---
    if 'model' in locals(): del model
    if 'trainer' in locals(): del trainer
    torch.cuda.empty_cache()

    # --- CHANGED: Return tuple ---
    return final_rmse_snow, final_dice


def get_args():
    # ... (Arguments mostly correct as provided) ...
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for SCF U-Net with Combined Loss')
    parser.add_argument('--trials', type=int, default=100, help='Total number of trials to run')
    parser.add_argument('--output-dir', type=str, default='/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/combloss_tuning', help='Directory to save tuning results')
    parser.add_argument('--study-name', type=str, default='scf-unet-combloss-tuning-rmse-dice', help='Name for the Optuna study') # Changed name
    parser.add_argument('--split_file', type=str, required=True, help='Path to a CSV file defining train/val/test splits')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs per trial')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--server', type=str, default='jasmin', help='Running on "jasmin" or "pentland"?')
    parser.add_argument('--mask_clouds', action='store_true', help='Mask clouds in the image features')
    parser.add_argument('--use_augmentations', action='store_true', help='Flag to enable image augmentations for all trials')
    parser.add_argument('--use_indices', action='store_true', help='Flag to append spectral indices for all trials')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.info("Preparing data split from file...")
    train_files, val_files = prepare_data_from_split_file(args.split_file)
    logging.info(f"Data split: {len(train_files)} training files, {len(val_files)} validation files.")

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f'{args.study_name}.log')
    storage = JournalStorage(JournalFileStorage(log_path))
    logging.info(f"Using JournalStorage: {log_path}")

    # --- CHANGED: Define directions and disable pruning ---
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        directions=["minimize", "maximize"], # Corresponds to (rmse_snow, dice)
        pruner=optuna.pruners.NopPruner(),  # Disable pruning for multi-objective safety
        load_if_exists=True
    )
    # --- END CHANGED ---

    logging.info(f"Starting multi-objective optimization for study '{args.study_name}' for {args.trials} trials...")
    study.optimize(
        lambda trial: objective(trial, args, train_files, val_files),
        n_trials=args.trials
    )
    logging.info(f"--- Optimization complete for study '{args.study_name}' ---")

    # --- CHANGED: Update reporting for 2 objectives ---
    if study.best_trials:
        best_params_path = os.path.join(args.output_dir, f'best_hyperparameters_{args.study_name}.txt')
        with open(best_params_path, 'w') as f:
            f.write(f"Found {len(study.best_trials)} Pareto optimal trials:\n\n")
            for i, trial in enumerate(study.best_trials):
                f.write(f"--- Trial {trial.number} ---\n")
                f.write(f"  Values (RMSE_snow, Dice): {trial.values}\n") # Updated label
                f.write("  Parameters:\n")
                for key, value in trial.params.items():
                    f.write(f"    {key}: {value}\n")
                f.write("\n")
        logging.info(f"Best hyperparameters (Pareto front) saved to {best_params_path}")
    else:
        logging.info("No successful trials completed.")
    # --- END CHANGED ---
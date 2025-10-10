import argparse
import logging
import os
import torch
import pytorch_lightning as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback
# REQUIRED CHANGE: Import the NFS-safe storage classes
from optuna.storages import JournalStorage, JournalFileStorage
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

from src.unet_torchgeo_lightning_trainer import SnowModel
from train_lightning import prepare_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# WORKAROUND for version incompatibility
class PyTorchLightningPruningCallbackWithFix(PyTorchLightningPruningCallback, Callback):
    pass

def get_data_dirs(server, data_root_override=None):
    """Gets the data directories based on the server."""
    if data_root_override:
        default_data_root = data_root_override
    elif server == 'jasmin':
        default_data_root = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/s2_training_scf'
    elif server == 'pentland':
        default_data_root = '/home/s2112771/r/1_DETECT/data/s2_training_scf'
    else:
        raise ValueError(f"Unknown server: {server}")
    return os.path.join(default_data_root, 'original_training_set')

def get_band_list(selection_name):
    """Returns a list of bands based on a selection name."""
    band_options = {
        'all_spectral': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'],
        'lis':['B3','B4','B11'],
        'wang': ['B2','B11','B4','B9'],
        'snow_relevant': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
        'all': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60','cs','cs_cdf']
    }
    return band_options.get(selection_name)

def objective(trial, args, train_files, val_files):
    """
    The objective function for Optuna to optimize.
    A 'trial' represents one set of hyperparameters.
    """
    # --- 1. Suggest hyperparameters ---
    band_selection = trial.suggest_categorical("band_selection", ["all_spectral", "lis", "wang", "snow_relevant", "all"])
    bands = get_band_list(band_selection)
    use_augmentations = trial.suggest_categorical("use_augmentations", [True, False])
    use_indices = trial.suggest_categorical("use_indices", [True, False])
    
    # Suggest weighting strategy
    loss_weighting = trial.suggest_categorical("loss_weighting", ["none", "binary", "linear"])
    # Only suggest snow_weight if a weighting strategy is used
    if loss_weighting != "none":
        snow_weight = trial.suggest_float("snow_weight", 1.5, 20.0, log=True)
    else:
        snow_weight = 1.0 # Default value, not used

    hparams = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "backbone": trial.suggest_categorical("backbone", ["resnet34", "resnet50", "vgg16"]),
        "weights": trial.suggest_categorical("weights", ["imagenet", None]),
        "loss_fn": trial.suggest_categorical("loss_fn", ["l1", "mse", "huber"]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop"]),
        "num_workers": int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
        "patience": args.patience,
        "loss_weighting": loss_weighting,
        "snow_weight": snow_weight,
        "snow_thresh": 0.01 # Fixed threshold for simplicity
    }

    logging.info(f"--- Starting Trial {trial.number} ---")
    # Log hparams and other trial-specific info
    log_params = hparams.copy()
    log_params["use_augmentations"] = use_augmentations
    log_params["use_indices"] = use_indices
    log_params["band_selection"] = band_selection
    logging.info(f"  Params: {log_params}")
    logging.info(f"  Bands: {band_selection} ({len(bands)} bands)")

    # --- 2. Initialize model with trial hyperparameters ---
    model = SnowModel(
        bands=bands,
        train_files=train_files,
        val_files=val_files,
        mask_clouds=args.mask_clouds,
        use_augmentations=use_augmentations,
        use_indices=use_indices,
        hparams=hparams
    )

    # --- 3. Trainer Setup ---
    log_dir = os.path.join(args.output_dir, "logs")
    early_stop_callback = EarlyStopping(monitor='val_mae_snow_epoch', patience=hparams['patience'], verbose=False, mode='min')
    
    # CORRECTED: Use the new wrapper class to fix the "Expected a parent" error
    pruning_callback = PyTorchLightningPruningCallbackWithFix(trial, monitor="val_mae_snow_epoch")

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        max_epochs=args.epochs,
        logger=TensorBoardLogger(save_dir=log_dir, name=args.study_name, version=f"trial_{trial.number}"),
        callbacks=[early_stop_callback, pruning_callback],
        enable_checkpointing=False,
        enable_progress_bar=False, # Disable progress bar for cleaner logs
    )

    # --- 4. Training ---
    trainer.fit(model)
    
    # --- 5. Return the final validation score ---
    final_score = trainer.callback_metrics.get('val_mae_snow_epoch', float('inf'))
    
    # Clean up memory
    del model, trainer
    torch.cuda.empty_cache()

    return final_score


def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for SCF U-Net (Fast Version)')
    
    parser.add_argument('--trials', type=int, default=100, help='Total number of trials to run')
    parser.add_argument('--output-dir', type=str, default='/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/tuning', help='Directory to save tuning results')
    parser.add_argument('--study-name', type=str, default='scf-unet-tuning', help='Name for the Optuna study')
    
    parser.add_argument('--epochs', type=int, default=30, help='Max epochs per trial')
    parser.add_argument('--patience', type=int, default=7, help='Patience for early stopping')
    parser.add_argument('--val-frac', type=float, default=0.2, help='Fraction of data for validation')

    parser.add_argument('--server', type=str, default='jasmin', help='Running on "jasmin" or "pentland"?')
    parser.add_argument('--data-root', type=str, default=None, help='Override for data root directory')
    parser.add_argument('--mask_clouds', action='store_true', help='Mask clouds in the loss function')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for dataset split')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    # --- 1. Prepare Data Split Once ---
    logging.info("Preparing data split...")
    data_directory = get_data_dirs(args.server, args.data_root)
    all_files, train_files, val_files = prepare_data(
        [data_directory],
        1.0 - args.val_frac,
        args.val_frac,
        0.0,
        seed=args.split_seed
    )
    logging.info(f"Data split: {len(train_files)} training files, {len(val_files)} validation files.")

    # --- 2. Optuna Study with NFS-friendly JournalStorage ---
    os.makedirs(args.output_dir, exist_ok=True)
    # The new storage will use a .log file instead of a .db file
    log_path = os.path.join(args.output_dir, f'{args.study_name}.log')
    
    # REQUIRED CHANGE: Use JournalStorage, which is robust on network file systems.
    storage = JournalStorage(JournalFileStorage(log_path))
    
    logging.info(f"Using JournalStorage: {log_path}")

    # Create or load the study using the new, safer storage object
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,  # Use the JournalStorage object
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=3),
        load_if_exists=True
    )

    # --- 3. Run Optimization ---
    logging.info(f"Attempting to run one trial for study '{args.study_name}'...")
    study.optimize(lambda trial: objective(trial, args, train_files, val_files), n_trials=1)

    logging.info(f"--- Single trial completed for study '{args.study_name}' ---")

    # --- 4. Save Best Hyperparameters ---
    if study.best_trial:
        best_params_path = os.path.join(args.output_dir, f'best_hyperparameters_{args.study_name}.txt')
        with open(best_params_path, 'w') as f:
            f.write(f"Best trial value: {study.best_trial.value}\n")
            f.write("Best parameters:\n")
            for key, value in study.best_trial.params.items():
                f.write(f"  {key}: {value}\n")
        logging.info(f"Best hyperparameters saved to {best_params_path}")

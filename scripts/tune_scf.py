import argparse
import logging
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import optuna
from sklearn.model_selection import KFold

from src.unet_torchgeo_lightning_trainer import SnowModel
from train_lightning import prepare_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    return [
        default_data_root,
        os.path.join(default_data_root, 'original_training_set')
    ]

def get_band_list(selection_name):
    """Returns a list of bands based on a selection name."""
    band_options = {
        'all_spectral': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'],
        'lis':['B3','B4','B11'],
        'wang': ['B2','B11','B4','B9'],
        'snow_relevant': ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
        'all': ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60', 'cs_cdf']
    }
    return band_options.get(selection_name)

def objective(trial, args):
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

    # --- 2. Create dataset ---
    data_directories = get_data_dirs(args.server, args.data_root)
    all_files, _, _ = prepare_data(data_directories, 1.0, 0.0, 0.0, seed=args.split_seed)
    
    # --- 3. K-Fold Cross-Validation ---
    k_folds = args.k_folds
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=args.split_seed)
    fold_mae_scores = []
    fold_rmse_scores = []
    fold_dice_scores = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_files)):
        logging.info(f"  -- Fold {fold+1}/{k_folds} --")
        
        train_files = [all_files[i] for i in train_ids]
        val_files = [all_files[i] for i in val_ids]

        # Initialize model for this fold
        model = SnowModel(
            bands=bands,
            train_files=train_files,
            val_files=val_files,
            mask_clouds=args.mask_clouds,
            use_augmentations=use_augmentations,
            use_indices=args.use_indices,
            hparams=hparams
        )

        # --- Trainer Setup ---
        # Unique dir for each trial and fold
        log_dir = os.path.join(args.output_dir, "logs", f"trial_{trial.number}", f"fold_{fold}")
        
        # Callbacks
        early_stop_callback = EarlyStopping(
            monitor='val_mae_snow_epoch',
            patience=hparams['patience'],
            verbose=False,
            mode='min'
        )
        
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            # devices=1,
            max_epochs=args.epochs,
            logger=TensorBoardLogger(save_dir=log_dir, name="lightning_logs"),
            callbacks=[early_stop_callback],
            enable_checkpointing=False, # Disable default checkpointing
            enable_progress_bar=True, # Set to False to reduce log clutter
        )

        # --- Training ---
        trainer.fit(model)
        
        # --- Evaluation ---
        # The validation epoch end logs 'val_mae_snow_epoch'. We can get it from the trainer.
        val_mae_snow = trainer.callback_metrics.get('val_mae_snow_epoch', float('inf'))
        val_rmse_snow = trainer.callback_metrics.get('val_rmse_snow_epoch', float('inf'))
        val_dice = trainer.callback_metrics.get('val_dice_epoch', 0.0) # Default to 0

        if val_mae_snow is None or val_mae_snow == float('inf'):
             logging.warning(f"  Fold {fold+1} did not produce a valid snow validation score. Skipping.")
             # Use a very high value to penalize this fold
             fold_mae_scores.append(float('inf'))
             fold_rmse_scores.append(float('inf'))
             fold_dice_scores.append(0.0)
             continue

        logging.info(f"  Fold {fold+1} Validation MAE on Snow: {val_mae_snow:.4f}, RMSE on Snow: {val_rmse_snow:.4f}, Dice: {val_dice:.4f}")
        fold_mae_scores.append(val_mae_snow)
        fold_rmse_scores.append(val_rmse_snow)
        fold_dice_scores.append(val_dice)

        # Report intermediate result to Optuna and enable pruning by fold
        current_avg = float(np.mean(fold_mae_scores))
        trial.report(current_avg, step=fold)
        if trial.should_prune():
            # Clean up before pruning
            del model, trainer
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        # Clean up memory for the next fold
        del model, trainer
        torch.cuda.empty_cache()

    # --- 4. Return the average score across folds ---
    if not fold_mae_scores:
        logging.warning(f"Trial {trial.number} completed without any valid fold scores.")
        return float('inf') # Return a large value if all folds failed

    average_mae_snow = np.mean([s for s in fold_mae_scores if s != float('inf')])
    average_rmse_snow = np.mean([s for s in fold_rmse_scores if s != float('inf')])
    average_dice_score = np.mean(fold_dice_scores)
    
    if np.isnan(average_mae_snow):
        average_mae_snow = float('inf') # If all folds failed, return infinity

    # Convert numpy floats to python floats for JSON serialization
    average_mae_snow = float(average_mae_snow)
    average_rmse_snow = float(average_rmse_snow)
    average_dice_score = float(average_dice_score)

    # Store the companion metric (RMSE) as a user attribute
    trial.set_user_attr("average_rmse_snow", average_rmse_snow)
    trial.set_user_attr("average_dice_score", average_dice_score)

    logging.info(f"--- Trial {trial.number} Finished ---")
    logging.info(f"  Average MAE on Snow: {average_mae_snow:.4f}")
    logging.info(f"  Average RMSE on Snow: {average_rmse_snow:.4f}")
    logging.info(f"  Average Dice Score: {average_dice_score:.4f}")
    
    return average_mae_snow


def get_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for SCF U-Net')
    # Optuna args
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials to run')
    parser.add_argument('--study-name', type=str, default='scf-unet-tuning', help='Name for the Optuna study')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs per trial per fold')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--k-folds', type=int, default=5, help='Number of folds for cross-validation')

    # Data and Model args (mostly fixed for the study)
    parser.add_argument('--server', type=str, default='jasmin', help='Running on "jasmin" or "pentland"?')
    parser.add_argument('--output-dir', type=str, default='/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/spun-scf/tuning', help='Directory for logs and study DB')
    parser.add_argument('--data_root', type=str, default=None, help='Override for data root directory')
    parser.add_argument('--bands', type=str, default='all_spectral_bands', help='Bands to use')
    parser.add_argument('--mask_clouds', action='store_true', help='Mask cloud pixels')
    parser.add_argument('--use_indices', action='store_true', help='Use indices in the dataset')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for dataset split and KFold')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    # --- Optuna Study with persistent storage ---
    os.makedirs(args.output_dir, exist_ok=True)
    storage_name = f"sqlite:///{os.path.join(args.output_dir, f'{args.study_name}.db')}"
    
    logging.info(f"Using storage: {storage_name}")

    # Create or load the study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1), # Prune after 2 folds (0-indexed)
        load_if_exists=True
    )

    # Calculate remaining trials
    n_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logging.info(f"Resuming study '{args.study_name}'. {n_completed_trials} trials already completed.")
    
    remaining_trials = args.trials - n_completed_trials
    if remaining_trials > 0:
        logging.info(f"Running {remaining_trials} more trials to reach the goal of {args.trials}.")
        study.optimize(lambda trial: objective(trial, args), n_trials=remaining_trials)
    else:
        logging.info("The study has already reached or exceeded the target number of trials.")

    print("\n--- Tuning Finished ---")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Avg. MAE on Snow): {trial.value}")
    if "average_rmse_snow" in trial.user_attrs:
        print(f"  Companion Metric (Avg. RMSE on Snow): {trial.user_attrs['average_rmse_snow']}")
    if "average_dice_score" in trial.user_attrs:
        print(f"  Companion Metric (Avg. Dice Score): {trial.user_attrs['average_dice_score']}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best parameters
    best_params_file = os.path.join(args.output_dir, f'best_hyperparameters_{args.study_name}.txt')
    with open(best_params_file, 'w') as f:
        f.write(f"Best Average MAE on Snow: {trial.value}\n")
        if "average_rmse_snow" in trial.user_attrs:
            f.write(f"Companion Metric (Avg. RMSE on Snow): {trial.user_attrs['average_rmse_snow']}\n")
        if "average_dice_score" in trial.user_attrs:
            f.write(f"Companion Metric (Avg. Dice Score): {trial.user_attrs['average_dice_score']}\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
    logging.info(f"Best hyperparameters saved to {best_params_file}")

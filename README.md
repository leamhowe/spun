# SPUN SCF Training

## Overview
`train_lightning.py` trains a UNet-based regression model (PyTorch Lightning) to predict Snow Cover Fraction (SCF) from Sentinel-2 `.zarr` tiles.  
Data are gathered from two directories:
- `<data_root>`
- `<data_root>/original_training_set`

On JASMIN the default `data_root` is:
```
/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/s2_training_scf
```
You can override with `--data_root`.

A reproducible random split is performed across all discovered `.zarr` files:
```
train_ratio = 1 - val_ratio - test_ratio
val_ratio = --val_ratio (default 0.15)
test_ratio = --test_ratio (default 0.15)
seed = --split_seed (default 42)
```

## Key Features
- Random train/val/test split over all `.zarr` files (deduplicated).
- Optional band set selection (`--bands`).
- Optional spectral indices (`--use_indices`).
- Optional augmentations (`--use_augmentations`).
- W&B logging (`project=spun_scf`).
- Early stopping + checkpointing on `val_mae`.
- Final regression metrics (MAE, RMSE) computed on test split (ignoring `label == 200`).

## Run (Local / Interactive)
```
cd scripts
python train_lightning.py \
  --model_name scf_run_001 \
  --server jasmin \
  --bands custom_bands \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --use_indices
```

Add augmentations:
```
python train_lightning.py --model_name scf_aug --use_augmentations
```

Resume:
```
python train_lightning.py --model_name scf_resume \
  --checkpoint_path /path/to/checkpoint.ckpt
```

## SLURM (JASMIN)
Submit:
```
sbatch train_lightning_jasmin.sh scf_run_002
```
Monitor:
```
squeue -u $USER
```

## Important Arguments
- `--model_name` : Run identifier.
- `--bands` : one of `all_bands | all_spectral_bands | wang_bands | perm_threshold_01 | custom_bands`.
- `--val_ratio`, `--test_ratio`, `--split_seed`.
- `--use_indices` : include derived indices in dataset (if available).
- `--use_augmentations` : enable augmentation pipeline.
- `--backbone`, `--weights` : model backbone + pretrained weights.
- `--loss_fn` : `l1 | mse | huber`.
- `--loss_weighting` : `none | binary | linear`.
- `--mask_clouds` : mask cloud pixels (set to ignore value).
- `--checkpoint_path` : resume training.

## Outputs
- Checkpoints: `<model_save_dir>/checkpoints/` (Lightning + final copy `<model_name>.ckpt`).
- State dict: `<model_save_dir>/<model_name>.pt`
- Metrics CSV: `<metrics_save_dir>/<model_name>_metrics.csv`
- W&B run directory: `../wandb/`
- Console prints split counts + final test MAE/RMSE.

## Example Minimal Run
```
python train_lightning.py --model_name quick_test --bands all_spectral_bands
```

## Notes
- Ensure the Conda env with PyTorch Lightning, albumentations, wandb, torchgeo is active.
- If W&B offline is needed: `export WANDB_MODE=offline` before running.
- Splits are in-memory only; persist manually if required.

## Future Enhancements (optional)
- Persist split lists to JSON for reproducibility across future runs.
- Add CLI flag for deterministic CuDNN behavior.
- Expose image size / crop size as arguments.





## my dump

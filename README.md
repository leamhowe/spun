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
python train_lightning_combloss.py --model_name 20251021_torchgeo_combloss_NOIGNORE_S2SplitsClean_BAllSpectral_PTWS2AllMoco_WBin10_pat50_lr1e-4_pad0_run001 --bands all_spectral_bands --split_file /home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv --weights sentinel2_all_moco --num_workers 24 --patience 50 --lr 1e-4 --batch_size 8 --snow_weight 10 --loss_weighting binary --train_with_clouds


python train_lightning_combloss.py --model_name 20251022_torchgeo_combloss_NOIGNORE_S2SplitsClean_Bsatlas_PTWSatlasMs_WInLin10_pat50_lr1e-4_pad0_run001 --bands satlas_ms --backbone resnet152  --weights satlas_ms --num_workers 24 --patience 50 --lr 1e-4 --batch_size 8 --snow_weight 10 --loss_weighting inverse_linear --train_with_clouds --split_file /home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv
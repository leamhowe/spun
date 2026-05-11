# SPUN SCF Training

![](spun_logo.png)

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

python train_lightning_combloss.py --model_name 20251111_torchgeo_combloss_NOIGNORE_S2SplitsClean_BAllSpectral_PTWS2SoftCon_WBin10_pat50_lr1e-4_pad0_run001 --bands all_spectral --split_file /home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv --weights sentinel2_softcon --num_workers 24 --patience 50 --lr 1e-4 --batch_size 8 --snow_weight 10 --loss_weighting binary --train_with_clouds

python train_lightning_combloss.py --model_name 20251111_torchgeo_combloss_NOIGNORE_S2SplitsClean_BAllSpectral_PTWS2AllMoco_WBin10_pat50_lr1e-4_pad0_run001 --bands all_spectral --split_file /home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv --weights sentinel2_all_moco --num_workers 24 --patience 50 --lr 1e-4 --batch_size 8 --snow_weight 10 --loss_weighting binary --train_with_clouds

python train_lightning_combloss_13bands.py --model_name 20251112_torchgeo_combloss_NOIGNORE_S2SplitsClean_BAllSpectral_PTWS2AllMoco_WBin10_pat50_lr1e-4_pad0_run001 --bands all_spectral --split_file /home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv --weights sentinel2_all_moco --num_workers 24 --patience 50 --lr 1e-4 --batch_size 8 --snow_weight 10 --loss_weighting binary --train_with_clouds

### tuned parameters

python train_lightning_combloss.py     
  --model_name "Final_Trial30_ResNet50_Softcon_v3"     
  --server "jasmin"     
  --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv"     
  --use_augmentations     
  --train_with_clouds     
  --bands "all_spectral"     
  --backbone "resnet50"     
  --weights "sentinel2_softcon"     
  --batch_size 8     
  --lr 1.1725161187672691e-05     
  --loss_fn "huber"     
  --loss_weighting "binary"     
  --snow_weight 7.040241273998283     
  --snow_thresh 0.3399060516710048     
  --dice_loss_weight 0.8809736082904772     
  --patience 20     
  --num_workers 4


**Trial 39**

python train_lightning_combloss.py \
    --model_name "Final_T39_AllDino_invlinear" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "sentinel2_all_dino" \
    --batch_size 8 \
    --lr 5.405207066353444e-05 \
    --loss_fn "mse" \
    --loss_weighting "inverse_linear" \
    --snow_weight 1.6472820171376819 \
    --snow_thresh 0.08910991771832737 \
    --dice_loss_weight 0.14556214009188198

**Trial 4**
python train_lightning_combloss.py \
    --model_name "Final_T4_AllDino_none" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "sentinel2_all_dino" \
    --batch_size 32 \
    --lr 0.0009698563793929831 \
    --loss_fn "l1" \
    --loss_weighting "none" \
    --snow_thresh 0.3320815509861747 \
    --dice_loss_weight 0.6095664535712578

**Trial 66**
python train_lightning_combloss.py \
    --model_name "Final_T66_ImageNet_binary" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "imagenet" \
    --batch_size 16 \
    --lr 7.81514168490619e-05 \
    --loss_fn "huber" \
    --loss_weighting "binary" \
    --snow_weight 2.6174735357743604 \
    --snow_thresh 0.4251087897646257 \
    --dice_loss_weight 0.8700951942444198

**Trial 143**
python train_lightning_combloss.py \
    --model_name "Final_T143_ImageNet_binary" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "imagenet" \
    --batch_size 32 \
    --lr 8.237703246555589e-05 \
    --loss_fn "huber" \
    --loss_weighting "binary" \
    --snow_weight 18.12120023327038 \
    --snow_thresh 0.4251087897646257 \
    --dice_loss_weight 0.8700951942444198
**Trial 152**
python train_lightning_combloss.py \
    --model_name "Final_T152_ImageNet_linear" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "imagenet" \
    --batch_size 32 \
    --lr 0.00016990484220539724 \
    --loss_fn "huber" \
    --loss_weighting "linear" \
    --snow_weight 3.418548843378609 \
    --snow_thresh 0.2097704039745673 \
    --dice_loss_weight 0.8700951942444198
**Trial 28**
python train_lightning_combloss.py \
    --model_name "Final_T28_Moco_linear" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "sentinel2_all_moco" \
    --batch_size 32 \
    --lr 0.00011154468326912748 \
    --loss_fn "mse" \
    --loss_weighting "linear" \
    --snow_weight 9.671855483590338 \
    --snow_thresh 0.034802366839596 \
    --dice_loss_weight 0.7068915020828858
**Trial 12**
python train_lightning_combloss.py \
    --model_name "Final_T12_Softcon_linear" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "sentinel2_softcon" \
    --batch_size 16 \
    --lr 5.9789815148777714e-05 \
    --loss_fn "huber" \
    --loss_weighting "linear" \
    --snow_weight 2.9512016190028687 \
    --snow_thresh 0.03650865323402854 \
    --dice_loss_weight 0.5005588405503213
**Trial 64**
python train_lightning_combloss.py \
    --model_name "Final_T64_Softcon_invlinear" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "sentinel2_softcon" \
    --batch_size 8 \
    --lr 0.00010767609233176974 \
    --loss_fn "huber" \
    --loss_weighting "inverse_linear" \
    --snow_weight 4.874022607722513 \
    --snow_thresh 0.26351518424865583 \
    --dice_loss_weight 0.3749431516118932


# Normalisation added

--- Trial 66 ---
  Values (RMSE_snow, Dice): [0.3597731590270996, 0.7101811170578003]
  Parameters:
    loss_weighting: inverse_linear
    snow_weight: 3.0533141102829875
    lr: 2.925316069302096e-05
    batch_size: 8
    loss_fn: mse
    snow_thresh: 0.07615339251914251
    dice_loss_weight: 0.28545900145292324


python train_lightning_combloss_13bands.py \
    --model_name "Final_T66_Softcon_invlinear_NORM" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "sentinel2_softcon" \
    --batch_size 8 \
    --lr 2.925316069302096e-05 \
    --loss_fn "mse" \
    --loss_weighting "inverse_linear" \
    --snow_weight 3.0533141102829875 \
    --snow_thresh 0.07615339251914251 \
    --dice_loss_weight 0.28545900145292324

--- Trial 59 ---
  Values (RMSE_snow, Dice): [0.36314302682876587, 0.7146376967430115]
  Parameters:
    loss_weighting: binary
    snow_weight: 16.91732391775266
    lr: 0.00016417313765636663
    batch_size: 32
    loss_fn: huber
    snow_thresh: 0.013805707032480621
    dice_loss_weight: 0.1290031348531521

python train_lightning_combloss_13bands.py \
    --model_name "Final_T59_Moco_binary_NORM" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --weights "sentinel2_all_moco" \
    --patience 50 \
    --num_workers 24 \
    --batch_size 32 \
    --lr 0.00016417313765636663 \
    --loss_fn "huber" \
    --loss_weighting "binary" \
    --snow_weight 16.91732391775266 \
    --snow_thresh 0.013805707032480621 \
    --dice_loss_weight 0.1290031348531521


## Scaling Added

python train_lightning_combloss_13bands.py \
    --model_name "Final_T77_Dino_linear_Scale10000" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --weights "sentinel2_all_dino" \
    --patience 50 \
    --num_workers 24 \
    --batch_size 32 \
    --lr 0.000130045 \
    --loss_fn "l1" \
    --loss_weighting "linear" \
    --snow_weight 13.4949 \
    --snow_thresh 0.452531 \
    --dice_loss_weight 0.793629

--- Top Model: Rank 24 (Trial 4, softcon) --- RMSE_snow: 0.3470, Dice: 0.7353

python train_lightning_combloss_13bands.py \
    --model_name "Final_T4_softcon_FinalScale" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "sentinel2_softcon" \
    --batch_size 32 \
    --lr 0.0002931482464615 \
    --loss_fn "huber" \
    --loss_weighting "inverse_linear" \
    --snow_weight 19.78087315383835 \
    --snow_thresh 0.1439946901683584 \
    --dice_loss_weight 0.10454034128721

T15

python train_lightning_combloss_13bands.py \
    --model_name "Final_T15_dino_none_scale10000" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "sentinel2_dino" \
    --batch_size 16 \
    --lr 0.000118 \
    --loss_fn "huber" \
    --loss_weighting "none" \
    --snow_thresh 0.399728 \
    --dice_loss_weight 0.166421



python train_lightning_combloss_13bands.py \
    --model_name "Final_T3_decur_binary_scale10000" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "sentinel2_all_decur" \
    --batch_size 32 \
    --lr 0.000074 \
    --loss_fn "mse" \
    --loss_weighting "binary" \
    --snow_weight 10.476566 \
    --snow_thresh 0.479745 \
    --dice_loss_weight 0.843779


python train_lightning_combloss_13bands.py \
    --model_name "Final_T3_imagenet_binary_scale10000" \
    --server "jasmin" \
    --split_file "/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --patience 50 \
    --num_workers 24 \
    --weights "imagenet" \
    --batch_size 32 \
    --lr 0.000229 \
    --loss_fn "huber" \
    --loss_weighting "binary" \
    --snow_weight 9.862176 \
    --snow_thresh 0.113775 \
    --dice_loss_weight 0.117788
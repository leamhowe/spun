# A Snow Patch U-Net (SPUN)

![spun logo](assets/spun_logo.png)

## Overview

**SPUN** (Snow Patch U-Net) is a deep learning pipeline built on PyTorch and TorchGeo, designed to predict continuous Snow Cover Fraction (SCF) from multi-spectral Sentinel-2 imagery.

Developed to handle the challenging marginal snow conditions and frequent cloud cover of the Scottish Highlands, SPUN leverages self-supervised pre-trained backbones (DINO) and a custom combined loss function (Huber + Dice) to generate high-resolution (10m) SCF maps.

## 📂 Repository Structure

The repository is organised to strictly separate source code, model training, and evaluation pipelines:

* **`src/`**: Core PyTorch modules, including the Lightning trainer, custom combined loss functions, and the `Snows2Dataset` PyTorch `Dataset` class.
* **`scripts/`**: Executable scripts for model training, hyperparameter tuning (`optuna`), and SLURM submission.
* **`training/`**: Stores training outputs, including model checkpoints, final `.pt` weights, hyperparameter tradeoff plots, and band importance logs.
* **`results/evaluation/`**: Contains the rigorous, reproducible evaluation pipelines comparing SPUN to baselines (NDSI, Let-It-Snow, Wang 2022).
    * `scotland/`: Pipeline for 1km × 1km chip-based evaluation (Multi-res metrics and Hydrological union-snow evaluation).
    * `global_wang2022/`: Pipeline for Scene-Level evaluation on massive contiguous tiles, including CloudScore+ mask ablation studies.


* **`example_notebooks/`**: Interactive tutorials, including a quickstart guide for running inference on new imagery.

## 🚀 Quickstart: Inference

If you just want to load the pre-trained SPUN model and run predictions on Sentinel-2 `.zarr` files, check out the interactive tutorial:
🔗 **[`example_notebooks/01_quickstart_inference.ipynb`](https://www.google.com/search?q=example_notebooks/01_quickstart_inference.ipynb)**

## 🏋️ Training the Model

Training is handled by `scripts/train.py`. The pipeline utilises Weights & Biases (`wandb`) for logging and Albumentations for data augmentation.

### The Dataset

Unlike earlier iterations that performed on-the-fly random splits, the dataset is now strictly defined by a centralised metadata file (`dataset_metadata.csv`). This ensures perfectly reproducible Train/Val/Test boundaries.

### Optimal Run Command

Based on our extensive hyperparameter tuning and ablation studies, the optimal model utilises all 13 Sentinel-2 bands, a ResNet-50 backbone pre-trained with DINO, and a combined Huber + Dice loss function to balance continuous fraction prediction with sharp boundary delineation.

Run the optimal configuration locally or via SLURM:

```bash
python -m scripts.train \
    --model_name "Final_T15_dino_none_scale10000" \
    --data_root "<path_to_zenodo_dataset>/SPUN_Dataset" \
    --split_file "dataset_metadata.csv" \
    --use_augmentations \
    --train_with_clouds \
    --bands "all_spectral" \
    --backbone "resnet50" \
    --weights "sentinel2_all_dino" \
    --loss_fn "huber" \
    --loss_weighting "none" \
    --dice_loss_weight 0.166421 \
    --snow_thresh 0.399728 \
    --lr 0.000118 \
    --batch_size 16 \
    --num_workers 24 \
    --patience 50

```

*Note: On JASMIN, you can submit this job using the provided SLURM wrapper: `sbatch scripts/train_lightning_jasmin.sh <run_name>*`

## 📊 Evaluation Pipelines

To ensure scientific rigor, SPUN is evaluated across two distinct environments. Please see the dedicated `README.md` files within each evaluation directory for specific bash commands and prerequisite data instructions.

### 1. Scottish Highlands 1km Chip Evaluation

**Location:** `results/evaluation/scotland/`
Compares SPUN against LIS, NDSI, and Wang 2022 on 1km × 1km chips centered on highly complex, patchy terrain.

* **Multi-Resolution Evaluation:** Evaluates native 10m performance, then downscales to 20m for direct comparison with the operational LIS baseline under clear-sky and all-sky conditions.
* **Hydrological Evaluation:** Assesses performance exclusively within the "Snow Union" mask (ignoring clear ground) to determine utility for downstream hydrological modeling.

### 2. Global Scene-Level Evaluation

**Location:** `results/evaluation/global_wang2022/`
Assesses macro-scale performance on massive, contiguous Sentinel-2 tiles drawn from the Wang 2022 dataset.

* **Global Benchmarking:** Calculates unified 3-class (Snow, Cloud, Background) macro-averages against previous literature.
* **Cloud Mask Ablation:** Sensitivity analysis and ablation studies justifying the integration of the continuous CloudScore+ probability mask (`cs_cdf`) into the final operational SPUN pipeline.

## ⚙️ Environment Setup

It is highly recommended to run this repository within a dedicated Conda environment. Key dependencies include:

* `torch` & `pytorch-lightning`
* `torchgeo` (Crucial for Sentinel-2 specific model weight loading)
* `albumentations`
* `zarr` & `xarray`
* `rasterio` & `geopandas`
* `wandb` (Set `export WANDB_MODE=offline` if working on a compute node without internet access).
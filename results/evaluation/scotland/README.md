# Scotland Test Set Evaluation Pipeline

This directory contains the pipeline for evaluating the SPUN model against established baseline methods (Wang2022, NDSI, and LIS) across the complex, patchy snow environments of the Scottish Highlands.

## 📂 Directory Structure

The repository is structured to separate model inference from global evaluation:

* **Model Subdirectories (`/spun`, `/wang2022`, `/ndsi`):** Each folder contains the specific `scripts/` required to generate predictions for that method, as well as a `test_set_predictions/` folder where the resulting `.zarr` outputs are saved.
* **External Data (`/lis`):** Contains scripts for processing and extracting the Let It Snow (LIS) baseline patches.
* **Root Scripts:** The Python scripts at the root of this directory ingest the `.zarr` files from all the subdirectories to calculate multi-resolution metrics, coverage statistics, and hydrological reporting.

Please see the individual `README.md` files within each subdirectory for detailed information on data sources, prerequisites, and model quirks.

## 🚀 How to Run the Pipeline

### Step 1: Prepare Baselines and Generate Predictions
Before running the comparisons, you must extract the LIS baseline patches and generate the prediction `.zarr` files for each comparative model. Navigate to the root of your project and run the following commands:

```bash
# 1. Prepare LIS Baseline Patches
python -m lis.scripts.patchify_lis \
    --data_root <path_to_zenodo_dataset> \
    --lis_dir <path_to_raw_lis_fsc> \
    --osgb_grid <path_to_os_bng_grids.gpkg> \
    --output_dir lis/lis_fsc_patches

# 2. Generate NDSI baselines
python ndsi/scripts/predict_ndsi_baseline.py \
    --data_root <path_to_zenodo_dataset> \
    --output_dir ndsi/test_set_predictions

# 3. Generate Wang2022 baselines (Requires Level-2A data)
python wang2022/scripts/predict_wang2022.py \
    --data_root <path_to_L2A_dataset> \
    --split_file <path_to_zenodo_dataset>/dataset_metadata.csv \
    --model_dir <path_to_wang_repo> \
    --model_path <path_to_wang_repo>/models/unet_4bands.pth \
    --output_dir wang2022/test_set_predictions

```

### Step 2: Run the Multi-Resolution Comparison

Once all predictions and patches are generated, run the multi-resolution evaluation script. This script compares the models at their native 10m resolution (Tier 1) and then downscales them to 20m to compare against the LIS baseline under Clear-Sky and All-Sky conditions (Tier 2).

```bash
python comparison_offline_10and20m.py \
    --data_root <path_to_zenodo_dataset> \
    --spun_dir spun/test_set_predictions \
    --lis_dir lis/lis_fsc_patches \
    --ndsi_dir ndsi/test_set_predictions \
    --wang_dir wang2022/test_set_predictions \
    --output_report final_multires_report_with_cloud_separation.txt

```

### Step 3: Run the SCA Evaluation

To evaluate the models purely on their intrinsic spatial resolution snow-detection capabilities focusing area of snow extract per 1 km image chip for hydrological applications, run the native resolution evaluation script:

```bash
python evaluate_per_tile_hydrology_unionsnow_nativeres.py \
    --data_root <path_to_zenodo_dataset> \
    --spun_dir spun/test_set_predictions \
    --ndsi_dir ndsi/test_set_predictions \
    --wang_dir wang2022/test_set_predictions \
    --output_report final_hydrology_native_report.txt

```

## 📊 Outputs

Running the evaluation scripts will generate detailed text reports (`.txt`) in this root directory containing:

1. **Continuous Metrics:** MAE, RMSE, Bias (Global and Snow-Only).
2. **Binary Classification:** F1-Score, Precision, Recall, Accuracy.
3. **Coverage Statistics:** Detected snow area (km²) and deviation from Ground Truth.


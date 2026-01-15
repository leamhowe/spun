# **Analysis & Comparison Workflow**

This directory contains the pipeline for evaluating the SPUN model, the operational LIS product, and performing a fair, direct comparison between them.

## **Root Directory (Comparison Pipeline)**

The core comparison is performed "offline" using pre-generated predictions to ensure efficiency and reproducibility.

* **`evaluate_comparison_offline.py`**: The primary script for comparing SPUN vs. LIS. It loads pre-computed SPUN predictions and LIS patches to calculate comprehensive metrics. It evaluates two distinct scenarios:
* **Scenario 1 (Clear-Sky):** A strict algorithm comparison that evaluates only pixels where *both* the Ground Truth and LIS are cloud-free. This measures pure retrieval accuracy.
* **Scenario 2 (All-Sky):** An operational product comparison that evaluates all valid Ground Truth pixels. LIS cloud masks are treated as 0% snow (missed detection) to quantify the real-world utility gap.
* **Metrics:** Calculates Accuracy, Precision, Recall, F1, Dice, MAE, and RMSE. It includes specific "Sensitivity" metrics (threshold > 1%) and "Snow-Only" error metrics to address class imbalance.


* **`final_comparison_report.txt`**: The output file from the offline evaluation. Contains formatted performance tables for both scenarios and a "Coverage Gap Analysis" quantifying the total snow volume detected by each product.


## **`spun/scripts` (Model Inference & Baseline)**

Scripts for running the SPUN model and establishing its standalone performance.

* **`generate_predictions.py`**: Prerequisite for the offline comparison. Runs SPUN inference *once* on the test set, applies necessary geometric corrections (flipping/cropping), and saves the raw 10m output and aligned ground truth to Zarr files.
* **`evaluate_spun.py`**: Evaluates the SPUN model at its native 10m resolution against the test set labels.
* **`evaluate_spun_20m.py`**: Evaluates SPUN performance after downscaling outputs to 20m. This establishes a baseline for how the model performs at the coarser resolution used by LIS, independent of the LIS comparison logic.


## **`lis/scripts` (LIS Processing & Benchmarking)**

Scripts for preparing LIS data and evaluating its standalone performance.

* **`patchify_lis.py`**: Processes original LIS product files to extract spatial patches corresponding to the test set tiles, saving them as Zarr files for efficient access.
* **`evaluate_lis.py`** / **`evaluate_lis_v2.py`**: Standalone scripts for evaluating the LIS product against ground truth. They handle alignment (e.g., coordinate flipping) and calculate standard metrics to benchmark LIS performance in isolation.
* **`check_zarr_contents.py`**: Utility to inspect Zarr file structures, shapes, and data types. Used to verify that SPUN predictions and LIS patches are correctly formatted before comparison.
* **`identify_missing_files.py`** & **`missing_lis_files.txt`**: Quality control script that checks the test set list against available LIS patches to identify and log any missing corresponding files.
* **`spatial_utils.py`**: Helper module containing reusable spatial functions (e.g., downscaling logic, coordinate extraction) shared across LIS processing scripts.
* **`debug.ipynb`**: Interactive notebook for visualizing data alignment and debugging patch extraction issues.
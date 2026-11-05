# A Snow Patch U-Net (SPUN)

![spun logo](assets/spun_logo.png)

## Overview

**SPUN** (Snow Patch U-Net) is a deep learning pipeline built on PyTorch and TorchGeo, designed to predict continuous Snow Cover Fraction (SCF) from multi-spectral Sentinel-2 imagery.

SPUN was developed for mapping patchy late-season snow and snow patches (decametre- to hectometre-scale features) in the Highlands of Scotland. Scotland has marginal snow conditions and frequent cloud cover, providing a tough test for optical remote sensing. SPUN leverages self-supervised pre-trained backbones (DINO) and a combined loss function (Huber + Dice) to generate SCF maps on a 10 m grid, and shows strong generalisation of snow detection outside the training domain.

**You can read more details about the SPUN model in our preprint:** [https://egusphere.copernicus.org/preprints/2026/egusphere-2026-3163/](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-3163/)

### Example use case - Cairngorms National Park snow persistence

SPUN was designed for mapping snow patches in the Scottish Highlands (as part of my PhD research studying these small decametre-scale features) but shows promise for wider applications in snow mapping and monitoring. As an example of the potential of SPUN, we mapped snow across the entire Sentinel-2 archive covering the Cairngorms National Park, Scotland. You can visualise these snow maps here:

- [leamhowe.github.io/spun](https://leamhowe.github.io/spun.html)
<!-- - [Dashboard of SPUN snow cover maps for the Cairngorms, 2016-2026](https://leamhowe.github.io/assets/dashboards/CNP_snow_dashboard_QCflags_sunangle22_cloud80.html). -->

We used these maps to generate an estimate of snow persistence, normalising by cloud-free views (i.e., total snow views / total clear views for each pixel). The plot below shows a comparison of the estimated Snow Cover Duration (eSCD) in 2018 using the SPUN model (right) against the best currently available baseline, the UK Met Office HadUK product (left):

![snow persistence cairngorms](assets/eSCD_SPUN_HadUK_comparison_CNP_2018.png)

The SPUN persistence map highlights both naturally occurring late-lying snow patches, such as Ciste Mhearad (Margaret's Coffin), and localised, wind-driven snow accumulation around the Cairngorm ski centre infrastructure. The sub-grid processes revealed here provide a much richer dataset for investigating localised ecological and hydrological processes.

## 📂 Repository Structure

The repository is organised to separate source code, model training, inference and evaluation:

* **`src/`**: Core PyTorch modules, including the Lightning trainer, combined loss functions, the `Snows2Dataset` PyTorch `Dataset` class, and inference utilities (`src/utils/inference_utils.py`).
* **`scripts/`**: Command-line tools, including `predict_s2_tile.py` for mapping a full Sentinel-2 tile (see below).
* **`training_scripts/`**: Model training and hyperparameter tuning (`optuna`), with example SLURM submission scripts.
* **`training/`**: Training outputs, including model checkpoints, final `.pt` weights, hyperparameter trade-off plots and band importance logs.
* **`results/evaluation/`**: Reproducible evaluation pipelines comparing SPUN to baselines (NDSI, Let-It-Snow, Wang et al., 2022).
    * `scotland/`: 1 km × 1 km chip-based evaluation (multi-resolution metrics and snow-covered-area evaluation), plus the additional analyses listed below.
    * `global_wang2022/`: scene-level evaluation on the global dataset of Wang et al. (2022), including CloudScore+ mask analyses.
* **`example_notebooks/`**: Interactive tutorials, including a quickstart guide for running inference on new imagery.

## 🚀 Quickstart: Inference

If you just want to load the pre-trained SPUN model and run predictions on Sentinel-2 `.zarr` files, check out the interactive tutorial:
🔗 **[`example_notebooks/01_quickstart_inference.ipynb`](example_notebooks/01_quickstart_inference.ipynb)**

## 🗺️ Mapping a full Sentinel-2 tile

`scripts/predict_s2_tile.py` produces a snow map for a whole Sentinel-2 tile (~110 × 110 km) directly from a tile identifier and date. It downloads Level-1C data and CloudScore+ from Google Earth Engine, runs SPUN, and writes GeoTIFFs on the tile's native 10 m UTM grid. You need a Google Earth Engine account: either authenticate once with `earthengine authenticate`, or pass a service-account key with `--credentials_path`.

```bash
# Which dates have full, clear coverage of a tile?
python scripts/predict_s2_tile.py --tile 30VVJ --list_scenes 2026-05-20 2026-05-30

# Snow map for one date, with water and built-up areas masked
python scripts/predict_s2_tile.py --tile 30VVJ --date 2026-05-26 \
    --model_path <path/to/spun_model.pt> --output_dir out --water_mask --urban_mask

# Only the true-colour image of the same scene (no model needed)
python scripts/predict_s2_tile.py --tile 30VVJ --date 2026-05-26 --rgb_only --output_dir out
```

Useful options: `--composite_days N` fills partial (swath-edge) tiles with the clearest pixel within ±N days; `--save_rgb` also saves a true-colour image; `--device cpu --cpu_threads 16` runs on CPU; `--max_chunks 4` processes a small corner for testing. Run `python scripts/predict_s2_tile.py --help` for all options.

A full tile takes ~3 min on a single GPU (NVIDIA A100) and ~5 min on 16 CPU cores, including data download.

### Output files

**`SPUN_<tile>_<date>_SCF.tif`**: the snow product. A single band with an embedded colour table, so it displays directly when dragged into QGIS or ArcGIS:

| Value | Meaning |
|---|---|
| 0–100 | Snow cover fraction (%); 0 = observed, no snow |
| 205 | Cloud (CloudScore+ `cs_cdf` < 0.5, adjustable with `--cs_threshold`) |
| 210 | Permanent water (ESA WorldCover; with `--water_mask`) |
| 215 | Built-up (ESA WorldCover; with `--urban_mask`) |
| 255 | No data |

**`SPUN_<tile>_<date>_QA.tif`**: supporting layers: `scf_raw` (SCF before any masking, e.g. to apply your own cloud threshold), `cs_cdf_x100` (CloudScore+ score × 100) and `day_offset_plus100` (acquisition date relative to `--date`, plus 100; useful with `--composite_days`). 255 = no data.

**`SPUN_<tile>_<date>_RGB.tif`** (with `--save_rgb` or `--rgb_only`): true-colour image (B4, B3, B2) of the input composite, with a fixed square-root stretch of TOA reflectance. 0 = no data.

## 🏋️ Training the Model

You can download the trained model from here: https://doi.org/10.5281/zenodo.20055817

If you want to train the model from scratch see [`training_scripts/README.md`](training_scripts/README.md) for more details. The example SLURM scripts contain placeholder paths (e.g. `<path/to/SPUN_Dataset>`) that need editing before use.

## 📊 Evaluation Pipelines

SPUN is evaluated on the [Scotland dataset](https://doi.org/10.5281/zenodo.20055817) and on an independent global dataset ([Wang et al., 2022](https://github.com/yiluyucheng/SnowCoverage/tree/main)). Please see the dedicated `README.md` files within each evaluation directory for specific commands and prerequisite data.

### 1. Scottish Highlands 1 km chip evaluation

**Location:** `results/evaluation/scotland/`
Compares SPUN against LIS, NDSI and Wang et al. (2022) on 1 km × 1 km chips in complex, patchy terrain.

* **Multi-resolution evaluation:** evaluates performance at native 10 m resolution (Tier 1), then aggregates to 20 m for comparison with the operational LIS product under clear-sky (Tier 2a) and all-sky (Tier 2b) conditions.
* **Snow-covered-area evaluation:** assesses performance on the area of estimated snow cover per chip, to determine utility for downstream hydrological modelling.

### 2. Global scene-level evaluation

**Location:** `results/evaluation/global_wang2022/`
Assesses performance on full Sentinel-2 scenes from the Wang et al. (2022) dataset. As these labels are categorical, this tests the transferability of snow detection, not of fractional retrieval.

* **Global benchmarking:** 3-class (snow, cloud, background) macro-averages for comparison with previous literature.
* **Cloud mask analyses:** sensitivity and ablation studies of the CloudScore+ (`cs_cdf`) mask used in the SPUN pipeline.

### 3. Additional analyses (manuscript revision)

| Folder | Analysis |
|---|---|
| `results/evaluation/scotland/cs_threshold_sensitivity/` | Pixel categories (clear, partially affected, CS+ cloud) and sensitivity of results to the CloudScore+ threshold |
| `results/evaluation/scotland/scf_stratified/` | Performance stratified by reference SCF, at 10 m and 20 m |
| `results/evaluation/scotland/split_overlap/` | Evaluation excluding test chips that share a Sentinel-2 acquisition with training chips |
| `results/evaluation/global_wang2022/cloudscore/` | CloudScore+ cloud-detection performance on the global dataset, and propagation of its errors into the final snow product |

## ⚙️ Environment Setup

It is highly recommended to run this repository within a dedicated Conda environment. Key dependencies include:

* `torch` & `pytorch-lightning`
* `torchgeo` (needed for Sentinel-2-specific model weight loading)
* `albumentations`
* `zarr` & `xarray`
* `rasterio` & `geopandas`
* `earthengine-api` & `google-auth` (for `scripts/predict_s2_tile.py`)
* `wandb`

I used [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) to create the environment used in development of SPUN. For Linux, macOS, or Git Bash on Windows install with:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

The environment can then be created using the command:

```bash
micromamba env create -n spun -f environment.yml
```

Once created, there is a known issue of the `simsimd` library not installing properly with micromamba, so install it separately:

```bash
micromamba activate spun
pip install simsimd
```

## Running example notebooks

Once you have created and activated the spun environment (see above), you might need to register the environment as a kernel before it is detected as a Jupyter environment:

```bash
python -m ipykernel install --user --name=spun
```

Once registered as a kernel, launch JupyterLab with:

```bash
jupyter lab
```

and JupyterLab should open in your browser with the spun kernel available.

## 🗺️ Roadmap & Ongoing Development

We are actively working to expand SPUN's capabilities and make it easier to use:

- [x] Create a quick inference tutorial notebook.
- [x] Create dashboard to visualise data for Cairngorms National Park.
- [x] Add script to generate a snow map directly from a Sentinel-2 tile identifier (`scripts/predict_s2_tile.py`).
- [x] Add optional water and built-up masks from ESA WorldCover.
- [ ] Create additional tutorial notebooks, e.g. create a snow persistence map, or apply SPUN to a full Sentinel-2 tile.
- [ ] Add a normalised DEM or modelled hillshade as an input (could help with topographic shadows).
- [ ] Add classes for common confusers (e.g. water, built-up) to the model output.
- [ ] Shaded snow as a different class? Identifying cloud and topographic shade separately?
- [ ] Add SCL to training data?

## Known issues to overcome in future development

- SPUN is a *Snow Patch* U-Net. It was developed for mapping patchy late-season snow and snow patches (decametre- to hectometre-scale features), and this is the [data](https://doi.org/10.5281/zenodo.20055817) it was trained on. Therefore, although it shows strong generalisation to complete snow cover, there are some issues with tough-to-identify examples that are far outside the training distribution. This includes when **low sun angles** lead to **harsh topographic and cloud shadows over complete snow cover**, leading to very contrasting viewing conditions which SPUN struggles with --- **these issues are widely seen across optical snow mapping products.**
- Occasional misclassification of bright buildings and the margins of water bodies as snow. This can be mitigated with the `--water_mask` and `--urban_mask` options (from [ESA WorldCover](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200#bands)), but future versions could include these as classes.
- Snow and ice, and snow on ice. Not so much of an issue in Scotland (except when frozen lakes have snow on them), but separating snow on glaciers would be really useful (e.g. as a proxy for the equilibrium line altitude).
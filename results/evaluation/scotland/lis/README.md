# Let It Snow (LIS) Baseline Preparation

This directory handles the preparation and extraction of the Let It Snow (LIS) baseline data for direct comparison against SPUN predictions.

The raw LIS snowmaps were generated using the default recommended configuration from the official Orfeo ToolBox remote module: 
🔗 [Let It Snow Repository](https://gitlab.orfeo-toolbox.org/remote_modules/let-it-snow)

## 🛠️ Scripts

### `scripts/patchify_lis.py`

**Purpose:** The standard LIS pipeline produces large regional tiles matching the Sentinel-2 tiling system. We needed to extract the 1km x 1km SPUN chips for direct comparison to our test set. This script automatically cross-references the SPUN test set with the raw LIS output. It finds the exact spatial and temporal match, dynamically reprojects the LIS data to the OSGB grid (EPSG:27700), and extracts a perfectly aligned 20m resolution patch.

**Key Features:**
* Uses `rioxarray` and bounding box intersection to efficiently find the correct overlapping LIS tile.
* Reprojects and crops the data to guarantee pixel-perfect alignment with the SPUN evaluation grids.
* Saves the extracted baseline patches as `.zarr` stores for fast, parallel loading in the evaluation scripts.

**Prerequisites:**
You will need the OSGB geopackage to define the spatial bounds, which can be accessed here:
🔗 [OS-British-National-Grids](https://github.com/OrdnanceSurvey/OS-British-National-Grids)

**Example Usage:**

```bash
python -m scripts.patchify_lis \
    --data_root ../data/SPUN_Dataset \
    --lis_dir ../data/raw/lis_fsc \
    --osgb_grid ../data/os_bng_grids.gpkg \
    --output_dir lis_fsc_patches

```

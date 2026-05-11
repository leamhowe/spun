### `scripts/patchify_lis.py`

**Purpose:** Prepares the LIS (Let It Snow) baseline data for direct comparison against SPUN predictions.

Because the raw LIS fractional snow cover products are massive regional GeoTIFFs in their native CRS, they cannot be directly compared to the 1km x 1km SPUN chips. This script automatically cross-references the SPUN test set with your raw LIS data directory. It finds the exact spatial and temporal match, dynamically reprojects the LIS data to the OSGB grid (EPSG:27700), and extracts a perfectly aligned 20m resolution patch.

**Key Features:**

* Uses `rioxarray` and bounding box intersection to efficiently find the correct overlapping LIS tile.
* Reprojects and crops the data to guarantee pixel-perfect alignment with the SPUN evaluation grids.
* Saves the extracted baseline patches as `.zarr` stores for fast, parallel loading in the evaluation scripts.

**Example Usage:**

```bash
python -m scripts.patchify_lis \
    --data_root data/SPUN_Dataset \
    --lis_dir data/raw/lis_fsc \
    --osgb_grid data/os_bng_grids.gpkg \
    --output_dir results/evaluation/lis_fsc_patches

```



OSGB geopackage can be accessed here: https://github.com/OrdnanceSurvey/OS-British-National-Grids
import pandas as pd
import os
import glob
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from tqdm import tqdm
from joblib import Parallel, delayed
from shapely.geometry import box
import warnings
import shutil
import traceback
from collections import defaultdict
import re
from rasterio.enums import Resampling

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SPLIT_CSV_PATH = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
LIS_DATA_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc'
GRID_GPKG_PATH = '/home/users/leamhowe/sensecdt/users/leamhowe/OSGB_data/os_bng_grids/os_bng_grids.gpkg'
OUTPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches'
N_JOBS = 8  

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---

def extract_tile_date_and_id(file_path):
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    tile_date = parts[0]
    tile_id = parts[-1].split('.')[0]
    return tile_date, tile_id

def extract_exact_osgb_patch_bounds(source_xr, bounds, target_res, crs="EPSG:27700"):
    # Unpack bounds tuple directly
    minx, miny, maxx, maxy = bounds
    
    width = int((maxx - minx) / target_res)
    height = int((maxy - miny) / target_res)
    
    x_coords = np.linspace(minx + target_res/2, maxx - target_res/2, width)
    y_coords = np.linspace(maxy - target_res/2, miny + target_res/2, height)

    template = xr.DataArray(
        data=np.zeros((height, width)), 
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x")
    ).rio.write_crs(crs)
    
    output_xr = source_xr.rio.reproject_match(
        template, 
        resampling=Resampling.nearest,
        nodata=255
    )
    return output_xr

def process_single_tile(test_file, lis_candidates, tile_bounds, output_dir):
    # Note: tile_bounds is now a simple tuple (minx, miny, maxx, maxy)
    try:
        tile_date, tile_id = extract_tile_date_and_id(test_file)
        
        output_zarr_name = f"{tile_date}_{tile_id}_LIS.zarr" 
        output_path = os.path.join(output_dir, output_zarr_name)

        # 1. Clean Overwrite
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        if tile_bounds is None:
            return f"failed_no_grid_{tile_id}"
            
        target_geom_box = box(*tile_bounds)

        # 2. Find Best LIS File
        best_snow_path = None
        max_valid_pixels = -1

        for candidate_snow_path in lis_candidates:
            if not candidate_snow_path.endswith('.tif'): continue

            try:
                with rioxarray.open_rasterio(candidate_snow_path, masked=True) as src:
                    src_bounds_box = box(*src.rio.bounds())
                    
                    # Coordinate transform check
                    if src.rio.crs and src.rio.crs.to_string() != "EPSG:27700":
                        # If simple bounds check fails, precise check needed? 
                        # For speed, usually we skip transformation if we trust the metadata, 
                        # but ideally we transform target to src CRS
                        # (Skipping complex transform logic here for speed if standard products)
                        pass
                        
                    if not src_bounds_box.intersects(target_geom_box):
                        continue

                    clipped_check = src.rio.clip([target_geom_box], all_touched=True, drop=True)
                    valid_pixels = clipped_check.count().item()
                    
                    if valid_pixels > max_valid_pixels:
                        max_valid_pixels = valid_pixels
                        best_snow_path = candidate_snow_path
            except Exception:
                continue

        # 3. Extraction
        if best_snow_path and max_valid_pixels > 0:
            lis_data = rioxarray.open_rasterio(best_snow_path, masked=False).squeeze()
            lis_data.rio.write_nodata(255, inplace=True)
            
            lis_50_osgb = extract_exact_osgb_patch_bounds(
                source_xr=lis_data, 
                bounds=tile_bounds, 
                target_res=20
            )

            lis_50_osgb.name = 'lis_snow'
            lis_50_osgb.attrs['source_file'] = best_snow_path
            
            # --- CRITICAL FIXES ---
            # A. Cast to float32 to ensure compatibility
            lis_50_osgb = lis_50_osgb.astype('float32')
            
            # B. WIPE ENCODING (Fixes "Expected integer... Got None")
            lis_50_osgb.encoding = {} 
            
            # C. Chunking (Optional but good for Zarr)
            # lis_50_osgb = lis_50_osgb.chunk({'y': 100, 'x': 100})
            
            lis_50_osgb.to_dataset().to_zarr(output_path, mode='w')
            return "success"
        else:
            return "failed_no_overlap"

    except Exception as e:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        # return traceback so we can see it in the counter keys if it fails
        return f"error_{str(e)}"

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    
    print("Loading split csv...")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()

    print("Indexing LIS files...")
    lis_files = glob.glob(f'{LIS_DATA_DIR}/*/LIS_S2-SNOW-FSC-TOC*.tif') 
    lis_lookup = defaultdict(list)
    for lf in lis_files:
        m = re.search(r'(\d{8})', os.path.basename(lf))
        if m:
            lis_lookup[m.group(1)].append(lf)

    print("Loading OSGB Grid...")
    bng_gdf = gpd.read_file(GRID_GPKG_PATH, layer='1km_grid')
    
    # OPTIMIZATION: Create a lightweight lookup dictionary for bounds
    # This avoids pickling the entire GeoDataFrame to 800 workers
    print("Building Grid Lookup Dictionary...")
    grid_bounds_lookup = {}
    
    # Filter bng_gdf to only the tiles we actually need (optional speedup)
    needed_ids = set([extract_tile_date_and_id(tf)[1] for tf in test_files])
    subset_gdf = bng_gdf[bng_gdf['tile_name'].isin(needed_ids)]
    
    for idx, row in subset_gdf.iterrows():
        # Store bounds as tuple (minx, miny, maxx, maxy)
        grid_bounds_lookup[row['tile_name']] = row.geometry.bounds

    print("Matching test files...")
    tasks = []
    for tf in test_files:
        t_date, t_id = extract_tile_date_and_id(tf)
        if t_date in lis_lookup:
            candidates = lis_lookup[t_date]
            # Get bounds tuple from dict, None if missing
            t_bounds = grid_bounds_lookup.get(t_id, None)
            tasks.append((tf, candidates, t_bounds))

    print(f"Starting processing for {len(tasks)} tiles...")

    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_single_tile)(tf, cands, bounds, OUTPUT_DIR) 
        for tf, cands, bounds in tqdm(tasks)
    )

    from collections import Counter
    counts = Counter(results)
    print("\nProcessing Complete.")
    print(counts)
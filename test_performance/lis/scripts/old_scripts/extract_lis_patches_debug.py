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
N_JOBS = 8  # Adjust based on your HPC allocation

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---

def extract_tile_date_and_id(file_path):
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    tile_date = parts[0]
    tile_id = parts[-1].split('.')[0]
    return tile_date, tile_id

def extract_exact_osgb_patch(source_xr, tile_gdf, target_res, crs="EPSG:27700"):
    minx, miny, maxx, maxy = tile_gdf.geometry.bounds.iloc[0]
    width = int((maxx - minx) / target_res)
    height = int((maxy - miny) / target_res)
    
    half_res = target_res / 2
    x_coords = np.linspace(minx + half_res, maxx - half_res, width)
    y_coords = np.linspace(maxy - half_res, miny + half_res, height)

    # 1. Match Dtype to source (uint8/int16)
    template = xr.DataArray(
        data=np.zeros((height, width), dtype=source_xr.dtype), 
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x")
    ).rio.write_crs(crs)
    
    # 2. Use Nearest Neighbor and Explicit Nodata
    output_xr = source_xr.rio.reproject_match(
        template, 
        resampling=Resampling.nearest,
        nodata=255
    )
    return output_xr

def process_single_tile(test_file, lis_candidates, bng_gdf, output_dir):
    """
    Worker function with Robust Zarr V3 Saving.
    """
    try:
        tile_date, tile_id = extract_tile_date_and_id(test_file)
        
        # 1. Output Check
        output_zarr_name = f"{tile_date}_{tile_id}_LIS.zarr" 
        output_path = os.path.join(output_dir, output_zarr_name)
        if os.path.exists(output_path):
            return "skipped"

        # 2. Geometry Lookup
        if tile_id not in bng_gdf.index:
            return f"failed_no_grid_{tile_id}"
        
        target_tile_row = bng_gdf.loc[[tile_id]]
        target_geom = target_tile_row.geometry.values[0]

        # 3. Candidate Selection
        best_snow_path = None
        max_valid_pixels = -1

        for candidate_snow_path in lis_candidates:
            try:
                # Construct S2 Path
                lis_s2_data_dir = candidate_snow_path.replace('lis_fsc', 's2l2a').split('/LIS_S2')[0]
                b4_filename = f'{lis_s2_data_dir.split("s2l2a/")[-1]}_SRE_B4.tif'
                candidate_s2_path = os.path.join(lis_s2_data_dir, b4_filename)
                
                if not os.path.exists(candidate_s2_path):
                    continue
                
                # Check bounds metadata
                with rioxarray.open_rasterio(candidate_s2_path, masked=True) as src:
                    s2_bounds = src.rio.bounds()
                    s2_box = box(*s2_bounds)
                    
                    if src.rio.crs != bng_gdf.crs:
                        target_series = gpd.GeoSeries([target_geom], crs=bng_gdf.crs)
                        target_poly_s2 = target_series.to_crs(src.rio.crs).iloc[0]
                    else:
                        target_poly_s2 = target_geom
                        
                    if not s2_box.intersects(target_poly_s2):
                        continue

                    clipped_check = src.rio.clip([target_poly_s2], all_touched=True, drop=True)
                    valid_pixels = clipped_check.count().item()
                    
                    if valid_pixels > max_valid_pixels:
                        max_valid_pixels = valid_pixels
                        best_snow_path = candidate_snow_path
            except:
                continue

        # 4. Extraction
        if best_snow_path and max_valid_pixels > 0:
            # FIX A: Raw Open (masked=False)
            lis_data = rioxarray.open_rasterio(best_snow_path, masked=False).squeeze()
            lis_data.rio.write_nodata(255, inplace=True)
            
            # Extract
            lis_50_osgb = extract_exact_osgb_patch(
                source_xr=lis_data, 
                tile_gdf=target_tile_row, 
                target_res=20
            )

            # Assign Metadata
            lis_50_osgb.name = 'lis_snow'
            lis_50_osgb.attrs['crs'] = 'EPSG:27700'
            lis_50_osgb.attrs['resolution'] = 20
            lis_50_osgb.attrs['source_file'] = best_snow_path
            lis_50_osgb.attrs['original_tile_id'] = tile_id
            
            ds = lis_50_osgb.to_dataset()
            
            # --- ZARR V3 CRASH FIX ---
            
            # 1. FORCE COMPUTE: Convert Dask array to Memory (NumPy). 
            # This is essential to stop lazy chunking from crashing Zarr v3.
            ds = ds.compute()
            
            # 2. NUCLEAR WIPE: Remove all encoding that came from the TIFF
            for var in ds.variables:
                ds[var].encoding = {}

            # 3. EXPLICIT ENCODING: Tell Zarr exactly how to chunk the file.
            encoding_opts = {
                'lis_snow': {'chunks': (50, 50)}
            }
            
            ds.to_zarr(output_path, mode='w', encoding=encoding_opts)
            # -------------------------
            
            return "success"
        else:
            return "failed_no_overlap"

    except Exception as e:
        # Print full trace if it crashes again to be sure
        import traceback
        traceback.print_exc()
        return f"error_{e}"

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    
    # 1. Load Split
    print("Loading split csv...")
    df = pd.read_csv(SPLIT_CSV_PATH)
    col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    test_files = df[df[col] == 'Test']['filepath'].tolist()

    # 2. Index LIS Files
    print("Indexing LIS files...")
    lis_files = glob.glob(f'{LIS_DATA_DIR}/*/LIS_S2-SNOW-FSC-TOC*')
    
    lis_lookup = defaultdict(list)
    for lf in lis_files:
        m = re.search(r'(\d{8})', os.path.basename(lf))
        if m:
            lis_lookup[m.group(1)].append(lf)

    # 3. Load Grid
    print("Loading OSGB Grid...")
    bng_gdf = gpd.read_file(GRID_GPKG_PATH, layer='1km_grid')
    if bng_gdf.crs is None: bng_gdf.set_crs(epsg=27700, inplace=True)
    bng_gdf = bng_gdf.set_index('tile_name')

    # 4. Prepare Job List
    print("Matching test files...")
    tasks = []
    
    for tf in test_files:
        t_date, t_id = extract_tile_date_and_id(tf)
        if t_date in lis_lookup:
            candidates = lis_lookup[t_date]
            tasks.append((tf, candidates))

    import random
    random.shuffle(tasks)

    print(f"Starting parallel processing for {len(tasks)} tiles with {N_JOBS} cores...")

    # 5. Run Parallel Loop
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_single_tile)(tf, cands, bng_gdf, OUTPUT_DIR) 
        for tf, cands in tasks
    )

    # 6. Report
    from collections import Counter
    counts = Counter(results)
    print("\nProcessing Complete.")
    print(counts)
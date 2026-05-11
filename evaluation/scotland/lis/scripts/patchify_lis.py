# Save this as debug_lis.py and run it: python debug_lis.py
import pandas as pd
import os
import glob
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from collections import defaultdict
import re
from shapely.geometry import box

# --- USE YOUR EXISTING CONFIG ---
SPLIT_CSV_PATH = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
LIS_DATA_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc'
GRID_GPKG_PATH = '/home/users/leamhowe/sensecdt/users/leamhowe/OSGB_data/os_bng_grids/os_bng_grids.gpkg'
OUTPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/theia/geodes/lis_fsc_patches' 

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    template = xr.DataArray(
        data=np.zeros((height, width)), 
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x")
    ).rio.write_crs(crs)
    output_xr = source_xr.rio.reproject_match(template, nodata=np.nan)
    return output_xr

# --- MAIN DEBUG LOGIC ---
print("Loading split csv...")
df = pd.read_csv(SPLIT_CSV_PATH)
col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
test_files = df[df[col] == 'Test']['filepath'].tolist()

print("Indexing LIS files...")
lis_files = glob.glob(f'{LIS_DATA_DIR}/*/LIS_S2-SNOW-FSC-TOC*')
lis_lookup = defaultdict(list)
for lf in lis_files:
    m = re.search(r'(\d{8})', os.path.basename(lf))
    if m: lis_lookup[m.group(1)].append(lf)

print("Loading OSGB Grid...")
bng_gdf = gpd.read_file(GRID_GPKG_PATH, layer='1km_grid')
if bng_gdf.crs is None: bng_gdf.set_crs(epsg=27700, inplace=True)

# --- THE DEBUG LOOP ---
print("\n--- STARTING DEBUG RUN ---")

count = 0
for test_file in test_files:
    # if count >= 3: break # Stop after 3 attempts
    
    tile_date, tile_id = extract_tile_date_and_id(test_file)
    
    # --- CHECK IF OUTPUT EXISTS START ---
    output_zarr_name = f"{tile_date}_{tile_id}_LIS.zarr" 
    output_path = os.path.join(OUTPUT_DIR, output_zarr_name)
    
    if os.path.exists(output_path):
        print(f"Skipping {tile_id} ({tile_date}): Output already exists.")
        continue
    # --- CHECK IF OUTPUT EXISTS END ---

    print(f"\nProcessing {tile_id} ({tile_date})...")

    # 1. Check Date Match
    if tile_date not in lis_lookup:
        print(f"  -> Skipping: No LIS files for date {tile_date}")
        continue
    
    lis_candidates = lis_lookup[tile_date]
    print(f"  -> Found {len(lis_candidates)} candidates.")

    # 2. Check Geometry
    target_tile_row = bng_gdf[bng_gdf['tile_name'] == tile_id]
    if target_tile_row.empty:
        print("  -> Failed: Tile ID not found in Grid GPKG.")
        continue
    target_geom = target_tile_row.geometry.values[0]

    # 3. Process Candidates
    best_snow_path = None
    
    for candidate_snow_path in lis_candidates:
        try:
            lis_s2_data_dir = candidate_snow_path.replace('lis_fsc', 's2l2a').split('/LIS_S2')[0]
            b4_filename = f'{lis_s2_data_dir.split("s2l2a/")[-1]}_SRE_B4.tif'
            candidate_s2_path = os.path.join(lis_s2_data_dir, b4_filename)
            
            if not os.path.exists(candidate_s2_path):
                print(f"    -> Warning: S2 file not found: {candidate_s2_path}")
                continue
            
            print(f"    -> Checking overlap with: {os.path.basename(candidate_s2_path)}")
            
            with rioxarray.open_rasterio(candidate_s2_path, masked=True) as src:
                s2_bounds = src.rio.bounds()
                s2_box = box(*s2_bounds)
                
                # Check intersection
                if src.rio.crs != bng_gdf.crs:
                    target_series = gpd.GeoSeries([target_geom], crs=bng_gdf.crs)
                    target_poly_s2 = target_series.to_crs(src.rio.crs).iloc[0]
                else:
                    target_poly_s2 = target_geom
                    
                if s2_box.intersects(target_poly_s2):
                    print("       -> INTERSECTION FOUND!")
                    best_snow_path = candidate_snow_path
                    break # Stop looking for this tile
                else:
                    print("       -> No intersection.")
        except Exception as e:
            print(f"    -> Error checking candidate: {e}")

    # 4. Extract and Save
    if best_snow_path:
        print(f"  -> Extracting from {os.path.basename(best_snow_path)}...")
        try:
            lis_data = rioxarray.open_rasterio(best_snow_path, masked=True).squeeze()
            lis_50_osgb = extract_exact_osgb_patch(lis_data, target_tile_row, 20)
            
            # (path definition moved to top of loop, but we use it here)
            
            lis_50_osgb.name = 'lis_snow'
            lis_50_osgb.to_dataset().to_zarr(output_path, mode='w')
            print(f"  -> SUCCESS! Saved to {output_path}")
            count += 1
        except Exception as e:
            print(f"  -> FAILED during extraction: {e}")
    else:
        print("  -> FAILED: No valid overlapping candidate found.")
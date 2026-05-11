import argparse
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

def get_args():
    parser = argparse.ArgumentParser(description="Extract LIS patches corresponding to the SPUN test set.")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the Zenodo dataset')
    parser.add_argument('--split_file', type=str, default='dataset_metadata.csv', help='Name of the metadata CSV')
    parser.add_argument('--lis_dir', type=str, required=True, help='Directory containing source LIS data')
    parser.add_argument('--osgb_grid', type=str, required=True, help='Path to os_bng_grids.gpkg')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted LIS patches')
    return parser.parse_args()

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

# --- MAIN LOGIC ---
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading split csv...")
    split_csv_path = os.path.join(args.data_root, args.split_file)
    if not os.path.exists(split_csv_path):
        raise FileNotFoundError(f"Split file not found: {split_csv_path}")

    df = pd.read_csv(split_csv_path)
    path_col = 'relative_path' if 'relative_path' in df.columns else 'filepath'
    split_col = 'split_cleaned' if 'split_cleaned' in df.columns else 'split'
    
    test_df = df[df[split_col] == 'Test']
    test_files = test_df[path_col].tolist()

    print("Indexing LIS files...")
    lis_files = glob.glob(f'{args.lis_dir}/*/LIS_S2-SNOW-FSC-TOC*')
    lis_lookup = defaultdict(list)
    for lf in lis_files:
        m = re.search(r'(\d{8})', os.path.basename(lf))
        if m: lis_lookup[m.group(1)].append(lf)

    print(f"Loading OSGB Grid from {args.osgb_grid}...")
    bng_gdf = gpd.read_file(args.osgb_grid, layer='1km_grid')
    if bng_gdf.crs is None: 
        bng_gdf.set_crs(epsg=27700, inplace=True)

    # --- THE EXTRACTION LOOP ---
    print("\n--- STARTING EXTRACTION RUN ---")

    count = 0
    for test_file in test_files:
        tile_date, tile_id = extract_tile_date_and_id(test_file)
        
        # --- CHECK IF OUTPUT EXISTS START ---
        output_zarr_name = f"{tile_date}_{tile_id}_LIS.zarr" 
        output_path = os.path.join(args.output_dir, output_zarr_name)
        
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
                
                lis_50_osgb.name = 'lis_snow'
                lis_50_osgb.to_dataset().to_zarr(output_path, mode='w')
                print(f"  -> SUCCESS! Saved to {output_path}")
                count += 1
            except Exception as e:
                print(f"  -> FAILED during extraction: {e}")
        else:
            print("  -> FAILED: No valid overlapping candidate found.")

    print(f"\nExtraction complete. Successfully processed {count} new tiles.")

if __name__ == "__main__":
    main()
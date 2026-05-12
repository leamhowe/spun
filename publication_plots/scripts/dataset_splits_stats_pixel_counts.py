import os
import argparse
import pandas as pd
import numpy as np
import zarr
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def get_args():
    parser = argparse.ArgumentParser(description="Extract and visualize dataset split composition (Macro & Micro)")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of the SPUN Dataset (Zenodo)')
    parser.add_argument('--split_file', type=str, default='dataset_metadata.csv', help='Name of the metadata CSV file')
    parser.add_argument('--osgb_grid', type=str, required=True, help='Path to the OS BNG Geopackage (os_bng_grids.gpkg)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output plot')
    parser.add_argument('--output_name', type=str, default='dataset_composition.png', help='Filename for the output plot')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # ==============================================================================
    # --- STEP 1: LOAD DEFINITIVE SPLITS & GEOMETRIES ---

    split_file_path = os.path.join(args.data_root, args.split_file)
    
    print("Loading the final, cleaned dataset split...")

    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"Could not find metadata at {split_file_path}")

    splits_df = pd.read_csv(split_file_path)

    # Handle the column name change (Zenodo uses 'relative_path', old used 'filepath')
    path_col = 'relative_path' if 'relative_path' in splits_df.columns else 'filepath'
    split_col = 'split_cleaned' if 'split_cleaned' in splits_df.columns else 'split'

    # Create ABSOLUTE paths for Zarr loading by stitching the root and relative path
    splits_df['absolute_path'] = splits_df[path_col].apply(lambda x: os.path.join(args.data_root, x))

    # Extract the tile name correctly
    splits_df['tile_name'] = splits_df[path_col].str.split('/').str[-1].str.split('_').str[1].str.replace('.zarr', '')
    print(f"Loaded exactly {len(splits_df)} tiles from the splits CSV.")

    bng_1km_gdf = gpd.read_file(args.osgb_grid, layer='1km_grid')

    merged_df = splits_df.merge(
        bng_1km_gdf[['tile_name', 'geometry']],
        on='tile_name',
        how='left'
    )
    merged_gdf_cleaned = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=bng_1km_gdf.crs)

    # ==============================================================================
    # --- STEP 2: EXTRACT MACRO (IMAGE) AND MICRO (PIXEL) SNOW STATS ---

    print("Extracting image and pixel-level snow statistics from Zarr files...")

    has_snow_list = []
    snow_pixel_counts = []
    cloud_pixel_counts = []
    bg_pixel_counts = []
    total_pixel_counts = []

    # Let's check the very first path before we even loop!
    first_path = merged_gdf_cleaned['absolute_path'].iloc[0]
    print("\n--- DEBUG INFO ---")
    print(f"Trying to open exact path: {first_path}")
    print(f"Does this file exist on the system?: {os.path.exists(first_path)}")
    print("------------------\n")

    for f in tqdm(merged_gdf_cleaned['absolute_path']):
        # REMOVED the try...except block so Python gives us the exact error!
        if pd.isna(f):
            continue
            
        if not os.path.exists(f):
            raise FileNotFoundError(f"STOPPING: Cannot find the zarr file at:\n{f}\nPlease check if the DATA_ROOT and relative_paths are combining correctly!")

        store = zarr.open(f, mode='r')
        
        # Check if 'label' actually exists in this Zarr store
        if 'label' not in store:
            raise KeyError(f"STOPPING: Found the file, but it doesn't contain a 'label' array. Available arrays: {list(store.array_keys())}")
            
        gt = store['label'][:] 
        
        total_px = gt.size
        
        cloud_px = np.sum(gt == 200)
        snow_px = np.sum((gt > 0) & (gt != 200))
        bg_px = total_px - cloud_px - snow_px
        
        has_snow_list.append(snow_px > 0)
        snow_pixel_counts.append(snow_px)
        cloud_pixel_counts.append(cloud_px)
        bg_pixel_counts.append(bg_px)
        total_pixel_counts.append(total_px)

    merged_gdf_cleaned['has_snow'] = has_snow_list
    merged_gdf_cleaned['snow_pixels'] = snow_pixel_counts
    merged_gdf_cleaned['cloud_pixels'] = cloud_pixel_counts
    merged_gdf_cleaned['bg_pixels'] = bg_pixel_counts
    merged_gdf_cleaned['total_pixels'] = total_pixel_counts

    # ==============================================================================
    # --- STEP 3: CALCULATE AND VISUALIZE TWO-TIERED SPLIT STATISTICS ---

    split_stats = merged_gdf_cleaned.groupby(split_col).agg(
        total_images=('tile_name', 'size'),
        snowy_images=('has_snow', 'sum'),
        total_all_px=('total_pixels', 'sum'),
        total_snow_px=('snow_pixels', 'sum'),
        total_cloud_px=('cloud_pixels', 'sum'),
        total_bg_px=('bg_pixels', 'sum')
    ).reset_index()

    split_stats.rename(columns={split_col: 'split'}, inplace=True)

    split_stats['split'] = pd.Categorical(split_stats['split'], categories=['Test', 'Validation', 'Train'], ordered=True)
    split_stats = split_stats.sort_values('split').reset_index(drop=True)

    split_stats['img_snow_pct'] = (split_stats['snowy_images'] / split_stats['total_images']) * 100
    split_stats['img_nosnow_pct'] = 100.0 - split_stats['img_snow_pct']

    split_stats['px_snow_pct'] = (split_stats['total_snow_px'] / split_stats['total_all_px']) * 100
    split_stats['px_bg_pct'] = (split_stats['total_bg_px'] / split_stats['total_all_px']) * 100
    split_stats['px_cloud_pct'] = (split_stats['total_cloud_px'] / split_stats['total_all_px']) * 100

    print("\n--- Final Cleaned Dataset Statistics (Image & Pixel Level) ---")
    print(split_stats.to_string(index=False))

    # --- Visualization ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Two-Tiered Class Imbalance in SPUN Dataset', fontsize=16, fontweight='bold', y=1.05)

    splits = split_stats['split']

    snow_color = '#00BCD4'   
    bg_color = '#E0E0E0'     

    # Panel A
    ax1.barh(splits, split_stats['img_snow_pct'], color=snow_color, edgecolor='black', label='Contains Snow')
    ax1.barh(splits, split_stats['img_nosnow_pct'], left=split_stats['img_snow_pct'], color=bg_color, edgecolor='black', label='Snow-Free')
    ax1.set_title('A. Macro-Scale: Image-Level Imbalance', fontweight='bold')
    ax1.set_xlabel('Percentage of Images (%)', fontweight='bold')
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_xlim(0, 100)

    for i, (snow_pct, snow_count, total_count) in enumerate(zip(split_stats['img_snow_pct'], split_stats['snowy_images'], split_stats['total_images'])):
        ax1.text(snow_pct / 2, i, f'{snow_pct:.1f}%\n({int(snow_count)})', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
        ax1.text(snow_pct + (100 - snow_pct) / 2, i, f'{100-snow_pct:.1f}%\n({int(total_count - snow_count)})', ha='center', va='center', color='black', fontsize=10)

    # Panel B
    cloud_color = '#FFFFFF' 
    cloud_edge = '#9E9E9E'  
    bg_color = '#BDBDBD'    
    snow_color = '#00BCD4'  

    ax2.barh(splits, split_stats['px_snow_pct'], color=snow_color, edgecolor='black', label='Snow')
    ax2.barh(splits, split_stats['px_bg_pct'], left=split_stats['px_snow_pct'], color=bg_color, edgecolor='black', label='Clear Ground')

    left_cloud = split_stats['px_snow_pct'] + split_stats['px_bg_pct']
    ax2.barh(splits, split_stats['px_cloud_pct'], left=left_cloud, color=cloud_color, edgecolor=cloud_edge, hatch='///', label='Cloud Cover')

    ax2.set_title('B. Micro-Scale: Pixel-Level Imbalance', fontweight='bold')
    ax2.set_xlabel('Percentage of Total Pixels (%)', fontweight='bold')
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_xlim(0, 100)

    for i, (snow_pct, bg_pct, cloud_pct) in enumerate(zip(split_stats['px_snow_pct'], split_stats['px_bg_pct'], split_stats['px_cloud_pct'])):
        x_snow = snow_pct + 2 if snow_pct < 5 else snow_pct / 2
        align_snow = 'left' if snow_pct < 5 else 'center'
        ax2.text(x_snow, i, f'{snow_pct:.2f}%', ha=align_snow, va='center', color='black', fontsize=10, fontweight='bold')
        ax2.text(snow_pct + (bg_pct / 2), i, f'{bg_pct:.1f}%', ha='center', va='center', color='black', fontsize=10)
        ax2.text(snow_pct + bg_pct + (cloud_pct / 2), i, f'{cloud_pct:.1f}%', ha='center', va='center', color='black', fontsize=10)

    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)

    plt.tight_layout()

    # Save locally to the specified output directory
    output_path = os.path.join(args.output_dir, args.output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {output_path}")

    plt.close()

if __name__ == "__main__":
    main()
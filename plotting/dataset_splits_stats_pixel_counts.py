import os
import pandas as pd
import numpy as np
import zarr
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ==============================================================================
# --- STEP 1: LOAD DEFINITIVE SPLITS & GEOMETRIES ---

print("Loading the final, cleaned dataset split...")
split_file_path = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
splits_df = pd.read_csv(split_file_path)

splits_df['tile_name'] = splits_df['filepath'].str.split('/').str[-1].str.split('_').str[1].str.replace('.zarr', '')
print(f"Loaded exactly {len(splits_df)} tiles from the splits CSV.")

os_bng_path = '~/sensecdt/users/leamhowe/OSGB_data/os_bng_grids/os_bng_grids.gpkg'
bng_1km_gdf = gpd.read_file(os_bng_path, layer='1km_grid')

merged_df = splits_df.merge(
    bng_1km_gdf[['tile_name', 'geometry']],
    on='tile_name',
    how='left'
)
merged_gdf_cleaned = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=bng_1km_gdf.crs)


# ==============================================================================
# --- STEP 2: EXTRACT MACRO (IMAGE) AND MICRO (PIXEL) SNOW STATS ---

print("Extracting image and pixel-level snow statistics from Zarr files...")

# 1. Initialize empty lists
has_snow_list = []
snow_pixel_counts = []
cloud_pixel_counts = []
bg_pixel_counts = []
total_pixel_counts = []

for f in tqdm(merged_gdf_cleaned['filepath']):
    # Safety check if filepath is missing
    if pd.isna(f) or not os.path.exists(f):
        has_snow_list.append(False)
        snow_pixel_counts.append(0)
        cloud_pixel_counts.append(0)
        bg_pixel_counts.append(0)
        total_pixel_counts.append(0)
        continue
        
    try:
        store = zarr.open(f, mode='r')
        gt = store['label'][:] # Load mask data
        
        total_px = gt.size
        
        # Calculate pixel distributions
        cloud_px = np.sum(gt == 200)
        snow_px = np.sum((gt > 0) & (gt != 200))
        bg_px = total_px - cloud_px - snow_px
        
        # EXACTLY ONE APPEND BLOCK PER IMAGE
        has_snow_list.append(snow_px > 0)
        snow_pixel_counts.append(snow_px)
        cloud_pixel_counts.append(cloud_px)
        bg_pixel_counts.append(bg_px)
        total_pixel_counts.append(total_px)
        
    except Exception as e:
        # Handle unloadable files gracefully by appending zeros
        has_snow_list.append(False)
        snow_pixel_counts.append(0)
        cloud_pixel_counts.append(0)
        bg_pixel_counts.append(0)
        total_pixel_counts.append(0)

# Add the correctly sized lists back to the dataframe
merged_gdf_cleaned['has_snow'] = has_snow_list
merged_gdf_cleaned['snow_pixels'] = snow_pixel_counts
merged_gdf_cleaned['cloud_pixels'] = cloud_pixel_counts
merged_gdf_cleaned['bg_pixels'] = bg_pixel_counts
merged_gdf_cleaned['total_pixels'] = total_pixel_counts


# ==============================================================================
# --- STEP 3: CALCULATE AND VISUALIZE TWO-TIERED SPLIT STATISTICS ---

# Aggregate counts
split_stats = merged_gdf_cleaned.groupby('split_cleaned').agg(
    total_images=('tile_name', 'size'),
    snowy_images=('has_snow', 'sum'),
    total_all_px=('total_pixels', 'sum'),
    total_snow_px=('snow_pixels', 'sum'),
    total_cloud_px=('cloud_pixels', 'sum'),
    total_bg_px=('bg_pixels', 'sum')
).reset_index()


split_stats.rename(columns={'split_cleaned': 'split'}, inplace=True)

# Order the splits: Train -> Validation -> Test (Reversed for bottom-to-top horizontal plotting)
split_stats['split'] = pd.Categorical(split_stats['split'], categories=['Test', 'Validation', 'Train'], ordered=True)
split_stats = split_stats.sort_values('split').reset_index(drop=True)

# Calculate Macro (Image) Percentages
split_stats['img_snow_pct'] = (split_stats['snowy_images'] / split_stats['total_images']) * 100
split_stats['img_nosnow_pct'] = 100.0 - split_stats['img_snow_pct']

# Calculate Micro (Pixel) Percentages (Now 3 classes)
split_stats['px_snow_pct'] = (split_stats['total_snow_px'] / split_stats['total_all_px']) * 100
split_stats['px_bg_pct'] = (split_stats['total_bg_px'] / split_stats['total_all_px']) * 100
split_stats['px_cloud_pct'] = (split_stats['total_cloud_px'] / split_stats['total_all_px']) * 100

print("\n--- Final Cleaned Dataset Statistics (Image & Pixel Level) ---")
print(split_stats.to_string(index=False))

# --- Visualization (Two-Panel Horizontal Stacked Bar) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Two-Tiered Class Imbalance in SPUN Dataset', fontsize=16, fontweight='bold', y=1.05)

splits = split_stats['split']

# Design choice: Use semantic colors instead of split colors to highlight the imbalance
snow_color = '#00BCD4'   # Cyan
bg_color = '#E0E0E0'     # Light Grey

# --- Panel A: Image-Level Imbalance ---
ax1.barh(splits, split_stats['img_snow_pct'], color=snow_color, edgecolor='black', label='Contains Snow')
ax1.barh(splits, split_stats['img_nosnow_pct'], left=split_stats['img_snow_pct'], color=bg_color, edgecolor='black', label='Snow-Free')

ax1.set_title('A. Macro-Scale: Image-Level Imbalance', fontweight='bold')
ax1.set_xlabel('Percentage of Images (%)', fontweight='bold')
ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
ax1.set_xlim(0, 100)

# Add text labels inside bars for Panel A
for i, (snow_pct, snow_count, total_count) in enumerate(zip(split_stats['img_snow_pct'], split_stats['snowy_images'], split_stats['total_images'])):
    # Label the snow section
    ax1.text(snow_pct / 2, i, f'{snow_pct:.1f}%\n({int(snow_count)})', ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    # Label the background section
    ax1.text(snow_pct + (100 - snow_pct) / 2, i, f'{100-snow_pct:.1f}%\n({int(total_count - snow_count)})', ha='center', va='center', color='black', fontsize=10)

# --- Panel B: Pixel-Level Imbalance (3 Classes) ---
cloud_color = '#FFFFFF' # White
cloud_edge = '#9E9E9E'  # Darker grey edge to make it visible
bg_color = '#BDBDBD'    # Solid grey for clear ground
snow_color = '#00BCD4'  # Cyan for snow

# 1. Plot Snow
ax2.barh(splits, split_stats['px_snow_pct'], color=snow_color, edgecolor='black', label='Snow')

# 2. Plot Clear Ground (Stacked on top of Snow)
ax2.barh(splits, split_stats['px_bg_pct'], left=split_stats['px_snow_pct'], 
         color=bg_color, edgecolor='black', label='Clear Ground')

# 3. Plot Clouds (Stacked on top of Snow + Ground)
left_cloud = split_stats['px_snow_pct'] + split_stats['px_bg_pct']
ax2.barh(splits, split_stats['px_cloud_pct'], left=left_cloud, 
         color=cloud_color, edgecolor=cloud_edge, hatch='///', label='Cloud Cover')

ax2.set_title('B. Micro-Scale: Pixel-Level Imbalance', fontweight='bold')
ax2.set_xlabel('Percentage of Total Pixels (%)', fontweight='bold')
ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
ax2.set_xlim(0, 100)

# Add text labels inside bars for Panel B (Adjusted for 3 classes)
for i, (snow_pct, bg_pct, cloud_pct) in enumerate(zip(split_stats['px_snow_pct'], split_stats['px_bg_pct'], split_stats['px_cloud_pct'])):
    
    # Snow label (Place outside if it's super tiny)
    x_snow = snow_pct + 2 if snow_pct < 5 else snow_pct / 2
    align_snow = 'left' if snow_pct < 5 else 'center'
    ax2.text(x_snow, i, f'{snow_pct:.2f}%', ha=align_snow, va='center', color='black', fontsize=10, fontweight='bold')
    
    # Ground label
    ax2.text(snow_pct + (bg_pct / 2), i, f'{bg_pct:.1f}%', ha='center', va='center', color='black', fontsize=10)
    
    # Cloud label
    ax2.text(snow_pct + bg_pct + (cloud_pct / 2), i, f'{cloud_pct:.1f}%', ha='center', va='center', color='black', fontsize=10)

# Add Legends
ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)

plt.tight_layout()
plt.savefig('two_tiered_imbalance.png', dpi=300, bbox_inches='tight')
plt.show()
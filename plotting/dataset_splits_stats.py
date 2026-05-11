import os
import pandas as pd
import numpy as np
import zarr
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==============================================================================
# --- STEP 1: LOAD DEFINITIVE SPLITS & GEOMETRIES ---

print("Loading the final, cleaned dataset split...")
split_file_path = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
splits_df = pd.read_csv(split_file_path)

# Extract tile_name directly from the filepath already present in your CSV
splits_df['tile_name'] = splits_df['filepath'].str.split('/').str[-1].str.split('_').str[1].str.replace('.zarr', '')

print(f"Loaded exactly {len(splits_df)} tiles from the splits CSV.")

# Load the 1km OSGB grid
os_bng_path = '~/sensecdt/users/leamhowe/OSGB_data/os_bng_grids/os_bng_grids.gpkg'
bng_1km_gdf = gpd.read_file(os_bng_path, layer='1km_grid')

# Merge the geometries directly onto your splits_df
# By using a 'left' merge with splits_df on the left, you guarantee exactly 2455 rows
merged_df = splits_df.merge(
    bng_1km_gdf[['tile_name', 'geometry']],
    on='tile_name',
    how='left'
)

# Convert back to a proper GeoDataFrame
merged_gdf_cleaned = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=bng_1km_gdf.crs)


# ==============================================================================
# --- STEP 2: CHECK ZARR FILES FOR SNOW (Only 2455 files now) ---

print("Checking Zarr files for snow presence (this will be much faster now)...")
has_snow_list = []

for f in tqdm(merged_gdf_cleaned['filepath']):
    # Safety check if filepath is missing
    if pd.isna(f) or not os.path.exists(f):
        has_snow_list.append(False)
        continue
        
    try:
        store = zarr.open(f, mode='r')
        gt = store['label'][:] # Load mask data
        # Check if there's any snow (gt > 0) excluding no-data/clouds (gt == 200)
        has_snow = np.any((gt > 0) & (gt != 200))
        has_snow_list.append(has_snow)
    except Exception as e:
        # Handle unloadable files gracefully
        has_snow_list.append(False)

# Add the boolean list back to the dataframe
merged_gdf_cleaned['has_snow'] = has_snow_list

total_snowy = merged_gdf_cleaned['has_snow'].sum()
print(f"\nSnow extraction complete: Found {total_snowy} snowy tiles out of {len(merged_gdf_cleaned)} total.")


# ==============================================================================
# --- STEP 3: CALCULATE AND VISUALIZE SPLIT STATISTICS ---

# Group by the 'split_cleaned' column
split_stats = merged_gdf_cleaned.groupby('split_cleaned').agg(
    tile_count=('tile_name', 'size'),
    snowy_count=('has_snow', 'sum')
).reset_index()

split_stats.rename(columns={'split_cleaned': 'split'}, inplace=True)

# Calculate percentages
total_tiles = len(merged_gdf_cleaned)
split_stats['snow_percentage'] = (split_stats['snowy_count'] / split_stats['tile_count']) * 100
split_stats['dataset_percentage'] = (split_stats['tile_count'] / total_tiles) * 100
split_stats['non_snowy_count'] = split_stats['tile_count'] - split_stats['snowy_count']

# Order the splits logically: Train -> Validation -> Test
split_stats['split'] = pd.Categorical(split_stats['split'], categories=['Train', 'Validation', 'Test'], ordered=True)
split_stats = split_stats.sort_values('split').reset_index(drop=True)

print("\n--- Final Cleaned Dataset Split Statistics ---")
print(split_stats.to_string(index=False))

# --- Visualization ---
color_map = {
    'Train': '#D81B60',      # Vivid Pink
    'Validation': '#1E88E5', # Bright Blue
    'Test': '#FFC107'        # Amber
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Visualization of Cleaned Dataset Split Statistics', fontsize=16, fontweight='bold', y=1.02)

splits = split_stats['split']
non_snowy = split_stats['non_snowy_count']
snowy = split_stats['snowy_count']
percentages = split_stats['snow_percentage']

bar_colors = [color_map[s] for s in splits]

# Plot 1: Stacked Bar Chart
ax1.bar(splits, non_snowy, color=bar_colors, alpha=0.3, edgecolor='black', linewidth=1, label='Non-Snowy')
ax1.bar(splits, snowy, bottom=non_snowy, color=bar_colors, alpha=1.0, edgecolor='black', linewidth=1, label='Snowy')

ax1.set_title('Tile Count per Split', fontweight='bold')
ax1.set_xlabel('Dataset Split', fontweight='bold')
ax1.set_ylabel('Number of Tiles', fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='grey', edgecolor='black', alpha=1.0, label='Snowy Tiles'),
    Patch(facecolor='grey', edgecolor='black', alpha=0.3, label='Non-Snowy Tiles')
]
ax1.legend(handles=legend_elements, loc='upper right')

for i, (total, snow) in enumerate(zip(split_stats['tile_count'], snowy)):
    y_offset = max(split_stats['tile_count']) * 0.02 
    ax1.text(i, total + y_offset, f'Total: {int(total)}\n(Snow: {int(snow)})',
             ha='center', va='bottom', fontsize=10)
    
ax1.set_ylim(0, max(split_stats['tile_count']) * 1.15)

# Plot 2: Percentage Chart
bars2 = ax2.bar(splits, percentages, color=bar_colors, edgecolor='black', linewidth=1)
ax2.set_title('Percentage of Snowy Tiles per Split', fontweight='bold')
ax2.set_xlabel('Dataset Split', fontweight='bold')
ax2.set_ylabel('Snowy Tiles (%)', fontweight='bold')

ax2.set_ylim(0, max(100, percentages.max() * 1.1)) 

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 2, f'{height:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('split_statistics_cleaned.png', dpi=300, bbox_inches='tight')
plt.show()
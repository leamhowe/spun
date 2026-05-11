import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch  # <-- NEW IMPORT
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

# ==============================================================================

# --- STEP 1: LOAD AND PREPARE THE FINAL CLEANED SPLIT DATA ---

import os
import glob

path = '/home/users/leamhowe/sensecdt/1_DETECT/data/s2_training_scf'
path_2 = '/home/users/leamhowe/sensecdt/1_DETECT/data/s2_training_scf/original_training_set'

def glob_from_both(pattern):
    # search both paths and return sorted unique results
    a = glob.glob(f'{path}/{pattern}')
    b = glob.glob(f'{path_2}/{pattern}')
    return sorted(set(a + b))

# total number of files
all_files = glob_from_both('*.zarr')
len(all_files)

my_files_df = pd.DataFrame({'filepath': all_files})

def extract_grid_ref(file_path):
    tile_name = os.path.basename(file_path).split('_')[-1].split('.')[0]
    return tile_name

# Apply the function to create a new column
my_files_df['tile_name'] = my_files_df['filepath'].apply(extract_grid_ref)

os_bng_path = '~/sensecdt/OSGB_data/os_bng_grids/os_bng_grids.gpkg'
bng_1km_gdf = gpd.read_file(os_bng_path,layer='1km_grid')

# read in osgb 100km grid for context
bng_100km_gdf = gpd.read_file(os_bng_path, layer='100km_grid')

print("Loading the final, cleaned dataset split...")

# This is the definitive file created by the reassignment script
split_file_path = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'
splits_df = pd.read_csv(split_file_path)

# Create a 'tile_name' column from the filepath to allow merging
splits_df['tile_name'] = splits_df['filepath'].str.split('/').str[-1].str.split('_').str[1].str.replace('.zarr', '')

merged_df = my_files_df.merge(
    bng_1km_gdf[['tile_name', 'geometry']],
    on='tile_name',
    how='left'   # 'left' keeps all your file rows; use 'inner' if you only want matches
)

merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=bng_1km_gdf.crs)

# Merge the split info with your main GeoDataFrame containing the 1km geometries
merged_gdf_cleaned = merged_gdf.merge(splits_df[['tile_name', 'split_cleaned']], on='tile_name', how='left')
merged_gdf_cleaned = merged_gdf_cleaned.dropna(subset=['split_cleaned'])


# ==============================================================================
# --- STEP 2: PLOTTING ---

def add_scale_bar_osgb(ax, length_km, location_frac=(0.5, 0.05), text_offset_m=5000):
    xlim = ax.get_xlim() 
    ylim = ax.get_ylim()
    start_x = xlim[0] + (xlim[1] - xlim[0]) * location_frac[0]
    start_y = ylim[0] + (ylim[1] - ylim[0]) * location_frac[1]
    ax.plot([start_x, start_x + length_km * 1000], [start_y, start_y],
            color='black', linewidth=3, transform=ax.projection, zorder=10)
    ax.text(start_x, start_y - text_offset_m, '0 km',
            ha='center', va='top', transform=ax.projection, fontsize=10, zorder=10)
    ax.text(start_x + length_km * 1000, start_y - text_offset_m, f'{length_km} km',
            ha='center', va='top', transform=ax.projection, fontsize=10, zorder=10)

def add_north_arrow_stylized(ax, location=(0.95, 0.95), size=0.08):
    x, y = location
    north_triangle = mpatches.Polygon([[x, y + size], [x - size*0.3, y], [x + size*0.3, y]],
                                      facecolor='black', edgecolor='black', transform=ax.transAxes, zorder=20)
    south_triangle = mpatches.Polygon([[x, y - size], [x - size*0.3, y], [x + size*0.3, y]],
                                      facecolor='white', edgecolor='black', transform=ax.transAxes, zorder=20)
    ax.add_patch(north_triangle)
    ax.add_patch(south_triangle)
    ax.text(x, y + size * 1.1, 'N', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=12, fontweight='bold', zorder=21)

# Enable KML driver for Geopandas
import fiona
fiona.drvsupport.supported_drivers['libkml'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# Define map projection 
projection = ccrs.OSGB()

# ==============================================================================
# --- Create figure with 2 subplots (1 row, 2 columns) ---
fig, (ax, ax_context) = plt.subplots(
    1, 2, 
    figsize=(14, 12), 
    gridspec_kw={'width_ratios': [4, 1.5], 'wspace': 0.05}, 
    subplot_kw={'projection': projection}
)
# ==============================================================================

# Define the main map extent
buffer = 20000
total_bounds_1km = merged_gdf_cleaned.total_bounds
map_extent = [total_bounds_1km[0] - buffer, total_bounds_1km[2] + buffer,
              total_bounds_1km[1] - buffer, total_bounds_1km[3] + buffer]
ax.set_extent(map_extent, crs=projection)

# Add geographic features and background terrain to MAIN map
ax.add_feature(cfeature.LAND.with_scale('10m'), zorder=0, facecolor='#EBEDEF')
ax.add_feature(cfeature.LAKES.with_scale('10m'), zorder=1, facecolor='#D6EAF8')

class EsriTerrain(cimgt.GoogleTiles):
    def _image_url(self, tile):
        x, y, z = tile
        return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}"

esri_terrain = EsriTerrain()
ax.add_image(esri_terrain, 9)

# --- Plot the 100km BNG grid boxes and labels (INTEGRATED) ---
map_extent_polygon = box(map_extent[0], map_extent[2], map_extent[1], map_extent[3])
relevant_100km_grid = bng_100km_gdf[bng_100km_gdf.intersects(map_extent_polygon)]

ax.add_geometries(relevant_100km_grid['geometry'], crs=projection, facecolor='none', edgecolor='gray', linewidth=1, linestyle='-', alpha=0.8, zorder=3)

manual_100km_tiles = ['NO', 'NJ'] 
label_locations_100km = {
    'NO': (320000, 850000), 
    'NJ': (320000, 750000)  
}

for idx, row in relevant_100km_grid.iterrows():
    tile_name = row['tile_name']
    if map_extent_polygon.contains(row.geometry.centroid) and tile_name not in manual_100km_tiles:
        label_locations_100km[tile_name] = (row.geometry.centroid.x, row.geometry.centroid.y)

for tile_name, (x, y) in label_locations_100km.items():
    ax.text(x, y, tile_name, 
            horizontalalignment='center', verticalalignment='center', 
            transform=projection, fontsize=12, fontweight='bold', 
            color='darkslategrey', zorder=4)

# --- PLOT AND LABEL ONLY THE S2 TILES USED IN THE DATASET ---
dataset_s2_tiles_with_T = splits_df['s2_tile_id'].unique().tolist()
dataset_s2_tiles_for_kml = [tile_id[1:] for tile_id in dataset_s2_tiles_with_T]

s2_kml_path = '/home/users/leamhowe/sensecdt/1_DETECT/data/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml'
s2_tiles_gdf = gpd.read_file(s2_kml_path, driver='KML')
s2_tiles_osgb = s2_tiles_gdf.to_crs(projection)
relevant_s2_tiles = s2_tiles_osgb[s2_tiles_osgb['Name'].isin(dataset_s2_tiles_for_kml)]

if not relevant_s2_tiles.empty:
    ax.add_geometries(relevant_s2_tiles['geometry'], crs=projection,
                          facecolor='none', edgecolor='red', linewidth=1.5,
                          linestyle='--', alpha=0.6, zorder=10)

    tiles_to_label_list = ['29VPD', '30VUJ', '30VVJ', '30VUH', '30VVH']
    good_tile_names = ['30VUJ', '30VVJ']
    good_tiles_gdf = relevant_s2_tiles[relevant_s2_tiles['Name'].isin(good_tile_names)]
    
    label_locations = {
        row['Name']: (row.geometry.centroid.x, row.geometry.centroid.y)
        for idx, row in good_tiles_gdf.iterrows()
    }

    try:
        vpd_centroid = relevant_s2_tiles[relevant_s2_tiles['Name'] == '29VPD'].geometry.iloc[0].centroid
        label_locations['29VPD'] = (vpd_centroid.x+10000, vpd_centroid.y - 10000) 
    except IndexError:
        pass

    vuh_centroid = relevant_s2_tiles[relevant_s2_tiles['Name'] == '30VUH'].geometry.iloc[0].centroid
    label_locations['30VUH'] = (vuh_centroid.x, 730000) 
    vvh_centroid = relevant_s2_tiles[relevant_s2_tiles['Name'] == '30VVH'].geometry.iloc[0].centroid
    label_locations['30VVH'] = (vvh_centroid.x, 730000) 

    for tile_name, (x, y) in label_locations.items():
        rotation_angle = 0 
        if tile_name == '29VPD':
            rotation_angle = 354 

        ax.text(x, y, tile_name,
                horizontalalignment='center', verticalalignment='center',
                transform=projection, fontsize=14, fontweight='bold', color='red',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'), 
                zorder=11, rotation=rotation_angle)

# --- Plot 1km tiles by their FINAL cleaned split ---
color_map = {
    'Train': '#D81B60',      
    'Validation': '#1E88E5', 
    'Test': '#FFC107'        
}

for split_group, group_color in color_map.items():
    group_gdf = merged_gdf_cleaned[merged_gdf_cleaned['split_cleaned'] == split_group]
    if not group_gdf.empty:
        ax.add_geometries(group_gdf['geometry'], crs=projection, facecolor=group_color, edgecolor='black', linewidth=0.5, alpha=0.8, zorder=5)

# --- ADD HIGHLIGHT BOXES ---
box_a_coords = [205000, 742000, 233000, 780000]
box_b_coords = [285000, 785000, 325000, 815000]
box_a_geom = box(box_a_coords[0], box_a_coords[1], box_a_coords[2], box_a_coords[3])
box_b_geom = box(box_b_coords[0], box_b_coords[1], box_b_coords[2], box_b_coords[3])
highlight_boxes_gdf = gpd.GeoDataFrame({'label': ['A', 'B']}, geometry=[box_a_geom, box_b_geom], crs="EPSG:27700")
ax.add_geometries(highlight_boxes_gdf['geometry'], crs=projection, facecolor='none', edgecolor='#004D40', linewidth=2, linestyle='--', zorder=15)

# --- Create combined legend ---
legend_patches = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
s2_grid_line = Line2D([0], [0], label='Sentinel-2 UTM Tile Grid', color='red', linestyle='--', linewidth=1.5,)
legend_patches.append(s2_grid_line)
highlight_line = Line2D([0], [0], label='Original Manual Labels', color='#004D40', linestyle='--', linewidth=1.5)
legend_patches.append(highlight_line)

leg = ax.legend(handles=legend_patches, 
                loc='upper right', 
                bbox_to_anchor=(1.0, 1.0),
                bbox_transform=ax.transAxes, 
                frameon=True, framealpha=0.9)
leg.set_zorder(20) 

# --- Add cartographic elements ---
gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.x_inline = False; gl.y_inline = False
gl.xlabel_style = {'size': 9}; gl.ylabel_style = {'size': 9}

add_scale_bar_osgb(ax, 50, location_frac=(0.735, 0.79)) 
add_north_arrow_stylized(ax, location=(0.68, 0.9), size=0.05) 
ax.set_title('Scotland study region and SCF dataset splits', fontsize=16, fontweight='bold', pad=20)


# ==============================================================================
# --- 8. CONTEXT MAP (RIGHT PLOT - SCOTLAND ONLY) ---
scotland_path = './LSCMAP_SCOTLAND.geojson'

if os.path.exists(scotland_path):
    scot_gdf = gpd.read_file(scotland_path)
    if scot_gdf.crs != "EPSG:27700":
        scot_gdf = scot_gdf.to_crs(epsg=27700)
    
    scot_outline = scot_gdf.dissolve()
    
    # Minimalist greyscale map
    scot_outline.plot(ax=ax_context, facecolor='#F2F2F2', edgecolor='black', linewidth=0.5, zorder=5)
    
    scot_bounds = scot_outline.total_bounds
    pad = 15000 
    ax_context.set_extent(
        [scot_bounds[0] - pad, scot_bounds[2] + pad, 
         scot_bounds[1] - pad, scot_bounds[3] + pad], 
        crs=projection
    )

    # Plot the Bounding Box in minimalist style
    bbox_gdf = gpd.GeoDataFrame({'geometry': [map_extent_polygon]}, crs="EPSG:27700")
    bbox_gdf.plot(ax=ax_context, facecolor='#555555', alpha=0.35, edgecolor='black', linewidth=1.5, zorder=10)

    # Bottom Title
    # ax_context.text(0.5, -0.05, 'Scotland', 
    #                 transform=ax_context.transAxes, 
    #                 fontsize=14, fontweight='bold', color='#333333', 
    #                 horizontalalignment='center', verticalalignment='top')
    ax_context.set_axis_off()

    # --- 9. DRAW CONNECTION LINES ---
    # Top Line: From Top-Right of Main Map to Top-Left of Context Bounding Box
    con1 = ConnectionPatch(
        xyA=(1.0, 1.0), coordsA=ax.transAxes,                   # Top-right of main ax
        xyB=(map_extent[0], map_extent[3]), coordsB=ax_context.transData,  # Top-left of context bbox
        axesA=ax, axesB=ax_context,
        color="black", linestyle=":", linewidth=1.2, alpha=0.6, zorder=15
    )
    
    # Bottom Line: From Bottom-Right of Main Map to Bottom-Left of Context Bounding Box
    con2 = ConnectionPatch(
        xyA=(1.0, 0.0), coordsA=ax.transAxes,                   # Bottom-right of main ax
        xyB=(map_extent[0], map_extent[2]), coordsB=ax_context.transData,  # Bottom-left of context bbox
        axesA=ax, axesB=ax_context,
        color="black", linestyle=":", linewidth=1.2, alpha=0.6, zorder=15
    )
    
    fig.add_artist(con1)
    fig.add_artist(con2)

else:
    print("Warning: scotland_path not found. Context map could not be rendered.")
    ax_context.set_axis_off()
# ==============================================================================


# --- Save the figure ---
output_filename = 'fig3_context_map_v6.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"\nFinal map with cleaned S2 tile split saved as {output_filename}")
plt.show()
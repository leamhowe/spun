import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os

# --- MOCK DATA FOR STANDALONE TESTING ---
map_extent_polygon = box(200000, 720000, 350000, 850000) 
projection = ccrs.OSGB()

# Create a standalone figure just for the context map
fig, ax_context = plt.subplots(figsize=(6, 8), subplot_kw={'projection': projection})

# ------------------------------------------------------------------------------
# CONTEXT MAP (SCOTLAND)
# ------------------------------------------------------------------------------
scotland_path = './LSCMAP_SCOTLAND.geojson'

if os.path.exists(scotland_path):
    scot_gdf = gpd.read_file(scotland_path)
    if scot_gdf.crs != "EPSG:27700":
        scot_gdf = scot_gdf.to_crs(epsg=27700)
    
    scot_bounds = scot_gdf.total_bounds
    pad = 15000
    ax_context.set_extent([scot_bounds[0] - pad, scot_bounds[2] + pad, 
                           scot_bounds[1] - pad, scot_bounds[3] + pad], crs=projection)

    # Plot simple Scotland outline (Clean light grey fill, crisp thin black edge)
    scot_outline = scot_gdf.dissolve()
    scot_outline.plot(ax=ax_context, facecolor='#F2F2F2', edgecolor='black', linewidth=0.5, zorder=5)

    # Plot Bounding Box in Context Map (Minimalist Greyscale)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [map_extent_polygon]}, crs="EPSG:27700")
    bbox_gdf.plot(ax=ax_context, 
        facecolor='#555555',  # Dark grey fill 
        alpha=0.35,           # Semi-transparent
        edgecolor='black',    # Crisp black border
        linewidth=1.5, 
        zorder=10)

    # --- REGIONAL TEXT LABELS (Optional) ---
    # (Updated to match the darker/minimalist theme if you decide to uncomment)
    # ax_context.text(140000, 790000, 'WEST', 
    #                 horizontalalignment='center', verticalalignment='center',
    #                 transform=projection, fontsize=14, fontweight='bold', 
    #                 color='#333333', alpha=0.8, zorder=15)
    
    # ax_context.text(360000, 790000, 'EAST', 
    #                 horizontalalignment='center', verticalalignment='center',
    #                 transform=projection, fontsize=14, fontweight='bold', 
    #                 color='#333333', alpha=0.8, zorder=15)

    ax_context.set_axis_off()
    
    # Clean bottom title using relative axis coordinates
    ax_context.text(0.5, -0.05, 'Scotland', 
                    transform=ax_context.transAxes, 
                    fontsize=14, 
                    fontweight='bold', 
                    color='#333333',  # Soft black 
                    horizontalalignment='center', 
                    verticalalignment='top')

else:
    print("Warning: scotland_path not found. Using dummy polygon to show layout.")
    ax_context.set_extent([100000, 450000, 500000, 1000000], crs=projection)
    
    bbox_gdf = gpd.GeoDataFrame({'geometry': [map_extent_polygon]}, crs="EPSG:27700")
    bbox_gdf.plot(ax=ax_context, facecolor='#555555', alpha=0.35, edgecolor='black', linewidth=1.5, zorder=10)
    
    ax_context.set_axis_off()
    ax_context.text(0.5, -0.05, 'Scotland', transform=ax_context.transAxes, fontsize=14, fontweight='bold', color='#333333', horizontalalignment='center', verticalalignment='top')

plt.savefig('context_map_test.png', dpi=300, bbox_inches='tight')
plt.show()
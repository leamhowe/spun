import numpy as np
import xarray as xr
import rioxarray
from scipy.ndimage import uniform_filter

def extract_exact_osgb_patch(source_xr, tile_gdf, target_res, crs="EPSG:27700"):
    """
    Extracts a patch from source_xr (UTM) that EXACTLY matches the 
    OSGB tile geometry at the specified resolution.
    """
    minx, miny, maxx, maxy = tile_gdf.geometry.bounds.iloc[0]
    
    # Calculate dimensions
    width = int((maxx - minx) / target_res)
    height = int((maxy - miny) / target_res)
    
    # Create Target Coordinates (North -> South for Y)
    half_res = target_res / 2
    x_coords = np.linspace(minx + half_res, maxx - half_res, width)
    y_coords = np.linspace(maxy - half_res, miny + half_res, height) # Descending Y

    # Create Template
    template = xr.DataArray(
        data=np.zeros((height, width)), 
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x")
    ).rio.write_crs(crs)
    
    # Warping
    output_xr = source_xr.rio.reproject_match(template)
    return output_xr

def downscale_label_masked(arr_100, factor=2, invalid_val=200):
    """
    Downscales a label array (e.g. 100x100 -> 50x50) while ignoring invalid values.
    Returns the downscaled array.
    """
    # Create mask (True where valid)
    valid_mask = (arr_100 != invalid_val)
    
    # Mask data (set invalid to NaN for calculation)
    arr_masked = np.where(arr_100 == invalid_val, np.nan, arr_100)
    
    # Reshape for block processing
    # shape: (50, 2, 50, 2)
    h, w = arr_masked.shape
    blocks = arr_masked.reshape(h // factor, factor, w // factor, factor)
    
    # Sum and Count valid pixels per block
    sum_ds = np.nansum(blocks, axis=(1, 3))
    count_ds = np.count_nonzero(~np.isnan(blocks), axis=(1, 3))
    
    # Calculate Mean
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ds = sum_ds / count_ds
        
    # Round to nearest class integer
    # If count is 0 (all pixels were invalid), set to invalid_val
    arr_final = np.where(count_ds == 0, invalid_val, np.round(mean_ds)).astype(arr_100.dtype)
    
    return arr_final
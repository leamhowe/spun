import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds

def export_prediction_to_geotiff(prediction_np, date_str, roi, export_dir='exports', prefix='SPUN_prediction'):
    """
    Exports a predicted snow cover array to a georeferenced GeoTIFF.
    """
    os.makedirs(export_dir, exist_ok=True)
    
    export_path = os.path.join(export_dir, f"{prefix}_{date_str}.tif")
    
    bounds = roi.bounds().getInfo()['coordinates'][0]
    lons = [c[0] for c in bounds]
    lats = [c[1] for c in bounds]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    height, width = prediction_np.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    
    with rasterio.open(
        export_path, 'w', 
        driver='GTiff',
        height=height, width=width, count=1,
        dtype=prediction_np.dtype, 
        crs='EPSG:4326',
        transform=transform, 
        compress='lzw',
        nodata=200
    ) as dst:
        dst.write(prediction_np, 1)
        
    print(f"💾 Exported GeoTIFF to: {export_path}")
    return export_path


def export_rgb_to_geotiff(img_array, date_str, roi, export_dir='exports', prefix='Sentinel2_RGB'):
    """
    Exports a brightened, 3-band RGB composite (Bands 4, 3, 2) to GeoTIFF.
    """
    os.makedirs(export_dir, exist_ok=True)
    
    # Use the custom prefix!
    export_path = os.path.join(export_dir, f"{prefix}_{date_str}.tif")
    
    bounds = roi.bounds().getInfo()['coordinates'][0]
    lons, lats = zip(*bounds)
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    height, width, _ = img_array.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
    
    rgb = np.stack([img_array[:, :, 3], img_array[:, :, 2], img_array[:, :, 1]], axis=0)
    rgb_bright = np.power(np.clip(rgb, 0, 1), 0.65) * 255.0
    rgb_uint8 = rgb_bright.astype(np.uint8)
    
    with rasterio.open(
        export_path, 'w', 
        driver='GTiff',
        height=height, width=width, count=3,
        dtype=rgb_uint8.dtype, 
        crs='EPSG:4326',       
        transform=transform, 
        compress='lzw',
        photometric='RGB'
    ) as dst:
        dst.write(rgb_uint8)
        
    print(f"🖼️ Exported RGB Basemap to: {export_path}")
    return export_path
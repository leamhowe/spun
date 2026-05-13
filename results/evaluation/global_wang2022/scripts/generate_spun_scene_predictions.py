import os
import sys
import re
import datetime
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import xarray as xr
import rioxarray as rio
import ee
import scipy.signal
from google.oauth2 import service_account
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from rasterio.transform import from_bounds
from geemap import ee_to_numpy

# --- Dynamic Path Resolution ---
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import the custom model from the root src/ directory
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel

# --- Constants ---
QA_BAND = 'cs_cdf'
CLEAR_THRESHOLD = 0.50

def instantiate_model(model_path, bands, backbone, device):
    """Instantiates and loads the pre-trained model."""
    C = len(bands)
    print(f"Instantiating model with {C} input channels.")
    hparams = {'backbone': backbone, 'weights': None, 'loss_fn': 'huber', 'lr': 1e-3, 'batch_size': 1, 'patience': 1, 'num_workers': 0}
    
    # Internal model loading parameters (use_indices removed!)
    model = SnowModel(
        bands=bands, 
        train_files=[], 
        val_files=[], 
        mask_clouds=False, 
        use_augmentations=False, 
        hparams=hparams, 
        in_channels=C, 
        train_with_clouds=True
    )
    
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print("Model loaded successfully.")
    return model

def get_gee_data(roi, start_date, end_date, bands, tile_id=None, pixel_size=10, mask_clouds=True):
    """Downloads Sentinel-2 TOA data. Filters by Tile, Mosaics, Reprojects, then Clips to ROI."""
    S2_TOA_COLLECTION = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
    
    print(f"Querying GEE for date: {start_date}")
    
    s2_col = S2_TOA_COLLECTION.filterBounds(roi).filterDate(start_date, end_date)
    
    if tile_id:
        print(f"Filtering by MGRS Tile: {tile_id}")
        clean_tile = tile_id[1:] if tile_id.startswith('T') else tile_id
        s2_col = s2_col.filter(ee.Filter.eq('MGRS_TILE', clean_tile))

    count = s2_col.size().getInfo()
    if count == 0:
        raise Exception(f"No S2 image found for date {start_date} and Tile {tile_id} (or ROI).")
    
    print(f"Found {count} optical tile(s). Processing...")

    ref_proj = s2_col.first().select('B2').projection()
    mosaic_image = s2_col.select(bands).mosaic()
    resampled = mosaic_image.reproject(ref_proj)
    final_image = resampled.clip(roi)
    
    image_array = ee_to_numpy(final_image, region=roi, scale=pixel_size)

    if image_array is None:
        raise Exception("Failed to download optical data from GEE.")

    cloud_mask_array = None
    if mask_clouds:
        print("Retrieving Cloud Score+ data (Masking Enabled)...")
        CS_PLUS_COLLECTION = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        qa_col = CS_PLUS_COLLECTION.filterBounds(roi).filterDate(start_date, end_date)
        
        if tile_id:
             clean_tile = tile_id[1:] if tile_id.startswith('T') else tile_id
             qa_col = qa_col.filter(ee.Filter.eq('MGRS_TILE', clean_tile))
        
        qa_count = qa_col.size().getInfo()
        if qa_count > 0:
            qa_mosaic = qa_col.select(QA_BAND).mosaic()
            qa_resampled = qa_mosaic.reproject(ref_proj)
            qa_final = qa_resampled.clip(roi)
            
            cloud_mask_array = ee_to_numpy(qa_final, region=roi, scale=pixel_size)
        else:
            print("Warning: No Cloud Score+ data found. Proceeding without mask.")
    else:
        print("Skipping Cloud Mask retrieval (Disabled by user).")

    return image_array, cloud_mask_array

class PatchDataset(Dataset):
    def __init__(self, image, coords, patch_size, transform):
        self.image = image
        self.coords = coords
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        patch_np = self.image[:, y:y + self.patch_size, x:x + self.patch_size]
        patch_tensor = self.transform(image=patch_np.transpose(1, 2, 0))['image']
        return patch_tensor

def predict_batched(model, image_array, device, batch_size, num_workers, cloud_mask_array=None):
    input_image_chw = image_array.transpose(2, 0, 1).astype(np.float32)
    print(f"Using numpy array of shape {input_image_chw.shape} for prediction.")

    PATCH_SIZE = 128
    STRIDE = 64
    C, H, W = input_image_chw.shape

    pad_h = (STRIDE - (H - PATCH_SIZE) % STRIDE) % STRIDE
    pad_w = (STRIDE - (W - PATCH_SIZE) % STRIDE) % STRIDE
    
    padded_image = np.pad(input_image_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0.0)
    padded_H, padded_W = padded_image.shape[1], padded_image.shape[2]

    patch_coords = []
    for y in range(0, padded_H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, padded_W - PATCH_SIZE + 1, STRIDE):
            patch_coords.append((y, x))

    transform = Compose([ToTensorV2()])
    dataset = PatchDataset(padded_image, patch_coords, PATCH_SIZE, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Starting batched GPU inference on {len(patch_coords)} patches...")
    results_list = []
    with torch.no_grad():
        for patch_batch in loader:
            patch_batch = patch_batch.to(device)
            pred_batch = model(patch_batch)
            results_list.append(pred_batch.squeeze(1).cpu())

    all_preds = torch.cat(results_list, dim=0).numpy() * 100.0
    print("Batched inference complete.")

    window = scipy.signal.windows.hann(PATCH_SIZE)
    window_2d = np.outer(window, window).astype(np.float32)
    prediction_canvas_np = np.zeros((H, W), dtype=np.float32)
    weight_canvas = np.zeros((H, W), dtype=np.float32)

    print("Accumulating results...")
    for i, (y, x) in enumerate(patch_coords):
        pred_np = all_preds[i]
        y_end, x_end = min(y + PATCH_SIZE, H), min(x + PATCH_SIZE, W)
        
        prediction_canvas_np[y:y_end, x:x_end] += pred_np[:y_end-y, :x_end-x] * window_2d[:y_end-y, :x_end-x]
        weight_canvas[y:y_end, x:x_end] += window_2d[:y_end-y, :x_end-x]

    with np.errstate(divide='ignore', invalid='ignore'):
        prediction_canvas_np = np.where(weight_canvas > 0, prediction_canvas_np / weight_canvas, 0)

    print("\nInference and blending complete.")

    if cloud_mask_array is not None:
        print("Applying cloud mask...")
        is_cloud = cloud_mask_array < CLEAR_THRESHOLD
        
        if is_cloud.ndim == 3:
            is_cloud = is_cloud.squeeze()
            
        prediction_canvas_np[is_cloud] = 200
        print(f"Cloud mask applied. Pixels masked: {np.sum(is_cloud)}")

    return prediction_canvas_np

def save_geotiff(prediction_array, bounds, crs, output_path):
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], prediction_array.shape[1], prediction_array.shape[0])
    
    prediction_da = xr.DataArray(
        data=np.expand_dims(prediction_array, axis=0),
        dims=("band", "y", "x"),
        coords={"band": [1]},
        attrs=dict(description="Snow Cover Fraction Prediction"),
    )
    prediction_da.rio.write_crs(crs, inplace=True)
    prediction_da.rio.write_transform(transform, inplace=True)
    prediction_da.rio.write_nodata(200, inplace=True)
    
    prediction_da.rio.to_raster(output_path, dtype='float32')
    print(f"Prediction saved to {output_path}")

def save_s2_geotiff(image_array_hwc, bands_list, bounds, crs, output_path):
    image_array_chw = image_array_hwc.transpose(2, 0, 1)
    height, width = image_array_chw.shape[1], image_array_chw.shape[2]
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    image_da = xr.DataArray(
        data=image_array_chw,
        dims=("band", "y", "x"),
        coords={"band": bands_list},
        attrs=dict(description="Sentinel-2 TOA Reflectance"),
    )
    image_da.rio.write_crs(crs, inplace=True)
    image_da.rio.write_transform(transform, inplace=True)
    image_da.rio.to_raster(output_path, dtype='float32')
    print(f"Source Sentinel-2 image saved to {output_path}")

def main(args):
    scopes = ['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/devstorage.read_write']
    credentials = service_account.Credentials.from_service_account_file(args.credentials_path, scopes=scopes)
    ee.Initialize(credentials)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Updated call: Removed args.use_indices
    model = instantiate_model(args.model_path, args.bands, args.backbone, device)

    with xr.open_dataset(args.input_geotiff, engine='rasterio') as ds:
        bounds = ds.rio.bounds()
        crs = ds.rio.crs.to_string()
    
    filename = os.path.basename(args.input_geotiff)
    
    date_str = filename.split('T')[0]
    image_date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    start_date = image_date.strftime('%Y-%m-%d')
    end_date = (image_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    mgrs_tile = None
    match = re.search(r'T\d{2}[A-Z]{3}', filename)
    if match:
        mgrs_tile = match.group(0)
    else:
        parts = filename.split('_')
        for part in parts:
            if len(part) == 6 and part.startswith('T'):
                mgrs_tile = part
                break
    
    if mgrs_tile:
        print(f"Detected MGRS Tile ID: {mgrs_tile}")
    else:
        print("Warning: Could not detect MGRS Tile ID from filename. GEE projection might mismatch.")

    roi = ee.Geometry.Polygon([[[bounds[0], bounds[1]], [bounds[2], bounds[1]], [bounds[2], bounds[3]], [bounds[0], bounds[3]], [bounds[0], bounds[1]]]], proj=crs, evenOdd=False)
    
    perform_masking = not args.no_cloud_mask
    
    image_array, cloud_mask_array = get_gee_data(roi, start_date, end_date, args.bands, tile_id=mgrs_tile, mask_clouds=perform_masking)

    S2_DIVISOR = 10000.0
    print(f"Applying standardization: dividing by {S2_DIVISOR} and clipping...")
    image_array = image_array / S2_DIVISOR
    image_array = np.clip(image_array, 0.0, None)

    prediction_np = predict_batched(model, image_array, device, args.batch_size, args.num_workers, cloud_mask_array=cloud_mask_array)

    save_geotiff(prediction_np, bounds, crs, args.output_geotiff)

    output_dir = os.path.dirname(args.output_geotiff)
    base_name = os.path.splitext(os.path.basename(args.output_geotiff))[0]
    s2_output_path = os.path.join(output_dir, f"{base_name}_s2.tif")
    save_s2_geotiff(image_array, args.bands, bounds, crs, s2_output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run fast snow cover prediction (Default: Clouds Masked).")
    parser.add_argument('--input_geotiff', type=str, required=True, help='Path to the input GeoTIFF to define ROI and date.')
    parser.add_argument('--output_geotiff', type=str, required=True, help='Path to save the output prediction GeoTIFF.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pt file.')
    parser.add_argument('--credentials_path', type=str, required=True, help='Path to your GEE service account credentials JSON file.')
    
    parser.add_argument('--no_cloud_mask', action='store_true', help='Use this flag to DISABLE cloud masking. (Masking is enabled by default)')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for GPU inference.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers for data loading.')
    parser.add_argument('--bands', nargs='+', default=['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'])
    parser.add_argument('--backbone', type=str, default='resnet50')
    
    # Removed --use_indices argument entirely

    args = parser.parse_args()
    main(args)
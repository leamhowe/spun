#!/usr/bin/env python3
"""
Batch inference for Snow Cover Fraction (SCF) and save outputs as GeoTIFFs.
Assumes input tiles are .zarr with arrays:
- 'COPERNICUS/S2_HARMONIZED' [T, C, H, W]
- 'band' listing band names
- 'label' with 0..100 valid and 200 for cloud/pad

Geo-referencing:
- If the zarr store contains an 'affine' and 'crs' array/attrs, they will be used.
- Otherwise, a worldfile .tfw/.prj next to the zarr will be used if present.
- As a last resort, saves a plain, non-georeferenced TIFF.
"""
import argparse
import os
import warnings
import json
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import zarr

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
except Exception:
    rasterio = None

from src.unet_torchgeo_lightning_trainer import SnowModel
from src.snow_s2_dataset import Snows2Dataset


def guess_geo_from_store(store) -> tuple[Optional[Affine], Optional[CRS]]:
    # Try common places to find transform/CRS
    transform = None
    crs = None

    try:
        if 'transform' in store.attrs:
            transform = Affine(*store.attrs['transform']) if rasterio else None
        if 'crs' in store.attrs:
            crs = CRS.from_string(store.attrs['crs']) if rasterio else None
    except Exception:
        pass

    # Sub-group attrs
    try:
        img = store['COPERNICUS/S2_HARMONIZED']
        if hasattr(img, 'attrs'):
            if transform is None and 'transform' in img.attrs:
                transform = Affine(*img.attrs['transform']) if rasterio else None
            if crs is None and 'crs' in img.attrs:
                crs = CRS.from_string(img.attrs['crs']) if rasterio else None
    except Exception:
        pass

    return transform, crs


def read_worldfile_sidecars(zarr_path: str):
    # Look for a .tfw (worldfile) and .prj next to the zarr
    base = os.path.splitext(zarr_path)[0]
    tfw = base + '.tfw'
    prj = base + '.prj'
    transform = None
    crs = None
    if rasterio and os.path.exists(tfw):
        try:
            with open(tfw, 'r') as f:
                vals = [float(x.strip()) for x in f.readlines()[:6]]
            transform = Affine(vals[0], vals[1], vals[4], vals[2], vals[3], vals[5])
        except Exception:
            transform = None
    if rasterio and os.path.exists(prj):
        try:
            with open(prj, 'r') as f:
                wkt = f.read()
            crs = CRS.from_wkt(wkt)
        except Exception:
            crs = None
    return transform, crs


def build_model(bands: List[str], ds: Snows2Dataset, checkpoint: str, backbone: str, use_indices: bool) -> SnowModel:
    C = len(ds[0]['image'])
    hparams = {
        'backbone': backbone, 'weights': None, 'loss_fn': 'l1', 'lr': 1e-3,
        'batch_size': 1, 'patience': 1, 'num_workers': 0
    }
    model = SnowModel.load_from_checkpoint(
        checkpoint,
        bands=bands, train_files=[], val_files=[], mask_clouds=False, use_augmentations=False, use_indices=use_indices,
        hparams=hparams, in_channels=C
    )
    model.eval()
    return model


def save_geotiff(out_path: str, data: np.ndarray, transform: Optional[Affine], crs: Optional[CRS]):
    # data shape: (H, W) or (1, H, W)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    H, W = data.shape[-2], data.shape[-1]
    if rasterio and transform is not None and crs is not None:
        profile = {
            'driver': 'GTiff', 'height': H, 'width': W, 'count': data.shape[0], 'dtype': data.dtype,
            'transform': transform, 'crs': crs, 'compress': 'lzw'
        }
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(data)
    else:
        # Fallback: plain TIFF without georeference if rasterio missing or no geo
        if rasterio:
            profile = {'driver': 'GTiff', 'height': H, 'width': W, 'count': data.shape[0], 'dtype': data.dtype}
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(data)
        else:
            import imageio
            imageio.imwrite(out_path, data.squeeze())


def run_inference(
    checkpoint: str,
    data_dir: str,
    out_dir: str,
    bands: List[str],
    backbone: str = 'resnet50',
    use_indices: bool = True,
    batch_size: int = 8,
    device: Optional[str] = None,
    ignore_value: float = 200.0,
):
    os.makedirs(out_dir, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.zarr')])
    if len(files) == 0:
        raise FileNotFoundError(f"No .zarr files in {data_dir}")

    ds = Snows2Dataset(files, bands=bands, train=False, mask_clouds=False, use_augmentations=False, use_indices=use_indices, target_type='scf')
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(bands, ds, checkpoint, backbone, use_indices).to(device)

    # Iterate tiles and save per-item GeoTIFFs
    idx = 0
    with torch.no_grad():
        for batch in dl:
            x = batch['image'].to(device)
            y_hat = model(x)
            y_hat_pct = (y_hat.squeeze(1) * 100.0).clamp(0, 100).cpu().numpy().astype(np.float32)  # [N,H,W]

            N = y_hat_pct.shape[0]
            for i in range(N):
                zarr_path = files[idx]
                idx += 1
                store = zarr.open(zarr_path, mode='r')
                transform, crs = (None, None)
                if rasterio:
                    t1, c1 = guess_geo_from_store(store)
                    t2, c2 = read_worldfile_sidecars(zarr_path)
                    transform = t1 or t2
                    crs = c1 or c2

                # If label exists, build a validity mask (label != ignore_value)
                mask = None
                try:
                    if 'label' in store:
                        lbl = store['label'][:].astype(np.float32)
                        mask = (lbl < (ignore_value - 0.5))
                except Exception:
                    mask = None

                out_arr = y_hat_pct[i]
                nodata_value = None
                if mask is not None and mask.shape == out_arr.shape:
                    # set nodata to -9999 for invalid pixels
                    nodata_value = -9999.0
                    arr = out_arr.copy()
                    arr[~mask] = nodata_value
                    # Prefer saving with nodata profile; if rasterio present, pass mask to write_mask
                    if rasterio:
                        H, W = arr.shape
                        profile = {
                            'driver': 'GTiff', 'height': H, 'width': W, 'count': 1, 'dtype': arr.dtype,
                            'compress': 'lzw'
                        }
                        if transform is not None and crs is not None:
                            profile.update({'transform': transform, 'crs': crs})
                        profile['nodata'] = nodata_value
                        base = os.path.splitext(os.path.basename(zarr_path))[0]
                        out_path = os.path.join(out_dir, f"{base}_scf.tif")
                        with rasterio.open(out_path, 'w', **profile) as dst:
                            dst.write(arr[np.newaxis, ...])
                            # write pixel mask (255 valid, 0 invalid)
                            dst.write_mask((mask.astype(np.uint8) * 255))
                        print(f"Saved {out_path}")
                        continue
                    else:
                        out_arr = arr  # fall back to plain write below

                base = os.path.splitext(os.path.basename(zarr_path))[0]
                out_path = os.path.join(out_dir, f"{base}_scf.tif")
                save_geotiff(out_path, out_arr, transform, crs)
                print(f"Saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description='SCF inference to GeoTIFFs')
    p.add_argument('--checkpoint', required=True, help='Path to Lightning .ckpt file')
    p.add_argument('--data_dir', required=True, help='Directory with .zarr tiles')
    p.add_argument('--out_dir', required=True, help='Output directory for GeoTIFFs')
    p.add_argument('--bands', default='all_spectral_bands', choices=['all_spectral_bands','wang_bands','custom_bands','all_bands'])
    p.add_argument('--backbone', default='resnet50')
    p.add_argument('--no-indices', action='store_true', help='Do not append spectral indices in dataset')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--device', default=None)
    p.add_argument('--ignore_value', type=float, default=200.0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.bands == 'all_spectral_bands':
        bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
    elif args.bands == 'wang_bands':
        bands = ['B2','B11','B4','B9']
    elif args.bands == 'all_bands':
        bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60']
    else:
        # custom_bands must match training session if used
        bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']

    run_inference(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        bands=bands,
        backbone=args.backbone,
        use_indices=not args.no_indices,
        batch_size=args.batch_size,
    device=args.device,
    ignore_value=args.ignore_value,
    )

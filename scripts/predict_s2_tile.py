"""
SPUN snow cover fraction (SCF) prediction for one full Sentinel-2 tile (MGRS, ~110 x 110 km).

The tile is processed on its native 10 m UTM grid in chunks (chunk size chosen automatically to fit the 48 MB request limit, + 510 m context
buffer). Input is a best-pixel composite (Earth Engine qualityMosaic on CloudScore+ cs_cdf, as in
the SPUN GitHub helpers) of all Sentinel-2 L1C scenes within +/- --composite_days of --date:
  --composite_days 0   same-day composite (all scenes/orbits acquired on that date)
  --composite_days N   clearest pixel within +/- N days, which fills partial (swath-edge) tiles;
                       a 'day_offset' band records which date each pixel comes from.
For each chunk, S2 (13 bands), cs_cdf, the acquisition day offset and (optionally) an ESA WorldCover
land-cover masks (water, built-up) are downloaded in ONE computePixels request; chunks are downloaded by parallel threads
while the model predicts (overlapping 128 x 128 patches with Hann-window feathering).

Earth Engine cannot return a whole tile in one interactive request (~3 GB for 13 bands; the
request limit is 48 MB), hence the chunking. For many tiles, exporting to Cloud Storage or using
local L1C SAFE files avoids the download cost entirely.

Outputs (uint8, LZW-compressed, tiled with overviews, in the tile's UTM CRS):

  SPUN_<tile>_<date>_SCF.tif  - the snow product: one band with an embedded colour table, so it
                                displays directly when dragged into QGIS/ArcGIS
      0-100  snow cover fraction (%)            (0 = observed, no snow)
      205    cloud (CloudScore+ cs_cdf < --cs_threshold; 205 follows the Theia/LIS convention)
      210    permanent water (ESA WorldCover, if --water_mask)
      215    built-up (ESA WorldCover, if --urban_mask)
      255    no data (transparent)

  SPUN_<tile>_<date>_QA.tif   - supporting layers
      band 1  scf_raw     SCF (0-100) before any masking, e.g. to apply your own cloud threshold
      band 2  cs_cdf      CloudScore+ cs_cdf x 100 (0-100)
      band 3  day_offset  acquisition date minus --date in days, plus 100 (100 = --date; useful
                          with --composite_days)
      255 = no data in all bands

  SPUN_<tile>_<date>_RGB.tif  - optional (--save_rgb): true-colour image (B4, B3, B2) of the input,
                                TOA reflectance with a fixed square-root stretch over 0-1 (so dates
                                are comparable and snow is not saturated); 0 = no data

A per-stage timing breakdown is printed and saved as CSV next to the output.

Examples
  # Which dates have full, clear coverage of a tile?
  python predict_s2_tile.py --tile 30VVJ --list_scenes 2023-03-01 2023-03-31 --credentials_path key.json
  # Same-day prediction
  python predict_s2_tile.py --tile 30VVJ --date 2023-03-15 --model_path spun.pt \
      --credentials_path key.json --output_dir out --water_mask --urban_mask
  # Only the true-colour image (e.g. for a date already predicted; no model needed)
  python predict_s2_tile.py --tile 30VVJ --date 2023-03-15 --rgb_only --credentials_path key.json --output_dir out
  # Full-tile composite (clearest pixel within +/- 3 days), CPU with 16 threads
  python predict_s2_tile.py ... --composite_days 3 --device cpu --cpu_threads 16

Timing: with --download_workers 1 every stage runs in series and the per-stage times add up to the
total (clean breakdown of inference vs data access). With more workers (default 8), downloads
overlap with each other and with inference; the wall-clock total is the realistic end-to-end time.
"""
import os
import sys
import time
import argparse
import contextlib
import platform
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

S2_COLLECTION = 'COPERNICUS/S2_HARMONIZED'
CS_COLLECTION = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
WORLDCOVER = 'ESA/WorldCover/v200'
WATER_CLASS = 80                     # ESA WorldCover: permanent water bodies
URBAN_CLASS = 50                     # ESA WorldCover: built-up
BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
CLOUD_VALUE, WATER_VALUE, URBAN_VALUE, NODATA = 205, 210, 215, 255
PX = 10                              # output pixel size (m)
CS_SCALE = 10000                     # cs_cdf is sent as uint16 (x 10000)
DT_OFFSET = 1000                     # day offset is sent as uint16 (+ 1000)
EE_REQUEST_LIMIT = 48 * 1024 ** 2    # computePixels payload limit (bytes)
EE_BYTES_PER_BAND = 3                # uint16 value + 1-byte mask per band per pixel (as counted by EE)


def auto_chunk_px(width, buffer_px, n_bands):
    """Largest chunk core (px) that splits the tile evenly and keeps each request under the EE limit."""
    for n in range(1, width + 1):
        core = -(-width // n)                                   # ceil(width / n)
        if (core + 2 * buffer_px) ** 2 * n_bands * EE_BYTES_PER_BAND <= EE_REQUEST_LIMIT:
            return core


# ----------------------------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------------------------
TIMINGS = {}
_LOCK = threading.Lock()


@contextlib.contextmanager
def timed(name, device=None):
    """Accumulate time per stage (synchronising CUDA so GPU work is attributed correctly).
    Thread-safe: with parallel downloads, the download stage is summed over worker threads."""
    cuda = device is not None and device.type == 'cuda'
    if cuda:
        import torch
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize()
        with _LOCK:
            TIMINGS[name] = TIMINGS.get(name, 0.0) + time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------
# Earth Engine
# ----------------------------------------------------------------------------------------------
def init_ee(credentials_path=None, project=None, high_volume=True):
    import ee
    url = 'https://earthengine-highvolume.googleapis.com' if high_volume else None
    kwargs = {k: v for k, v in (('project', project), ('opt_url', url)) if v}
    if credentials_path:
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=['https://www.googleapis.com/auth/earthengine'])
        ee.Initialize(creds, **kwargs)
    else:
        ee.Initialize(**kwargs)


def tile_collection(tile, start, end):
    """Scenes of one MGRS tile. Metadata used for reporting are copied to properties that default to
    -1 when missing, because reduceColumns(toList) silently drops scenes with any missing property."""
    import ee

    def with_defaults(img):
        names = img.propertyNames()
        for src, dst in (('NODATA_PIXEL_PERCENTAGE', 'nodata_pct'), ('CLOUDY_PIXEL_PERCENTAGE', 'cloudy_pct')):
            img = img.set(dst, ee.Algorithms.If(names.contains(src), img.get(src), -1))
        return img

    return (ee.ImageCollection(S2_COLLECTION)
            .filter(ee.Filter.eq('MGRS_TILE', tile.upper().lstrip('T')))
            .filterDate(start, end)
            .map(with_defaults))


def list_scenes(tile, start, end):
    """Print the scenes of a tile with their no-data and cloud percentages (to choose a date)."""
    import ee
    col = tile_collection(tile, ee.Date(start), ee.Date(end).advance(1, 'day'))
    props = col.reduceColumns(ee.Reducer.toList(4), ['system:time_start', 'PRODUCT_ID',
                                                     'nodata_pct', 'cloudy_pct']).get('list').getInfo()
    if not props:
        print(f"No scenes for tile {tile} between {start} and {end}")
        return
    df = pd.DataFrame(props, columns=['time', 'product_id', 'nodata_%', 'cloudy_%'])
    df['date'] = pd.to_datetime(df['time'], unit='ms').dt.strftime('%Y-%m-%d')
    df['orbit'] = df['product_id'].str.extract(r'_(R\d{3})_')
    print(df[['date', 'orbit', 'nodata_%', 'cloudy_%', 'product_id']]
          .sort_values('date').round(1).to_string(index=False))
    print("\nnodata_% > 0: partial (swath-edge) scene -> use --composite_days, or pick another date. "
          "-1 = not recorded in the metadata.")


def find_grid(tile, date, product_id, composite_days):
    """Native 10 m grid of the tile, the target date, and the scenes contributing to the composite."""
    import ee
    if product_id:
        col = ee.ImageCollection(S2_COLLECTION).filter(ee.Filter.eq('PRODUCT_ID', product_id))
        info = col.first().getInfo()
        if info is None:
            raise SystemExit(f"No Sentinel-2 L1C image found for PRODUCT_ID {product_id}")
        tile = info['properties']['MGRS_TILE']
        date = time.strftime('%Y-%m-%d', time.gmtime(info['properties']['system:time_start'] / 1000))
    d = ee.Date(date)
    col = tile_collection(tile, d.advance(-composite_days, 'day'), d.advance(composite_days + 1, 'day'))
    scenes = col.reduceColumns(ee.Reducer.toList(2), ['system:index', 'nodata_pct']).get('list').getInfo()
    ids = col.aggregate_array('PRODUCT_ID').getInfo()
    if len(ids) == len(scenes):
        scenes = [(pid, nod) for pid, (_, nod) in zip(ids, scenes)]
    if not scenes:
        # Help diagnose: nearby dates for this tile, and which tiles were acquired over it on this date
        near = tile_collection(tile, d.advance(-10, 'day'), d.advance(11, 'day'))
        near_dates = sorted(set(near.aggregate_array('system:time_start').map(
            lambda t: ee.Date(t).format('YYYY-MM-dd')).getInfo()))
        any_img = ee.ImageCollection(S2_COLLECTION).filter(ee.Filter.eq('MGRS_TILE', tile.upper().lstrip('T'))).first()
        same_day = (ee.ImageCollection(S2_COLLECTION).filterBounds(any_img.geometry().centroid(100))
                    .filterDate(d, d.advance(1, 'day')).aggregate_array('PRODUCT_ID').getInfo())
        raise SystemExit(
            f"No Sentinel-2 L1C scene of tile {tile} within {composite_days} day(s) of {date} in {S2_COLLECTION}.\n"
            f"  Dates with tile {tile} within +/-10 days: {', '.join(near_dates) or 'none'}\n"
            f"  Scenes over the centre of {tile} on {date} (any tile): {', '.join(same_day) or 'none'}")
    band = ee.Image(col.first()).select('B2').getInfo()['bands'][0]
    grid = {'crs': band['crs'], 'transform': band['crs_transform'],
            'width': band['dimensions'][0], 'height': band['dimensions'][1]}
    return grid, tile.upper().lstrip('T'), date, scenes


def build_source(grid, date, composite_days, water_mask, urban_mask, context_m):
    """One uint16 image per request: 13 S2 bands, cs_cdf (x 1e4), day offset (+1000)
    [, land-cover mask: 0 = none, 1 = water, 2 = built-up]."""
    import ee
    a, _, x0, _, e, y0 = grid['transform'][:6]
    x1, y1 = x0 + grid['width'] * a, y0 + grid['height'] * e
    region = ee.Geometry.Rectangle([x0 - context_m, y1 - context_m, x1 + context_m, y0 + context_m],
                                   proj=grid['crs'], geodesic=False)
    d = ee.Date(date)
    start, end = d.advance(-composite_days, 'day'), d.advance(composite_days + 1, 'day')
    cs = ee.ImageCollection(CS_COLLECTION).filterBounds(region).filterDate(start, end)

    def add_offset(img):
        dt = ee.Image.constant(img.date().difference(d, 'day').round()).toInt16().rename('day_offset')
        return img.addBands(dt)

    col = (ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate(start, end)
           .linkCollection(cs, ['cs_cdf'])
           .filter(ee.Filter.listContains('system:band_names', 'cs_cdf'))     # drop scenes without CS+
           .map(add_offset))
    best = col.qualityMosaic('cs_cdf')                                          # clearest pixel per location
    stack = (best.select(BANDS).toUint16()
             .addBands(best.select('cs_cdf').multiply(CS_SCALE).round().toUint16())
             .addBands(best.select('day_offset').add(DT_OFFSET).toUint16()))
    if water_mask or urban_mask:
        wc = ee.ImageCollection(WORLDCOVER).first().select('Map')
        lc = ee.Image.constant(0).toUint16()
        if water_mask:
            lc = lc.where(wc.eq(WATER_CLASS), 1)
        if urban_mask:
            lc = lc.where(wc.eq(URBAN_CLASS), 2)
        stack = stack.addBands(lc.toUint16().rename('landcover'))
    return stack, len(BANDS) + 2 + int(water_mask or urban_mask)


def compute_pixels(image, grid, row0, col0, h, w, retries=4, backoff=2.0):
    """Download an (h, w) window of the tile grid (nearest-neighbour) as a (bands, h, w) array."""
    import ee
    a, _, x0, _, e, y0 = grid['transform'][:6]
    request = {
        'expression': image,
        'fileFormat': 'NUMPY_NDARRAY',
        'grid': {
            'dimensions': {'width': int(w), 'height': int(h)},
            'affineTransform': {'scaleX': a, 'shearX': 0, 'translateX': x0 + col0 * a,
                                'shearY': 0, 'scaleY': e, 'translateY': y0 + row0 * e},
            'crsCode': grid['crs'],
        },
    }
    for attempt in range(retries):
        try:
            arr = ee.data.computePixels(request)
            return np.stack([arr[name] for name in arr.dtype.names], axis=0)
        except Exception as err:   # transient EE errors (rate limits, timeouts)
            if attempt == retries - 1 or 'request size' in str(err).lower():
                raise
            wait = backoff * 2 ** attempt
            print(f"    computePixels failed ({err}); retrying in {wait:.0f} s", flush=True)
            time.sleep(wait)


# ----------------------------------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------------------------------
def chunk_windows(height, width, core, buffer):
    """Yield (core_row, core_col, core_h, core_w, read_row, read_col, read_h, read_w)."""
    for r in range(0, height, core):
        for c in range(0, width, core):
            ch, cw = min(core, height - r), min(core, width - c)
            yield r, c, ch, cw, r - buffer, c - buffer, ch + 2 * buffer, cw + 2 * buffer


def fetch_chunk(stack, grid, win):
    """One request per chunk: all bands over the buffered window."""
    _, _, _, _, rr, rc, rh, rw = win
    with timed('1. Download (S2 + CS+ [+ land cover], 1 request/chunk)'):
        return compute_pixels(stack, grid, rr, rc, rh, rw)


def process_chunk(data, win, predict_fn, device, cs_threshold):
    """Return uint8 arrays (scf_final, scf_raw, cs_cdf_pct, day_offset) and the true-colour
    (3, h, w) uint8 image for the chunk core."""
    r, c, ch, cw, rr = win[:5]
    b = r - rr
    core = (slice(b, b + ch), slice(b, b + cw))
    nb = len(BANDS)
    img = data[:nb].astype(np.float32)
    cs_cdf = data[nb][core].astype(np.float32) / CS_SCALE
    day = data[nb + 1][core].astype(np.int32) - DT_OFFSET
    lc = data[nb + 2][core] if data.shape[0] > nb + 2 else np.zeros((ch, cw), np.uint16)

    nodata = (img[(slice(None),) + core] == 0).all(axis=0)
    if nodata.all():
        empty = np.full((ch, cw), NODATA, np.uint8)
        return empty, empty, empty, empty, np.zeros((3, ch, cw), np.uint8)

    with timed('2. Inference (patching, model, feathered mosaicking)', device):
        pred = predict_fn(np.clip(img / 10000.0, 0, None))                            # (H, W), 0-100
    raw = np.round(np.clip(pred[core], 0, 100)).astype(np.uint8)

    with timed('3. Cloud / land-cover masking'):
        final = raw.copy()
        final[cs_cdf < cs_threshold] = CLOUD_VALUE
        final[lc == 1] = WATER_VALUE
        final[lc == 2] = URBAN_VALUE
        cs_pct = np.round(np.clip(cs_cdf * 100, 0, 100)).astype(np.uint8)
        day_u8 = np.clip(day + 100, 0, 254).astype(np.uint8)
        for arr in (final, raw, cs_pct, day_u8):
            arr[nodata] = NODATA
        rgb = true_colour(img[[3, 2, 1]][(slice(None),) + core] / 10000.0)
        rgb[:, nodata] = 0
    return final, raw, cs_pct, day_u8, rgb


# ----------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------
def scf_colormap():
    """Colour table for the SCF product: brown (no snow) -> light blue -> white (100 % snow);
    cloud grey, water dark blue, built-up red, no data transparent."""
    stops = [(0, (115, 90, 60)), (50, (150, 200, 230)), (100, (255, 255, 255))]
    cmap = {}
    for v in range(101):
        for (v0, c0), (v1, c1) in zip(stops, stops[1:]):
            if v0 <= v <= v1:
                t = (v - v0) / (v1 - v0)
                cmap[v] = tuple(int(round(a + (b - a) * t)) for a, b in zip(c0, c1)) + (255,)
                break
    for v in range(101, 256):
        cmap[v] = (0, 0, 0, 0)
    cmap[CLOUD_VALUE] = (128, 128, 128, 255)
    cmap[WATER_VALUE] = (20, 60, 160, 255)
    cmap[URBAN_VALUE] = (200, 40, 40, 255)
    cmap[NODATA] = (0, 0, 0, 0)
    return cmap


def true_colour(refl):
    """(3, h, w) TOA reflectance -> uint8 RGB with a fixed square-root stretch over 0-1; values are
    kept in 1-255 so that 0 can mark no data."""
    return (1 + np.round(np.sqrt(np.clip(refl, 0, 1)) * 254)).astype(np.uint8)


def write_geotiff(path, grid, bands, descriptions, tags, colormap=None, nodata=NODATA, rgb=False):
    """Write uint8 bands to a tiled, LZW-compressed GeoTIFF with overviews. The file is written under a
    temporary name and renamed when complete, so a reader (e.g. QGIS) never sees a half-written file."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    a, b_, x0, d_, e, y0 = grid['transform'][:6]
    profile = dict(driver='GTiff', height=grid['height'], width=grid['width'], count=len(bands),
                   dtype='uint8', crs=grid['crs'], transform=Affine(a, b_, x0, d_, e, y0), nodata=nodata,
                   compress='lzw', tiled=True, blockxsize=512, blockysize=512, BIGTIFF='IF_SAFER')
    if colormap is not None:
        profile['photometric'] = 'palette'
    elif rgb:
        profile['photometric'] = 'rgb'
    tmp_path = path + f".tmp{os.getpid()}"
    with rasterio.open(tmp_path, 'w', **profile) as dst:
        for j, (arr, desc) in enumerate(zip(bands, descriptions), 1):
            dst.write(arr, j)
            dst.set_band_description(j, desc)
        if colormap is not None:
            dst.write_colormap(1, colormap)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
        # nearest-neighbour overviews keep the codes intact and make large tiles quick to display
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns='rio_overview', resampling='nearest')
    os.replace(tmp_path, path)
    return path


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group('scene')
    g.add_argument('--tile', help='MGRS tile, e.g. 30VVJ')
    g.add_argument('--date', help='Target date YYYY-MM-DD')
    g.add_argument('--product_id', help='Alternatively, a Sentinel-2 PRODUCT_ID (sets tile and date)')
    g.add_argument('--composite_days', type=int, default=0,
                   help='Best-pixel (max cs_cdf) composite over +/- N days (0 = same day only)')
    g.add_argument('--list_scenes', nargs=2, metavar=('START', 'END'),
                   help='List scenes of --tile between two dates (no-data and cloud %%) and exit')
    g = p.add_argument_group('model')
    g.add_argument('--model_path')
    g.add_argument('--repo_root', default=None,
                   help='Root of the spun-scf-pub repository (default: the parent of this script\'s folder)')
    g.add_argument('--backbone', default='resnet50')
    g.add_argument('--batch_size', type=int, default=64)
    g.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader workers for patching (0 is usually fastest: data are already in memory)')
    g.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    g.add_argument('--cpu_threads', type=int, default=None, help='torch threads when running on CPU')
    g = p.add_argument_group('processing')
    g.add_argument('--cs_threshold', type=float, default=0.5, help='cs_cdf below this is cloud')
    g.add_argument('--water_mask', action='store_true', help='Flag ESA WorldCover permanent water as 210')
    g.add_argument('--urban_mask', action='store_true', help='Flag ESA WorldCover built-up areas as 215')
    g.add_argument('--save_rgb', action='store_true', help='Also save a true-colour (B4, B3, B2) image')
    g.add_argument('--rgb_only', action='store_true',
                   help='Only save the true-colour image of the same composite (no model, no prediction)')
    g.add_argument('--chunk_m', type=int, default=None,
                   help='Chunk core size (m); default: largest even split of the tile within the 48 MB request limit')
    g.add_argument('--context_m', type=int, default=510, help='Context buffer around each chunk (m)')
    g.add_argument('--max_chunks', type=int, default=None, help='Process only the first N chunks (testing)')
    g.add_argument('--download_workers', type=int, default=8,
                   help='Parallel Earth Engine download threads, overlapping with inference (1 = strictly serial)')
    g = p.add_argument_group('Earth Engine / output')
    g.add_argument('--credentials_path', default=None, help='Service account JSON (else default credentials)')
    g.add_argument('--ee_project', default=None)
    g.add_argument('--no_high_volume', action='store_true', help='Use the standard EE endpoint')
    g.add_argument('--output_dir')
    args = p.parse_args()
    if args.list_scenes:
        if not args.tile:
            p.error('--list_scenes needs --tile')
        return args
    if not args.product_id and not (args.tile and args.date):
        p.error('Provide --tile and --date, or --product_id')
    if not args.output_dir:
        p.error('--output_dir is required')
    if not (args.model_path or args.rgb_only):
        p.error('--model_path is required (unless --rgb_only)')
    return args


def rgb_only(args):
    """Save only the true-colour image of the composite (no model): 3 bands, no context buffer, so
    chunks are large and a full tile needs few requests."""
    t0 = time.perf_counter()
    init_ee(args.credentials_path, args.ee_project, not args.no_high_volume)
    grid, tile, date, scenes = find_grid(args.tile, args.date, args.product_id, args.composite_days)
    stack, _ = build_source(grid, date, args.composite_days, False, False, 0)
    stack = stack.select(['B4', 'B3', 'B2'])
    print(f"Tile {tile}, target date {date}, composite window +/- {args.composite_days} day(s): "
          f"{', '.join(pid for pid, _ in scenes)}")
    H, W = grid['height'], grid['width']
    core = auto_chunk_px(W, 0, 3)
    windows = list(chunk_windows(H, W, core, 0))
    if args.max_chunks:
        windows = windows[:args.max_chunks]
    rgb = np.zeros((3, H, W), np.uint8)
    with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as pool:
        futs = {pool.submit(compute_pixels, stack, grid, w[0], w[1], w[2], w[3]): w for w in windows}
        for i, fut in enumerate(futs, 1):
            r, c, ch, cw = futs[fut][:4]
            data = fut.result().astype(np.float32)
            block = true_colour(data / 10000.0)
            block[:, (data == 0).all(axis=0)] = 0
            rgb[:, r:r + ch, c:c + cw] = block
            print(f"  chunk {i}/{len(windows)} done", flush=True)
    stem = f"SPUN_{tile}_{date.replace('-', '')}" + (f"_pm{args.composite_days}d" if args.composite_days else '')
    path = write_geotiff(
        os.path.join(args.output_dir, f"{stem}_RGB.tif"), grid, list(rgb), ['red_B4', 'green_B3', 'blue_B2'],
        dict(tile=tile, target_date=date, composite_days=args.composite_days,
             scenes=';'.join(pid for pid, _ in scenes),
             stretch="TOA reflectance, sqrt stretch over 0-1 to 1-255; 0 = no data"),
        nodata=0, rgb=True)
    print(f"Saved {path} ({len(windows)} chunks, {time.perf_counter() - t0:.0f} s)")


def main():
    args = get_args()
    if args.list_scenes:
        init_ee(args.credentials_path, args.ee_project, not args.no_high_volume)
        list_scenes(args.tile, *args.list_scenes)
        return

    t_start = time.perf_counter()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.rgb_only:
        rgb_only(args)
        return

    t_imp = time.perf_counter()
    import torch
    import rasterio
    from rasterio.transform import Affine
    # Model and inference helpers come from the repository (<repo>/src); this script is in <repo>/scripts
    from pathlib import Path
    sys.path.insert(0, args.repo_root or str(Path(__file__).resolve().parents[1]))
    from src.utils.inference_utils import instantiate_model, predict_batched
    TIMINGS['0. Library imports'] = time.perf_counter() - t_imp

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    if device.type == 'cpu' and args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)

    with timed('0a. Earth Engine initialisation'):
        init_ee(args.credentials_path, args.ee_project, not args.no_high_volume)
    with timed('0b. Scene search + metadata'):
        grid, tile, date, scenes = find_grid(args.tile, args.date, args.product_id, args.composite_days)
        stack, n_bands = build_source(grid, date, args.composite_days, args.water_mask, args.urban_mask, args.context_m)
    print(f"Tile {tile}, target date {date}, composite window +/- {args.composite_days} day(s):")
    for pid, nod in scenes:
        print(f"  {pid}  (no data: {f'{nod:.1f}%' if nod >= 0 else 'not recorded'})")
    if args.composite_days == 0 and all(nod > 0 for _, nod in scenes):
        print("  NOTE: partial tile on this date; consider --composite_days or --list_scenes")
    print(f"Grid: {grid['crs']}, {grid['width']} x {grid['height']} px")

    buf_px = args.context_m // PX
    core_px = args.chunk_m // PX if args.chunk_m else auto_chunk_px(grid['width'], buf_px, n_bands)
    req_bytes = n_bands * EE_BYTES_PER_BAND * (core_px + 2 * buf_px) ** 2
    if req_bytes > EE_REQUEST_LIMIT:
        raise SystemExit(f"Each request would be {req_bytes / 1024 ** 2:.1f} MB (> 48 MB Earth Engine limit); "
                         f"reduce --chunk_m or omit it to choose automatically")

    with timed('0c. Model loading + warm-up', device):
        model = instantiate_model(args.model_path, BANDS, args.backbone, device)
        with torch.no_grad():
            model(torch.zeros(1, len(BANDS), 128, 128, device=device))

    def predict_fn(x):
        # inference_utils.predict_batched expects (H, W, bands); cloud masking is done separately below
        return predict_batched(model, x.transpose(1, 2, 0), device, args.batch_size, args.num_workers)

    H, W = grid['height'], grid['width']
    names = ('scf', 'scf_raw', 'cs_cdf', 'day_offset')
    out = {k: np.full((H, W), NODATA, np.uint8) for k in names}
    rgb_out = np.zeros((3, H, W), np.uint8) if args.save_rgb else None
    windows = list(chunk_windows(H, W, core_px, buf_px))
    if args.max_chunks:
        windows = windows[:args.max_chunks]
    nw = max(1, args.download_workers)
    print(f"Processing {len(windows)} chunks on {device} ({nw} download worker(s), "
          f"{req_bytes / 1024 ** 2:.0f} MB per request)")

    # Downloads run in a thread pool ahead of inference (bounded look-ahead to cap memory);
    # with --download_workers 1 everything runs strictly in series (each chunk is downloaded only
    # after the previous one has been predicted), so download and inference times add up.
    t_loop = time.perf_counter()
    n_pred = 0
    serial = nw == 1
    with ThreadPoolExecutor(max_workers=nw) as pool:
        pending, it = deque(), iter(windows)
        for win in it:
            pending.append((win, pool.submit(fetch_chunk, stack, grid, win)))
            if len(pending) >= (1 if serial else 2 * nw):
                break
        i = 0
        while pending:
            win, fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None and not serial:
                pending.append((nxt, pool.submit(fetch_chunk, stack, grid, nxt)))
            t0 = time.perf_counter()
            with timed('Waiting for downloads (wall clock)'):
                data = fut.result()
            res = process_chunk(data, win, predict_fn, device, args.cs_threshold)
            r, c, ch, cw = win[:4]
            for k, arr in zip(names, res[:4]):
                out[k][r:r + ch, c:c + cw] = arr
            if rgb_out is not None:
                rgb_out[:, r:r + ch, c:c + cw] = res[4]
            n_pred += int((res[1] != NODATA).any())
            if nxt is not None and serial:
                pending.append((nxt, pool.submit(fetch_chunk, stack, grid, nxt)))
            i += 1
            print(f"  chunk {i}/{len(windows)} done in {time.perf_counter() - t0:.1f} s", flush=True)
    t_loop = time.perf_counter() - t_loop

    stem = f"SPUN_{tile}_{date.replace('-', '')}" + (f"_pm{args.composite_days}d" if args.composite_days else '')
    common_tags = dict(tile=tile, target_date=date, composite_days=args.composite_days,
                       scenes=';'.join(pid for pid, _ in scenes), cs_threshold=args.cs_threshold,
                       water_mask=str(args.water_mask), urban_mask=str(args.urban_mask),
                       model=os.path.basename(args.model_path))
    with timed('4. Writing GeoTIFFs'):
        scf_path = write_geotiff(
            os.path.join(args.output_dir, f"{stem}_SCF.tif"), grid, [out['scf']], ['snow_cover_fraction'],
            dict(common_tags, codes=f"0-100 snow cover fraction (%); {CLOUD_VALUE} cloud; {WATER_VALUE} water; "
                                    f"{URBAN_VALUE} built-up; {NODATA} no data"),
            colormap=scf_colormap())
        qa_path = write_geotiff(
            os.path.join(args.output_dir, f"{stem}_QA.tif"), grid,
            [out['scf_raw'], out['cs_cdf'], out['day_offset']], ['scf_raw', 'cs_cdf_x100', 'day_offset_plus100'],
            dict(common_tags, scf_raw="SCF (0-100) before cloud and land-cover masking",
                 cs_cdf_x100="CloudScore+ cs_cdf x 100", day_offset_plus100=
                 "acquisition date minus target_date in days, plus 100 (100 = target date)",
                 nodata="255 in all bands"))
        if rgb_out is not None:
            rgb_path = write_geotiff(
                os.path.join(args.output_dir, f"{stem}_RGB.tif"), grid, list(rgb_out), ['red_B4', 'green_B3', 'blue_B2'],
                dict(common_tags, stretch="TOA reflectance, sqrt stretch over 0-1 to 1-255; 0 = no data"),
                nodata=0, rgb=True)
    print(f"Saved {scf_path}\n      {qa_path}" + (f"\n      {rgb_path}" if rgb_out is not None else ''))

    # ---- timing report ----
    total = time.perf_counter() - t_start
    valid = (out['scf_raw'] != NODATA).mean()
    hw = torch.cuda.get_device_name(0) if device.type == 'cuda' else (platform.processor() or platform.machine())
    rows = [(k, v) for k, v in sorted(TIMINGS.items())]
    rows.append(('Chunk loop (wall clock)', t_loop))
    rows.append(('TOTAL (wall clock)', total))
    print(f"\nTiming | device: {device} ({hw}) | torch threads: {torch.get_num_threads()} | "
          f"chunks: {len(windows)} ({n_pred} predicted) | valid pixels: {valid:.1%} | "
          f"batch size: {args.batch_size} | download workers: {nw}")
    if valid < 0.99:
        print("  NB: part of the tile has no data, so inference time is lower than for a full tile.")
    if nw > 1:
        print(f"  NB: stage 1 is summed over {nw} parallel threads and overlaps with inference; "
              f"stages do not add up to the wall-clock total.")
    for k, v in rows:
        print(f"  {k:55s} {v:9.1f} s  {v / 60:7.2f} min")
    df = pd.DataFrame(rows, columns=['stage', 'seconds'])
    df['device'], df['hardware'], df['threads'] = device.type, hw, torch.get_num_threads()
    df['chunks'], df['valid_fraction'] = len(windows), valid
    df['download_workers'], df['batch_size'] = nw, args.batch_size
    df.to_csv(os.path.join(args.output_dir, f"{stem}_{device.type}_dl{nw}_timings.csv"), index=False)


if __name__ == '__main__':
    main()
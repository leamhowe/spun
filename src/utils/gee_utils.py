import ee
import numpy as np
from geemap import ee_to_numpy
from google.oauth2 import service_account

def authenticate_ee(credentials_path):
    """Authenticates and initialises Earth Engine using a service account JSON."""
    try:
        scopes = ['https://www.googleapis.com/auth/earthengine', 'https://www.googleapis.com/auth/devstorage.read_write']
        credentials = service_account.Credentials.from_service_account_file(credentials_path, scopes=scopes)
        ee.Initialize(credentials)
        print("Earth Engine Initialised Successfully!")
    except Exception as e:
        print(f"Failed to initialise Earth Engine: {e}")


def get_individual_s2_tiles(roi, start_date, end_date, bands, max_cloud_cover=100.0, clear_threshold=0.50):
    """
    Retrieves individual Sentinel-2 tiles. Skips tiles where ROI cloud cover exceeds max_cloud_cover.
    """
    s2_col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(roi).filterDate(start_date, end_date)
    qa_col = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED').filterBounds(roi).filterDate(start_date, end_date)

    count = s2_col.size().getInfo()
    print(f"Found {count} individual optical tile(s). Evaluating cloud cover...")

    image_list = []
    if count > 0:
        s2_list = s2_col.toList(count)
        
        for i in range(count):
            img = ee.Image(s2_list.get(i))
            date_msec = img.get('system:time_start').getInfo()
            date_str = ee.Date(date_msec).format('YYYY-MM-dd HH:mm:ss').getInfo()
            ref_proj = img.select('B2').projection()
            
            print(f"Analyzing Tile {i+1}/{count} from {date_str}...")
            
            # 1. Fetch Cloud Mask FIRST
            img_id = img.get('system:index').getInfo()
            qa_img = qa_col.filter(ee.Filter.eq('system:index', img_id)).first()
            
            qa_np = None
            cloud_percent = 0.0
            
            if qa_img is not None:
                qa_clipped = ee.Image(qa_img).select('cs_cdf').reproject(ref_proj).clip(roi)
                qa_np = ee_to_numpy(qa_clipped, region=roi, scale=10)
                if qa_np is not None:
                    # Calculate percentage of pixels below the clear threshold
                    cloud_percent = np.sum(qa_np < clear_threshold) / qa_np.size * 100.0
            
            # 2. Check Threshold
            if cloud_percent > max_cloud_cover:
                print(f"  -> Skipping: ROI is {cloud_percent:.1f}% cloudy (Limit: {max_cloud_cover}%).")
                continue
                
            print(f"  -> ROI is {cloud_percent:.1f}% cloudy. Downloading 13-band optical data...")
            
            # 3. Fetch Heavy Optical Data ONLY if it passes the check
            img_clipped = img.select(bands).reproject(ref_proj).clip(roi)
            img_np = ee_to_numpy(img_clipped, region=roi, scale=10)
            
            if img_np is not None:
                img_np = np.clip(img_np / 10000.0, 0.0, None)
                image_list.append({'date': date_str, 'optical': img_np, 'cloud_mask': qa_np})

    return image_list


def get_daily_cloud_optimized_mosaics(roi, start_date, end_date, bands, max_cloud_cover=100.0, clear_threshold=0.50):
    """
    Retrieves cloud-optimized mosaics. Skips days where ROI cloud cover exceeds max_cloud_cover.
    """
    s2_col = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(roi).filterDate(start_date, end_date)
    qa_col = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED').filterBounds(roi).filterDate(start_date, end_date)

    linked_col = s2_col.linkCollection(qa_col, ['cs_cdf'])
    count = linked_col.size().getInfo()

    if count == 0:
        print("No images found in this date range.")
        return []

    def get_date(img):
        return ee.Image(img).date().format('YYYY-MM-dd')
    
    unique_dates = linked_col.toList(count).map(get_date).distinct().getInfo()
    print(f"Found {count} raw tiles grouped into {len(unique_dates)} unique day(s).")
    
    image_list = []
    for date_str in unique_dates:
        print(f"Analyzing daily composite for: {date_str}...")
        
        day_start = ee.Date(date_str)
        day_end = day_start.advance(1, 'day')
        daily_col = linked_col.filterDate(day_start, day_end)
        
        best_pixel_mosaic = daily_col.qualityMosaic('cs_cdf')
        ref_proj = daily_col.first().select('B2').projection()
        final_img = best_pixel_mosaic.reproject(ref_proj).clip(roi)
        
        # 1. Fetch Cloud Mask FIRST
        qa_np = ee_to_numpy(final_img.select('cs_cdf'), region=roi, scale=10)
        cloud_percent = 0.0
        
        if qa_np is not None:
            cloud_percent = np.sum(qa_np < clear_threshold) / qa_np.size * 100.0
            
        # 2. Check Threshold
        if cloud_percent > max_cloud_cover:
            print(f"  -> Skipping: Composite is {cloud_percent:.1f}% cloudy (Limit: {max_cloud_cover}%).")
            continue
            
        print(f"  -> Composite is {cloud_percent:.1f}% cloudy. Downloading 13-band optical data...")
        
        # 3. Fetch Heavy Optical Data
        img_np = ee_to_numpy(final_img.select(bands), region=roi, scale=10)
        if img_np is not None:
            img_np = np.clip(img_np / 10000.0, 0.0, None)
            image_list.append({'date': date_str, 'optical': img_np, 'cloud_mask': qa_np})
        
    return image_list
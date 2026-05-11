import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import zarr
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

# --- 1. YOUR NORMALIZATION CLASS ---
class ScaleSentinel2(ImageOnlyTransform):
    """
    Divides Sentinel-2 DNs by 10,000 to get Reflectance.
    Clips negative values to 0, but allows values > 1.0 (clouds/specular highlights).
    """
    def __init__(self, divisor=10000.0, p=1.0):
        super().__init__(p=p)
        self.divisor = divisor

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        scaled = img / self.divisor
        return np.clip(scaled, 0.0, None).astype(np.float32)

# --- 2. YOUR DATASET CLASS ---
class Snows2Dataset(Dataset):
    def __init__(self, files, bands, train=False, use_augmentations=False, mask_clouds=False,
                 rmcloud100=False, target_type: str = 'class',
                 train_with_clouds: bool = False):
        
        self.files = files
        self.bands = bands
        self.use_augmentations = use_augmentations
        self.mask_clouds = mask_clouds
        self.train = train
        self.rmcloud100 = rmcloud100
        self.target_type = target_type
        self.train_with_clouds = train_with_clouds
        self._should_duplicate = train and use_augmentations

        self.scaler = ScaleSentinel2(divisor=10000.0, p=1.0)
        mask_fill_value = 200 if self.target_type == 'scf' else 0

        self.standard_augmentation = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, 
                          fill=0, fill_mask=mask_fill_value),
            self.scaler,
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

        self.train_augmentation = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, 
                          fill=0, fill_mask=mask_fill_value),
            A.Rotate(limit=90, p=0.5, border_mode=cv2.BORDER_REFLECT, fill_mask=mask_fill_value),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            self.scaler,
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.files) * 2 if self._should_duplicate else len(self.files)

    def _load_base_bands(self, store, bands_requested):
        shape_ref = None
        for k in store.array_keys():
            shape_ref = store[k].shape
            break
        if shape_ref is None:
            raise RuntimeError("Empty Zarr store.")
        H, W = shape_ref
        
        data = np.zeros((len(bands_requested), H, W), dtype=np.float32)

        for i, band in enumerate(bands_requested):
            if band in store:
                data[i] = store[band][:].astype(np.float32)
            else:
                raise KeyError(f"Requested band '{band}' not found in store.")
        return data 

    def __getitem__(self, idx):
        is_augmented = self._should_duplicate and idx >= len(self.files)
        original_idx = idx % len(self.files)
        zarr_path = self.files[original_idx]
        store = zarr.open(zarr_path, mode='r')

        bands_data = self._load_base_bands(store, self.bands)
        labels_data = store['label'][:].astype(np.int64)

        if self.mask_clouds:
            cloud_value = 0 if self.target_type == 'class' else 200
            cloud_mask = labels_data == cloud_value
            if cloud_mask.any():
                fill_val = -100.0 if self.rmcloud100 else 0.0
                bands_data[:, cloud_mask] = fill_val

        if self.target_type == 'scf' and self.train_with_clouds:
            labels_data[labels_data == 200] = 0
            
        aug = self.train_augmentation if is_augmented else self.standard_augmentation
        augmented = aug(image=bands_data.transpose(1, 2, 0), mask=labels_data)

        image_tensor = augmented['image']        
        labels_tensor = augmented['mask'].long() 

        return {"image": image_tensor, "label": labels_tensor, "mask": labels_tensor}


# --- 3. EXPORT SCRIPT LOGIC ---
BANDS_13 = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

def save_individual_chips(date_id, tile_id, zarr_filepath, spun_dir, output_dir):
    """Loads padded image chip + SPUN prediction and saves all 15 matrices as pure PNGs."""
    print(f"Loading data from: {zarr_filepath}")
    
    # Initialize the dataset for just this single file
    dataset = Snows2Dataset(
        files=[zarr_filepath],
        bands=BANDS_13,
        train=False,
        use_augmentations=False,
        target_type='class' # Change to 'scf' if needed depending on your masks
    )

    try:
        item = dataset[0]
    except Exception as e:
        print(f"Failed to load dataset item: {e}")
        return

    # Extract tensors and convert to NumPy 
    image_np = item['image'].numpy() # Shape: (13, 128, 128)
    label_np = item['label'].numpy() # Shape: (128, 128)

    # Ensure output directory exists
    chip_output_dir = os.path.join(output_dir, f"{date_id}_{tile_id}")
    os.makedirs(chip_output_dir, exist_ok=True)

    # 1. Save the 13 input bands
    for i, band_name in enumerate(BANDS_13):
        band_data = image_np[i]
        # Clip to [0, 1] to prevent blowout from clouds/glare when saving to 8-bit PNG
        clipped_data = np.clip(band_data, 0, 1)
        
        out_name = os.path.join(chip_output_dir, f"{band_name}.png")
        plt.imsave(out_name, clipped_data, cmap='gray')

    # 2. Save the Label (Ground Truth Snowmap)
    out_name_label = os.path.join(chip_output_dir, "label.png")
    plt.imsave(out_name_label, label_np, cmap='viridis', vmin=0, vmax=100)

    # 3. Load, Pad, and Save SPUN Prediction
    spun_path = os.path.join(spun_dir, f"{date_id}_{tile_id}_SPUN.zarr")
    if os.path.exists(spun_path):
        z_spun = zarr.open(spun_path, mode='r')
        spun_raw = z_spun['spun_scf'][:] # Raw unpadded prediction
        
        # Apply the exact same padding Albumentations uses for the dataset
        pad_transform = A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, fill_mask=0)
        # We pass it as a mask so Albumentations handles its single-channel layout natively
        spun_padded = pad_transform(image=np.zeros_like(spun_raw), mask=spun_raw)['mask']
        
        out_name_spun = os.path.join(chip_output_dir, "spun_prediction.png")
        plt.imsave(out_name_spun, spun_padded, cmap='viridis', vmin=0, vmax=100)
        print(f"Success! Saved 15 separate images (13 bands + label + SPUN) to {chip_output_dir}/")
    else:
        print(f"Warning: SPUN prediction not found at {spun_path}")
        print(f"Success! Saved 14 separate images (13 bands + label) to {chip_output_dir}/")


def main():
    # --- CONFIGURATION / INPUTS ---
    # TARGET_DATE = "20170508"
    # TARGET_TILE = "NN3884"
    TARGET_DATE = "20210621"
    TARGET_TILE = "NO0099"

    SPUN_PREDICTIONS_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/test_set_predictions'
    OUTPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/single_chip_exports'
    SPLIT_CSV_PATH = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

    try:
        df = pd.read_csv(SPLIT_CSV_PATH)
        matched_rows = df[df['filepath'].str.contains(f"{TARGET_DATE}_{TARGET_TILE}", na=False)]
        
        if matched_rows.empty:
            print(f"Error: Could not find {TARGET_DATE}_{TARGET_TILE} in the CSV.")
            return
            
        zarr_path = matched_rows.iloc[0]['filepath']
    except FileNotFoundError:
        print(f"Error: Could not locate CSV at {SPLIT_CSV_PATH}")
        return

    # Run the export
    save_individual_chips(TARGET_DATE, TARGET_TILE, zarr_path, SPUN_PREDICTIONS_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()
import torch
from torch.utils.data import Dataset
import zarr
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

# --- 1. NEW NORMALIZATION CLASS ---
class ScaleSentinel2(ImageOnlyTransform):
    """
    Divides Sentinel-2 DNs by 10,000 to get Reflectance.
    Clips negative values to 0, but allows values > 1.0 (clouds/specular highlights).
    """
    def __init__(self, divisor=10000.0, p=1.0):
        super().__init__(p=p)
        self.divisor = divisor

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # img shape: (H, W, C)
        # Divide by fixed divisor (10,000)
        scaled = img / self.divisor
        # Optional: Clip min to 0 to remove negative noise, but keep max open
        return np.clip(scaled, 0.0, None).astype(np.float32)

class Snows2Dataset(Dataset):
    # removed "bands" from init args if it was just for normalization, 
    # but kept it if you use it for loading logic.
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

        # --- 2. INITIALIZE SCALER ---
        # No lists needed anymore. Just the physics constant.
        self.scaler = ScaleSentinel2(divisor=10000.0, p=1.0)

        mask_fill_value = 200 if self.target_type == 'scf' else 0

        # --- 3. UPDATE AUGMENTATION PIPELINES ---
        self.standard_augmentation = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, 
                          fill=0, fill_mask=mask_fill_value),
            self.scaler, # Apply fixed scaling
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

        self.train_augmentation = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, 
                          fill=0, fill_mask=mask_fill_value),
            
            # Spatial augs happen on raw DNs (integers) or floats, both are fine.
            A.Rotate(limit=90, p=0.5, border_mode=cv2.BORDER_REFLECT, fill_mask=mask_fill_value),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Note: RandomBrightnessContrast behaves differently on 0-1 vs 0-10000 data.
            # It is usually safer to do Brightness/Contrast AFTER scaling if using default settings,
            # OR explicitly set the limit to work with high values.
            # However, for satellite data, it's safer to augment geometric first, then scale.
            
            self.scaler, # Apply fixed scaling
            
            # If you want brightness augs, put them here (after scaling to 0-1 range)
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # 2. Safe Physics Augmentation
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  # Keep this LOW to preserve SWIR darkness
                contrast_limit=0.2,    # Simulates different sun intensities
                p=0.5                  # Apply to 50% of images
            ),
            
            # Blur works fine on scaled or unscaled
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.files) * 2 if self._should_duplicate else len(self.files)

    def _load_base_bands(self, store, bands_requested):
        # ... (Keep this logic exactly the same) ...
        # Ensure you verify the bands requested match the channel count logic elsewhere
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
        # ... (Keep logic same until augmentation) ...
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
        
        # Transpose is critical because Albumentations expects HWC
        augmented = aug(image=bands_data.transpose(1, 2, 0), mask=labels_data)

        image_tensor = augmented['image']        
        labels_tensor = augmented['mask'].long() 

        return {"image": image_tensor, "label": labels_tensor, "mask": labels_tensor}
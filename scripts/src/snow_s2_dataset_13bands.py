import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
import torch.nn as nn

## AUGMENTATIONS
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform

class MinMaxNormalize(ImageOnlyTransform):
    """Normalize channels to the range [0, 1] using min/max values. Works with HWC numpy arrays."""

    def __init__(self, mins, maxs, p=1.0):
        super().__init__(p=p)
        # Convert mins and maxs to numpy arrays and reshape to (1, 1, C)
        self.mins = np.asarray(mins, dtype=np.float32).reshape(1, 1, -1)
        self.maxs = np.asarray(maxs, dtype=np.float32).reshape(1, 1, -1)
        
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # img shape: (H, W, C)
        return (img - self.mins) / (self.maxs - self.mins + 1e-10)
    
    def get_transform_init_args_names(self):
        return ("mins", "maxs")

# --- Define mins and maxs for the 13 spectral bands ---
mins_list = [
    993.0000, 682.6670, 460.6021, 288.6343, 287.0000, 289.0000, 334.0000,
    310.7522, 288.0000, 99.0000, 7.0000, 155.0000, 90.0000
]
maxs_list = [
    10658.0000, 11470.4238, 11206.9004, 12399.5781, 12555.0000, 12712.0000,
    12884.0000, 13001.7539, 13062.0000, 8686.0000, 1186.0000, 8579.0000,
    7064.0000
]


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

        self.min_max_norm = MinMaxNormalize(mins=mins_list, maxs=maxs_list, p=1.0)

        # --- CORRECT MASK FILL VALUE ---
        # The model's loss function ignores 200
        # So, we must pad the *mask* with 200.
        # The *image* is padded with 0 (value=0).
        mask_fill_value = 200 if self.target_type == 'scf' else 0

        # --- Add self.min_max_norm before ToTensorV2() ---
        self.standard_augmentation = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, 
                          fill=0, fill_mask=mask_fill_value),
            self.min_max_norm, # 2. Apply normalization
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

        self.train_augmentation = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, 
                          fill=0, fill_mask=mask_fill_value), # 1. Pad first
            A.Rotate(limit=90, p=0.5, border_mode=cv2.BORDER_REFLECT, fill_mask=mask_fill_value),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            self.min_max_norm, # 2. Apply normalization
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
        
        if len(bands_requested) != len(mins_list):
            raise ValueError(f"Requested {len(bands_requested)} bands, but normalization is set for {len(mins_list)} bands.")
            
        data = np.zeros((len(bands_requested), H, W), dtype=np.float32)

        for i, band in enumerate(bands_requested):
            if band in store:
                data[i] = store[band][:].astype(np.float32)
            else:
                raise KeyError(f"Requested band '{band}' not found in store.")
        return data  # (C,H,W)

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

        # This logic runs *before* padding
        if self.target_type == 'scf' and self.train_with_clouds:
            labels_data[labels_data == 200] = 0
            
        aug = self.train_augmentation if is_augmented else self.standard_augmentation
        
        augmented = aug(image=bands_data.transpose(1, 2, 0), mask=labels_data)

        image_tensor = augmented['image']        
        labels_tensor = augmented['mask'].long() 

        return {"image": image_tensor, "label": labels_tensor, "mask": labels_tensor}
import torch
from torch.utils.data import Dataset
import zarr
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


class ScaleSentinel2(ImageOnlyTransform):
    """
    Divides Sentinel-2 Digital Numbers (DN) by a scaling factor to derive reflectance.
    Values are clipped at a minimum of 0.0, but left unclipped at the maximum to 
    preserve saturated pixels (e.g., clouds, specular highlights).
    """
    def __init__(self, divisor=10000.0, p=1.0):
        super().__init__(p=p)
        self.divisor = divisor

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        scaled = img / self.divisor
        return np.clip(scaled, 0.0, None).astype(np.float32)


class Snows2Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading Sentinel-2 Zarr stores for Snow Cover Fraction mapping.
    
    Handles cloud masking, dataset splitting, on-the-fly spatial padding, 
    and physics-safe data augmentations via Albumentations.
    """
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

        # Standard pipeline for validation/testing (Padding + Scaling)
        self.standard_augmentation = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, 
                          fill=0, fill_mask=mask_fill_value),
            self.scaler,
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

        # Augmented pipeline for training
        self.train_augmentation = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, 
                          fill=0, fill_mask=mask_fill_value),
            A.Rotate(limit=90, p=0.5, border_mode=cv2.BORDER_REFLECT, fill_mask=mask_fill_value),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            self.scaler,
            A.RandomBrightnessContrast(
                brightness_limit=0.1,  
                contrast_limit=0.2,    
                p=0.5                  
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.files) * 2 if self._should_duplicate else len(self.files)

    def _load_base_bands(self, store, bands_requested):
        """Extracts the requested bands from a Zarr store into a numpy array."""
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

        # Apply Cloud Masking
        if self.mask_clouds:
            cloud_value = 0 if self.target_type == 'class' else 200
            cloud_mask = labels_data == cloud_value
            if cloud_mask.any():
                fill_val = -100.0 if self.rmcloud100 else 0.0
                bands_data[:, cloud_mask] = fill_val

        # Include clouds as 0% Snow Cover Fractional (SCF) values
        if self.target_type == 'scf' and self.train_with_clouds:
            labels_data[labels_data == 200] = 0
            
        aug = self.train_augmentation if is_augmented else self.standard_augmentation
        
        # Albumentations expects inputs in HWC format
        augmented = aug(image=bands_data.transpose(1, 2, 0), mask=labels_data)

        image_tensor = augmented['image']        
        labels_tensor = augmented['mask'].long() 

        return {"image": image_tensor, "label": labels_tensor, "mask": labels_tensor}
import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
import torch.nn as nn

## AUGMENTATIONS

# from torchgeo.transforms import AppendNDSI
# import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
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

# define mins and maxs for each band based on the dataset 
from torch import Tensor
mins, maxs = (
    Tensor([993.0000, 682.6670, 460.6021, 288.6343, 287.0000, 289.0000, 334.0000,
            310.7522, 288.0000,  99.0000,   7.0000, 155.0000,  90.0000,   0.0000,
            0.0000,   0.0000]),
    Tensor([10658.0000, 11470.4238, 11206.9004, 12399.5781, 12555.0000, 12712.0000,
                        12884.0000, 13001.7539, 13062.0000,  8686.0000,  1186.0000,  8579.0000,
                        7064.0000,     0.0000,     0.0000,  2048.0000])
)


class Snows2Dataset(Dataset):
    def __init__(self, files, bands, train=False, use_augmentations=False, mask_clouds=False,
                 use_indices=True, rmcloud100=False, target_type: str = 'class',
                 train_with_clouds: bool = False): # <-- NEW ARGUMENT
        """
        Args:
            files (list): List of file paths to zarr files.
            bands (list): List of band names to be used in the dataset.
            train (bool): Whether the dataset is for training or not.
            use_augmentations (bool): Whether to apply data augmentation.
            mask_clouds (bool): Whether to mask cloud pixels in the *images* (input features).
            use_indices (bool): Whether to append spectral indices to the image tensor.
            rmcloud100 (bool): If True, set image cloud pixels to -100. Else set to 0.
            target_type (str): 'class' for 3-class labels {0,1,2}, 'scf' for 0..100 with clouds=200.
            train_with_clouds (bool): If True and target_type='scf', relabel clouds (200) 
                                      to 0 (0% SCF) instead of ignoring them.
        """
        self.files = files
        self.bands = bands
        self.use_augmentations = use_augmentations
        self.mask_clouds = mask_clouds
        self.train = train
        self.use_indices = use_indices
        self.rmcloud100 = rmcloud100
        self.target_type = target_type
        self.train_with_clouds = train_with_clouds # <-- STORE NEW ARGUMENT
        self._should_duplicate = train and use_augmentations

    def __len__(self):
        return len(self.files) * 2 if self._should_duplicate else len(self.files)

    def _load_base_bands(self, store, bands_requested):
        """
        Load requested bands from new flat Zarr structure.
        """
        # Determine spatial shape from first available array
        shape_ref = None
        for k in store.array_keys():
            shape_ref = store[k].shape
            break
        if shape_ref is None:
            raise RuntimeError("Empty Zarr store.")
        H, W = shape_ref
        data = np.zeros((len(bands_requested), H, W), dtype=np.float32)

        for i, band in enumerate(bands_requested):
            if band == 'NDSI':
                # Compute on the fly: needs B3 & B11
                if 'B3' in store and 'B11' in store:
                    b3 = store['B3'][:].astype(np.float32)
                    b11 = store['B11'][:].astype(np.float32)
                    ndsi = (b3 - b11) / (b3 + b11 + 1e-6)
                    data[i] = ndsi
                else:
                    raise KeyError("Cannot compute NDSI (need B3 and B11).")
            elif band in store:
                data[i] = store[band][:].astype(np.float32)
            elif band == 'cs_cdf' and 'cs_cdf' in store:
                data[i] = store['cs_cdf'][:].astype(np.float32)
            elif band == 'cs' and 'cs' in store:
                data[i] = store['cs'][:].astype(np.float32)
            else:
                raise KeyError(f"Requested band '{band}' not found in store.")
        return data  # (C,H,W)

    def __getitem__(self, idx):
        is_augmented = self._should_duplicate and idx >= len(self.files)
        original_idx = idx % len(self.files)
        zarr_path = self.files[original_idx]
        store = zarr.open(zarr_path, mode='r')

        # Load image bands
        bands_data = self._load_base_bands(store, self.bands)  # (C,H,W)

        # Labels
        labels_data = store['label'][:].astype(np.int64)

        # Optional cloud masking (on the *input features*)
        # This runs *before* we modify the labels, so it correctly finds clouds.
        if self.mask_clouds:
            cloud_value = 0 if self.target_type == 'class' else 200
            cloud_mask = labels_data == cloud_value
            if cloud_mask.any():
                fill_val = -100.0 if self.rmcloud100 else 0.0
                bands_data[:, cloud_mask] = fill_val

        # --- NEW LOGIC ---
        # If training with clouds, set cloud labels (200) to 0 (snow-free)
        # This happens *after* input feature masking but *before* augmentation.
        if self.target_type == 'scf' and self.train_with_clouds:
            labels_data[labels_data == 200] = 0
        # --- END NEW LOGIC ---
            
        # This logic is now correct:
        # If train_with_clouds=True, labels_data has 0-100. pad=200.
        # If train_with_clouds=False, labels_data has 0-100 AND 200 (cloud). pad=200.
        # mask_fill_value = 200 if self.target_type == 'scf' else (0 if (self.mask_clouds and not self.rmcloud100) else 1)
        mask_fill_value=0

        standard_augmentation = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=mask_fill_value),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

        train_augmentation = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=mask_fill_value),
            A.Rotate(limit=90, p=0.5, border_mode=cv2.BORDER_REFLECT, fill_mask=mask_fill_value),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

        aug = train_augmentation if is_augmented else standard_augmentation
        # Albumentations expects HWC
        augmented = aug(image=bands_data.transpose(1, 2, 0), mask=labels_data)

        image_tensor = augmented['image']        # (C,H,W)
        labels_tensor = augmented['mask'].long() # (H,W)

        # Optionally append derived indices (computed from already augmented values)
        if self.use_indices:
            # Build a lookup for channel positions
            channel_names = self.bands[:]  # original order
            name_to_idx = {name: i for i, name in enumerate(channel_names)}

            def safe_get(b):
                if b in name_to_idx:
                    return image_tensor[name_to_idx[b]].float()
                return None

            extras = []
            b3 = safe_get('B3')
            b11 = safe_get('B11')
            b8 = safe_get('B8')
            b8a = safe_get('B8A')
            b4 = safe_get('B4')

            # NDVI
            if b8 is not None and b4 is not None:
                ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
                extras.append(ndvi.unsqueeze(0))
            # NDSI (if not already a requested band)
            if b3 is not None and b11 is not None and 'NDSI' not in channel_names:
                ndsi = (b3 - b11) / (b3 + b11 + 1e-6)
                extras.append(ndsi.unsqueeze(0))
            # NDWI
            if b3 is not None and b8 is not None:
                ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
                extras.append(ndwi.unsqueeze(0))
            # NDBI
            if b11 is not None and b8 is not None:
                ndbi = (b11 - b8) / (b11 + b8 + 1e-6)
                extras.append(ndbi.unsqueeze(0))

            if extras:
                image_tensor = torch.cat([image_tensor] + extras, dim=0)

        return {"image": image_tensor, "label": labels_tensor, "mask": labels_tensor}
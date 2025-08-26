import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
from torchgeo.transforms import indices
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
    def __init__(self, files, bands, train=False, use_augmentations=False, mask_clouds=False, use_indices=True, rmcloud100=False, target_type: str = 'class'):
        """
        Args:
            files (list): List of file paths to zarr files.
            bands (list): List of band names to be used in the dataset.
            train (bool): Whether the dataset is for training or not.
            use_augmentations (bool): Whether to apply data augmentation.
            mask_clouds (bool): Whether to mask cloud pixels in the images.
            use_indices (bool): Whether to append spectral indices to the image tensor.
            rmcloud100 (bool): If True, set image cloud pixels to -100. Else set to 0.
            target_type (str): 'class' for 3-class labels {0,1,2}, 'scf' for 0..100 with clouds=200.
        """
        self.files = files
        self.bands = bands
        self.use_augmentations = use_augmentations
        self.mask_clouds = mask_clouds
        self.train = train
        self.use_indices = use_indices
        self.rmcloud100 = rmcloud100
        self.target_type = target_type

        # Duplicate when training with augmentations (original + augmented)
        self._should_duplicate = train and use_augmentations

    def __len__(self):
        return len(self.files) * 2 if self._should_duplicate else len(self.files)

    def __getitem__(self, idx):
        # Determine original vs augmented sample
        is_augmented = self._should_duplicate and idx >= len(self.files)
        original_idx = idx % len(self.files)

        zarr_path = self.files[original_idx]
        zarr_store = zarr.open(zarr_path, mode='r')

        # Load bands
        bands_data = np.zeros((len(self.bands), 100, 100), dtype=np.float32)
        for i, band in enumerate(self.bands):
            if band in zarr_store['band'][:]:
                band_index = list(zarr_store['band'][:]).index(band)
                bands_data[i, :, :] = zarr_store['COPERNICUS/S2_HARMONIZED'][-1, band_index, :, :].astype(np.float32)
            elif band == 'NDSI':
                # Compute NDSI if possible
                if 'B3' in zarr_store['band'][:] and 'B11' in zarr_store['band'][:]:
                    b3_idx = list(zarr_store['band'][:]).index('B3')
                    b11_idx = list(zarr_store['band'][:]).index('B11')
                    b3 = zarr_store['COPERNICUS/S2_HARMONIZED'][-1, b3_idx, :, :].astype(np.float32)
                    b11 = zarr_store['COPERNICUS/S2_HARMONIZED'][-1, b11_idx, :, :].astype(np.float32)
                    ndsi = (b3 - b11) / (b3 + b11 + 1e-10)
                    bands_data[i, :, :] = ndsi
            elif band == 'cs_cdf' and 'cs_cdf' in zarr_store:
                bands_data[i, :, :] = zarr_store['cs_cdf'][:, :].astype(np.float32)
            elif band == 'SCL' and 'SCL' in zarr_store['band'][:]:
                scl_index = list(zarr_store['band'][:]).index('SCL')
                bands_data[i, :, :] = zarr_store['COPERNICUS/S2_HARMONIZED'][-1, scl_index, :, :].astype(np.float32)

        # Labels
        labels_data = zarr_store['label'][:].astype(np.int64)

        # Optionally mask cloud pixels in the imagery
        if self.mask_clouds:
            cloud_value = 0 if self.target_type == 'class' else 200
            cloud_mask = labels_data == cloud_value
            if cloud_mask.any():
                if self.rmcloud100:
                    bands_data[:, cloud_mask] = -100.0
                else:
                    bands_data[:, cloud_mask] = 0.0

        # Choose mask fill value for padding/augs
        mask_fill_value = 200 if self.target_type == 'scf' else (0 if (self.mask_clouds and not self.rmcloud100) else 1)

        # Augmentations
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

        if is_augmented:
            augmented_data = train_augmentation(image=bands_data.transpose(1, 2, 0), mask=labels_data)
        else:
            augmented_data = standard_augmentation(image=bands_data.transpose(1, 2, 0), mask=labels_data)

        image_tensor = augmented_data['image']
        labels_tensor = augmented_data['mask'].long()

        # Optionally append indices
        if self.use_indices:
            add_difference_indices = nn.Sequential(
                indices.AppendNDBI(index_swir=12, index_nir=7),
                indices.AppendNDSI(index_green=3, index_swir=12),
                indices.AppendNDVI(index_nir=7, index_red=3),
                indices.AppendNDWI(index_green=2, index_nir=7),
            )
            image_tensor = add_difference_indices(image_tensor).squeeze(0)

        # Return both keys for trainer compatibility
        return {"image": image_tensor, "label": labels_tensor, "mask": labels_tensor}



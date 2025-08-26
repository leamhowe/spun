import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
# from torchgeo.transforms import AppendNDSI
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Snows2Dataset(Dataset):
    def __init__(self, files, bands, augmentation=None, mask_clouds=False):
        self.files = files
        self.bands = bands  # List of band names e.g., ['B2', 'B11', 'B4', 'B9']
        # self.original_size = 100  # Original size of the images
        # # self.clip_size = 96  # Desired size after clipping, must be < original_size
        # self.pad_size = 128  # Desired size after clipping, must be < original_size
        self.augmentation = augmentation
        self.mask_clouds = mask_clouds

    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        zarr_path = self.files[idx]
        zarr_store = zarr.open(zarr_path, mode='r')
        
        # Fetch band names from zarr and calculate indices
        band_list = list(zarr_store['band'][:])
        indices = [band_list.index(band) for band in self.bands]

        # Fetch band data and labels from zarr
        try:
            bands_data = zarr_store['COPERNICUS/S2_HARMONIZED'][-1, indices, :, :].astype(np.float32)
        except KeyError:
            bands_data = zarr_store['COPERNICUS/S2_SR_HARMONIZED'][-1, indices, :, :].astype(np.float32)
        labels_data = zarr_store['label'][:].astype(np.int64)

        if self.mask_clouds==True:
            # Set cloud pixels to -100
            cloud_mask = labels_data == 0
            for c in range(bands_data.shape[0]):  # Looping through each channel
                bands_data[c][cloud_mask] = -100.0

        if self.augmentation!=None:
            augmented_data = self.augmentation(image = bands_data.transpose(1, 2, 0), mask=labels_data)
            image_tensor = augmented_data['image']
            labels_tensor = augmented_data['mask'].long()  # Ensure labels are of type int64
        else:
            image_tensor = bands_data.torch()
            labels_tensor = labels_data.long()  # Ensure labels are of type int64
        
        # # Calculate padding size
        # padding_size = (self.pad_size - self.original_size) // 2

        # # Pad data
        # image_tensor_padded = F.pad(image_tensor, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
        # labels_tensor_padded = F.pad(labels_tensor, (padding_size, padding_size, padding_size, padding_size), mode='constant', value=255)
        # # labels_data_padded = F.pad(torch.from_numpy(labels_data), (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)

        # return {"image": image_tensor_padded, "label": labels_tensor_padded}
        return {"image": image_tensor, "label": labels_tensor}
        
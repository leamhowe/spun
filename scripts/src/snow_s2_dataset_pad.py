import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
# from torchgeo.transforms import AppendNDSI
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2


import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
# from torchgeo.transforms import AppendNDSI
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Snows2Dataset(Dataset):
    def __init__(self, files, bands, augmentation=None):
        self.files = files
        self.bands = bands  # List of band names e.g., ['B2', 'B11', 'B4', 'B9']
        self.original_size = 100  # Original size of the images
        # self.clip_size = 96  # Desired size after clipping, must be < original_size
        self.pad_size = 128  # Desired size after clipping, must be < original_size
        self.augmentation = augmentation

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

        # Calculate padding size
        padding_size = (self.pad_size - self.original_size) // 2

        # Pad data
        bands_data_padded = F.pad(torch.from_numpy(bands_data), (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
        # labels_data_padded = F.pad(torch.from_numpy(labels_data), (padding_size, padding_size, padding_size, padding_size), mode='constant', value=255)
        labels_data_padded = F.pad(torch.from_numpy(labels_data), (padding_size, padding_size, padding_size, padding_size), mode='constant', value=1)  # Assuming 1 is the background class, adjust as needed
        # labels_data_padded = F.pad(torch.from_numpy(labels_data), (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)

        if self.augmentation!=None:
            augmented_data = self.augmentation(image = bands_data_padded.permute(1, 2, 0).numpy(), mask=labels_data_padded.numpy())
            image_tensor = augmented_data['image']
            labels_tensor = augmented_data['mask'].long()  # Ensure labels are of type int64
        else:
            image_tensor = bands_data_padded
            labels_tensor = labels_data_padded

        return {"image": image_tensor, "label": labels_tensor}
        

# class Snows2Dataset(Dataset):
#     def __init__(self, files, bands, augmentation=None):
#         self.files = files
#         self.bands = bands  # List of band names e.g., ['B2', 'B11', 'B4', 'B9']
#         self.original_size = 100  # Original size of the images
#         # self.clip_size = 96  # Desired size after clipping, must be < original_size
#         self.pad_size = 128  # Desired size after clipping, must be < original_size
#         self.augmentation = augmentation

#     def __len__(self):
#         return len(self.files)


#     def __getitem__(self, idx):
#         zarr_path = self.files[idx]
#         zarr_store = zarr.open(zarr_path, mode='r')
        
#         # Fetch band names from zarr and calculate indices
#         band_list = list(zarr_store['band'][:])
#         indices = [band_list.index(band) for band in self.bands]

#         # Fetch band data and labels from zarr
#         try:
#             bands_data = zarr_store['COPERNICUS/S2_HARMONIZED'][-1, indices, :, :].astype(np.float32)
#         except KeyError:
#             bands_data = zarr_store['COPERNICUS/S2_SR_HARMONIZED'][-1, indices, :, :].astype(np.float32)
#         labels_data = zarr_store['label'][:].astype(np.int64)

#         # Calculate padding size
#         padding_size = (self.pad_size - self.original_size) // 2

#         # Pad data
#         bands_data_padded = F.pad(torch.from_numpy(bands_data), (padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
#         labels_data_padded = F.pad(torch.from_numpy(labels_data), (padding_size, padding_size, padding_size, padding_size), mode='constant', value=255)
#         # bands_data_padded = F.pad(torch.from_numpy(bands_data), (padding_size, padding_size, padding_size, padding_size), mode='reflect')
#         # labels_data_padded = F.pad(torch.from_numpy(labels_data), (padding_size, padding_size, padding_size, padding_size), mode='reflect')

#         if self.augmentation:
#             # Note: Need a custom transform here since albumentations expects three channel images
#             # bands_data_padded = bands_data_padded.transpose(1, 2, 0)  # Shape to (H, W, C)
#             bands_data_padded = bands_data_padded.permute(1, 2, 0)  # Shape to (H, W, C)
#             augmented = self.augmentation(image=bands_data_padded, mask=labels_data_padded)
#             bands_data_padded = augmented['image']
#             labels_data_padded = augmented['mask']
#             # Swap axes back to (C, H, W) as PyTorch expects channels first
#             # bands_data_padded = bands_data_padded.permute(1, 2, 0)

#             bands_tensor = bands_data_padded
#             labels_tensor = labels_data_padded.long() # convert to the expected long int64 format for discrete labels
#         else:
#             bands_tensor = bands_data_padded
#             labels_tensor = labels_data_padded

#         return {"image": bands_tensor, "label": labels_tensor}




        
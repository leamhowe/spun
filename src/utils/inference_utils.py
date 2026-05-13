import torch
import numpy as np
import scipy.signal
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

# Import your core model architecture
from src.unet_torchgeo_combloss_lightning_trainer_13bands import SnowModel

def instantiate_model(model_path, bands, backbone, device):
    """Instantiates and loads the pre-trained SPUN model."""
    C = len(bands)
    hparams = {'backbone': backbone, 'weights': None, 'loss_fn': 'huber', 'lr': 1e-3, 'batch_size': 1, 'patience': 1, 'num_workers': 0}
    
    model = SnowModel(
        bands=bands, 
        train_files=[], 
        val_files=[], 
        mask_clouds=False, 
        use_augmentations=False, 
        hparams=hparams, 
        in_channels=C, 
        train_with_clouds=True
    )
    
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model

class PatchDataset(Dataset):
    """PyTorch Dataset to serve overlapping patches from a large image array."""
    def __init__(self, image, coords, patch_size, transform):
        self.image = image
        self.coords = coords
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        patch_np = self.image[:, y:y + self.patch_size, x:x + self.patch_size]
        patch_tensor = self.transform(image=patch_np.transpose(1, 2, 0))['image']
        return patch_tensor

def predict_batched(model, image_array, device, batch_size=16, num_workers=0, cloud_mask_array=None, clear_threshold=0.5):
    """Performs fast, batched, overlapping-patch prediction to prevent edge artifacts."""
    input_image_chw = image_array.transpose(2, 0, 1).astype(np.float32)

    PATCH_SIZE = 128
    STRIDE = 64
    C, H, W = input_image_chw.shape

    # Pad image to fit stride
    pad_h = (STRIDE - (H - PATCH_SIZE) % STRIDE) % STRIDE
    pad_w = (STRIDE - (W - PATCH_SIZE) % STRIDE) % STRIDE
    padded_image = np.pad(input_image_chw, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0.0)
    padded_H, padded_W = padded_image.shape[1], padded_image.shape[2]

    # Generate coordinates
    patch_coords = [(y, x) for y in range(0, padded_H - PATCH_SIZE + 1, STRIDE) 
                           for x in range(0, padded_W - PATCH_SIZE + 1, STRIDE)]

    transform = Compose([ToTensorV2()])
    dataset = PatchDataset(padded_image, patch_coords, PATCH_SIZE, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Batched Inference
    results_list = []
    with torch.no_grad():
        for patch_batch in loader:
            patch_batch = patch_batch.to(device)
            pred_batch = model(patch_batch)
            results_list.append(pred_batch.squeeze(1).cpu())

    all_preds = torch.cat(results_list, dim=0).numpy() * 100.0

    # Hann Window Blending
    window = scipy.signal.windows.hann(PATCH_SIZE)
    window_2d = np.outer(window, window).astype(np.float32)
    prediction_canvas_np = np.zeros((H, W), dtype=np.float32)
    weight_canvas = np.zeros((H, W), dtype=np.float32)

    for i, (y, x) in enumerate(patch_coords):
        pred_np = all_preds[i]
        y_end, x_end = min(y + PATCH_SIZE, H), min(x + PATCH_SIZE, W)
        
        prediction_canvas_np[y:y_end, x:x_end] += pred_np[:y_end-y, :x_end-x] * window_2d[:y_end-y, :x_end-x]
        weight_canvas[y:y_end, x:x_end] += window_2d[:y_end-y, :x_end-x]

    with np.errstate(divide='ignore', invalid='ignore'):
        prediction_canvas_np = np.where(weight_canvas > 0, prediction_canvas_np / weight_canvas, 0)

    # Apply Cloud Mask
    if cloud_mask_array is not None:
        is_cloud = cloud_mask_array < clear_threshold
        if is_cloud.ndim == 3:
            is_cloud = is_cloud.squeeze()
        prediction_canvas_np[is_cloud] = 200

    return prediction_canvas_np
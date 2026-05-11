import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import zarr
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

# --- 1. YOUR NORMALIZATION CLASS ---
class ScaleSentinel2(ImageOnlyTransform):
    def __init__(self, divisor=10000.0, p=1.0):
        super().__init__(p=p)
        self.divisor = divisor

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        scaled = img / self.divisor
        return np.clip(scaled, 0.0, None).astype(np.float32)

# --- 2. YOUR DATASET CLASS ---
class Snows2Dataset(Dataset):
    def __init__(self, files, bands):
        self.files = files
        self.bands = bands
        self.scaler = ScaleSentinel2(divisor=10000.0, p=1.0)
        self.aug = A.Compose([
            A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
            self.scaler,
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        store = zarr.open(self.files[idx], mode='r')
        
        shape_ref = None
        for k in store.array_keys():
            shape_ref = store[k].shape
            break
        H, W = shape_ref
        
        data = np.zeros((len(self.bands), H, W), dtype=np.float32)
        for i, b in enumerate(self.bands):
            data[i] = store[b][:]
            
        labels = store['label'][:]
        
        augmented = self.aug(image=data.transpose(1, 2, 0), mask=labels)
        return {"image": augmented['image'], "label": augmented['mask']}


# --- 3. 3D REAL-IMAGE STACK SCRIPT ---
BANDS_13 = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

def create_real_image_stack(date_id, tile_id, zarr_filepath, output_dir):
    print(f"Loading real image data from: {zarr_filepath}")
    
    # 1. Load the actual image
    dataset = Snows2Dataset(files=[zarr_filepath], bands=BANDS_13)
    try:
        image_np = dataset[0]['image'].numpy() # Shape: (13, 128, 128)
    except Exception as e:
        print(f"Failed to load dataset item: {e}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # 2. Isometric transform parameters
    # This matrix skews and rotates the flat 2D image to look 3D
    transform = Affine2D().skew_deg(-30, 15).scale(1, 0.5)
    
    # Spacing between layers
    x_step, y_step = 15, 25 
    
    # Render from bottom (B13) to top (B1) so they overlap correctly
    for i in reversed(range(13)):
        band_data = image_np[i]
        clipped_data = np.clip(band_data, 0, 1) # Normalize for display
        
        # Calculate offset for this layer
        x_offset = i * x_step
        y_offset = i * y_step

        # Apply the offset to the base geometric transform
        layer_transform = transform + Affine2D().translate(x_offset, y_offset) + ax.transData

        # Show the image using the affine transform to tilt it
        im = ax.imshow(clipped_data, cmap='gray', origin='lower',
                       transform=layer_transform, 
                       extent=[0, 128, 0, 128], # Original image size
                       alpha=0.85)              # Slight transparency

        # Draw a black border around each image layer
        rect = plt.Rectangle((0, 0), 128, 128, fill=False, edgecolor='black', linewidth=0.5, transform=layer_transform)
        ax.add_patch(rect)

        # Label the top and bottom bands for context
        if i == 0:
            ax.text(x_offset - 20, y_offset + 140, 'B1', fontsize=12, fontweight='bold', transform=ax.transData)
        elif i == 12:
            ax.text(x_offset - 20, y_offset + 140, 'B12', fontsize=12, fontweight='bold', transform=ax.transData)

    # Automatically adjust axes to fit the skewed images
    ax.autoscale()
    ax.axis('off')

    # Save logic
    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.join(output_dir, f"{date_id}_{tile_id}_unet_input_stack.png")
    
    plt.tight_layout()
    plt.savefig(out_name, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Saved real-image diagram to {out_name}")


def main():
    # --- CONFIGURATION / INPUTS ---
    # TARGET_DATE = "20170508"
    # TARGET_TILE = "NN3884"
    TARGET_DATE = "20210621"
    TARGET_TILE = "NO0099"

    OUTPUT_DIR = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/spun_scf/diagrams'
    SPLIT_CSV_PATH = '/home/users/leamhowe/phd/1_DETECT/SPUN/spun-scf/dev_notebooks/dataset_splits_cleaned_reassigned.csv'

    # Read CSV to locate the correct file path
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

    # Generate the diagram stack
    create_real_image_stack(TARGET_DATE, TARGET_TILE, zarr_path, OUTPUT_DIR)

if __name__ == "__main__":
    main()
import argparse
import os
import torch

import pytorch_lightning as pl
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore", 
                      category=UserWarning,
                      message="The codec `vlen-utf8` is currently not part")

# Define which model to use

# from src.snowpatch_model import SnowModel
from src.unet_torchgeo_lightning_trainer import SnowModel
# from src.unet_torchgeo_PTwang import SnowModel

# Set matrix multiplication precision to 'high'
torch.set_float32_matmul_precision('high')

def get_args():
    parser = argparse.ArgumentParser(description="Run UNet model training.")
    parser.add_argument('--model_name', type=str, default='unet_torchgeo_default',
                        help='Model name for saving outputs')
    parser.add_argument('--use_augmentations', action='store_true', 
                        help='Flag to enable image augmentations')
    parser.add_argument('--bands', type=str, default='all_spectral_bands',
                        help='Specify the bands to use in the model')
    # Hyperparameters
    # parser.add_argument('--backbone', type=str, default='resnet34',
    #                     help='Backbone model for the UNet')
    # parser.add_argument('--weights', type=str, default='imagenet',
    #                     help='Pretrained weights for the backbone')
    parser.add_argument('--loss_fn', type=str, default='l1',
                        help="Regression loss: 'l1' (default), 'mse', or 'huber'")
    parser.add_argument('--loss_weighting', type=str, default='none',
                        help="Per-pixel weighting: 'none'|'binary'|'linear' to emphasize snow")
    parser.add_argument('--snow_weight', type=float, default=5.0,
                        help='Weight for snow pixels (binary) or 1+alpha (linear)')
    parser.add_argument('--snow_thresh', type=float, default=0.05,
                        help='Snow threshold in [0,1] for binary weighting')
    parser.add_argument('--ignore_value', type=float, default=200.0,
                        help='Ignore value in labels (cloud/pad)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone model for the UNet')
    parser.add_argument('--weights', type=str, default=None,
                        help='Pretrained weights for the backbone')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and validation')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader')
    parser.add_argument('--server', type=str, default='jasmin',
                        help='Are you running on Jasmin or Pentland?')
    parser.add_argument('--mask_fill_value', type=int, default=1, help='Fill value for padding and augmenting the label dataset') 
    parser.add_argument('--image_fill_value', type=int, default=0, help='Fill value for padding and augmenting the image dataset')
    parser.add_argument('--mask_clouds', action='store_true', help='Mask cloud pixels in the images and set them to -100')
    parser.add_argument('--use_indices', action='store_true', help='Use indices in the dataset')
    parser.add_argument('--dataset', type=str, default='cs_threshold_50', help='Dataset to use for training and validation. Default is "cs_threshold_50".')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to a checkpoint file to resume training or load model weights')

    return parser.parse_args()

def prepare_data(directory):
    
    l1c_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.zarr')]
    # l2a_dir = f'{os.path.dirname(os.path.dirname(directory))}/l2a/{directory.split("/")[-1]}'  # Assuming the L2A directory is two levels up from the L1C directory
    l2a_dir = f'{os.path.dirname(os.path.dirname(directory))}/l2a/cs_threshold_50_binary_local_mean_threshold_00'  # Assuming the L2A directory is two levels up from the L1C directory
    l2a_files = [os.path.join(l2a_dir, f) for f in os.listdir(l2a_dir) if f.endswith('.zarr')]

    # Extract just the file names (without directory) for both lists
    l1c_names = set(os.path.basename(f) for f in l1c_files)
    l2a_names = set(os.path.basename(f) for f in l2a_files)

    # Find the intersection (matching file names)
    matching_names = l1c_names & l2a_names   

    # Filter the original lists to keep only the matching files
    all_files = [f for f in l1c_files if os.path.basename(f) in matching_names]



    # all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.zarr')]
    # subset_files = [f for f in all_files if 'nevis' not in f and ('cairngorms' in f or 'glencoe' in f)]

    # Split the data based on the region specified in the filename
    cairngorms_files = [f for f in all_files if 'cairngorms' in f]
    train_files = cairngorms_files
    # train_files = [f for f in all_files if 'nevis' not in f and ('cairngorms' in f or 'glencoe' in f)]
    # train_files = [f for f in all_files if 'cairngorms' not in f and ('nevis' in f or 'glencoe' in f)]
    # val_files = [f for f in all_files if 'cairngorms' not in f and ('nevis' in f or 'glencoe' in f)]
    # val_files = [f for f in all_files if 'nevis' in f]
    val_files = [f for f in all_files if 'glencoe' in f]
    # val_files = cairngorms_files

    # train_files = [f for f in all_files if 'glencoe' in f]
    # val_files = [f for f in all_files if 'nevis' in f]

    # # random split
    # import random
    # random.seed(42)  # For reproducibility
    # random.shuffle(all_files)
    # split_index = int(0.8 * len(all_files))  # 80% for training, 20% for validation
    # train_files = all_files[:split_index]
    # val_files = all_files[split_index:]

    # import json
    # def load_splits(save_dir):
    #     with open(f"{save_dir}/train_files.json") as f:
    #         train_files = json.load(f)
    #     with open(f"{save_dir}/val_files.json") as f:
    #         val_files = json.load(f)
    #     return train_files, val_files
    # train_files, val_files = load_splits('/home/s2112771/r/1_DETECT/unet/dataset_splits')

    # train_files = [path.replace("cs_threshold_50", "cs_threshold_50_binary_local_mean_threshold_00") 
    #          for path in train_files]
    # val_files = [path.replace("cs_threshold_50", "cs_threshold_50_binary_local_mean_threshold_00") 
    #          for path in val_files]

    test_files = [f for f in all_files if 'nevis' in f]

    return train_files, val_files, test_files


# Define augmentations
# def augmentation_pipeline(mask_fill_value=1, image_fill_value=0):
#     """
#     Define the augmentation pipeline using Albumentations.
#     Args:
#         mask_fill_value (int): Fill value for the mask during augmentation.
#         image_fill_value (int): Fill value for the image during augmentation.
#     Returns:
#         A.Compose: An Albumentations Compose object containing the augmentation pipeline.
#     """
#     # Define the augmentation pipeline
#     # Note: ToTensorV2() handles putting channels as the first dimension when converting back to tensor
#     # and also converts numpy arrays to PyTorch tensors.
#     return A.Compose([
#     A.Resize(128,128),
#     A.ShiftScaleRotate(p=0.9, fill_mask=mask_fill_value, fill=image_fill_value),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=1),
#     A.RandomGamma(gamma_limit=120, p=1), 
#     ToTensorV2()  # Handles putting channels as the first dimension when converting back to tensor
#     ], additional_targets={'image':'label'} #  This line ensures that the label is also transformed similarly to the image
#     )

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
# Define augmentations
# def augmentation_pipeline(mask_fill_value=1, image_fill_value=0):
#     """
#     Define the augmentation pipeline using Albumentations.
#     Args:
#         mask_fill_value (int): Fill value for the mask during augmentation.
#         image_fill_value (int): Fill value for the image during augmentation.
#     Returns:
#         A.Compose: An Albumentations Compose object containing the augmentation pipeline.
#     """
#     # Define the augmentation pipeline
#     # Note: ToTensorV2() handles putting channels as the first dimension when converting back to tensor
#     # and also converts numpy arrays to PyTorch tensors.
#     return A.Compose([
#         # A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, fill=0),
#         A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_REFLECT),
#         A.Resize(height=128,width=128, #interpolation=cv2.INTER_NEAREST), # nearest neighbour interpolation best for masks and preserves spectral info
#                 ),
#         # A.ShiftScaleRotate(p=0.9, fill_mask=255),
#         A.ShiftScaleRotate(p=0.9, border_mode=cv2.BORDER_REFLECT),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=1),
#         A.RandomGamma(gamma_limit=120, p=1), 
#         # A.RandomGamma(gamma_limit=(80, 120), p=1), 
#         # A.RandomContrast(p=1),
#         # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=False, always_apply=None, p=1),
#         # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
#         # A.RandomBrightnessContrast(
#         #     # contrast_limit=0.1, 
#         #     brightness_by_max=True, 
#         #     p=1),
#         ToTensorV2()  # Handles putting channels as the first dimension when converting back to tensor
#         ], additional_targets={'image':'label'}
#         # strict=True,
#         # seed=137,
#     )


################ MAIN CODE ##############################################################################################################################################################
import wandb
from pytorch_lightning.loggers import WandbLogger

def main():
    args = get_args()
    # Initialize W&B logger
    # wandb_logger = WandbLogger(name=args.model_name, project='snowpatches_detect')

    # Setup Weights & Biases Logger
    wandb_logger = WandbLogger(
        # project="snowpatches_detect",
        project="spun_scf",
        name=args.model_name,  # You can dynamically adjust this
        config={
            "learning_rate": args.lr,
            "architecture": args.backbone,
            # "dataset": "Specific-Dataset-Name",
            "epochs": 1000,
            "batch_size": args.batch_size,
            # Include any other hyperparameters or metadata
        },
        save_dir="/home/s2112771/r/1_DETECT/trained_models/unet/wandb/"  # Adjust the save directory as needed
    )

    
    model_name = args.model_name
    
    if args.server == 'jasmin':
        # directory = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/s2_labelled/l1c'
        # directory = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/s2_labelled/l1c/cs_threshold_50'
        
        data_directory = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/s2_labelled/l1c'
        # directory = '/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/data/s2_labelled/l1c/cs_threshold_50_plus'
    elif args.server == 'pentland':
        data_directory = '/home/s2112771/r/1_DETECT/data/training_data/s2/l1c'
        # data_directory = '/home/s2112771/r/1_DETECT/data/training_data/s2/l1c/cs_threshold_50'
        # data_directory = '/home/s2112771/r/1_DETECT/data/training_data/s2/l1c/cs_threshold_50_binary_local_mean_threshold_00_snowonly'
        
    train_files, val_files, test_files = prepare_data(f'{data_directory}/{args.dataset}')  # Adjust the directory path as needed

    # Define the bands and filepaths for your model
    # all_bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60']
    # wang_bands = ['B2','B11','B4','B9']
    # BANDS = list(zarr_store['band'][:]) 

    # Define the bands and filepaths for your model
    if args.bands == 'all_bands':
        bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60'] # All bands
    elif args.bands == 'all_spectral_bands':
        bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'] # All spectral bands
    elif args.bands == 'wang_bands':
        bands = ['B2','B11','B4','B9'] # Wang et al. (2020) band combination
    elif args.bands == 'perm_threshold_01':
        bands = ['B2', 'B3', 'B4', 'B5', 'B8A', 'B11', 'B12']
    elif args.bands == 'custom_bands':
        # bands = ['B2', 'B3', 'B4', 'B9', 'B11','SCL', 'MSK_CLDPRB', 'MSK_SNWPRB', 'NDSI']
        # bands = ['B2', 'B3', 'B4', 'B9', 'B11','SCL', 'NDSI']
        bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12','QA10','QA20','QA60', 'NDSI', 'cs_cdf'] # All bands

        # bands = ['B2', 'B3', 'B4', 'B5', 'B8A', 'B9', 'B11', 'B12']
        # bands = ['B2', 'B3', 'B4', 'B5', 'B8A', 'B11', 'B12']
        # bands = ['B2', 'B3', 'B4', 'B9', 'B11']
        # bands = ['B2', 'B3', 'B4', 'B9', 'B11','NDSI', 'cs_cdf']

    else:
        bands = []  # Default value if args.bands doesn't match any band combination

    # ## AUGMENTATIONS
    # import numpy as np
    # from albumentations.core.transforms_interface import ImageOnlyTransform

    # class MinMaxNormalize(ImageOnlyTransform):
    #     """Normalize channels to the range [0, 1] using min/max values. Works with HWC numpy arrays."""

    #     def __init__(self, mins, maxs, p=1.0):
    #         super().__init__(p=p)
    #         # Convert mins and maxs to numpy arrays and reshape to (1, 1, C)
    #         self.mins = np.asarray(mins, dtype=np.float32).reshape(1, 1, -1)
    #         self.maxs = np.asarray(maxs, dtype=np.float32).reshape(1, 1, -1)
            
    #     def apply(self, img: np.ndarray, **params) -> np.ndarray:
    #         # img shape: (H, W, C)
    #         return (img - self.mins) / (self.maxs - self.mins + 1e-10)
        
    #     def get_transform_init_args_names(self):
    #         return ("mins", "maxs")

    # # define mins and maxs for each band based on the dataset 
    # from torch import Tensor
    # mins, maxs = (
    #     Tensor([993.0000, 682.6670, 460.6021, 288.6343, 287.0000, 289.0000, 334.0000,
    #             310.7522, 288.0000,  99.0000,   7.0000, 155.0000,  90.0000,   0.0000,
    #             0.0000,   0.0000]),
    #     Tensor([10658.0000, 11470.4238, 11206.9004, 12399.5781, 12555.0000, 12712.0000,
    #                         12884.0000, 13001.7539, 13062.0000,  8686.0000,  1186.0000,  8579.0000,
    #                         7064.0000,     0.0000,     0.0000,  2048.0000])
    # )

    # # augmentations = augmentation_pipeline(mask_fill_value=args.mask_fill_value, image_fill_value=args.image_fill_value) if args.use_augmentations else None
    # standard_augmentation = A.Compose([
    #     MinMaxNormalize(mins= mins[:len(bands)], maxs=maxs[:len(bands)]),
    #     # A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, fill=0),
    #     A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_REFLECT),
    #     # A.Resize(height=128,width=128, #interpolation=cv2.INTER_NEAREST), # nearest neighbour interpolation best for masks and preserves spectral info
    #     #         ),
    #     ToTensorV2()  # Handles putting channels as the first dimension when converting back to tensor
    #     ], additional_targets={'image':'label'}
    #     )

    # # If augmentations are used, define the augmentation pipeline
    # if args.use_augmentations:
    #     # Define train and validation augmentations
    #     train_augmentation =  A.Compose([
    #             MinMaxNormalize(mins= mins[:len(bands)], maxs=maxs[:len(bands)]),
    #             # A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_CONSTANT, fill=0),
    #             A.PadIfNeeded(min_height=128, min_width=128, border_mode=cv2.BORDER_REFLECT),
    #             # A.Resize(height=128,width=128, #interpolation=cv2.INTER_NEAREST), # nearest neighbour interpolation best for masks and preserves spectral info
    #             #         ),
    #             # A.ShiftScaleRotate(p=0.9, fill_mask=255),
    #             A.ShiftScaleRotate(p=0.9, border_mode=cv2.BORDER_REFLECT),
    #             A.HorizontalFlip(p=0.5),
    #             A.VerticalFlip(p=1),
    #             # A.RandomGamma(gamma_limit=120, p=1), 
    #             # A.RandomGamma(gamma_limit=(80, 120), p=1), 
    #             # A.RandomContrast(p=1),
    #             # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0d.2), contrast_limit=(-0.2, 0.2), brightness_by_max=False, always_apply=None, p=1),
    #             # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
    #             # A.RandomBrightnessContrast(
    #             #     # contrast_limit=0.1, 
    #             #     brightness_by_max=True, 
    #             #     p=1),
    #             ToTensorV2()  # Handles putting channels as the first dimension when converting back to tensor
    #             ], additional_targets={'image':'label'}
    #     )

    # else:
    #     # If no augmentations are used, set train_augmentation to None
    #     train_augmentation = standard_augmentation
        
    # val_augmentation = standard_augmentation  # Use standard augmentation for validation


    
    # Adjust arguments passed into SnowModel according to parsed arguments
    model_hyperparams = {

        "backbone": args.backbone,
        "weights": args.weights,
        "loss_fn": args.loss_fn,  # Cross-entropy loss
        "loss_weighting": args.loss_weighting,
        "snow_weight": args.snow_weight,
        "snow_thresh": args.snow_thresh,
        "ignore_value": args.ignore_value,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "num_workers": args.num_workers,  # Number of workers for DataLoader
        # "train_augmentation": train_augmentation,
        # "val_augmentation": val_augmentation
    }
    
    # Instantiate the SnowModel
    snow_model = SnowModel(
        # bands=all_bands,
        bands=bands,
        train_files=train_files,
        val_files=val_files,
        mask_clouds=args.mask_clouds,  # Set to True if you want to mask clouds
        use_augmentations=args.use_augmentations,
        use_indices=args.use_indices,  # Set to True if you want to use spectral indices in the dataset
        hparams = model_hyperparams,
        # hparams={"backbone": "resnet34", "lr": 1e-4, "batch_size": 16}  # Example hyperparameters
        # augmentation=augmentations
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_mae",  # monitor regression MAE
        mode="min",
        dirpath="/home/s2112771/r/1_DETECT/trained_models/unet/model_checkpoints/",
        filename=f'{model_name}-{{epoch}}-{{val_mae:.4f}}'
    )


    # Setup the EarlyStopping callback
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_mae',
        min_delta=0.0,
        patience=args.patience,
        verbose=True,
        mode='min'
    )

    # Set up the Trainer. Specify whether to use GPU and other training options.
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Set up the Trainer. Specify whether to use GPU and other training options.
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=1000,
        callbacks=[checkpoint_callback, early_stopping_callback],
        # check_val_every_n_epochs = 1,
        # progress_bar_refresh_rate=20,  # Frequency of progress bar update
        logger=wandb_logger
    )

    # Fit the model
    if args.checkpoint_path:
        trainer.fit(snow_model, ckpt_path=args.checkpoint_path)
    else:
        trainer.fit(snow_model)

    # save the model
    # trainer.save_checkpoint(f"/home/users/leamhowe/phd/1_DETECT/NeuralNetworks/unet/models/{model_name}.ckpt") # Example to save the final model
    if args.server == 'jasmin':
        trainer.save_checkpoint(f"/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/models/{model_name}.ckpt")
        trainer.save_checkpoint(f"/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/models/{model_name}.ckpt") # Example to save the final model
        torch.save(snow_model.state_dict(), f"/home/users/leamhowe/sensecdt/users/leamhowe/1_DETECT/ml/unet/models/{model_name}.pt")
    elif args.server == 'pentland':
        trainer.save_checkpoint(f"/home/s2112771/r/1_DETECT/trained_models/unet/checkpoints/{model_name}.ckpt")
        torch.save(snow_model.state_dict(), f"/home/s2112771/r/1_DETECT/trained_models/unet/{model_name}.pt")
    # Optionally turn off W&B sync if the script is ending or you're running offline experiments
    wandb.finish()

    
    
    ### Inference on test data
    from src.snow_s2_dataset import Snows2Dataset  # Adjust import as needed
    # import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    batch_size=140
    mask_clouds = args.mask_clouds
    use_indices = args.use_indices

    snow_model.eval().to(device)  # Set the model to evaluation mode if needed

    test_dataset = Snows2Dataset(test_files, bands=bands, mask_clouds=mask_clouds, use_indices=use_indices, target_type='scf')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions_list = []
    labels_list = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch.get('mask', batch.get('label')).to(device).float()
            outputs = snow_model(images)  # (N,1,H,W) in [0,1]
            predictions = (outputs.squeeze(1) * 100.0)  # percent 0..100
            predictions_list.append(predictions.cpu())
            labels_list.append(labels.cpu())

    # predictions_array = predictions.cpu().numpy().astype(int)
    # labels_array = labels.cpu().numpy().astype(int)

    # Regression metrics on valid pixels (labels != 200)
    import pandas as pd
    import numpy as np
    if len(predictions_list) > 0:
        predictions = torch.cat(predictions_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        valid = (labels != 200)
        y_true = labels[valid]
        y_pred = predictions[valid]
        mae = (y_pred - y_true).abs().mean().item()
        rmse = float(torch.sqrt(((y_pred - y_true) ** 2).mean()).item())
        print({'mae': mae, 'rmse': rmse})
        df = pd.DataFrame([{'mae': mae, 'rmse': rmse}])
    else:
        df = pd.DataFrame([{'mae': np.nan, 'rmse': np.nan}])

    metrics_save_dir = '/home/s2112771/r/1_DETECT/trained_models/test_results'
    df.to_csv(os.path.join(metrics_save_dir, f"{model_name.split('.')[0]}_metrics.csv"))
    print(df)


if __name__ == "__main__":
    main()


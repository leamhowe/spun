import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from src.snow_s2_dataset import Snows2Dataset  # Make sure this path is correct
import os
import segmentation_models_pytorch as smp
import torch.nn as nn

# Import the torchgeo weights enums
from torchgeo.models import ResNet50_Weights 

# Environment for geospatial libs if needed
os.environ['PROJ_LIB'] = '/home/users/leamhowe/miniconda3/envs/torchgeeo/share/proj'


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss for regression tasks. (Copied from your combloss file)
    Assumes inputs and targets are in the range [0, 1].
    """
    def __init__(self, smooth=1.0, eps=1e-7):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs (torch.Tensor): Predictions from the model. Shape (N, C, H, W)
            targets (torch.Tensor): Ground truth labels. Shape (N, C, H, W)
            mask (torch.Tensor, optional): Boolean mask to ignore certain pixels. 
                                           Shape (N, 1, H, W). Defaults to None.
        """
        # Apply mask if provided
        if mask is not None:
            inputs = inputs * mask
            targets = targets * mask

        # Flatten inputs and targets
        inputs_flat = inputs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        # Calculate intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        
        # Using sum is more standard for the soft dice coefficient
        total = inputs_flat.sum() + targets_flat.sum()

        dice_coeff = (2. * intersection + self.smooth) / (total + self.smooth + self.eps)
        
        # The loss is 1 - the Dice coefficient
        return 1 - dice_coeff


class SnowModel(pl.LightningModule):
    """
    MERGED MODEL:
    - Uses TorchGeo pretrained weights for the encoder.
    - Uses a combined Regression + SoftDiceLoss.
    - Handles 'train_with_clouds' in the dataset.
    """

    def __init__(self, bands, train_files, val_files, mask_clouds, use_augmentations, use_indices, train_with_clouds, hparams, in_channels=None):
        super().__init__()
        self.save_hyperparameters(hparams)

        # --- Dataset Initialization (from torchgeo file) ---
        self.train_dataset = Snows2Dataset(train_files, bands=bands, train=True,
                                           mask_clouds=mask_clouds, use_augmentations=use_augmentations,
                                           use_indices=use_indices, target_type='scf', train_with_clouds=train_with_clouds)
        self.val_dataset = Snows2Dataset(val_files, bands=bands, train=False,
                                         mask_clouds=mask_clouds, use_augmentations=False,
                                         use_indices=use_indices, target_type='scf', train_with_clouds=train_with_clouds) # Note: train_with_clouds has no effect if train=False

        if in_channels is None:
            in_channels = len(self.train_dataset[0]['image'])

        # --- Model Definition (from torchgeo file) ---
        encoder_name = self.hparams.get('backbone', 'resnet50')
        weights_str = self.hparams.get('weights', None)

        smp_encoder_weights = None
        if weights_str == 'imagenet':
            smp_encoder_weights = 'imagenet'
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=smp_encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

        if weights_str and weights_str != 'imagenet':
            print(f"Manually loading torchgeo weights: {weights_str}")
            try:
                torchgeo_weights_enum = None
                if weights_str == 'sentinel2_all_moco':
                    if not encoder_name.startswith('resnet50'):
                        raise ValueError(f"Backbone '{encoder_name}' not compatible with 'sentinel2_all_moco'")
                    if in_channels != 13:
                        print(f"Warning: 'sentinel2_all_moco' weights expect 13 input channels, but model has {in_channels}.")
                    torchgeo_weights_enum = ResNet50_Weights.SENTINEL2_ALL_MOCO
                
                elif weights_str == 'sentinel2_rgb_moco':
                    if not encoder_name.startswith('resnet50'):
                        raise ValueError(f"Backbone '{encoder_name}' not compatible with 'sentinel2_rgb_moco'")
                    if in_channels != 3:
                        print(f"Warning: 'sentinel2_rgb_moco' weights expect 3 input channels, but model has {in_channels}.")
                    torchgeo_weights_enum = ResNet50_Weights.SENTINEL2_RGB_MOCO
                
                elif weights_str == 'sentinel2_all_dino':
                    if not encoder_name.startswith('vit_small'):
                        raise ValueError(f"Backbone '{encoder_name}' not compatible with 'sentinel2_all_dino' weights (requires 'vit_small...').")
                    if in_channels != 13:
                        print(f"Warning: 'sentinel2_all_dino' weights expect 13 input channels, but model has {in_channels}.")
                    torchgeo_weights_enum = ViTSmall_Weights.SENTINEL2_ALL_DINO
                
                else:
                    print(f"Warning: Unrecognized torchgeo weights string '{weights_str}'. No weights loaded.")

                if torchgeo_weights_enum:
                    state_dict = torchgeo_weights_enum.get_state_dict(progress=True)
                    self.model.encoder.load_state_dict(state_dict)
                    print(f"Successfully loaded {weights_str} into smp.Unet encoder.")

            except Exception as e:
                print(f"Error manually loading torchgeo weights '{weights_str}': {e}")
                print(f"Proceeding with '{smp_encoder_weights}' weights.")

        # --- Combined Loss Definition (from combloss file) ---

        # 1. Define constants for metrics and losses
        self.eps = 1e-8
        self.dice_smooth = 1.0
        self.dice_thresh = self.hparams.get('dice_thresh', 0.5)

        # 2. Define the loss functions
        # Regression loss component
        loss_name = self.hparams.get('loss_fn', 'l1').lower()
        if loss_name == 'mse':
            self.reg_crit = torch.nn.MSELoss(reduction='none')
        elif loss_name in ('huber', 'smoothl1', 'smooth_l1'):
            self.reg_crit = torch.nn.SmoothL1Loss(beta=0.1, reduction='none')
        else:
            self.reg_crit = torch.nn.L1Loss(reduction='none')
        
        # Soft Dice loss component
        self.dice_loss_fn = SoftDiceLoss(smooth=self.dice_smooth)
        self.dice_loss_weight = self.hparams.get('dice_loss_weight', 0.5) # e.g., 0.5 for 50/50 split

        # --- Accumulators (from both) ---
        self._train_acc = self._create_accumulator_dict()
        self._val_acc = self._create_accumulator_dict()


    def _create_accumulator_dict(self):
        """Helper to create a clean accumulator dictionary."""
        return {
            'abs_err': 0.0, 'sq_err': 0.0, 'sum_y': 0.0, 'sum_y2': 0.0,
            'sum_err': 0.0, 'n': 0, 'abs_err_snow': 0.0, 'sq_err_snow': 0.0, 'n_snow': 0,
            'dice_inter': 0.0, 'dice_union': 0.0
        }

    def forward(self, x):
        # Bound to [0,1]
        return torch.sigmoid(self.model(x))

    def _calculate_metrics(self, y_hat, y_raw):
        """
        Calculates all metrics AND combined loss for a given batch.
        (Logic from combloss file)
        """
        valid = (y_raw != 200)
        if not valid.any():
            z = y_hat.new_tensor(0.0)
            return {
                "loss": z, "loss_reg": z, "loss_dice": z, "mae": z, "rmse": z,
                "mae_snow": z, "rmse_snow": z, "dice": z,
                "dice_intersection": z, "dice_union": z
            }

        y = (y_raw.clamp(0, 100) / 100.0).unsqueeze(1)
        valid_mask = valid.unsqueeze(1)

        # --- Combined Loss Calculation ---
        
        # 1. Regression Loss Component
        loss_map_reg = self.reg_crit(y_hat, y)
        
        # Apply weighting to the regression loss map
        weighting_mode = self.hparams.get('loss_weighting', 'none')
        if weighting_mode != 'none':
            snow_thresh_for_weighting = self.hparams.get('snow_thresh', 0.01)
            snow_mask = (y > snow_thresh_for_weighting)
            
            if weighting_mode == 'binary':
                snow_weight = self.hparams.get('snow_weight', 2.0)
                weights = torch.ones_like(y)
                weights[snow_mask] = snow_weight
                loss_map_reg = loss_map_reg * weights
            
            elif weighting_mode == 'linear':
                alpha = self.hparams.get('snow_weight', 1.0)
                weights = 1 + alpha * y
                loss_map_reg = loss_map_reg * weights

            elif weighting_mode == 'inverse_linear':
                alpha = self.hparams.get('snow_weight', 1.0)
                weights = torch.ones_like(y)
                # Apply high weight to low SCF, low weight to high SCF
                weights[snow_mask] = 1.0 + alpha * (1.0 - y[snow_mask])
                loss_map_reg = loss_map_reg * weights            

        # Calculate final regression loss on valid pixels
        loss_reg = loss_map_reg[valid_mask].mean()
        
        # 2. Soft Dice Loss Component (Applied on valid pixels via mask)
        loss_dice = self.dice_loss_fn(y_hat, y, mask=valid_mask)

        # 3. Combine the losses
        loss = (1 - self.dice_loss_weight) * loss_reg + self.dice_loss_weight * loss_dice

        # --- Metric Calculation (unweighted, same as before) ---
        err = (y_hat - y)
        abs_err = err.abs()
        sq_err = err.pow(2)
        
        mae = abs_err[valid_mask].mean()
        rmse = torch.sqrt(sq_err[valid_mask].mean() + self.eps)
        
        snow_pixels = (y > 0) & valid_mask
        if snow_pixels.any():
            mae_snow = abs_err[snow_pixels].mean()
            rmse_snow = torch.sqrt(sq_err[snow_pixels].mean() + self.eps)
        else:
            mae_snow = y_hat.new_tensor(0.0)
            rmse_snow = y_hat.new_tensor(0.0)

        # Binarized Dice Score for METRICS
        y_hat_bin = (y_hat > self.dice_thresh).float()
        y_bin = (y > self.dice_thresh).float()

        intersection = (y_hat_bin * y_bin)[valid_mask].sum()
        union = y_hat_bin[valid_mask].sum() + y_bin[valid_mask].sum()
        
        batch_dice = (2. * intersection + self.dice_smooth) / (union + self.dice_smooth)

        return {
            "loss": loss,
            "loss_reg": loss_reg,
            "loss_dice": loss_dice,
            "mae": mae, "rmse": rmse, "mae_snow": mae_snow, "rmse_snow": rmse_snow,
            "dice": batch_dice,
            "dice_intersection": intersection,
            "dice_union": union
        }

    def _update_accumulators(self, acc, y_hat, y_raw, metrics):
        """Helper to update the accumulator dictionaries."""
        valid = (y_raw != 200)
        if not valid.any():
            return
        
        y = (y_raw.clamp(0, 100) / 100.0).unsqueeze(1)
        v = valid.unsqueeze(1)
        err = (y_hat - y)
        
        acc['abs_err'] += err.abs()[v].sum().item()
        acc['sq_err'] += (err.pow(2))[v].sum().item()
        acc['sum_y'] += y[v].sum().item()
        acc['sum_y2'] += (y[v].pow(2)).sum().item()
        acc['sum_err'] += err[v].sum().item()
        acc['n'] += int(v.sum().item())
        
        # For epoch-level Dice
        acc['dice_inter'] += metrics['dice_intersection'].item()
        acc['dice_union'] += metrics['dice_union'].item()

        snow_pixels = (y > 0) & v
        if snow_pixels.any():
            acc['abs_err_snow'] += err.abs()[snow_pixels].sum().item()
            acc['sq_err_snow'] += (err.pow(2))[snow_pixels].sum().item()
            acc['n_snow'] += int(snow_pixels.sum().item())

    def training_step(self, batch, batch_idx):
        # (from combloss file)
        x = batch['image']
        y_raw = batch.get('mask', batch.get('label')).float()
        y_hat = self(x)
        
        metrics = self._calculate_metrics(y_hat, y_raw)
        
        self.log('train_loss', metrics['loss'], prog_bar=True)
        self.log('train_loss_reg', metrics['loss_reg'])   # Log component
        self.log('train_loss_dice', metrics['loss_dice']) # Log component
        self.log('train_mae', metrics['mae'])
        self.log('train_rmse', metrics['rmse'])
        self.log('train_mae_snow', metrics['mae_snow'])
        self.log('train_rmse_snow', metrics['rmse_snow'])
        self.log('train_dice_step', metrics['dice']) # Log per-step dice

        self._update_accumulators(self._train_acc, y_hat, y_raw, metrics)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        # (from combloss file)
        x = batch['image']
        y_raw = batch.get('mask', batch.get('label')).float()
        y_hat = self(x)

        metrics = self._calculate_metrics(y_hat, y_raw)
        
        self.log('val_loss', metrics['loss'], prog_bar=True)
        self.log('val_loss_reg', metrics['loss_reg'])   # Log component
        self.log('val_loss_dice', metrics['loss_dice']) # Log component
        self.log('val_mae', metrics['mae'])
        self.log('val_rmse', metrics['rmse'])
        self.log('val_mae_snow', metrics['mae_snow'], prog_bar=True)
        self.log('val_rmse_snow', metrics['rmse_snow'], prog_bar=True)
        self.log('val_dice_step', metrics['dice'], prog_bar=True) # Log per-step dice

        self._update_accumulators(self._val_acc, y_hat, y_raw, metrics)
        return metrics['loss']

    def on_train_epoch_end(self):
        self._log_epoch_metrics(self._train_acc, 'train')

    def on_validation_epoch_end(self):
        self._log_epoch_metrics(self._val_acc, 'val')

    def _log_epoch_metrics(self, acc, prefix):
        """
        Helper to compute and log metrics at the end of an epoch.
        (Using the version from torchgeo file with sync_dist=True)
        """
        n = acc.get('n', 0)
        if n > 0:
            mae = acc['abs_err'] / n
            rmse = (acc['sq_err'] / n)**0.5
            # Add sync_dist=True to ensure metrics are correctly aggregated across all GPUs
            self.log(f'{prefix}_mae_epoch', mae, prog_bar=(prefix=='val'), sync_dist=True)
            self.log(f'{prefix}_rmse_epoch', rmse, sync_dist=True)

        n_snow = acc.get('n_snow', 0)
        if n_snow > 0:
            mae_snow = acc['abs_err_snow'] / n_snow
            rmse_snow = (acc['sq_err_snow'] / n_snow)**0.5
            self.log(f'{prefix}_mae_snow_epoch', mae_snow, prog_bar=(prefix=='val'), sync_dist=True)
            self.log(f'{prefix}_rmse_snow_epoch', rmse_snow, prog_bar=(prefix=='val'), sync_dist=True)
        
        inter = acc.get('dice_inter', 0)
        union = acc.get('dice_union', 0)
        if union > 0:
            dice = (2. * inter + self.dice_smooth) / (union + self.dice_smooth)
            self.log(f'{prefix}_dice_epoch', dice, prog_bar=(prefix=='val'), sync_dist=True)

    def on_train_epoch_start(self):
        # Reset accumulators at the start of each epoch
        self._train_acc = self._create_accumulator_dict()

    def on_validation_epoch_start(self):
        self._val_acc = self._create_accumulator_dict()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'], shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'],
                          num_workers=2, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
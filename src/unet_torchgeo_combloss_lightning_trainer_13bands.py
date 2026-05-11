import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.snow_s2_dataset_13bands import Snows2Dataset  
from torchgeo.models import ResNet50_Weights, ResNet152_Weights


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss optimized for regression and soft-classification tasks.
    Assumes model predictions and target labels are bounded in the range [0, 1].
    """
    def __init__(self, smooth=1.0, eps=1e-7):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, inputs, targets, mask=None):
        """
        Calculates the Soft Dice Loss between predictions and targets.

        Args:
            inputs (torch.Tensor): Model predictions. Shape (N, C, H, W).
            targets (torch.Tensor): Ground truth labels. Shape (N, C, H, W).
            mask (torch.Tensor, optional): Boolean mask to exclude specific pixels 
                                           (e.g., clouds, no-data). Shape (N, 1, H, W).

        Returns:
            torch.Tensor: Computed Soft Dice loss (1 - Dice Coefficient).
        """
        if mask is not None:
            inputs = inputs * mask
            targets = targets * mask

        inputs_flat = inputs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        total = inputs_flat.sum() + targets_flat.sum()

        dice_coeff = (2. * intersection + self.smooth) / (total + self.smooth + self.eps)
        
        return 1 - dice_coeff


class SnowModel(pl.LightningModule):
    """
    PyTorch Lightning Module for Snow Cover Fraction (SCF) mapping using U-Net.
    
    Features:
    - Integrates Earth Observation-specific pretrained weights (via TorchGeo).
    - Employs a dynamically weighted combination of Regression and Soft Dice Loss.
    - Handles robust accumulation of snow-specific and global accuracy metrics.
    """

    def __init__(self, bands, train_files, val_files, mask_clouds, use_augmentations, train_with_clouds, hparams, in_channels=None):
        super().__init__()
        self.save_hyperparameters(hparams)

        # --- Dataset Initialization ---
        self.train_dataset = Snows2Dataset(
            train_files, bands=bands, train=True,
            mask_clouds=mask_clouds, use_augmentations=use_augmentations,
            target_type='scf', train_with_clouds=train_with_clouds
        )
        self.val_dataset = Snows2Dataset(
            val_files, bands=bands, train=False,
            mask_clouds=mask_clouds, use_augmentations=False,
            target_type='scf', train_with_clouds=train_with_clouds 
        ) 

        if in_channels is None:
            in_channels = len(self.train_dataset[0]['image'])

        # --- Model Definition ---
        encoder_name = self.hparams.get('backbone', 'resnet50')
        weights_str = self.hparams.get('weights', None)

        smp_encoder_weights = 'imagenet' if weights_str == 'imagenet' else None
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=smp_encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

        # --- TorchGeo Weights Parsing ---
        if weights_str and weights_str != 'imagenet':
            print(f"Manually loading torchgeo weights: {weights_str}")
            try:
                torchgeo_weights_enum = None
                
                if weights_str == 'sentinel2_all_moco':
                    if not encoder_name.startswith('resnet50'):
                        raise ValueError(f"Backbone '{encoder_name}' not compatible with 'sentinel2_all_moco'")
                    if in_channels != 13:
                        print(f"Warning: '{weights_str}' weights expect 13 input channels, but model has {in_channels}.")
                    torchgeo_weights_enum = ResNet50_Weights.SENTINEL2_ALL_MOCO
                
                elif weights_str == 'sentinel2_rgb_moco':
                    if not encoder_name.startswith('resnet50'):
                        raise ValueError(f"Backbone '{encoder_name}' not compatible with 'sentinel2_rgb_moco'")
                    if in_channels != 3:
                        print(f"Warning: '{weights_str}' weights expect 3 input channels, but model has {in_channels}.")
                    torchgeo_weights_enum = ResNet50_Weights.SENTINEL2_RGB_MOCO
                
                elif weights_str == 'sentinel2_all_dino':
                    if not encoder_name.startswith('resnet50'):
                        raise ValueError(f"Backbone '{encoder_name}' not compatible with 'sentinel2_all_dino' weights.")
                    if in_channels != 13:
                        print(f"Warning: 'sentinel2_all_dino' weights expect 13 input channels, but model has {in_channels}.")
                    torchgeo_weights_enum = ResNet50_Weights.SENTINEL2_ALL_DINO

                elif weights_str == 'sentinel2_all_decur':
                    if not encoder_name.startswith('resnet50'):
                        raise ValueError(f"Backbone '{encoder_name}' not compatible with 'sentinel2_all_decur' weights.")
                    if in_channels != 13:
                        print(f"Warning: 'sentinel2_all_decur' weights expect 13 input channels, but model has {in_channels}.")
                    torchgeo_weights_enum = ResNet50_Weights.SENTINEL2_ALL_DECUR
                
                elif weights_str == 'sentinel2_softcon':
                    if not encoder_name.startswith('resnet50'):
                        raise ValueError(f"Backbone '{encoder_name}' not compatible with 'sentinel2_softcon' weights.")
                    if in_channels != 13:
                        print(f"Warning: 'sentinel2_softcon' weights expect 13 input channels, but model has {in_channels}.")
                    torchgeo_weights_enum = ResNet50_Weights.SENTINEL2_ALL_SOFTCON

                elif weights_str == 'satlas_ms':
                    if not encoder_name.startswith('resnet152'):
                        raise ValueError(f"Backbone '{encoder_name}' not compatible with 'satlas_ms' weights.")
                    if in_channels != 9:
                        print(f"Warning: 'satlas_ms' weights expect 9 input channels, but model has {in_channels}.")
                    torchgeo_weights_enum = ResNet152_Weights.SENTINEL2_SI_MS_SATLAS

                else:
                    print(f"Warning: Unrecognized torchgeo weights string '{weights_str}'. No weights loaded.")

                if torchgeo_weights_enum:
                    state_dict = torchgeo_weights_enum.get_state_dict(progress=True)
                    self.model.encoder.load_state_dict(state_dict)
                    print(f"Successfully loaded {weights_str} into smp.Unet encoder.")

            except Exception as e:
                print(f"Error manually loading torchgeo weights '{weights_str}': {e}")
                print(f"Proceeding with '{smp_encoder_weights}' weights.")

        # --- Loss Function Configuration ---
        self.eps = 1e-8
        self.dice_smooth = 1.0
        self.dice_thresh = self.hparams.get('dice_thresh', 0.5)

        loss_name = self.hparams.get('loss_fn', 'l1').lower()
        if loss_name == 'mse':
            self.reg_crit = nn.MSELoss(reduction='none')
        elif loss_name in ('huber', 'smoothl1', 'smooth_l1'):
            self.reg_crit = nn.SmoothL1Loss(beta=0.1, reduction='none')
        else:
            self.reg_crit = nn.L1Loss(reduction='none')
        
        self.dice_loss_fn = SoftDiceLoss(smooth=self.dice_smooth)
        self.dice_loss_weight = self.hparams.get('dice_loss_weight', 0.5) 

        # --- Metric Accumulators ---
        self._train_acc = self._create_accumulator_dict()
        self._val_acc = self._create_accumulator_dict()

    def _create_accumulator_dict(self):
        """Initializes a dictionary to accumulate epoch-level metrics."""
        return {
            'abs_err': 0.0, 'sq_err': 0.0, 'sum_y': 0.0, 'sum_y2': 0.0,
            'sum_err': 0.0, 'n': 0, 'abs_err_snow': 0.0, 'sq_err_snow': 0.0, 'n_snow': 0,
            'dice_inter': 0.0, 'dice_union': 0.0
        }

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def _calculate_metrics(self, y_hat, y_raw):
        """
        Computes the combined loss (Regression + Dice) and all step metrics.
        Masks out ignore-values (e.g., 200) prior to calculation.
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

        # 1. Regression Loss Component
        loss_map_reg = self.reg_crit(y_hat, y)
        
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
                weights[snow_mask] = 1.0 + alpha * (1.0 - y[snow_mask])
                loss_map_reg = loss_map_reg * weights            

        loss_reg = loss_map_reg[valid_mask].mean()
        
        # 2. Soft Dice Loss Component
        loss_dice = self.dice_loss_fn(y_hat, y, mask=valid_mask)

        # 3. Combined Loss
        loss = (1 - self.dice_loss_weight) * loss_reg + self.dice_loss_weight * loss_dice

        # --- Metric Calculation ---
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

        # Binarised Dice Score
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
        """Aggregates metrics across batches for epoch-level evaluation."""
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
        
        acc['dice_inter'] += metrics['dice_intersection'].item()
        acc['dice_union'] += metrics['dice_union'].item()

        snow_pixels = (y > 0) & v
        if snow_pixels.any():
            acc['abs_err_snow'] += err.abs()[snow_pixels].sum().item()
            acc['sq_err_snow'] += (err.pow(2))[snow_pixels].sum().item()
            acc['n_snow'] += int(snow_pixels.sum().item())

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y_raw = batch.get('mask', batch.get('label')).float()
        y_hat = self(x)
        
        metrics = self._calculate_metrics(y_hat, y_raw)
        
        self.log('train_loss', metrics['loss'], prog_bar=True)
        self.log('train_loss_reg', metrics['loss_reg'])
        self.log('train_loss_dice', metrics['loss_dice'])
        self.log('train_mae', metrics['mae'])
        self.log('train_rmse', metrics['rmse'])
        self.log('train_mae_snow', metrics['mae_snow'])
        self.log('train_rmse_snow', metrics['rmse_snow'])
        self.log('train_dice_step', metrics['dice'])

        self._update_accumulators(self._train_acc, y_hat, y_raw, metrics)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y_raw = batch.get('mask', batch.get('label')).float()
        y_hat = self(x)

        metrics = self._calculate_metrics(y_hat, y_raw)
        
        self.log('val_loss', metrics['loss'], prog_bar=True, sync_dist=True)
        self.log('val_loss_reg', metrics['loss_reg'], sync_dist=True)
        self.log('val_loss_dice', metrics['loss_dice'], sync_dist=True)
        self.log('val_mae', metrics['mae'], sync_dist=True)
        self.log('val_rmse', metrics['rmse'], sync_dist=True)
        self.log('val_mae_snow', metrics['mae_snow'], prog_bar=True, sync_dist=True)
        self.log('val_rmse_snow', metrics['rmse_snow'], prog_bar=True, sync_dist=True)
        self.log('val_dice_step', metrics['dice'], prog_bar=True, sync_dist=True)

        self._update_accumulators(self._val_acc, y_hat, y_raw, metrics)
        return metrics['loss']

    def on_train_epoch_end(self):
        self._log_epoch_metrics(self._train_acc, 'train')

    def on_validation_epoch_end(self):
        self._log_epoch_metrics(self._val_acc, 'val')

    def _log_epoch_metrics(self, acc, prefix):
        """Computes and logs aggregated metrics at the end of an epoch."""
        n = acc.get('n', 0)
        if n > 0:
            mae = acc['abs_err'] / n
            rmse = (acc['sq_err'] / n)**0.5
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
        self._train_acc = self._create_accumulator_dict()

    def on_validation_epoch_start(self):
        self._val_acc = self._create_accumulator_dict()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams['batch_size'],
            num_workers=self.hparams['num_workers'], shuffle=True, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams['batch_size'],
            num_workers=2, shuffle=False, pin_memory=True
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
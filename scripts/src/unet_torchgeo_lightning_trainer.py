import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from src.snow_s2_dataset import Snows2Dataset
import os

# Environment for geospatial libs if needed
os.environ['PROJ_LIB'] = '/home/users/leamhowe/miniconda3/envs/torchgeeo/share/proj'


class SnowModel(pl.LightningModule):
    """UNet for Snow Cover Fraction regression (0..1) with masked loss (ignore 200/cloud)."""

    def __init__(self, bands, train_files, val_files, mask_clouds, use_augmentations, use_indices, hparams, in_channels=None):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Datasets return integer masks; target_type='scf' ensures padding uses 200 and cloud value is 200
        self.train_dataset = Snows2Dataset(train_files, bands=bands, train=True,
                                           mask_clouds=mask_clouds, use_augmentations=use_augmentations,
                                           use_indices=use_indices, target_type='scf')
        self.val_dataset = Snows2Dataset(val_files, bands=bands, train=False,
                                         mask_clouds=mask_clouds, use_augmentations=False,
                                         use_indices=use_indices, target_type='scf')

        if in_channels is None:
            in_channels = len(self.train_dataset[0]['image'])

        # Build UNet directly via segmentation_models_pytorch
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "segmentation_models_pytorch is required for SCF regression UNet.\n"
                "Install it in your environment, e.g.: pip install segmentation-models-pytorch"
            ) from e

        encoder_name = self.hparams.get('backbone', 'resnet50')
        encoder_weights = self.hparams.get('weights', None)  # e.g., 'imagenet' or None
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )

        # Loss
        loss_name = self.hparams.get('loss_fn', 'l1').lower()
        if loss_name == 'mse':
            self.crit = torch.nn.MSELoss(reduction='none')
        elif loss_name in ('huber', 'smoothl1', 'smooth_l1'):
            self.crit = torch.nn.SmoothL1Loss(beta=0.1, reduction='none')
        else:
            self.crit = torch.nn.L1Loss(reduction='none')

        self.eps = 1e-8
        self.dice_smooth = 1.0

        # Accumulators for epoch metrics
        self._train_acc = {
            'abs_err': 0.0, 'sq_err': 0.0, 'sum_y': 0.0, 'sum_y2': 0.0,
            'sum_err': 0.0, 'n': 0, 'abs_err_snow': 0.0, 'sq_err_snow': 0.0, 'n_snow': 0,
            'dice_inter': 0.0, 'dice_union': 0.0
        }
        self._val_acc = {
            'abs_err': 0.0, 'sq_err': 0.0, 'sum_y': 0.0, 'sum_y2': 0.0,
            'sum_err': 0.0, 'n': 0, 'abs_err_snow': 0.0, 'sq_err_snow': 0.0, 'n_snow': 0,
            'dice_inter': 0.0, 'dice_union': 0.0
        }

    def forward(self, x):
        # Bound to [0,1]
        return torch.sigmoid(self.model(x))

    def _calculate_metrics(self, y_hat, y_raw):
        # y_raw: (N,H,W) with 0..100 valid, 200=cloud/pad
        valid = (y_raw != 200)
        if not valid.any():
            z = y_hat.new_tensor(0.0)
            return {"loss": z, "mae": z, "rmse": z, "mae_snow": z}

        y = (y_raw.clamp(0, 100) / 100.0).unsqueeze(1)
        
        # --- Loss Calculation ---
        loss_map = self.crit(y_hat, y)
        
        # Apply weighting to the loss
        weighting_mode = self.hparams.get('loss_weighting', 'none')
        if weighting_mode != 'none':
            snow_thresh_for_weighting = self.hparams.get('snow_thresh', 0.01)
            snow_mask = (y > snow_thresh_for_weighting)
            
            if weighting_mode == 'binary':
                snow_weight = self.hparams.get('snow_weight', 2.0)
                weights = torch.ones_like(y)
                weights[snow_mask] = snow_weight
                loss_map = loss_map * weights
            
            elif weighting_mode == 'linear':
                alpha = self.hparams.get('snow_weight', 1.0) # Here snow_weight acts as alpha
                weights = 1 + alpha * y
                loss_map = loss_map * weights

        # Final loss on valid pixels
        loss = loss_map[valid.unsqueeze(1)].mean()

        # --- Metric Calculation (unweighted) ---
        err = (y_hat - y).abs()
        
        # Overall MAE
        mae = err[valid.unsqueeze(1)].mean()
        
        # Overall RMSE
        rmse = torch.sqrt(((y_hat - y) ** 2)[valid.unsqueeze(1)].mean() + self.eps)
        
        # MAE on snow pixels only
        snow_pixels = (y > 0) & valid.unsqueeze(1)
        if snow_pixels.any():
            mae_snow = err[snow_pixels].mean()
            rmse_snow = torch.sqrt(((y_hat - y) ** 2)[snow_pixels].mean() + self.eps)
        else:
            mae_snow = y_hat.new_tensor(0.0) # No snow pixels in this batch
            rmse_snow = y_hat.new_tensor(0.0)

        # --- Dice Score Calculation ---
        # Use a threshold to binarize predictions and labels for Dice
        dice_thresh = 0.5 # Example threshold, can be tuned
        y_hat_bin = (y_hat > dice_thresh).float()
        y_bin = (y > dice_thresh).float()

        # Intersection and union for Dice, on valid pixels
        intersection = (y_hat_bin * y_bin)[valid.unsqueeze(1)].sum()
        union = y_hat_bin[valid.unsqueeze(1)].sum() + y_bin[valid.unsqueeze(1)].sum()
        dice = (2. * intersection + self.dice_smooth) / (union + self.dice_smooth)

        return {"loss": loss, "mae": mae, "rmse": rmse, "mae_snow": mae_snow, "rmse_snow": rmse_snow, "dice": dice}

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y_raw = batch.get('mask', batch.get('label')).float()
        y_hat = self(x)
        
        metrics = self._calculate_metrics(y_hat, y_raw)
        
        self.log('train_loss', metrics['loss'], prog_bar=True)
        self.log('train_mae', metrics['mae'])
        self.log('train_rmse', metrics['rmse'])
        self.log('train_mae_snow', metrics['mae_snow'])
        self.log('train_rmse_snow', metrics['rmse_snow'])
        self.log('train_dice', metrics['dice'])

        # Update accumulators
        valid = (y_raw != 200)
        if valid.any():
            y = (y_raw.clamp(0, 100) / 100.0).unsqueeze(1)
            v = valid.unsqueeze(1)
            err = (y_hat - y)
            self._train_acc['abs_err'] += err.abs()[v].sum().item()
            self._train_acc['sq_err'] += (err.pow(2))[v].sum().item()
            self._train_acc['sum_y'] += y[v].sum().item()
            self._train_acc['sum_y2'] += (y[v].pow(2)).sum().item()
            self._train_acc['sum_err'] += err[v].sum().item()
            self._train_acc['n'] += int(v.sum().item())
            
            # For epoch-level Dice
            dice_thresh = 0.5
            y_hat_bin = (y_hat > dice_thresh).float()
            y_bin = (y > dice_thresh).float()
            self._train_acc['dice_inter'] += (y_hat_bin * y_bin)[v].sum().item()
            self._train_acc['dice_union'] += (y_hat_bin[v].sum() + y_bin[v].sum()).item()

            snow_pixels = (y > 0) & v
            if snow_pixels.any():
                self._train_acc['abs_err_snow'] += err.abs()[snow_pixels].sum().item()
                self._train_acc['sq_err_snow'] += (err.pow(2))[snow_pixels].sum().item()
                self._train_acc['n_snow'] += int(snow_pixels.sum().item())

        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y_raw = batch.get('mask', batch.get('label')).float()
        y_hat = self(x)

        metrics = self._calculate_metrics(y_hat, y_raw)

        self.log('val_loss', metrics['loss'], prog_bar=True)
        self.log('val_mae', metrics['mae'])
        self.log('val_rmse', metrics['rmse'])
        self.log('val_mae_snow', metrics['mae_snow'], prog_bar=True)
        self.log('val_rmse_snow', metrics['rmse_snow'], prog_bar=True)
        self.log('val_dice', metrics['dice'], prog_bar=True)

        # Update accumulators
        valid = (y_raw != 200)
        if valid.any():
            y = (y_raw.clamp(0, 100) / 100.0).unsqueeze(1)
            v = valid.unsqueeze(1)
            err = (y_hat - y)
            self._val_acc['abs_err'] += err.abs()[v].sum().item()
            self._val_acc['sq_err'] += (err.pow(2))[v].sum().item()
            self._val_acc['sum_y'] += y[v].sum().item()
            self._val_acc['sum_y2'] += (y[v].pow(2)).sum().item()
            self._val_acc['sum_err'] += err[v].sum().item()
            self._val_acc['n'] += int(v.sum().item())

            # For epoch-level Dice
            dice_thresh = 0.5
            y_hat_bin = (y_hat > dice_thresh).float()
            y_bin = (y > dice_thresh).float()
            self._val_acc['dice_inter'] += (y_hat_bin * y_bin)[v].sum().item()
            self._val_acc['dice_union'] += (y_hat_bin[v].sum() + y_bin[v].sum()).item()

            snow_pixels = (y > 0) & v
            if snow_pixels.any():
                self._val_acc['abs_err_snow'] += err.abs()[snow_pixels].sum().item()
                self._val_acc['sq_err_snow'] += (err.pow(2))[snow_pixels].sum().item()
                self._val_acc['n_snow'] += int(snow_pixels.sum().item())

        return metrics['loss']

    def on_train_epoch_end(self):
        n = self._train_acc.get('n', 0)
        if n > 0:
            mae = self._train_acc['abs_err'] / n
            self.log('train_mae_epoch', mae)
            # ... other metrics ...
        
        n_snow = self._train_acc.get('n_snow', 0)
        if n_snow > 0:
            mae_snow = self._train_acc['abs_err_snow'] / n_snow
            rmse_snow = (self._train_acc['sq_err_snow'] / n_snow)**0.5
            self.log('train_mae_snow_epoch', mae_snow)
            self.log('train_rmse_snow_epoch', rmse_snow)
        
        # Calculate and log epoch-level Dice
        inter = self._train_acc.get('dice_inter', 0)
        union = self._train_acc.get('dice_union', 0)
        if union > 0:
            dice = (2. * inter + self.dice_smooth) / (union + self.dice_smooth)
            self.log('train_dice_epoch', dice)

    def on_validation_epoch_end(self):
        n = self._val_acc.get('n', 0)
        if n > 0:
            mae = self._val_acc['abs_err'] / n
            rmse = (self._val_acc['sq_err'] / n) ** 0.5
            self.log('val_mae_epoch', mae, prog_bar=True)
            self.log('val_rmse_epoch', rmse)
            # ... other metrics ...

        n_snow = self._val_acc.get('n_snow', 0)
        if n_snow > 0:
            mae_snow = self._val_acc['abs_err_snow'] / n_snow
            rmse_snow = (self._val_acc['sq_err_snow'] / n_snow)**0.5
            self.log('val_mae_snow_epoch', mae_snow, prog_bar=True)
            self.log('val_rmse_snow_epoch', rmse_snow, prog_bar=True)

        # Calculate and log epoch-level Dice
        inter = self._val_acc.get('dice_inter', 0)
        union = self._val_acc.get('dice_union', 0)
        if union > 0:
            dice = (2. * inter + self.dice_smooth) / (union + self.dice_smooth)
            self.log('val_dice_epoch', dice)

    def on_train_epoch_start(self):
        # Reset accumulators at the start of each epoch
        for acc in [self._train_acc, self._val_acc]:
            for k in acc:
                acc[k] = 0.0 if isinstance(acc[k], float) else 0

    def on_validation_epoch_start(self):
        # It's good practice to also reset val accumulator here if you have one
        for k in self._val_acc:
            self._val_acc[k] = 0.0 if isinstance(self._val_acc[k], float) else 0

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'], shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'],
                          num_workers=2, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

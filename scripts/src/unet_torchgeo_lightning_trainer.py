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

        # Accumulators for epoch metrics
        self._train_acc = {
            'abs_err': 0.0,
            'sq_err': 0.0,
            'sum_y': 0.0,
            'sum_y2': 0.0,
            'sum_err': 0.0,
            'n': 0,
        }
        self._val_acc = {
            'abs_err': 0.0,
            'sq_err': 0.0,
            'sum_y': 0.0,
            'sum_y2': 0.0,
            'sum_err': 0.0,
            'n': 0,
        }

    def forward(self, x):
        # Bound to [0,1]
        return torch.sigmoid(self.model(x))

    def _masked_loss_metrics(self, y_hat, y_raw):
        # y_raw: (N,H,W) with 0..100 valid, 200=cloud/pad
        valid = (y_raw != 200)
        if not valid.any():
            z = y_hat.new_tensor(0.0)
            return z, {"mae": z, "rmse": z}

        y = (y_raw.clamp(0, 100) / 100.0).unsqueeze(1)
        loss_map = self.crit(y_hat, y)
        loss = loss_map[valid.unsqueeze(1)].mean()

        err = (y_hat - y).abs()
        mae = err[valid.unsqueeze(1)].mean()
        rmse = torch.sqrt(((y_hat - y) ** 2)[valid.unsqueeze(1)].mean() + self.eps)
        return loss, {"mae": mae, "rmse": rmse}

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y_raw = batch.get('mask', batch.get('label')).float()
        y_hat = self(x)
        loss, m = self._masked_loss_metrics(y_hat, y_raw)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae', m['mae'])
        self.log('train_rmse', m['rmse'])

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
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y_raw = batch.get('mask', batch.get('label')).float()
        y_hat = self(x)
        loss, m = self._masked_loss_metrics(y_hat, y_raw)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', m['mae'], prog_bar=True)
        self.log('val_rmse', m['rmse'])

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
        return loss

    def on_train_epoch_end(self):
        n = self._train_acc['n']
        if n > 0:
            mae = self._train_acc['abs_err'] / n
            rmse = (self._train_acc['sq_err'] / n) ** 0.5
            y_mean = self._train_acc['sum_y'] / n
            sst = max(self._train_acc['sum_y2'] - n * (y_mean ** 2), self.eps)
            r2 = 1.0 - (self._train_acc['sq_err'] / sst)
            bias = self._train_acc['sum_err'] / n
            self.log('train_mae_epoch', mae)
            self.log('train_rmse_epoch', rmse)
            self.log('train_r2_epoch', r2)
            self.log('train_bias_epoch', bias)
        # reset
        for k in self._train_acc:
            self._train_acc[k] = 0.0 if k != 'n' else 0

    def on_validation_epoch_end(self):
        n = self._val_acc['n']
        if n > 0:
            mae = self._val_acc['abs_err'] / n
            rmse = (self._val_acc['sq_err'] / n) ** 0.5
            y_mean = self._val_acc['sum_y'] / n
            sst = max(self._val_acc['sum_y2'] - n * (y_mean ** 2), self.eps)
            r2 = 1.0 - (self._val_acc['sq_err'] / sst)
            bias = self._val_acc['sum_err'] / n
            # Show on the progress bar for easier interpretation
            self.log('val_mae_epoch', mae, prog_bar=True)
            self.log('val_rmse_epoch', rmse)
            self.log('val_r2_epoch', r2, prog_bar=True)
            self.log('val_bias_epoch', bias)
        # reset
        for k in self._val_acc:
            self._val_acc[k] = 0.0 if k != 'n' else 0

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'],
                          num_workers=self.hparams['num_workers'], shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'],
                          num_workers=2, shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])

import lightning.pytorch as pl
import numpy as np
import torch

from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from classification_code.ResidualBlock import ResidualBlock

from .Encoder import Encoder
from .Decoder import Decoder

class Autoencoder(pl.LightningModule):
        
    def __init__(
            self, input_dim: int = None, latent_dim: int= None,
            hidden_dims: np.ndarray = None,
            act_fn: nn.Module=None, optimizer: Optimizer = None,
            config_optimizer: dict = None, dropout: float = 0.4, 
            lr_scheduler: LRScheduler = None, config_lr_scheduler: dict = None,
            monitor: str = "val_loss",
            noise: Tensor = None, noise_level: float = 0.01, beta: float = 1.0
    ):
    
        """Autoencoder class for DAE (Denoising Autoencoder) model.
        
        Args:
            input_dim (int): Input dimension.
            latent_dim (int): Latent dimension.
            hidden_dims (np.ndarray): Hidden layer dimensionality.
            act_fn : Activation function.
            optimizer : Optimizer.
            config_optimizer : Optimizer configuration.
            dropout : Dropout rate.
            lr_scheduler : Learning rate scheduler.
            config_lr_scheduler : Learning rate scheduler configuration.
            monitor : Monitor.
            noise (Tensor): Noise tensor.
            noise_level (float): Noise level.
            beta (float): Beta value.
        """

        super().__init__()
        if input_dim is None:
            return
        self.save_hyperparameters()
        self.loss_function = nn.MSELoss(reduction="mean")
        self.eps = 1e-8  # Small constant for numerical stability
        self.warmup_epochs = 1
        self.smoothing = 0.1
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims, act_fn, dropout, noise, noise_level)
        self._init_weights(self.encoder.encoder)
        self._init_weights(self.encoder.latent_layer)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dims, act_fn, dropout)
        self._init_weights(self.decoder.decoder)
        self._init_weights(self.decoder.output_layer)
        self.kl_weight = 0.01  # Add KL divergence weight
        self.max_beta = beta

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer
        if self.hparams.config_optimizer is not None:
            self.optimizer = optimizer(self.parameters(), **self.hparams.config_optimizer)
        else:
            self.optimizer = optimizer()
        
        lr_scheduler = self.hparams.lr_scheduler
        
        dict_ret = {"optimizer":  self.optimizer, "monitor": self.hparams.monitor}
        
        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler( self.optimizer, **self.hparams.config_lr_scheduler)

            dict_ret["lr_scheduler"] = {
                "scheduler": lr_scheduler,
                "monitor": self.hparams.monitor,
            }

    
        return dict_ret
    
    def update_current_epoch(self, current_epoch: int):
        self.encoder.current_epoch = current_epoch
        self.decoder.current_epoch = current_epoch
    
    def _get_reconstruction_loss(self, x: Tensor) -> Tensor:
        """
        Calculate the total reconstruction loss including KL divergence and regularization.
        
        Args:
            x (Tensor): Input tensor to reconstruct
            
        Returns:
            Tensor: Total loss value
        """
       

        
         # Get encoded representation
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        # MSE reconstruction loss with gradient clipping
        mse_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
        l1_loss = nn.functional.l1_loss(x_hat, x, reduction='mean')
        recon_loss = 0.8 * mse_loss + 0.2 * l1_loss
        # print(f"Debug - x_hat stats: min={x_hat.min()}, max={x_hat.max()}, mean={x_hat.mean()}")
        # print(f"Debug - x stats: min={x.min()}, max={x.max()}, mean={x.mean()}")
        # print(f"Debug - z stats: min={z.min()}, max={z.max()}, mean={z.mean()}")

        # 3. Feature-weighted loss 
        # Give more weight to important features
        feature_weights = torch.tensor([
            0.15,  # points
            0.15,  # uci_points
            0.1,   # length
            0.1,   # climb_total
            0.15,  # profile
            0.1,   # startlist_quality
            0.075, # cyclist_age
            0.075  # delta
        ]).to(x.device)
        
        weighted_loss = torch.mean(feature_weights * (x_hat - x).pow(2))

        # Improved KL divergence calculation
        z_var = torch.var(z, dim=1).clamp(min=self.eps)
        z_mean = torch.mean(z, dim=1)
        
        kl_loss = -0.5 * torch.mean(1 + torch.log(z_var) - z_mean.pow(2) - z_var)
        
        # Dynamic beta scheduling
        beta = self.max_beta * min(1.0, self.current_epoch / self.warmup_epochs)
        
        # Lighter L2 regularization
        l2_reg = 0.0001 * sum(p.pow(2.0).sum() for p in self.parameters())
        
        # Weighted loss combination
        total_loss = (
            0.7 * recon_loss + 
            0.3 * weighted_loss +
            beta * kl_loss + 
            l2_reg
        )
 
        
        # Only log metrics during training
        if self.training:
            self.log_dict({
                "mse_recon_loss": mse_loss,
                "kl_loss": kl_loss,
                "l2_reg": l2_reg,
                "total_loss": total_loss
            }, on_epoch=True)

        # print(f"Debug - total_loss={total_loss} - recon_loss={recon_loss} - kl_loss={self.kl_weight * kl_loss} - l2_reg={l2_reg} ")

        
        return total_loss
        


      
    
    def _init_weights_helper(self, model: nn.Module):
        for layer in model:
            if isinstance(layer, nn.Linear):
                # Use larger init range
                nn.init.xavier_uniform_(layer.weight, gain=0.2)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.normal_(layer.weight,mean=1.0,std=0.002)
                nn.init.zeros_(layer.bias)

            elif isinstance(layer, ResidualBlock):
                self._init_weights(layer.main_branch)
                self._init_weights(layer.skip)
            
    def _init_weights(self, model: nn.Module):
        if isinstance(model, nn.ModuleList) or isinstance(model, nn.Sequential):
            self._init_weights_helper(model)
        else:
            print(f"Model: {model}")
            
        
        
    
        
    def training_step(self, batch: Tensor, batch_idx: int):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        if batch_idx == 0:  # Log first batch of each epoch
            x = batch
            x_hat = self.forward(x)
            self.logger.experiment.add_histogram("input_distribution", x, self.current_epoch)
            self.logger.experiment.add_histogram("output_distribution", x_hat, self.current_epoch)
            
        # Log learning rate
        self.log("lr", self.optimizers().param_groups[0]["lr"])
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch: Tensor, batch_idx: int):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch import rand_like
from .logging import setup_logger
from .validation import validate_dimensions
from .ResidualBlock import ResidualBlock

logger = setup_logger("Autoencoder")


class Encoder(nn.Module):
    def __init__(
            self, input_dim: int, latent_dim: int,
            scale_factor: np.ndarray,
            act_fn: nn.Module, dropout: float = 0.4,
            noise: Tensor = None, noise_level: float = 0.01
        ):
        super().__init__()
        
        # Input validation
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("Dimensions must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("Dropout must be between 0 and 1")
        
        # Initialize attributes
        if noise is not None:
            assert noise.size(0) == input_dim, "Noise size must match input dimension."

        if isinstance(act_fn, str):
            act_fn = getattr(nn, act_fn)()
        else:
            act_fn = act_fn()

        self.input_dim = input_dim
        self.num_hidden_layers = len(scale_factor)
        self.scale_factor = scale_factor
        self.noise = noise
        self.act_fn = act_fn
        self.noise_level = noise_level
        self.latent_dim = latent_dim
        self.hidden_dims = []
        self.encoder = nn.ModuleList()
        self.current_epoch = 0
        
        for i in range(self.num_hidden_layers):
            next_dim = validate_dimensions(input_dim, scale_factor[i], 'multiply')
            next_dim = max(next_dim, latent_dim)
            
            # Add residual block
            self.encoder.append(ResidualBlock(input_dim, next_dim, act_fn))
            self.encoder.append(nn.Dropout(dropout))
            
            self.hidden_dims.append(next_dim)
            # print(f"Encoder Layer {i}: {input_dim} -> {next_dim}")

            input_dim = next_dim
            

        self.latent_layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            act_fn,
            nn.LayerNorm(latent_dim),

        )
        # print(f"Encoder Latent Layer: {input_dim} -> {latent_dim}")

        # Initialize weights
        # self.apply(self._init_weights)
        

    def _init_weights(self, module):
        # print(f"Initializing weights for {module}")
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
                
    def _log_architecture(self):
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Encoder Architecture:")
        logger.info(f"Input dimension: {self.input_dim}")
        logger.info(f"Latent dimension: {self.latent_dim}")
        logger.info(f"Hidden dimensions: {self.hidden_dims}")
        logger.info(f"Total parameters: {total_params}")

    def add_noise(self, x: Tensor) -> Tensor:
        if self.noise is not None:
            return x + self.noise * self.noise_level
        # Use smaller noise and decay over time
        noise_factor = self.noise_level * (0.995 ** self.current_epoch)
        return x + torch.randn_like(x) * noise_factor

    def forward(self, x: Tensor) -> Tensor:
        # Add noise
        x = self.add_noise(x)
        
        # Pass through encoder layers
        for layer in self.encoder:
            x = layer.forward(x)
            
        # Pass through latent layer
        return self.latent_layer.forward(x)
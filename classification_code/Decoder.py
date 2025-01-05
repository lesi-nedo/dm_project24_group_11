import numpy as np
import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict
from .logging import setup_logger
from .validation import validate_dimensions
from .ResidualBlock import ResidualBlock

logger = setup_logger("Autoencoder")

class Decoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: np.ndarray, 
                 act_fn: nn.Module, dropout: float = 0.4):
        super().__init__()
        
        # Input validation
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("Dimensions must be positive")
        if not 0 <= dropout <= 1:
            raise ValueError("Dropout must be between 0 and 1")

        # Initialize attributes
        if isinstance(act_fn, str):
            act_fn = getattr(nn, act_fn)()
        else:
            act_fn = act_fn()

        self.hidden_dims = []
        self.latent_dim = latent_dim
        self.num_hidden_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.act_fn = act_fn
        self.input_dim = input_dim
        self.current_epoch = 0

        self.decoder = nn.ModuleList()
        prev_dim = input_dim
        for i in range(self.num_hidden_layers):
            next_dim = hidden_dims[i]
            self.hidden_dims.append(next_dim)
            prev_dim = next_dim
        
        self.hidden_dims = self.hidden_dims[::-1]
        prev_dim = latent_dim
        # Calculate hidden dimensions
        for i in range(self.num_hidden_layers):
            next_dim = self.hidden_dims[i]
            
            # Add residual block
            self.decoder.append(ResidualBlock(prev_dim, next_dim, act_fn))
            self.decoder.append(nn.Dropout(dropout))
            
            # print(f"Decoder Layer {i}: {prev_dim} -> {next_dim}")
            prev_dim = next_dim        
    
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()  
        )
        # print(f"Decoder Output Layer: {prev_dim} -> {input_dim}")

        # Initialize weights with smaller values
        # self.apply(self._init_weights)
        
        # Log architecture

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def _log_architecture(self):
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Decoder Architecture:")
        logger.info(f"Latent dimension: {self.latent_dim}")
        logger.info(f"Hidden dimensions: {self.hidden_dims}")
        logger.info(f"Output dimension: {self.input_dim}")
        logger.info(f"Total parameters: {total_params}")

    def forward(self, x: Tensor) -> Tensor:
        # Pass through decoder layers
        for layer in self.decoder:
            x = layer(x)
        
        # Pass through output layer
        result = self.output_layer(x)
        return result

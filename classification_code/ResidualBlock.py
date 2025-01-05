import torch
import torch.nn as nn
from torch import Tensor

class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, act_fn: nn.Module):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Main branch with scaling factor
        self.main_branch = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            act_fn
        )
        
        # Residual branch with adaptive scaling
        self.skip = nn.Sequential(
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity(),
            nn.LayerNorm(out_dim)
        )
        
        # Learnable scaling factors
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)  # Main path scaling
        self.beta = nn.Parameter(torch.ones(1) * 0.5)   # Skip path scaling
        
        
        # Add safeguard epsilon
        self.eps = 1e-8
        
    def forward(self, x: Tensor) -> Tensor:
        # Use less aggressive clamping
        x = torch.clamp(x, min=-5, max=5)
        
        # Allow more signal through
        alpha = torch.sigmoid(self.alpha) 
        beta = torch.sigmoid(self.beta)
        
        main = self.main_branch(x)
        skip = self.skip(x)
        
        # Add skip connection with less normalization
        output = alpha * main + beta * skip
        return output

    def extra_repr(self) -> str:
        return f'in_dim={self.in_dim}, out_dim={self.out_dim}'

import torch
import torch.nn as nn
from .AdaLayerNorm import AdaLayerNorm

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block with singer conditioning.
    """
    def __init__(self, latent_dim, intermediate_dim, layer_scale_init_value=1e-6):
        """
        latent_dim: number of latent dimensions (64 in your case)
        intermediate_dim: expansion size (e.g., 256)
        layer_scale_init_value: for training stability
        """
        super().__init__()
        self.dwconv = nn.Conv1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=7, padding=3, groups=latent_dim)

        self.norm = AdaLayerNorm(latent_dim, latent_dim)

        self.mix = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, latent_dim)
        )

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(latent_dim), requires_grad=True) if layer_scale_init_value > 0 else None


    def forward(self, x, condition):
        residual = x
        
        # Apply AdaLayerNorm with conditioning
        x = torch.transpose(x, -1, -2) # [B, T, C]
        x = self.norm(x, condition)     # [B, T, C]
        x = torch.transpose(x, -1, -2)  # [B, C, T]
        
        # Apply depthwise conv on temporal patterns
        x = self.dwconv(x)  # [B, C, T]
        
        # Mix channels
        x = torch.transpose(x, -1, -2)  # [B, T, C]
        x = self.mix(x)                 # [B, T, C]
        
        # Apply layer scale
        x = self.gamma * x              # [B, T, C]
        
        # Residual connection
        x = torch.transpose(x, -1, -2)  # [B, C, T]
        x = residual + x
        
        return x

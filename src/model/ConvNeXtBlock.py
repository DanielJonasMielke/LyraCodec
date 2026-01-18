import torch
import torch.nn as nn
from .AdaLayerNorm import AdaLayerNorm

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block with adaptive conditioning.
    
    Flow:
    1. Depthwise conv (temporal patterns per dimension)
    2. AdaLayerNorm (FiLM conditioning)
    3. Pointwise conv with expansion (mix dimensions)
    4. Residual connection
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
        """
        x: [B, C, T] - latent features (channels FIRST)
        condition: [B, C] - singer embedding
        
        Returns: [B, C, T]
        """
        residual = x
        
        # Step 1: Depthwise conv
        x = self.dwconv(x)  # [B, C, T] -> [B, C, T]
        
        # Step 2: Transpose for LayerNorm (needs channels LAST)
        x = torch.transpose(x, -1, -2)
        
        # Step 3: Apply AdaLayerNorm with condition
        x = self.norm(x, condition)
        
        # Step 4: Pointwise expansion and mixing
        x = self.mix(x)
        
        # Step 5: Apply layer scale if it exists
        x = self.gamma * x
        
        # Step 6: Transpose back to channels-first
        x = torch.transpose(x, -1, -2)

        # Step 7: Residual connection
        x = residual + x
        
        return x

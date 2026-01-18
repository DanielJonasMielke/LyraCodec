import torch.nn as nn
from .ConvNeXtBlock import ConvNeXtBlock

class SingerConditioningModule(nn.Module):
    """
    Applies FiLM conditioning with ConvNeXt blocks to VAE latents.
    
    Takes:
        z: [B, C, T] - VAE latents
        singer_emb: [B, C] - projected singer identity
    
    Returns:
        conditioned_z: [B, C, T]
    """
    def __init__(self, latent_dim=64, intermediate_dim=256, num_blocks=2):
        """
        latent_dim: your VAE latent dimension (64)
        intermediate_dim: expansion size in ConvNeXt (256)
        num_blocks: how many ConvNeXt blocks to stack (2 is reasonable)
        """
        super().__init__()
    
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim

        convnextblocks = []

        for _ in range(num_blocks):
            convnextblocks.append(ConvNeXtBlock(latent_dim, intermediate_dim))

        self.blocks = nn.ModuleList(convnextblocks)
    
    def forward(self, z, singer_emb):
        """
        z: [B, C, T]
        singer_emb: [B, C]
        
        Returns: [B, C, T]
        """ 
        x = z       
        for block in self.blocks:
            x = block(x, singer_emb)
        return x

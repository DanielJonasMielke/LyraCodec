import math
import torch
import torch.nn as nn

from .SnakeActivation import SnakeActivation
from .ResidualUnit import ResidualUnit

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels, dilation=1), # Local scale Residual dilation
            ResidualUnit(in_channels, dilation=3), # mid-scale Residual dilation
            ResidualUnit(in_channels, dilation=9), # Larger Scale Residual dilation

            SnakeActivation(in_channels),

            # Downsample to desired size
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2)
            )
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class VAE(nn.Module):
    def __init__(self,
                 in_channels=2,
                 base_channels=128,
                 latent_dim=64,
                 c_mults=[
                     (1, 1), (1, 2), (2, 4), (4, 8), (8, 16) # Each tuple is (in_mult, out_mult)
                     ],
                 strides=[2, 4, 4, 8, 8]):
        super().__init__()
        
        self.in_channels = in_channels
        
        # First layer: stereo audio (2 channels) -> 128 feature channels
        self.encoder_init = nn.Conv1d(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=7,
            padding=3 
        )
        
        encoder_blocks = []

        for i in range(len(strides)):
            in_chan = base_channels * c_mults[i][0]
            out_chan = base_channels * c_mults[i][1]

            encoder_blocks.append(
                EncoderBlock(in_chan, out_chan, strides[i])
                )
            
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        self.encoder_final_activation = SnakeActivation(base_channels * c_mults[-1][1])

        # Final latent projection
        self.distribution_conv = nn.Conv1d(
            in_channels=base_channels * c_mults[-1][1],
            out_channels=2 * latent_dim, # 64 channels for mu and 64 for var
            kernel_size=3,
            padding=1
        )

    def encode(self, x):
        x = self.encoder_init(x) # 2 -> 128 channels

        for block in self.encoder_blocks:
            # Block 0: (128, 88200) -> (128, 44100)
            # Block 1: (128, 44100) -> (256, 11025)
            # Block 2: (256, 11025) -> (512, 2756)
            # Block 3: (512, 2756)  -> (1024, 344)
            # Block 4: (1024, 344)  -> (2048, 43)
            x = block(x)

        x = self.encoder_final_activation(x)

        x = self.distribution_conv(x) # (2048, 43) -> (128, 43)
        
        mu, logvar = torch.chunk(x, 2, dim=1) # Each of shape (64, 43)

        return mu, logvar

    def reparameterization(self, mu, logvar):
        """Sample z from the distributrion using the reparameterization trick

        Args:
            mu (Tensor): the mean
            logvar (Tensor): the variance stored in logspace
        """
        eps = torch.randn_like(mu)
        z = mu + eps * torch.sqrt(torch.exp(logvar))
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        return mu, logvar, z


if __name__ == "__main__":
    vae = VAE()
    audio = torch.randn(1, 2, 88200)
    mu, logvar, z = vae(audio)
    print(f"mu shape: {mu.shape}")  # Should be [1, 64, 43]
    print(f"logvar shape: {logvar.shape}")  # Should be [1, 64, 43]

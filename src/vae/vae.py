import math
import torch
import torch.nn as nn

from .ResidualUnit import ResidualUnit
from .SnakeActivation import SnakeActivation



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        self.res1 = ResidualUnit(in_channels, dilation=1)
        self.res2 = ResidualUnit(in_channels, dilation=3)
        self.res3 = ResidualUnit(in_channels, dilation=9)
        
        # Activation BEFORE downsampling
        self.activation = SnakeActivation(in_channels)
        
        self.downsample = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2)
        )
    
    def forward(self, x):
        # Apply all residual units
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        # Activation
        x = self.activation(x)
        # Downsample
        x = self.downsample(x)
        return x

class AudioVAE(nn.Module):
    def __init__(self,
             in_channels=2,
             base_channels=128,
             latent_dim=64,
             c_mults=[1, 2, 4, 8, 16],
             strides=[2, 4, 4, 8, 8]):
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        c_mults = [1] + c_mults
        
        # Initial convolution
        self.encoder_init = nn.Conv1d(
            in_channels=in_channels,
            out_channels=base_channels,  # base_channels * c_mults[0] = 128 * 1
            kernel_size=7,
            padding=3
        )
        
        # Build encoder blocks
        encoder_blocks = []
        for i in range(len(strides)):
            in_ch = base_channels * c_mults[i]    
            out_ch = base_channels * c_mults[i + 1]  
            
            encoder_blocks.append(
                EncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=strides[i]
                )
            )
        
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        
        # Final layers after all encoder blocks
        final_channels = base_channels * c_mults[-1]  # 128 * 16 = 2048
        self.encoder_final_activation = SnakeActivation(final_channels)
        
        self.encoder_final_conv = nn.Conv1d(
            in_channels=final_channels,
            out_channels=latent_dim * 2,
            kernel_size=3,
            padding=1
        )
    
    def encode(self, x):
        """
        Encodes audio to latent representation.
        
        Args:
            x: [batch, 2, 88200] - stereo audio
            
        Returns:
            mu: [batch, 64, 43] - mean of latent distribution
            logvar: [batch, 64, 43] - log-variance of latent distribution
        """
        # Initial conv: [batch, 2, 88200] → [batch, 128, 88200]
        x = self.encoder_init(x)
        
        # Apply encoder blocks (downsampling)
        for block in self.encoder_blocks:
            x = block(x)
        # After all blocks: [batch, 2048, 43]
        
        # Final activation and conv
        x = self.encoder_final_activation(x)
        x = self.encoder_final_conv(x)
        # Result: [batch, 128, 43] (latent_dim * 2)
        
        # Split into mean and log-variance
        mu, logvar = torch.chunk(x, 2, dim=1)
        # mu: [batch, 64, 43]
        # logvar: [batch, 64, 43]
        
        return mu, logvar
    
    def forward(self, x):
        # For now, just encode
        mu, logvar = self.encode(x)
        return mu, logvar


if __name__ == "__main__":
    vae = AudioVAE()
    
    audio = torch.randn(1, 2, 88200)
    
    # Encode
    mu, logvar = vae(audio)
    
    print(f"Input shape: {audio.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"\nExpected latent rate: 44100 / 2048 = {44100/2048:.2f} Hz")
    print(f"Actual latent rate: 88200 / {mu.shape[2]} = {88200/mu.shape[2]:.2f} Hz")
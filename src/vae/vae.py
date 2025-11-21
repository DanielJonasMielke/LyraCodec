import torch
import torch.nn as nn

from .SnakeActivation import SnakeActivation
from .EncoderBlock import EncoderBlock
from .DecoderBlock import DecoderBlock

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
        
        # ----------------- ENCODER PROPERTIES ----------------- 
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

        # Final Encoder latent projection
        self.distribution_conv = nn.Conv1d(
            in_channels=base_channels * c_mults[-1][1],
            out_channels=2 * latent_dim, # 64 channels for mu and 64 for var
            kernel_size=3,
            padding=1
        )

        # -------------------------------------------------------
        # -----------------  DECODER PROPERTIES ----------------- 
        # projection layer to upscale from the latent channels (64, 43) -> (2048, 43) 
        self.decoder_projection = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=base_channels * c_mults[-1][1],
            kernel_size=3,
            padding=1
        )

        decoder_blocks = []

        for i in range(len(strides)):
            in_channels = base_channels * c_mults[-1-i][1]
            out_channels = base_channels * c_mults[-1-i][0]

            decoder_blocks.append(
                DecoderBlock(in_channels, out_channels, strides[-1-i])
                )
            
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.decoder_final_activation = SnakeActivation(base_channels)

        self.decoder_final_projection = nn.Conv1d(
            in_channels=base_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            padding=3
            )


    def encode(self, x):
        shapes = []  # Clear previous shapes
        x = self.encoder_init(x) # 2 -> 128 channels

        for block in self.encoder_blocks:
            # Block 0: (128, 90112) -> (128, 45056)
            # Block 1: (128, 45056) -> (256, 11264)
            # Block 2: (256, 11264) -> (512, 2816)
            # Block 3: (512, 2816)  -> (1024, 352)
            # Block 4: (1024, 352)  -> (2048, 44)
            shapes.append(x.shape)
            x = block(x)

        x = self.encoder_final_activation(x)

        x = self.distribution_conv(x) # (2048, 44) -> (128, 44)
        
        mu, logvar = torch.chunk(x, 2, dim=1) # Each of shape (64, 44)

        return mu, logvar, shapes

    def reparameterization(self, mu, logvar):
        """Sample z from the distributrion using the reparameterization trick

        Args:
            mu (Tensor): the mean
            logvar (Tensor): the variance stored in logspace
        """
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        return z
    
    def decode(self, z, encoder_shapes):
        z = self.decoder_projection(z) # (64, 44) -> (2048, 44)

        for i, block in enumerate(self.decoder_blocks):
            # Block 0: (2048, 44) -> (1024, 352)
            # Block 1: (1024, 352) -> (512, 2816)
            # Block 2: (512, 2816) -> (256, 11264)
            # Block 3: (256, 11264)  -> (128, 45056)
            # Block 4: (128, 45056)  -> (128, 90112)
            z = block(z)
            target_length = encoder_shapes[-(i+1)][-1]
            current_length = z.shape[-1]
            if current_length < target_length:
                # Pad with zeros
                padding_needed = target_length - current_length
                z = torch.nn.functional.pad(z, (0, padding_needed))
            elif current_length > target_length:
                # Crop
                z = z[..., :target_length]

        z = self.decoder_final_activation(z)

        x_recon = self.decoder_final_projection(z) # (128, 90112) -> (2, 90112)

        return x_recon

    def forward(self, x):
        mu, logvar, encoder_shapes = self.encode(x)
        z = self.reparameterization(mu, logvar)
        x_recon = self.decode(z, encoder_shapes)
        return mu, logvar, z, x_recon


if __name__ == "__main__":
    vae = VAE()
    audio = torch.randn(1, 2, 90112)
    mu, logvar, z, x_recon = vae(audio)
    print(f"mu shape: {mu.shape}")  # Should be [1, 64, 44]
    print(f"logvar shape: {logvar.shape}")  # Should be [1, 64, 44]
    print(f"z shape: {z.shape}")  # Should be [1, 64, 44]
    print(f"x_recon shape: {x_recon.shape}")  # Should be [1, 2, 90112]

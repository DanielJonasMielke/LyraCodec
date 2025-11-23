import math
import torch.nn as nn

from .SnakeActivation import SnakeActivation
from .ResidualUnit import ResidualUnit

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, dilation=1), # Local scale Residual dilation
            ResidualUnit(in_channels, dilation=3), # mid-scale Residual dilation
            ResidualUnit(in_channels, dilation=9), # Larger Scale Residual dilation

            SnakeActivation(in_channels),

            # Upsample to desired size
            nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2
            )
        )

    def forward(self, x):
        x = self.layers(x)
        return x
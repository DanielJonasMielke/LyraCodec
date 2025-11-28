import torch.nn as nn

from .SnakeActivation import SnakeActivation

class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation, kernel_size=7, num_groups=None):
        super().__init__()

        # padding formula: (dilation * (kernel_size - 1)) // 2
        padding = (dilation * (kernel_size - 1)) // 2

        if num_groups is None:
            num_groups = min(32, channels) if channels >= 2 else 1

        self.layers = nn.Sequential(
            SnakeActivation(channels),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            ), # Scan on different timescales
            nn.GroupNorm(num_groups=num_groups, num_channels=channels, affine=True),
            SnakeActivation(channels),
            nn.Conv1d(
                in_channels=channels, 
                out_channels=channels,      
                kernel_size=1
            ), # Mix channels
            nn.GroupNorm(num_groups=num_groups, num_channels=channels, affine=True)
        )

    def forward(self, x):
        residual = x
        x = self.layers(x)
        return x + residual

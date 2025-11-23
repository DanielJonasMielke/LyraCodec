import torch.nn as nn

from .SnakeActivation import SnakeActivation

class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation, kernel_size=7):
        super().__init__()

        # padding formula: (dilation * (kernel_size - 1)) // 2
        padding = (dilation * (kernel_size - 1)) // 2

        self.layers = nn.Sequential(
            SnakeActivation(channels),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            ), # Scan on different timescales
            SnakeActivation(channels),
            nn.Conv1d(
                in_channels=channels, 
                out_channels=channels,      
                kernel_size=1
            ) # Mix channels
        )

    def forward(self, x):
        residual = x
        x = self.layers(x)
        return x + residual

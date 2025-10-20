import torch.nn as nn

from .SnakeActivation import SnakeActivation


class ResidualUnit(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        
        # Calculate padding to keep length the same
        # For kernel_size=7 and dilation, padding = (dilation * (7-1)) // 2
        padding = (dilation * 6) // 2
        
        self.layers = nn.Sequential(
            SnakeActivation(channels),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding
            ),
            SnakeActivation(channels),
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1 
            )
        )
    
    def forward(self, x):
        residual = x
        x = self.layers(x)
        return x + residual
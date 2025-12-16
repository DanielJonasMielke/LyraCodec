import torch
import torch.nn as nn
import torch.nn.functional as F

class WNConv1d(nn.Module):
    """1D Convolution with weight normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, 
                     stride=stride, padding=padding, groups=groups)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
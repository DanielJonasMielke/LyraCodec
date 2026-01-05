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
    
class ScaleDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.convs = nn.ModuleList([
            WNConv1d(in_channels, 16, kernel_size=15, stride=1, padding=7),
            WNConv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            WNConv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
            WNConv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
            WNConv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
            WNConv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
        ])

        self.conv_post = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        features = []
        
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        
        x = self.conv_post(x)
        features.append(x)
        
        score = x.flatten(1).mean(dim=1)
        
        return score, features

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=1, n_scales=[1, 2, 4]):
        super().__init__()
        self.n_scales = n_scales
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(in_channels) for _ in range(len(n_scales))
        ])
        
    def forward(self, x):
        scores = []
        features_list = []
        x_orig = x
        
        for i, discriminator in enumerate(self.discriminators):
            x = F.avg_pool1d(x_orig, kernel_size=self.n_scales[i], stride=self.n_scales[i])
            score, features = discriminator(x)
            scores.append(score)
            features_list.append(features)
        
        return scores, features_list
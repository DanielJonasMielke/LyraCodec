import torch
import torch.nn as nn
import torch.nn.functional as F
    
class WNConv2d(nn.Module):
    """2D Convolution with weight normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        )
        self.activation = nn.LeakyReLU(0.2) # Why use leaky relu here?

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    
class PeriodDiscriminator(nn.Module):
    """Discriminator for single period"""
    def __init__(self, period, in_channels=1):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            WNConv2d(in_channels, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            WNConv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            WNConv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            WNConv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0)),
            WNConv2d(1024, 1024, kernel_size=(5, 1), stride=1, padding=(2, 0)),
        ])
        
        self.conv_post = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

    def forward(self, x):
        # x shape: (batch, channels, time)
        features = []
        
        # Pad so length is divisible by period
        t = x.shape[-1]
        if t % self.period != 0:
            padding = self.period - (t % self.period)
            x = F.pad(x, (0, padding), mode='reflect')
        
        # Reshape: (batch, channels, time) -> (batch, channels, time//period, period)
        x = x.reshape(x.shape[0], x.shape[1], -1, self.period)
        
        # Pass through conv layers, collecting intermediate features
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        
        # Final layer for score
        x = self.conv_post(x)
        features.append(x)
        
        # Flatten score to single value per batch
        score = x.flatten(1).mean(dim=1)
        
        return score, features
    
class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, in_channels, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period, in_channels) for period in periods
        ])

    def forward(self, x):
        features_list = []
        prediction_scores = []

        for discriminator in self.discriminators:
            prediction, features = discriminator(x)
            prediction_scores.append(prediction)
            features_list.append(features)
        return prediction_scores, features_list
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiPeriodDiscriminator(in_channels=1).to(device)
    x = torch.randn(2, 1, 71680).to(device)
    scores, features = model(x)

    print("Output scores and features:")
    for i, score in enumerate(scores):
        print(f"Discriminator {i} score shape: {score.shape}") # Discriminator 0 score shape: torch.Size([2])
    for i, feature_set in enumerate(features):
        print(f"Discriminator {i} features:")
        for j, feature in enumerate(feature_set):
            print(f"  Feature {j} shape: {feature.shape}")

# Discriminator 0 features:
#   Feature 0 shape: torch.Size([2, 32, 11947, 2])
#   Feature 1 shape: torch.Size([2, 128, 3983, 2])
#   Feature 2 shape: torch.Size([2, 512, 1328, 2])
#   Feature 3 shape: torch.Size([2, 1024, 443, 2])
#   Feature 4 shape: torch.Size([2, 1024, 443, 2])
#   Feature 5 shape: torch.Size([2, 1, 443, 2])
# Discriminator 1 features:
#   Feature 0 shape: torch.Size([2, 32, 7965, 3])
#   Feature 1 shape: torch.Size([2, 128, 2655, 3])
#   Feature 2 shape: torch.Size([2, 512, 885, 3])
#   Feature 3 shape: torch.Size([2, 1024, 295, 3])
#   Feature 4 shape: torch.Size([2, 1024, 295, 3])
#   Feature 5 shape: torch.Size([2, 1, 295, 3])
# ...
# (batch, channels, time_periods, period)
# [a, b, c] # Period 0
# [d, e, f] # Period 1
# [g, h, i] # Period 2
# ...
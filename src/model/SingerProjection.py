import torch.nn as nn

class SingerProjection(nn.Module):
    def __init__(self, input_dim=1000, output_dim=64, linear=False):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim

        if linear:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.SiLU(),
                nn.Linear(256, output_dim)
            )

    def forward(self, x):
        return self.projection(x)
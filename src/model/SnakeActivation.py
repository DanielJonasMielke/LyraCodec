import torch
import torch.nn as nn

class SnakeActivation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Initialize in LOG space (as zeros)
        self.log_alpha = nn.Parameter(torch.zeros(channels))
        self.log_beta = nn.Parameter(torch.zeros(channels))
    
    def forward(self, x):
        log_alpha = self.log_alpha.unsqueeze(0).unsqueeze(-1) # unsqueeze adds dimension at the start (batch) and unsqueeze(-1) at the end (time)
        log_beta = self.log_beta.unsqueeze(0).unsqueeze(-1) # --> so [1, channels, 1]
        
        # Convert from log-space to actual values using exp
        # exp(anything) is always positive
        alpha = torch.exp(log_alpha)  # exp(0) = 1.0 initially
        beta = torch.exp(log_beta)    
        
        return x + (1.0 / (beta + 1e-9)) * torch.sin(alpha * x) ** 2
    



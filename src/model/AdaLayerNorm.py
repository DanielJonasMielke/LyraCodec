import torch
import torch.nn as nn

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization - FiLM.
    
    Takes:
        - x: features [B, T, C] (channels LAST)
        - condition: singer embedding [B, C]
    
    Returns:
        - conditioned features [B, T, C]
    """
    def __init__(self, condition_dim, embedding_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        
        self.scale = nn.Linear(condition_dim, embedding_dim)
        self.shift = nn.Linear(condition_dim, embedding_dim)
        
        # Initialize scale to output 1s, shift to output 0s
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.scale.bias)
        torch.nn.init.zeros_(self.shift.weight)
        torch.nn.init.zeros_(self.shift.bias)
    
    def forward(self, x, condition):
        scale = self.scale(condition)  # [B, C]
        shift = self.shift(condition)  # [B, C]
        
        normalized = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)  # [B, T, C]
        
        # Broadcast scale/shift across time dimension
        scale = scale.unsqueeze(1)  # [B, 1, C]
        shift = shift.unsqueeze(1)  # [B, 1, C]
        
        # Apply FiLM
        conditioned = scale * normalized + shift  # [B, T, C]
        return conditioned
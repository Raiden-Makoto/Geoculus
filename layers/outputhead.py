import torch
import torch.nn as nn

class OutputHead(nn.Module):
    """
    Standard regression head: Dense -> Softplus -> Dense
    """
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
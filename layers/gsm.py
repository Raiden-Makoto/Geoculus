import torch
import torch.nn as nn

class GaussianSmearing(nn.Module):
    """
    Gaussian smearing is a technique in machine learning and data processing
    that involves applying a Gaussian (normal distribution) function to
    data points to create a smoother, continuous representation.
    Here we expand a distance scalar into a vector of RBFs (radial basis functions).
    """
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        # The width (coeff) is determined by the spacing between centers
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        # dist shape: [num_edges, 1]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
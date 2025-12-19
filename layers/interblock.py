import torch
import torch.nn as nn
from torch_geometric.nn import CGConv

class InteractionBlock(nn.Module):
    """
    A single Message Passing Layer.
    Hot-swappable for other GNN layers (GAT, Transformer).
    """
    def __init__(self, atom_features, rbf_features):
        super().__init__()
        self.conv = CGConv(channels=atom_features, dim=rbf_features, aggr='mean')
        self.bn = nn.BatchNorm1d(atom_features) # Batch Norm helps stability

    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        return x
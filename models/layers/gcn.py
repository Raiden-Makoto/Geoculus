import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GatedGCN(MessagePassing):
    """
    True Gated Graph Convolution.
    Formula: 
       h_in = W_src*h_j + W_dst*h_i + W_edge*e_ij
       gate = Sigmoid(h_in_gate)
       feat = SiLU(h_in_feat)
       msg  = gate * feat
    """
    def __init__(self, dim):
        super().__init__(aggr='add')
        
        # We project inputs into 2*dim (half for Feature, half for Gate)
        self.linear_src = nn.Linear(dim, 2 * dim)
        self.linear_dst = nn.Linear(dim, 2 * dim)
        self.linear_edge = nn.Linear(dim, 2 * dim)
        
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: Target Node Features
        # x_j: Source Node Features
        # edge_attr: Edge Features
        
        # 1. Combine Information
        z_src = self.linear_src(x_j)
        z_dst = self.linear_dst(x_i)
        z_edge = self.linear_edge(edge_attr)
        
        z = z_src + z_dst + z_edge
        
        # 2. Split into Feature (h) and Gate (g)
        # z shape: [E, 2*dim] -> h: [E, dim], g: [E, dim]
        h, g = torch.chunk(z, chunks=2, dim=-1)
        
        # 3. Gated Activation
        # This allows the model to "ignore" irrelevant angles/bonds
        return F.silu(h) * torch.sigmoid(g)

    def update(self, aggr_out, x):
        # Residual Connection + Batch Norm
        # x_new = x_old + BN(Sum(messages))
        return x + self.bn(aggr_out)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class GatedGCN(MessagePassing):
    """
    The workhorse of ALIGNN. 
    Updates Node Features (h) using Edge Features (e).
    Formula: h_i = h_i + Sum( ReLU(BatchNotm(W1 * h_j + W2 * e_ij)) * Sigmoid(...) )
    """
    def __init__(self, dim):
        super().__init__(aggr='add')
        self.dim = dim
        
        # Gates and Sources
        self.linear_src = nn.Linear(dim, dim)
        self.linear_dst = nn.Linear(dim, dim)
        self.linear_edge = nn.Linear(dim, dim)
        
        # Batch Norms are crucial for deep GNNs
        self.bn_node = nn.BatchNorm1d(dim)
        self.bn_edge = nn.BatchNorm1d(dim)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, dim]
        # edge_attr: [E, dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # The signal: A + B * C
        # But ALIGNN uses a Gated mechanism
        
        # Simple Gated Conv implementation:
        # z = A * x_j + B * edge_attr
        z = self.linear_src(x_j) + self.linear_edge(edge_attr)
        
        # Activation
        z = F.silu(self.bn_node(z)) # SiLU (Swish) is standard in ALIGNN
        
        return z

    def update(self, aggr_out, x):
        # Residual Connection
        return x + aggr_out

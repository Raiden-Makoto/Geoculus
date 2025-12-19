import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from layers import *

class CrystallGNN(nn.Module):
    def __init__(
        self,
        n_atom_feats: int=64,
        n_rbf: int=10,
        n_conv: int=3,
        n_hidden_head: int=64
    ):
        super().__init__()
        # 1. Embedding Layer
        # 95 is safe for elements up to Am (Z=95)
        self.embedding = nn.Embedding(95, n_atom_feats)
        self.rbf = GaussianSmearing(start=0.0, stop=6.0, num_gaussians=n_rbf)
        
        # 2. Backbone (Message Passing Stack)
        self.blocks = nn.ModuleList([
            InteractionBlock(n_atom_feats, n_rbf) for _ in range(n_conv)
        ])
        
        # 3. Global Pooling
        # Converts graph-level atoms -> single crystal vector
        self.pooling = global_mean_pool
        
        # 4. Regression Heads (Multi-Task)
        # Shared dense layer before splitting
        self.fc_shared = nn.Linear(n_atom_feats, n_hidden_head * 2)
        
        # Task 1: Band Gap
        self.head_bandgap = OutputHead(n_hidden_head * 2, n_hidden_head)
        
        # Task 2: Formation Energy
        self.head_ehull = OutputHead(n_hidden_head * 2, n_hidden_head)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # 1. Embeddings
        x = self.embedding(x.squeeze())         # Atoms: [N] -> [N, 64]
        edge_attr = self.rbf(edge_attr)         # Edges: [E, 1] -> [E, 50]
        
        # 2. Message Passing Backbone
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
            x = F.softplus(x) # Non-linearity between blocks
            
        # 3. Global Aggregation (Crystal Fingerprint)
        # x is [N_atoms, 64], batch is [N_atoms] -> c is [Batch_Size, 64]
        c = self.pooling(x, batch)
        
        # 4. Prediction
        c = F.relu(self.fc_shared(c))
        
        out_bg = self.head_bandgap(c)
        out_ehull = self.head_ehull(c)
        
        return out_bg, out_ehull
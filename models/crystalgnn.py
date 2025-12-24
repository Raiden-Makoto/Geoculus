import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from .layers import *

class CrystallGNN(nn.Module):
    def __init__(
        self,
        n_atom_input: int=4,  # Physical features: [EN, Radius, Mass, Melting]
        n_atom_feats: int=64,
        n_global_feats: int=2,  # Global features: [tolerance_factor, packing_fraction]
        n_rbf: int=10,
        n_conv: int=3,
        n_hidden_head: int=64
    ):
        super().__init__()
        # 1. Embedding Layer - Project physical features into hidden dimension
        # Replaced nn.Embedding with nn.Linear to handle float features
        self.embedding = nn.Linear(n_atom_input, n_atom_feats)
        self.rbf = GaussianSmearing(start=0.0, stop=6.0, num_gaussians=n_rbf)
        
        # 2. Backbone (Message Passing Stack)
        self.blocks = nn.ModuleList([
            InteractionBlock(n_atom_feats, n_rbf) for _ in range(n_conv)
        ])
        
        # 3. Global Pooling
        # Converts graph-level atoms -> single crystal vector
        self.pooling = global_mean_pool
        
        # 4. Regression Heads (Multi-Task)
        # Fusion Layer: Concatenate Graph Vector (64) + Global Features (2)
        fusion_dim = n_atom_feats + n_global_feats
        self.fc_shared = nn.Linear(fusion_dim, n_hidden_head * 2)
        
        # Task 1: Band Gap
        self.head_bandgap = OutputHead(n_hidden_head * 2, n_hidden_head)
        
        # Task 2: Formation Energy
        self.head_ehull = OutputHead(n_hidden_head * 2, n_hidden_head)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # 1. Embeddings
        # x is now [N, 4] (physical features) instead of [N, 1] (atomic numbers)
        x = self.embedding(x)                   # Atoms: [N, 4] -> [N, 64]
        edge_attr = self.rbf(edge_attr)         # Edges: [E, 1] -> [E, 50]
        
        # 2. Message Passing Backbone
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
            x = F.softplus(x) # Non-linearity between blocks
            
        # 3. Global Aggregation (Crystal Fingerprint)
        # x is [N_atoms, 64], batch is [N_atoms] -> c is [Batch_Size, 64]
        c = self.pooling(x, batch)
        
        # --- NEW: Inject Global Features ---
        # data.u is [Batch, 2] after batching (PyG automatically stacks them)
        # Handle both single graph [1, 2] and batched [Batch, 2] cases
        u = data.u
        if u.dim() == 3:
            u = u.squeeze(1)  # [Batch, 1, 2] -> [Batch, 2]
        elif u.dim() == 1:
            u = u.unsqueeze(0)  # [2] -> [1, 2] for single graph
        # Ensure u is on same device as c
        u = u.to(c.device)
        c = torch.cat([c, u], dim=1)
        
        # 4. Prediction
        c = F.relu(self.fc_shared(c))
        
        # Band gap: output in log space (log1p) to handle long tail distribution
        out_bg_raw = self.head_bandgap(c)
        out_bg = torch.log1p(torch.clamp(out_bg_raw, min=0.0))  # Ensure non-negative before log
        
        # e_hull: output in original space
        out_ehull = self.head_ehull(c)
        
        return out_bg, out_ehull
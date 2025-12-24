import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from .layers import GaussianSmearing, GatedGCN #

class ALIGNN(nn.Module):
    def __init__(self, n_atom_input=4, hidden_dim=128, n_layers=4):
        super().__init__()
        
        # 1. Embeddings
        self.atom_embedding = nn.Linear(n_atom_input, hidden_dim)
        
        # RBF Expanders
        self.rbf_bond = GaussianSmearing(start=0.0, stop=6.0, num_gaussians=hidden_dim)
        self.rbf_angle = GaussianSmearing(start=0.0, stop=180.0, num_gaussians=hidden_dim)
        
        # 2. ALIGNN Layers (The Sandwich)
        self.atom_layers = nn.ModuleList()
        self.line_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.atom_layers.append(GatedGCN(hidden_dim))
            self.line_layers.append(GatedGCN(hidden_dim))
            
        # 3. Output Heads
        self.pool = global_mean_pool
        self.head_bg = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.head_hull = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data):
        # A. Initial Feature Embedding
        h = self.atom_embedding(data.x)  # Atom Features [N, 128]
        m = self.rbf_bond(data.edge_attr) # Bond Features [E, 128]
        a = self.rbf_angle(data.angle_attr) # Angle Features [Triplets, 128]
        
        # B. Iterative Updates
        for i in range(len(self.atom_layers)):
            # 1. Update Bonds (using Angles)
            # The "Nodes" of the Line Graph are the "Edges" of the Atom Graph (m)
            # The "Edges" of the Line Graph are the Angles (a)
            m = self.line_layers[i](x=m, edge_index=data.edge_index_lg, edge_attr=a)
            
            # 2. Update Atoms (using new Bond info)
            h = self.atom_layers[i](x=h, edge_index=data.edge_index, edge_attr=m)
            
        # C. Readout
        c = self.pool(h, data.batch)
        
        # D. Predict
        # Remember: Log-Space for Bandgap!
        out_bg_log = self.head_bg(c)
        out_hull = self.head_hull(c)
        
        return out_bg_log, out_hull
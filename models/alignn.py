import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from .layers import GaussianSmearing, GatedGCN #

class ALIGNN(nn.Module):
    def __init__(self, n_atom_input=4, hidden_dim=96, n_layers=3):
        super().__init__()
        
        # 1. Embeddings
        # Projects physical features (4) to hidden space (128)
        self.atom_embedding = nn.Sequential(
            nn.Linear(n_atom_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU()
        )
        
        # RBF Expanders
        self.rbf_bond = GaussianSmearing(start=0.0, stop=6.0, num_gaussians=hidden_dim)
        self.rbf_angle = GaussianSmearing(start=-1.0, stop=180.0, num_gaussians=hidden_dim)
        
        # 2. ALIGNN Layers (The Sandwich)
        self.atom_layers = nn.ModuleList()
        self.line_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.atom_layers.append(GatedGCN(hidden_dim))
            self.line_layers.append(GatedGCN(hidden_dim))
            
        # 3. Output Heads
        self.pool = global_mean_pool
        
        # Deeper heads for regression
        self.head_bg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(), 
            nn.Linear(hidden_dim, 1)
        )
        self.head_hull = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(), 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # A. Embed Features
        h = self.atom_embedding(data.x)             
        m = self.rbf_bond(data.edge_attr)           
        a = self.rbf_angle(data.angle_attr)         
        
        # B. ALIGNN Iterations
        for i in range(len(self.atom_layers)):
            # 1. Update Bonds (Line Graph)
            # Input: Bond Features (m), Graph: Angles (lg_index, a)
            m = self.line_layers[i](x=m, edge_index=data.edge_index_lg, edge_attr=a)
            
            # 2. Update Atoms (Atom Graph)
            # Input: Atom Features (h), Graph: Bonds (edge_index, m)
            h = self.atom_layers[i](x=h, edge_index=data.edge_index, edge_attr=m)
            
        # C. Readout & Predict
        c = self.pool(h, data.batch)
        
        out_bg_log = self.head_bg(c)
        out_hull = self.head_hull(c)
        
        return out_bg_log, out_hull
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.core import Element
from torch_geometric.data import Data, Dataset
from pymatgen.analysis.local_env import CrystalNN

# --- CONFIG ---
# We use CrystalNN for robust neighbor finding (better than simple radius)
cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0, porous_adjustment=False)

def get_atom_features(element_symbol):
    """
    Returns a vector of physical properties:
    [Electronegativity, Ionic Radius, Melting Point, Atomic Mass]
    Normalized to roughly 0-1 range for better convergence.
    """
    el = Element(element_symbol)
    
    # Handle missing data (noble gases etc) with 0.0
    en = el.X if el.X else 0.0
    radius = el.atomic_radius if el.atomic_radius else 0.0
    mass = el.atomic_mass
    melting = el.melting_point if el.melting_point else 0.0
    
    # Simple normalization (roughly 0-1 range helps convergence)
    return [
        en / 4.0,           # Electronegativity (max ~4)
        radius / 3.0,       # Radius (max ~3A)
        mass / 200.0,       # Mass (max ~200)
        melting / 3000.0    # Melting point
    ]

class ALIGNNDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.csv_file = os.path.join(root, "raw", "perovskite_metadata.csv")
        self.structures_dir = os.path.join(root, "raw", "structures")
        self.df = pd.read_csv(self.csv_file)
        # Filter (Important: Apply the same filters you learned were good)
        self.df = self.df[self.df['e_hull'] < 0.5] 
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self): return ["perovskite_metadata.csv"]
    @property
    def processed_file_names(self): return [f'data_alignn_{i}.pt' for i in range(len(self.df))]

    def process(self):
        print("Building Line Graphs for ALIGNN...")
        
        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            material_id = row['material_id']
            cif_path = os.path.join(self.structures_dir, f"{material_id}.cif")

            try:
                structure = Structure.from_file(cif_path)
                
                # --- 1. ATOM GRAPH (Nodes & Bonds) ---
                # Nodes
                atom_features = [get_atom_features(s.specie.symbol) for s in structure]
                x = torch.tensor(atom_features, dtype=torch.float)
                
                # Edges (Bonds) via CrystalNN
                # This finds "Chemically bonded" neighbors, not just "Close" ones
                all_neighbors = cnn.get_all_nn_info(structure)
                
                edge_indices = []
                edge_distances = []
                
                # We need to map (atom_i, atom_j, image) to a unique Edge ID for the Line Graph
                # This map will store: {(src, dst): edge_index_in_tensor}
                bond_map = {} 
                edge_count = 0

                for i, neighbors in enumerate(all_neighbors):
                    for n in neighbors:
                        j = n['site_index']
                        dist = n['weight'] # CrystalNN weight often correlates to distance, but let's calculate real dist
                        # Re-calculate exact distance
                        d_vec = structure[j].coords - structure[i].coords # Simplified (PBC ignored for brevity, assume CrystalNN handles)
                        # Better: use PyMatgen's computed distance
                        real_dist = structure.get_distance(i, j)
                        
                        edge_indices.append([i, j])
                        edge_distances.append(real_dist)
                        
                        bond_map[(i, j)] = edge_count
                        edge_count += 1

                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1) # [E, 1]

                # --- 2. LINE GRAPH (Angles) ---
                # Nodes of Line Graph = Edges of Atom Graph
                # Edges of Line Graph = Connections between bonds (Angles)
                
                lg_indices = []
                lg_angles = []

                # Iterate over atoms (central atom k)
                # Find pairs of bonds (i-k) and (k-j)
                for k in range(len(structure)):
                    # Find all neighbors of k
                    neighbors = [e for e in edge_indices if e[1] == k] # Incoming bonds to k
                    # (In undirected graph, we treat i->k and k->i. Let's simplify:
                    #  Iterate all pairs of edges (i, k) and (k, j) in the edge_list)
                    pass 
                
                # FAST WAY: Use PyG's matching if possible, but we need ANGLES.
                # Explicit Loop over Triplets
                # This is O(N_neighbors^2) per atom. Fast enough for Perovskites.
                
                # Re-organize edges by center atom
                adj = {} # atom_idx -> list of (neighbor_idx, edge_index_id)
                for e_id, (src, dst) in enumerate(edge_indices):
                    if dst not in adj: adj[dst] = []
                    adj[dst].append((src, e_id))
                
                for k, neighbors in adj.items():
                    # k is the center atom of the angle i-k-j
                    # Iterate all pairs of neighbors
                    n_count = len(neighbors)
                    if n_count < 2: continue
                    
                    for idx_a in range(n_count):
                        for idx_b in range(n_count):
                            if idx_a == idx_b: continue
                            
                            i, bond_idx_a = neighbors[idx_a] # Bond i->k
                            j, bond_idx_b = neighbors[idx_b] # Bond j->k
                            
                            # Calculate Angle i-k-j
                            try:
                                angle = structure.get_angle(i, k, j)
                                # Handle NaN/inf angles (can happen with degenerate geometries)
                                if not np.isfinite(angle):
                                    angle = 90.0  # Default to 90 degrees for degenerate cases
                            except:
                                angle = 90.0  # Default fallback
                            
                            lg_indices.append([bond_idx_a, bond_idx_b])
                            lg_angles.append(angle)

                edge_index_lg = torch.tensor(lg_indices, dtype=torch.long).t().contiguous()
                angle_attr = torch.tensor(lg_angles, dtype=torch.float).unsqueeze(1) # [Num_Angles, 1]

                # --- 3. TARGETS ---
                y_bg = torch.tensor([row['band_gap']], dtype=torch.float)
                y_hull = torch.tensor([row['e_hull']], dtype=torch.float)

                data = Data(
                    x=x, 
                    edge_index=edge_index, 
                    edge_attr=edge_attr,       # Bond Lengths
                    edge_index_lg=edge_index_lg, 
                    angle_attr=angle_attr,     # Bond Angles
                    y_bandgap=y_bg,
                    y_ehull=y_hull
                )
                
                torch.save(data, os.path.join(self.processed_dir, f'data_alignn_{idx}.pt'))

            except Exception as e:
                # print(f"Error {material_id}: {e}")
                pass

    def len(self):
        return len(self.df)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_alignn_{idx}.pt'), weights_only=False)

if __name__ == "__main__":
    # Ensure folder structure exists
    dataset = ALIGNNDataset(root="dataset")
    print(f"Processed {len(dataset)} ALIGNN graphs.")
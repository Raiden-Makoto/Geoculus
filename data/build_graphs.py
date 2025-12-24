import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.core import Element
from torch_geometric.data import Data, Dataset

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
        self.df = self.df[self.df['e_hull'] < 0.5] 
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self): return ["perovskite_metadata.csv"]
    @property
    def processed_file_names(self): return [f'data_alignn_{i}.pt' for i in range(len(self.df))]

    def process(self):
        print("Building CORRECTED ALIGNN Graphs (Vector Math)...")
        
        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            material_id = row['material_id']
            cif_path = os.path.join(self.structures_dir, f"{material_id}.cif")

            try:
                structure = Structure.from_file(cif_path)
                
                # --- 1. ATOM GRAPH ---
                radius = 5.0
                # Get neighbors WITH vectors
                # neighbor structure: (site, distance, index, image)
                neighbors = structure.get_all_neighbors(r=radius, include_index=True)
                
                atom_features = [get_atom_features(s.specie.symbol) for s in structure]
                x = torch.tensor(atom_features, dtype=torch.float)
                
                edge_indices = []
                edge_attrs = []
                
                # Store vectors for angle calculation later
                # key: bond_id -> vector (numpy array)
                bond_vectors = {} 
                
                # Group bonds by center atom for line graph
                # key: center_atom_idx -> list of bond_ids
                center_to_bonds = {i: [] for i in range(len(structure))}
                
                bond_count = 0

                for i, neighbor_list in enumerate(neighbors):
                    neighbor_list.sort(key=lambda x: x[1])
                    
                    for n in neighbor_list[:8]: # Max 8 neighbors (6 B-X + 2 A-site)
                        dist = n[1]
                        j = n[2]
                        neighbor_site = n[0]
                        
                        # Vector from Center (i) to Neighbor (j)
                        # We must use the specific coords of the neighbor image
                        v = neighbor_site.coords - structure[i].coords
                        
                        edge_indices.append([i, j])
                        edge_attrs.append([dist])
                        
                        # Store info for Line Graph
                        bond_vectors[bond_count] = v
                        center_to_bonds[i].append(bond_count)
                        
                        bond_count += 1
                
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)

                # --- 2. LINE GRAPH (True Geometric Angles) ---
                lg_indices = []
                lg_angles = []
                
                for k in range(len(structure)):
                    bonds = center_to_bonds[k]
                    if len(bonds) < 2: continue
                    
                    # Double loop to find all pairs (angles) around atom k
                    for idx_a in range(len(bonds)):
                        for idx_b in range(len(bonds)):
                            if idx_a == idx_b: continue
                            
                            bond_id_a = bonds[idx_a]
                            bond_id_b = bonds[idx_b]
                            
                            v1 = bond_vectors[bond_id_a]
                            v2 = bond_vectors[bond_id_b]
                            
                            # Calculate Angle using Dot Product
                            # theta = arccos( (v1 . v2) / (|v1| * |v2|) )
                            dot = np.dot(v1, v2)
                            norm_mul = np.linalg.norm(v1) * np.linalg.norm(v2)
                            
                            # Clip to handle floating point noise > 1.0 or < -1.0
                            cosine = np.clip(dot / (norm_mul + 1e-8), -1.0, 1.0)
                            angle_rad = np.arccos(cosine)
                            angle_deg = np.degrees(angle_rad)
                            
                            lg_indices.append([bond_id_a, bond_id_b])
                            lg_angles.append([angle_deg])

                edge_index_lg = torch.tensor(lg_indices, dtype=torch.long).t().contiguous()
                angle_attr = torch.tensor(lg_angles, dtype=torch.float)

                # --- 3. Save ---
                y_bg = torch.tensor([row['band_gap']], dtype=torch.float)
                y_hull = torch.tensor([row['e_hull']], dtype=torch.float)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                            edge_index_lg=edge_index_lg, angle_attr=angle_attr,
                            y_bandgap=y_bg, y_ehull=y_hull, material_id=material_id)
                
                torch.save(data, os.path.join(self.processed_dir, f'data_alignn_{idx}.pt'))

            except Exception as e:
                pass

    def len(self):
        return len(self.df)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_alignn_{idx}.pt'), weights_only=False)

if __name__ == "__main__":
    # Ensure folder structure exists
    dataset = ALIGNNDataset(root="dataset")
    print(f"Processed {len(dataset)} ALIGNN graphs.")
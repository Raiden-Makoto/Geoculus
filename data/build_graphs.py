import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
warnings.filterwarnings('ignore')

class PerovskiteDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root: Directory where the dataset should be stored.
              Structure:
              /root
                  /raw
                      perovskite_metadata.csv
                      /structures (contains .cif files)
                  /processed
                      data_0.pt, data_1.pt, ...
        """
        self.csv_file = os.path.join(root, "raw", "perovskite_metadata.csv")
        self.structures_dir = os.path.join(root, "raw", "structures")
        
        # Load the dataframe to get targets (Labels)
        self.df = pd.read_csv(self.csv_file)
        
        super(PerovskiteDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # PyG looks for these files to decide if it needs to download anything
        return ["perovskite_metadata.csv"]

    @property
    def processed_file_names(self):
        # If these exist, process() is skipped. 
        return [f'data_{i}.pt' for i in range(len(self.df))]

    def process(self):
        print("Converting CIF files to PyTorch Graphs...")
        
        # We define a cutoff radius for edges (neighbors)
        radius = 4.0  # Angstroms (Standard for Perovskites)
        max_neighbors = 12 # Limit max edges per node to save memory

        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            material_id = row['material_id']
            cif_path = os.path.join(self.structures_dir, f"{material_id}.cif")

            try:
                # 1. Load Structure
                crystal = Structure.from_file(cif_path)

                # 2. Extract Node Features (Atomic Numbers)
                # We simply use the atomic number. The Embedding layer in the model will handle the rest.
                atomic_numbers = [site.specie.number for site in crystal]
                x = torch.tensor(atomic_numbers, dtype=torch.long).unsqueeze(1) 

                # 3. Extract Edges (Connectivity)
                # Get all neighbors within radius
                neighbors = crystal.get_all_neighbors(r=radius, include_index=True)
                
                edge_indices = []
                edge_attrs = []

                for i, neighbor_list in enumerate(neighbors):
                    # Sort by distance to keep only closest interactions
                    neighbor_list.sort(key=lambda x: x[1]) 
                    
                    for neighbor in neighbor_list[:max_neighbors]:
                        # neighbor is (site, distance, index)
                        dist = neighbor[1]
                        j = neighbor[2] # Index of neighbor atom
                        
                        edge_indices.append([i, j])
                        edge_attrs.append([dist])

                # Convert to Tensors
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

                # 4. Extract Targets (Labels)
                y_bandgap = torch.tensor([row['band_gap']], dtype=torch.float)
                y_ehull = torch.tensor([row['e_hull']], dtype=torch.float)
                
                # Create PyG Data Object
                data = Data(x=x, 
                            edge_index=edge_index, 
                            edge_attr=edge_attr, 
                            y_bandgap=y_bandgap,
                            y_ehull=y_ehull,
                            material_id=material_id)

                # Save the single graph
                torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))

            except Exception as e:
                print(f"Failed to process {material_id}: {e}")
                pass

    def len(self):
        return len(self.df)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=False)

# --- EXECUTION ---
if __name__ == "__main__":
    # Get the script's directory and build absolute path to dataset
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # Go up from data/ to project root
    dataset_root = project_root / "dataset"
    
    # Ensure your folder structure matches:
    # ./dataset/raw/perovskite_metadata.csv
    # ./dataset/raw/structures/*.cif
    # PyTorch Geometric is very strict
    dataset = PerovskiteDataset(root=str(dataset_root))
    print(f"Created dataset with {len(dataset)} graphs.")
    print(f"Sample Graph: {dataset[67]}")
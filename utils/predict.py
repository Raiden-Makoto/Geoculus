import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from pymatgen.core import Structure

# --- IMPORTS FROM YOUR MODULES ---
import sys
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data"))

from models.alignn import ALIGNN
from build_graphs import get_atom_features # We still use the basic atom featurizer

# --- CONFIG ---
CANDIDATES_DIR = project_root / "candidates"
MODEL_PATH = project_root / "checkpoints/best_model.pth"
OUTPUT_FILE = project_root / "final_predictions_alignn.csv"

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

def load_model():
    print(f"Loading ALIGNN model from {MODEL_PATH}...")
    # MUST match training params
    model = ALIGNN(
        n_atom_input=4,
        hidden_dim=96,
        n_layers=3
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Did you use checkpointing? The state_dict might need key adjustment if saved wrapped.")
        exit()
        
    model.eval()
    return model

def cif_to_alignn_graph(cif_path):
    """
    Constructs an ALIGNN graph (Atom + Line Graph) for a single CIF.
    Matches the 'Lightweight' logic: Radius 5.0, Max 8 Neighbors.
    """
    try:
        structure = Structure.from_file(cif_path)
        
        # --- 1. ATOM GRAPH ---
        radius = 5.0
        # Get neighbors (site, distance, index, image)
        neighbors = structure.get_all_neighbors(r=radius, include_index=True)
        
        atom_features = [get_atom_features(s.specie.symbol) for s in structure]
        x = torch.tensor(atom_features, dtype=torch.float)
        
        edge_indices = []
        edge_attrs = []
        
        bond_vectors = {} 
        center_to_bonds = {i: [] for i in range(len(structure))}
        bond_count = 0

        for i, neighbor_list in enumerate(neighbors):
            neighbor_list.sort(key=lambda x: x[1])
            
            # LIMIT TO 8 NEIGHBORS (Lightweight)
            for n in neighbor_list[:8]:
                dist = n[1]
                j = n[2]
                neighbor_site = n[0]
                
                # Vector for angle calc
                v = neighbor_site.coords - structure[i].coords
                
                edge_indices.append([i, j])
                edge_attrs.append([dist])
                
                bond_vectors[bond_count] = v
                center_to_bonds[i].append(bond_count)
                
                bond_count += 1
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
        
        # --- 2. LINE GRAPH (Angles) ---
        lg_indices = []
        lg_angles = []
        
        for k in range(len(structure)):
            bonds = center_to_bonds[k]
            if len(bonds) < 2: continue
            
            for idx_a in range(len(bonds)):
                for idx_b in range(len(bonds)):
                    if idx_a == idx_b: continue
                    
                    bond_id_a = bonds[idx_a]
                    bond_id_b = bonds[idx_b]
                    v1 = bond_vectors[bond_id_a]
                    v2 = bond_vectors[bond_id_b]
                    
                    # Dot product angle
                    dot = np.dot(v1, v2)
                    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                    cosine = np.clip(dot / (norm + 1e-8), -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cosine))
                    
                    lg_indices.append([bond_id_a, bond_id_b])
                    lg_angles.append([angle_deg])

        edge_index_lg = torch.tensor(lg_indices, dtype=torch.long).t().contiguous()
        angle_attr = torch.tensor(lg_angles, dtype=torch.float).unsqueeze(1)
        
        # Batch vector (all zeros for single graph)
        batch = torch.zeros(len(x), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    edge_index_lg=edge_index_lg, angle_attr=angle_attr,
                    batch=batch)
        
    except Exception as e:
        # print(f"Failed on {cif_path}: {e}")
        return None

def predict():
    model = load_model()
    
    if not CANDIDATES_DIR.exists():
        print(f"Candidates dir not found: {CANDIDATES_DIR}")
        return

    files = [f for f in os.listdir(CANDIDATES_DIR) if f.endswith(".cif")]
    print(f"Screening {len(files)} candidates using ALIGNN...")
    
    results = []
    
    with torch.no_grad():
        for filename in tqdm(files):
            filepath = CANDIDATES_DIR / filename
            
            data = cif_to_alignn_graph(filepath)
            if data is None: continue
            
            data = data.to(DEVICE)
            
            # Predict
            pred_bg_raw, pred_hull = model(data)
            
            # UN-LOG BANDGAP (same as training: clamp, log1p, then expm1)
            # Model outputs raw values, we apply log1p transform, then reverse it
            pred_bg_log = torch.log1p(torch.clamp(pred_bg_raw.squeeze(), min=0.0))
            pred_bg_eV = torch.expm1(pred_bg_log).item()
            
            results.append({
                "Material": filename.replace(".cif", ""),
                "Pred_Bandgap": pred_bg_eV,
                "Pred_Stability": pred_hull.item()
            })
            
    df = pd.DataFrame(results)
    df = df.sort_values(by="Pred_Stability")
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print(f"  ALIGNN DISCOVERY RESULTS")
    print("="*40)
    
    # Filter: Stability < 0.10 eV AND Bandgap 0.9 - 1.8 eV
    winners = df[
        (df['Pred_Stability'] < 0.10) & 
        (df['Pred_Bandgap'] > 0.9) & 
        (df['Pred_Bandgap'] < 1.8)
    ]
    
    if len(winners) > 0:
        print(f"Found {len(winners)} potential solar materials:")
        print(winners.head(15).to_string(index=False))
    else:
        print("No perfect matches. Showing most stable non-metals (>0.5 eV):")
        semiconductors = df[df['Pred_Bandgap'] > 0.5].sort_values(by="Pred_Stability")
        print(semiconductors.head(10).to_string(index=False))

if __name__ == "__main__":
    predict()
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

from models.crystalgnn import CrystallGNN
from build_graphs import get_atom_features, calculate_tolerance_factor

# --- CONFIG ---
CANDIDATES_DIR = "candidates" 
MODEL_PATH = "checkpoints/best_model.pth"
OUTPUT_FILE = "final_predictions.csv"
# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

def load_model():
    model_path = project_root / MODEL_PATH
    print(f"Loading model from {model_path}...")
    model = CrystallGNN(
        n_atom_input=4,       
        n_atom_feats=64, 
        n_global_feats=2,     
        n_rbf=50, 
        n_conv=3
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def cif_to_graph(cif_path):
    try:
        crystal = Structure.from_file(cif_path)
        
        # 1. Atom Features
        atom_features = [get_atom_features(site.specie.symbol) for site in crystal]
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # 2. Edges (Radius = 6.0)
        radius = 6.0
        max_neighbors = 20
        neighbors = crystal.get_all_neighbors(r=radius, include_index=True)
        
        edge_indices, edge_attrs = [], []
        for i, neighbor_list in enumerate(neighbors):
            neighbor_list.sort(key=lambda x: x[1])
            for neighbor in neighbor_list[:max_neighbors]:
                dist = neighbor[1]
                j = neighbor[2]
                edge_indices.append([i, j])
                edge_attrs.append([dist])
                
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        # 3. Global Features
        t = calculate_tolerance_factor(crystal)
        pack = len(crystal) / crystal.volume
        u = torch.tensor([[t, pack]], dtype=torch.float)
        
        batch = torch.zeros(len(x), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)
        
    except Exception as e:
        return None

def predict():
    model = load_model()
    candidates_dir = project_root / CANDIDATES_DIR
    files = [f for f in os.listdir(candidates_dir) if f.endswith(".cif")]
    print(f"Screening {len(files)} candidates...")
    
    results = []
    
    with torch.no_grad():
        for filename in tqdm(files):
            filepath = candidates_dir / filename
            
            data = cif_to_graph(str(filepath))
            if data is None: continue
            data = data.to(DEVICE)
            
            # Predict
            pred_bg_log, pred_ehull = model(data)
            
            # --- CRITICAL: REVERSE THE LOG TRANSFORM ---
            # Model outputs log space, convert back to eV
            pred_bg_eV = torch.expm1(pred_bg_log.squeeze()).item()  # exp(x) - 1
            
            results.append({
                "Material": filename.replace(".cif", ""),
                "Pred_Bandgap": pred_bg_eV,       # Real eV value
                "Pred_Stability": pred_ehull.item(),
                "Tolerance_Factor": data.u[0][0].item()
            })
            
    # Save Results
    df = pd.DataFrame(results)
    df = df.sort_values(by="Pred_Stability") 
    output_path = project_root / OUTPUT_FILE
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "="*40)
    print(f"  TOP CANDIDATES (Goldilocks Zone)")
    print("="*40)
    
    # Filter: Stability < 0.10 eV AND Bandgap 0.9 - 1.8 eV
    winners = df[
        (df['Pred_Stability'] < 0.10) & 
        (df['Pred_Bandgap'] > 0.9) & 
        (df['Pred_Bandgap'] < 1.8)
    ]
    
    if len(winners) > 0:
        print(winners[['Material', 'Pred_Bandgap', 'Pred_Stability']].head(15).to_string(index=False))
    else:
        print("No strict matches. Showing best stable candidates:")
        print(df[['Material', 'Pred_Bandgap', 'Pred_Stability']].head(10).to_string(index=False))

if __name__ == "__main__":
    predict()
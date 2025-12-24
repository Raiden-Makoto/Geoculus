import torch
import os
from pathlib import Path

def check_data():
    # Find the first processed data file
    processed_dir = Path("dataset/processed")
    if not processed_dir.exists():
        print(f"Error: {processed_dir} not found.")
        return
    
    # Find first data file
    data_files = list(processed_dir.glob("data_alignn_*.pt"))
    if not data_files:
        print(f"Error: No processed data files found in {processed_dir}")
        return
    
    path = data_files[0]
    print(f"Checking: {path}")
    
    data = torch.load(path, weights_only=False)
    
    # Get material ID if available
    material_id = getattr(data, 'material_id', 'unknown')
    print(f"\n--- DIAGNOSIS: {material_id} ---")
    
    # 1. Check Atom Graph
    print(f"\n[Atom Graph]")
    print(f"  Atoms: {data.x.shape[0]}")
    print(f"  Atom Features: {data.x.shape[1]}D")
    print(f"  Bonds: {data.edge_index.shape[1]}")
    if hasattr(data, 'edge_attr'):
        print(f"  Bond Features: {data.edge_attr.shape}")
        if data.edge_attr.shape[0] > 0:
            print(f"  Sample Bond Length: {data.edge_attr[0].item():.4f} Å")
    
    # 2. Check Line Graph (CRITICAL)
    print(f"\n[Line Graph]")
    if hasattr(data, 'edge_index_lg'):
        num_triplets = data.edge_index_lg.shape[1]
        print(f"  Angles (Line Graph Edges): {num_triplets}")
        
        if num_triplets == 0:
            print("\n  [FAIL] ❌ The Line Graph is EMPTY!")
            print("         Your model is blind to angles. It cannot learn Band Gaps.")
            print("         The model will fall back to being a standard GNN.")
        else:
            print(f"\n  [PASS] ✅ Line Graph has {num_triplets} connections.")
            if hasattr(data, 'angle_attr') and data.angle_attr.shape[0] > 0:
                print(f"         Sample Angle Feature: {data.angle_attr[0].item():.4f}°")
                print(f"         Angle Range: [{data.angle_attr.min().item():.2f}°, {data.angle_attr.max().item():.2f}°]")
            
            # Check ratio
            bond_count = data.edge_index.shape[1]
            angle_count = num_triplets
            if bond_count > 0:
                ratio = angle_count / bond_count
                print(f"         Angle-to-Bond Ratio: {ratio:.1f}x")
                if ratio < 2.0:
                    print("         ⚠️  Warning: Low angle count. Expected ~3-5x bonds.")
    else:
        print("\n  [FAIL] ❌ No 'edge_index_lg' found in data object.")
        print("         The dataset was not processed for ALIGNN.")
    
    # 3. Check Targets
    print(f"\n[Targets]")
    if hasattr(data, 'y_bandgap'):
        print(f"  Band Gap: {data.y_bandgap.item():.4f} eV")
    if hasattr(data, 'y_ehull'):
        print(f"  e_hull: {data.y_ehull.item():.4f} eV/atom")
    
    # 4. Check multiple samples
    print(f"\n[Dataset Statistics]")
    all_files = list(processed_dir.glob("data_alignn_*.pt"))
    print(f"  Total samples: {len(all_files)}")
    
    # Sample a few more
    empty_line_graphs = 0
    total_angles = 0
    total_bonds = 0
    
    for i, data_file in enumerate(all_files[:10]):  # Check first 10
        sample = torch.load(data_file, weights_only=False)
        bonds = sample.edge_index.shape[1]
        total_bonds += bonds
        
        if hasattr(sample, 'edge_index_lg'):
            angles = sample.edge_index_lg.shape[1]
            total_angles += angles
            if angles == 0:
                empty_line_graphs += 1
        else:
            empty_line_graphs += 1
    
    if len(all_files) > 0:
        print(f"  Samples checked: {min(10, len(all_files))}")
        print(f"  Empty line graphs: {empty_line_graphs}")
        if total_bonds > 0:
            avg_ratio = total_angles / total_bonds if total_angles > 0 else 0
            print(f"  Average angle-to-bond ratio: {avg_ratio:.2f}x")
        
        if empty_line_graphs > 0:
            print(f"\n  ⚠️  WARNING: {empty_line_graphs} samples have empty line graphs!")
            print(f"     These samples cannot learn band gaps properly.")

if __name__ == "__main__":
    check_data()


import os
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Lattice, Element

# --- CONFIGURATION ---
# 1. Define your "Menu" of ingredients
# We include oxidation states to ensure charge balance
A_sites = {
    'Cs': (+1, 1.88), 'Rb': (+1, 1.72), 'K': (+1, 1.64), 
    'Ba': (+2, 1.61), 'Sr': (+2, 1.44)
}

B_sites = {
    'Sn': (+2, 1.18), 'Ge': (+2, 0.73), # Lead-free group IV
    'Bi': (+3, 1.03), 'Sb': (+3, 0.76), # Trivalent
    'Ti': (+4, 0.605), 'Zr': (+4, 0.72) # Quadrivalent
}

X_sites = {
    'I': (-1, 2.20), 'Br': (-1, 1.96), 'Cl': (-1, 1.81), # Halides
    'O': (-2, 1.35), 'S': (-2, 1.84) # Chalcogenides
}

def calculate_tolerance_factor(r_a, r_b, r_x):
    """
    Goldschmidt Tolerance Factor (t).
    Ideal cubic perovskites have 0.8 < t < 1.0
    """
    return (r_a + r_x) / (np.sqrt(2) * (r_b + r_x))

def generate_hypothetical_materials():
    # Get the script's directory and build paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # Go up from scripts/ to project root
    output_dir = project_root / "candidates"
    
    output_dir.mkdir(exist_ok=True)
    
    candidates = []
    
    # Iterate through every possible combination (Cartesian Product)
    for a_sym, b_sym, x_sym in itertools.product(A_sites, B_sites, X_sites):
        
        # Unpack properties
        a_chg, r_a = A_sites[a_sym]
        b_chg, r_b = B_sites[b_sym]
        x_chg, r_x = X_sites[x_sym]
        
        # --- FILTER 1: Charge Neutrality ---
        # A + B + 3*X must equal 0
        net_charge = a_chg + b_chg + (3 * x_chg)
        if net_charge != 0:
            continue # Skip invalid physics (e.g., Cs+ Sn+2 O-2 is not neutral)

        # --- FILTER 2: Geometric Stability (Tolerance Factor) ---
        t = calculate_tolerance_factor(r_a, r_b, r_x)
        if t < 0.8 or t > 1.1:
            continue # Structure likely won't form a perovskite

        # --- STRUCTURE GENERATION ---
        # We estimate lattice parameter 'a' based on the B-X bond
        # In an ideal cubic perovskite, a = 2 * (r_b + r_x)
        a_est = 2.0 * (r_b + r_x)
        
        # Create the Cubic Perovskite (Space Group Pm-3m, #221)
        # Positions: A=(0,0,0), B=(0.5,0.5,0.5), X=(0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        lattice = Lattice.from_parameters(a_est, a_est, a_est, 90, 90, 90)
        species = [a_sym, b_sym, x_sym, x_sym, x_sym]
        coords = [
            [0, 0, 0],          # A-site (Corner)
            [0.5, 0.5, 0.5],    # B-site (Center)
            [0.5, 0.5, 0.0],    # X-site (Face Center 1)
            [0.5, 0.0, 0.5],    # X-site (Face Center 2)
            [0.0, 0.5, 0.5]     # X-site (Face Center 3)
        ]
        
        struct = Structure(lattice, species, coords)
        
        # Save to file
        formula = f"{a_sym}{b_sym}{x_sym}3"
        filename = f"{formula}.cif"
        struct.to(filename=str(output_dir / filename))
        
        candidates.append({
            "formula": formula,
            "filename": filename,
            "t_factor": t,
            "lattice_param": a_est
        })

    # Save summary registry
    df = pd.DataFrame(candidates)
    csv_path = output_dir / "candidates.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} physically valid candidates in '{output_dir}'")
    print(df.head())

if __name__ == "__main__":
    generate_hypothetical_materials()
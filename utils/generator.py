import os
import numpy as np
from pymatgen.core import Structure, Lattice, Element
from pymatgen.analysis.structure_prediction.substitutor import Substitutor

# --- CONFIG ---
OUTPUT_DIR = "candidates_exotic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the "Exotic" Search Space
# We focus on two charge-balanced families:

# FAMILY 1: Chalcogenides (A2+ B4+ X3) 
# Target: Solar Absorbers
exotic_chalcogenides = {
    "A": ["Ba", "Sr", "Ca", "Eu", "Yb"],  # Eu/Yb are Rare Earths that can be 2+
    "B": ["Hf", "Zr", "Ti", "Ce", "Th"],  # Hf is the main target here
    "X": ["S", "Se"]
}

# FAMILY 2: Halides (A1+ B2+ X3)
# Target: Magnetic Semiconductors / exotic electronics
exotic_halides = {
    "A": ["Cs", "Rb", "K"], 
    "B": ["V", "Mn", "Fe", "Co", "Ni", "Cu", "Mg", "Zn"], # 3d Transition Metals
    "X": ["Cl", "Br", "I"]
}

def calculate_tolerance_factor(r_a, r_b, r_x):
    """Goldschmidt Tolerance Factor: t = (r_a + r_x) / sqrt(2)*(r_b + r_x)"""
    return (r_a + r_x) / (np.sqrt(2) * (r_b + r_x))

def generate():
    count = 0
    print(f"Generating Exotic Candidates into {OUTPUT_DIR}...")

    # --- GENERATE CHALCOGENIDES ---
    for a in exotic_chalcogenides["A"]:
        for b in exotic_chalcogenides["B"]:
            for x in exotic_chalcogenides["X"]:
                
                # 1. Physics Check (Radii)
                try:
                    r_a = Element(a).atomic_radius
                    r_b = Element(b).atomic_radius
                    r_x = Element(x).atomic_radius
                    
                    if None in [r_a, r_b, r_x]: continue
                    
                    t = calculate_tolerance_factor(r_a, r_b, r_x)
                    
                    # Filter: Keep only geometrically plausible perovskites
                    # Relaxed range: 0.8 < t < 1.1 (Slightly distorted is fine)
                    if not (0.8 <= t <= 1.1):
                        continue
                        
                except:
                    continue

                # 2. Build Crystal (Ideal Cubic Pm-3m as starting point)
                # Lattice constant guess: 2 * (r_b + r_x) is a rough approximation
                a_lat = 2.0 * (r_b + r_x)
                lattice = Lattice.from_parameters(a_lat, a_lat, a_lat, 90, 90, 90)
                
                # Perovskite Coordinates (Pm-3m)
                # A: (0,0,0), B: (0.5, 0.5, 0.5), X: (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
                species = [a, b, x, x, x]
                coords = [
                    [0, 0, 0],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5]
                ]
                
                struct = Structure(lattice, species, coords)
                
                # 3. Save
                formula = struct.composition.reduced_formula
                filename = os.path.join(OUTPUT_DIR, f"{formula}.cif")
                struct.to(filename=filename)
                count += 1

    # --- GENERATE HALIDES ---
    for a in exotic_halides["A"]:
        for b in exotic_halides["B"]:
            for x in exotic_halides["X"]:
                # Repeat Logic (Condensed)
                try:
                    r_a = Element(a).atomic_radius
                    r_b = Element(b).atomic_radius
                    r_x = Element(x).atomic_radius
                    if not (0.8 <= calculate_tolerance_factor(r_a, r_b, r_x) <= 1.1): continue
                    
                    a_lat = 2.0 * (r_b + r_x)
                    lattice = Lattice.from_parameters(a_lat, a_lat, a_lat, 90, 90, 90)
                    struct = Structure(lattice, [a, b, x, x, x], [[0,0,0], [0.5,0.5,0.5], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]])
                    
                    filename = os.path.join(OUTPUT_DIR, f"{struct.composition.reduced_formula}.cif")
                    struct.to(filename=filename)
                    count += 1
                except: pass

    print(f"✅ Successfully generated {count} exotic candidates.")

if __name__ == "__main__":
    generate()
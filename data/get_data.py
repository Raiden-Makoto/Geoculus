import os

from dotenv import load_dotenv
from mp_api.client import MPRester
import pandas as pd
from pymatgen.io.cif import CifWriter #type: ignore

# ---------------- CONFIGURATION ---------------- #
load_dotenv()
API_KEY = os.getenv("MATERIALS_API_KEY")

# Desired search parameters
FORMULA_ANONYMOUS = "ABC3"  # Finds generic ABX3 stoichiometry
EXCLUDE_ELEMENTS = ["Pb"]   # explicitly remove Lead
FIELDS = [
    "material_id", 
    "formula_pretty", 
    "structure", 
    "band_gap", 
    "formation_energy_per_atom", 
    "energy_above_hull", 
    "symmetry"
]

def download_data():
    print("STEP 1: Fetching Metadata (Lightweight)...")
    
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            formula=FORMULA_ANONYMOUS,
            exclude_elements=EXCLUDE_ELEMENTS,
            fields=FIELDS,
        )
    print(f"Found {len(docs)} candidates.")
    
    data = []
    for d in docs:
        data.append({
            "material_id": d.material_id,
            "formula": d.formula_pretty,
            "band_gap": d.band_gap,
            "e_hull": d.energy_above_hull,
            "formation_energy": d.formation_energy_per_atom
        })
    df = pd.DataFrame(data)
    
    initial_count = len(df)
    df = df[df['e_hull'] < 0.5] 
    print(f"Filtered out extremely unstable materials. Reduced from {initial_count} to {len(df)}.")
    
    df.to_csv("perovskite_metadata.csv", index=False)
    
    print("STEP 2: Downloading Structures in Batches...")
    os.makedirs("structures", exist_ok=True)
    
    material_ids = df['material_id'].tolist()
    batch_size = 1000
    
    with MPRester(API_KEY) as mpr:
        for i in range(0, len(material_ids), batch_size):
            batch_ids = material_ids[i:i+batch_size]
            print(f"Downloading batch {i} to {i+batch_size}...")
            
            try:
                batch_docs = mpr.materials.summary.search(
                    material_ids=batch_ids, 
                    fields=["material_id", "structure"]
                )
                
                for doc in batch_docs:
                    cif_path = os.path.join("structures", f"{doc.material_id}.cif")
                    CifWriter(doc.structure).write_file(cif_path)
                    
            except Exception as e:
                print(f"Error in batch {i}: {e}")

    print("Download complete.")

if __name__ == "__main__":
    download_data()
import os
from pathlib import Path

from dotenv import load_dotenv
from mp_api.client import MPRester
import pandas as pd
from pymatgen.io.cif import CifWriter  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

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
    """Download perovskite data from Materials Project API."""
    # Get the script's directory and build path relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    dataset_dir = project_root / "dataset" / "raw"
    structures_dir = dataset_dir / "structures"
    
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
    
    # Save CSV to dataset/raw directory
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / "perovskite_metadata.csv"
    df.to_csv(csv_path, index=False)
    
    print("STEP 2: Downloading Structures in Batches...")
    structures_dir.mkdir(parents=True, exist_ok=True)
    
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
                    cif_path = structures_dir / f"{doc.material_id}.cif"
                    CifWriter(doc.structure).write_file(str(cif_path))
                    
            except Exception as e:
                print(f"Error in batch {i}: {e}")

    print("Download complete.")


def visualize_dataset(csv_path=None):
    """Visualize the perovskite dataset with plots and statistics."""
    # Get the script's directory and build path relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if csv_path is None:
        csv_path = project_root / "dataset" / "raw" / "perovskite_metadata.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} materials.")
    except FileNotFoundError:
        print("Error: CSV not found. Run the download script first!")
        return

    # Set graphical style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- PLOT 1: Band Gap Distribution ---
    sns.histplot(df['band_gap'], bins=50, kde=True, ax=axes[0], color='teal')
    axes[0].set_title("Distribution of Band Gaps ($E_g$)")
    axes[0].set_xlabel("Band Gap (eV)")
    axes[0].axvline(1.0, color='r', linestyle='--', alpha=0.5)
    axes[0].axvline(1.5, color='r', linestyle='--', alpha=0.5)
    axes[0].text(1.25, axes[0].get_ylim()[1]*0.9, "Solar\nTarget", 
                 ha='center', color='red', fontsize=10)

    # --- PLOT 2: Stability Distribution ---
    # We zoom in on the range relevant to synthesis (< 0.5 eV)
    sns.histplot(df['e_hull'], bins=50, kde=True, ax=axes[1], color='orange')
    axes[1].set_title("Thermodynamic Stability ($E_{hull}$)")
    axes[1].set_xlabel("Energy Above Hull (eV/atom)")
    axes[1].set_xlim(-0.05, 0.5) # Focus on potentially stable region
    axes[1].axvline(0.0, color='green', linestyle='-', linewidth=2, label="Stable")
    axes[1].axvline(0.05, color='green', linestyle='--', label="Metastable")
    axes[1].legend()

    # --- PLOT 3: The Discovery Map (Scatter) ---
    sns.scatterplot(data=df, x='band_gap', y='e_hull', alpha=0.6, s=20, ax=axes[2], color='purple')
    axes[2].set_title("Stability vs. Band Gap")
    axes[2].set_xlabel("Band Gap (eV)")
    axes[2].set_ylabel("Energy Above Hull (eV/atom)")
    axes[2].set_ylim(-0.1, 1.0) # Zoom in Y-axis
    axes[2].set_xlim(0, 4.0)    # Zoom in X-axis

    # Highlight the "Goldilocks Zone"
    # Rect(x, y, width, height)
    rect = patches.Rectangle((1.0, 0), 0.5, 0.05, linewidth=2, edgecolor='red', facecolor='none')
    axes[2].add_patch(rect)
    axes[2].text(1.25, 0.08, "Target Region", color='red', ha='center', fontweight='bold')

    plt.tight_layout()
    # Build output path relative to project root
    output_path = project_root / "data" / "bandgap_plot.png"
    plt.savefig(output_path)

    # --- STATS REPORT ---
    metals = df[df['band_gap'] == 0]
    insulators = df[df['band_gap'] > 3.0]
    perfect_candidates = df[(df['band_gap'] >= 1.0) & 
                            (df['band_gap'] <= 1.5) & 
                            (df['e_hull'] <= 0.05)]

    print(f"--- Data Health Report ---")
    print(f"Total entries: {len(df)}")
    print(f"Metals (Eg = 0): {len(metals)} ({len(metals)/len(df):.1%} of data)")
    print(f"Insulators (Eg > 3.0): {len(insulators)}")
    print(f"Candidate Count (Stable + Ideal Eg): {len(perfect_candidates)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_data()
    else:
        visualize_dataset()

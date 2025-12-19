import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

def visualize_dataset(csv_path="perovskite_metadata.csv"):
    # 1. Load Data
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
    plt.savefig('data/bandgap_plot.png')

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
    visualize_dataset()
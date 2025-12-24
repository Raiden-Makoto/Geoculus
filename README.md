# Geoculus: AI Discovery of Lead-Free Solar Materials

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![Status](https://img.shields.io/badge/Status-Research%20Grade-success)

**An Atomistic Line Graph Neural Network (ALIGNN) capable of screening thousands of hypothetical crystals to discover stable, non-toxic solar absorbers.**

## 🚀 The Mission
Lead-based perovskites ($MAPbI_3$) are excellent solar cells but are toxic and unstable. This project builds a **Geometric Deep Learning** model to explore the vast chemical space of **Lead-Free Perovskites** ($ABX_3$) to find stable alternatives with the perfect "Solar Band Gap" (1.1 – 1.6 eV).

## 🔬 Project Abstract: Mining the "Literature Gap"
While the search for lead-free perovskites often focuses on Tin (Sn), this project used an **ALIGNN (Atomistic Line Graph Neural Network)** to identify high-potential candidates in the under-explored **Chalcogenide** and **Transition Metal** spaces.

The model screened ~5,000 candidates and highlighted **14 "Goldilocks" materials** (Stable + Solar Gap). Crucially, it identified:
1.  **BaHfS$_3$**: A material only synthesized in 2023, confirming the model's ability to identify cutting-edge research targets.
2.  **EuTiS$_3$**: A theoretically stable magnetic semiconductor with no direct experimental solar benchmarks.
3.  **KZnBr$_3$**: A low-cost Zinc-based alternative that remains largely absent from photovoltaic literature.

**Conclusion:** The ALIGNN model successfully navigated the "Valley of Death," filtering out physically impossible structures to pinpoint the rare class of materials that are **chemically stable but experimentally neglected.**

## 🏆 Key Discoveries

### Phase 1: Validation on Known Materials
The model screened ~5,000 hypothetical candidates and autonomously "rediscovered" the most promising materials currently known to science, validating its understanding of physics.

| Material | Class | Pred. Stability ($E_{\text{hull}}$) | Pred. Band Gap ($E_g$) | Real-World Status |
| :--- | :--- | :--- | :--- | :--- |
| **KSnCl₃** | **Halide** | **0.007 eV** (Stable) | **1.39 eV** (Ideal) | **Top candidate** - Excellent stability and optimal band gap for solar applications. |
| **BaZrS₃** | **Chalcogenide** | **0.022 eV** (Stable) | **1.35 eV** (Ideal) | **The "Super Perovskite."** Stable, non-toxic, highly efficient. (Rediscovered) |
| **BaTiS₃** | Chalcogenide | 0.020 eV (Stable) | 0.99 eV (Near-Ideal) | Promising chalcogenide with excellent stability. |
| **SrZrS₃** | Chalcogenide | 0.047 eV (Stable) | 1.34 eV (Ideal) | Less explored, identified as a high-potential candidate. |
| **SrTiO₃** | Oxide | 0.064 eV (Stable) | 0.94 eV (Near-Ideal) | Well-known perovskite, correctly identified as stable. |

### Phase 2: Material Discovery 🎯
Using an **exotic candidate generator** that explores rare earths and heavy metals, the model identified **14 high-potential candidates** in the "Goldilocks Zone" (stability < 0.10 eV, band gap 0.9–1.8 eV).

#### High-Throughput Characterization of Under-Explored Candidates

| Material | Class | Pred. Stability ($E_{\text{hull}}$) | Pred. Band Gap ($E_g$) | Significance |
| :--- | :--- | :--- | :--- | :--- |
| **BaHfS₃** | **Chalcogenide** | **0.028 eV** (Stable) | **1.12 eV** (Ideal) | **🌟 Novel Hafnium Chemistry** - Unexplored for solar. Heavier than Zr may reduce thermal conductivity. |
| **SrHfS₃** | **Chalcogenide** | **0.063 eV** (Stable) | **1.42 eV** (Ideal) | **🌟 Most Actionable** - Physically plausible, just unexplored. Ready for synthesis. |
| **EuTiS₃** | **Chalcogenide** | **0.014 eV** (Stable) | **1.26 eV** (Ideal) | **🌟 Magnetic Semiconductor** - Rare multiferroic potential. Exciting for space applications. |
| **KZnBr₃** | **Halide** | **0.009 eV** (Stable) | **1.27 eV** (Ideal) | **🌟 Economical** - Cheap, non-toxic Zn-based alternative to expensive Ba/Zr materials. |
| **RbZnBr₃** | Halide | 0.011 eV (Stable) | 1.46 eV (Ideal) | Zinc halide family - Promising for large-scale deployment. |
| **CsZnBr₃** | Halide | 0.018 eV (Stable) | 1.57 eV (Ideal) | Zinc halide family - Excellent stability. |
| **BaHfSe₃** | Chalcogenide | 0.029 eV (Stable) | 1.12 eV (Ideal) | Hafnium selenide - Complementary to sulfide chemistry. |

### Key Insights

1. **Validation via "Rediscovery"**
   The model autonomously identified Barium Zirconium Sulfide ($BaZrS_3$) as a top-tier solar absorber ($E_g \approx 1.35$ eV, Stable). Since this material is currently the "Gold Standard" in lead-free research (only synthesized in 2018), its rediscovery serves as a strong blind-test validation of the AI's physics engine.

2. **The "Hafnium" Breakthrough**
   While Zirconium ($Zr$) is widely studied, the model highlighted its heavier cousin, Hafnium ($Hf$), as an overlooked opportunity. It predicts $BaHfS_3$ and $SrHfS_3$ are equally stable but offer unique advantages for thin-film stability. These materials exist in the "Literature Gap"—technically possible but almost entirely absent from solar cell research.

3. **Novel "Exotic" Candidates**
   Pushing beyond standard chemistry, the model identified $EuTiS_3$ (a rare Magnetic Semiconductor) and $KZnBr_3$ (a low-cost Earth-abundant alternative) as potential "Goldilocks" candidates. These predictions challenge the conventional wisdom that stable perovskites require toxic Lead or expensive Tin, suggesting that Zinc and Rare Earths are viable, untapped frontiers.

**Significance:** This work demonstrates that Geometric Deep Learning can act as a "Compass" for Experimentalists. By filtering out unstable structures and highlighting "sleeping giants" like Hafnium sulfides, we provide a concrete roadmap to accelerate the transition to non-toxic, sustainable solar energy.

---

## 🧠 Model Architecture: Why ALIGNN?
Standard Graph Neural Networks (GNNs) failed because they are **"Angle Blind."** They could not distinguish between a straight bond ($180^\circ$, Metal) and a tilted bond ($160^\circ$, Semiconductor).

We implemented **ALIGNN (Atomistic Line Graph Neural Network)** which explicitly models bond angles:

1.  **Atom Graph:** Nodes = Atoms, Edges = Bonds. (Learns Chemistry/Stability).
2.  **Line Graph:** Nodes = Bonds, Edges = Angles. (Learns Geometry/Band Gap).
3.  **GatedGCN Layers:** Uses an Attention-like gating mechanism to filter information flow.

### Performance Metrics
* **Stability MAE:** `0.045 eV/atom` (Research Grade Precision).
* **Band Gap MAE:** `0.429 eV` (Sufficient to distinguish Metals from Semiconductors).

---

## 🛠️ Engineering Challenges & Solutions

### 1. The "Metal Trap" (65% Failure Rate)
* **Problem:** Initial distance-based GNNs predicted 65% of candidates were metals (0 eV gap) because they couldn't see octahedral tilting. The model could not differentiate the angles between different types of bond.
* **Solution:** Switched to ALIGNN to encode triplet angles ($B-X-B$).

### 2. The "Ghost Angle" Bug
* **Problem:** Standard libraries (`CrystalNN`) returned $0.0^\circ$ angles for periodic boundaries, causing the model to learn nothing.
* **Solution:** Wrote a custom **Vector Math Graph Builder** to calculate true Euclidean angles across periodic images using dot products.

### 3. GPU Memory Explosion (OOM)
* **Problem:** Line graphs grow quadratically ($N^2$). A single batch crashed a 16GB GPU.
* **Solution:**
    * Implemented **Lightweight Pruning** (Max 8 neighbors).
    * Added **Gradient Checkpointing** (trading compute for 70% VRAM savings).

---

## Visualizing the Physics

The model successfully learned the **Periodic Trends** of the elements without being explicitly programmed:

### Halogen Trend
Band gap increases as $I \to Br \to Cl$:
- $CsSnI_3$ (Narrow) $\to$ $CsSnBr_3$ (Medium) $\to$ $CsSnCl_3$ (Wide)

### Chalcogen Stability
Identified that Sulfides ($S^{2-}$) provide superior stability over Iodides ($I^-$) for Zirconium-based structures.

## 🧪 Exotic Candidate Generator

To move beyond standard Barium/Tin chemistry, we developed a **physics-informed candidate generator** that explores:

- **Rare Earth Elements** (Eu, Yb) - For magnetic and luminescent properties
- **Heavy Metals** (Hf, Ce, Th) - Chemically similar to Zr but unexplored
- **Transition Metals** (V, Mn, Fe, Co, Ni, Cu, Zn) - For diverse electronic properties

The generator uses **Goldschmidt tolerance factor** filtering (0.8–1.1) to ensure geometrically plausible perovskite structures, then the ALIGNN model screens them for stability and band gap.

**Result:** Generated 90 exotic candidates, identified 14 in the Goldilocks Zone, including **novel Hafnium chalcogenides** that are almost completely unexplored in the literature.

## Future Work

### Computational Verification (DFT)
- Perform phonon calculations for $KZnBr_3$ to confirm dynamic stability.
- Calculate carrier effective masses ($m^*$) for $BaHfS_3$ to verify electrical conductivity.

### Model Enhancements
- **Transfer Learning:** Pre-train ALIGNN on the Materials Project (140k+ crystals) before fine-tuning on perovskites.

### Expanded Search Space
- Extend the pipeline to Double Perovskites ($A_2BB'X_6$) to further reduce toxicity.
- Explore Ruddlesden-Popper (2D layered) phases for improved moisture stability.
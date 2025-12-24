# Geoculus: AI Discovery of Lead-Free Solar Materials

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange)
![Status](https://img.shields.io/badge/Status-Research%20Grade-success)

**An Atomistic Line Graph Neural Network (ALIGNN) capable of screening thousands of hypothetical crystals to discover stable, non-toxic solar absorbers.**

## 🚀 The Mission
Lead-based perovskites ($MAPbI_3$) are excellent solar cells but are toxic and unstable. This project builds a **Geometric Deep Learning** model to explore the vast chemical space of **Lead-Free Perovskites** ($ABX_3$) to find stable alternatives with the perfect "Solar Band Gap" (1.1 – 1.6 eV).

## 🏆 Key Discoveries
The model screened ~5,000 hypothetical candidates and autonomously "rediscovered" the most promising materials currently known to science, validating its understanding of physics.

| Material | Class | Pred. Stability ($E_{hull}$) | Pred. Band Gap ($E_g$) | Real-World Status |
| :--- | :--- | :--- | :--- | :--- |
| **BaZrS$_3$** | **Chalcogenide** | **0.025 eV** (Stable) | **1.35 eV** (Ideal) | **The "Super Perovskite."** Stable, non-toxic, highly efficient. (Rediscovered) |
| **CsSnI$_3$** | Halide | 0.052 eV (Metastable) | 0.00 - 0.5 eV | The "Holy Grail" of Sn-halides. Conductive/Black phase identified. |
| **SrZrS$_3$** | Chalcogenide | 0.049 eV (Stable) | 1.27 eV (Ideal) | Less explored, identified as a high-potential candidate. |
| **CsSnCl$_3$** | Halide | -0.017 eV (Stable) | 2.19 eV (Wide) | Correctly identified as a transparent wide-gap material. |

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

## Conclusion
This project demonstrates an end-to-end workflow in Materials Informatics. By combining custom graph engineering with state-of-the-art deep learning, we successfully filtered thousands of materials to find BaZrS$_3$, a material that could define the next generation of solar energy.
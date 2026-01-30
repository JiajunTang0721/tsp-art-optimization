# Optimization Meets Art: TSP-Based Portrait Rendering

This project recreates a portrait image as a **single continuous line drawing** using the **Traveling Salesperson Problem (TSP)**.  
By combining **image-aware point sampling**, **degree-2 TSP optimization**, **geometric subtour merging**, and **2-opt local refinement**, the final result transforms a static image into a visually coherent optimization-driven artwork.

---

## Project Overview

- **Goal**: Generate a recognizable portrait using a single continuous tour produced by TSP-style optimization.
- **Pipeline**: Image → weighted sampling → degree-2 TSP (subtours) → geometric merging → 2-opt → line art
- **Tech**: Python · NumPy · OpenCV · Matplotlib · Gurobi

---

## Results

### 1) Image-Aware Sampling Visualization
![Sampling Visualization](results/sampling_visualization.png)

### 2) Final TSP Art Output
![TSP Art Result](results/tsp_art_result.png)

---

## Methodology

### Step 1 — Image-Aware Point Sampling
Points are sampled **non-uniformly** to preserve perceptually important structures.

- **Edge strength**: Sobel magnitude + Laplacian magnitude  
- **Local contrast**: local darkness relative to heavily blurred neighborhood  
- **Region boosts**:
  - Haar cascade face detection
  - Elliptical face mask
  - Gaussian emphasis on facial feature region
  - Rectangle boost for hair region
- **Background suppression**: reduce weights for blue-toned background pixels

The combined weight map is normalized into a probability distribution and used for weighted sampling.

---

### Step 2 — Degree-2 TSP Optimization (Relaxed Model)
Solve a degree-constrained MILP:

- Binary edge variables \(x_{ij}\)
- Minimize total Euclidean length
- Degree constraints:
  \[
  \sum_{j \neq i} x_{ij} = 2 \quad \forall i
  \]

This guarantees each node has degree 2 but typically returns **multiple disjoint cycles (subtours)**.

---

### Step 3 — Subtour Merging via Geometry
Instead of adding expensive subtour-elimination constraints, subtours are merged post-solve:

- pick the closest pair of nodes between two cycles
- rotate cycles to align
- try concatenation and reversal variants
- keep the shortest merged ordering
- repeat until one tour remains

---

### Step 4 — 2-opt Local Refinement
Apply 2-opt to reduce crossings and improve local structure:

- randomly pick two edges
- reverse segment if length decreases
- stop after many non-improving steps (patience threshold)

---

## Challenges & Solutions

- **Uniform sampling loses recognizability**
  - **Fix**: edge + local darkness weighting; face/hair region boosts; background suppression
- **Degree-2 model yields many subtours**
  - **Fix**: geometric subtour merging to form one continuous tour
- **Merged tour has crossings / clutter**
  - **Fix**: large-budget 2-opt with patience stopping
- **Scalability (dense \(O(N^2)\) edges)**
  - **Mitigation**: reduce `num_points`; future extension: kNN sparse edges / coarse-to-fine refinement

---

## Repository Structure

```text
.
├── src/
│   └── tsp_art.py
├── assets/
│   └── input_portrait.jpg
├── results/
│   ├── sampling_visualization.png
│   └── tsp_art_result.png
└── README.md

# Portrait TSP Art (Edge-Aware Sampling + Degree-2 Optimization + Tour Heuristics)

This project converts a portrait image into **single-stroke line art** by:
1) sampling thousands of points from the image **with carefully designed pixel-level weights**,  
2) building a **degree-2 minimum-length graph** (a multi-cycle “2-factor” structure) via Gurobi,  
3) merging cycles using a **geometry-driven heuristic**, and  
4) refining the final route using **2-opt** local search, then rendering the tour as a continuous polyline.

---

## Table of Contents
- [Overview](#overview)
- [Pipeline](#pipeline)
  - [1) Weighted Point Sampling from the Portrait](#1-weighted-point-sampling-from-the-portrait)
  - [2) Degree-2 Optimization Model (2-Factor) with Gurobi](#2-degree2-optimization-model-2factor-with-gurobi)
  - [3) Cycle Extraction and Geometric Merging](#3-cycle-extraction-and-geometric-merging)
  - [4) 2-opt Tour Improvement](#4-2opt-tour-improvement)
  - [5) Rendering / Export](#5-rendering--export)
- [Key Design Choices (Why This Works)](#key-design-choices-why-this-works)
- [Parameters You Can Tune](#parameters-you-can-tune)
- [Complexity & Practical Notes](#complexity--practical-notes)
- [Reproducibility](#reproducibility)
- [Requirements](#requirements)

---

## Overview
The core idea is to **allocate points where visual structure is dense** (edges, hair, facial features, local shadows) and suppress points in low-information regions (e.g., uniform background).  
A tour is then constructed so that connecting these points produces a line drawing that visually resembles the portrait.

---

## Pipeline

### 1) Weighted Point Sampling from the Portrait
**Goal:** Sample `N` points so that the point cloud concentrates on meaningful structures (contours, hair strands, facial details), rather than wasting samples on flat background.

**Steps & Methods**

1. **Image preprocessing**
   - Resize the input image to a fixed width (aspect ratio preserved) using a high-quality interpolation method.
   - Convert to grayscale for stable edge/contrast computations.

2. **Edge-aware weighting (structure emphasis)**
   - Compute **Sobel gradients** in x/y and combine them into a gradient magnitude map.
     - **What it captures:** strong boundaries and contours (face outline, edges of features).
   - Compute **Laplacian response magnitude**.
     - **What it captures:** high-frequency details and sharp intensity changes (fine textures, hair detail).
   - Normalize both maps to comparable scales and form a weighted combination.
     - **Problem solved:** avoids one detector dominating due to scale differences; uses Sobel for robust contours and Laplacian for fine detail.

3. **Local darkness / shadow detail weighting (contrast emphasis)**
   - Convert grayscale to a “darkness map” (darker pixels → larger values).
   - Apply a **large-kernel Gaussian blur** to obtain a local baseline (low-frequency background illumination).
   - Subtract blurred darkness from raw darkness and clamp to keep only positive residuals.
     - **What it captures:** pixels that are **locally darker than their neighborhood** (shadow edges, local texture).
     - **Problem solved:** preserves important tonal structure beyond pure edges (e.g., shading on face/hair).

4. **Region boosting using face detection (semantic emphasis)**
   - Detect the primary face using a **Haar cascade frontal-face detector**.
   - If a face is found, construct three region masks:
     - **Face ellipse mask:** boosts the overall face region.
     - **Facial-feature Gaussian mask:** smoothly boosts central feature area (eyes/nose/mouth region) without hard boundaries.
     - **Hair rectangle mask:** strongly boosts the hair region above and around the face bounding box.
   - Combine these masks into a single **multiplicative region boost**.
     - **Problem solved:** concentrates sampling on the subject (especially hair and features), producing a clearer portrait-like stroke outcome.

5. **Background suppression (color-based heuristic)**
   - Use simple RGB/BGR thresholding to identify a “blue-ish” background region (common in portrait photos).
   - Down-weight these pixels substantially.
     - **Problem solved:** prevents sample waste in flat background; reduces noisy strokes.

6. **Final probability distribution and sampling**
   - Aggregate weights: **edges + local darkness**, then multiply by **region boost**, then apply **background suppression**.
   - Add a small epsilon for numerical stability and normalize to a probability distribution over all pixels.
   - Sample `N` pixels using the normalized probabilities (**importance sampling**).
   - Convert sampled pixel coordinates to normalized `[0,1]×[0,1]` coordinates (with y-axis flipped for plotting consistency).

**Output:** A set of normalized 2D points representing the portrait’s most informative structures.

---

### 2) Degree-2 Optimization Model (2-Factor) with Gurobi
**Goal:** Connect points with minimal total length while ensuring each point has exactly two incident edges, forming one or more cycles.

**Model Structure**
- Define binary decision variables for undirected edges between point pairs.
- Objective: minimize the sum of selected edge lengths (Euclidean distances).
- Constraint: for every node, the number of incident selected edges must equal **2**.

**What this produces**
- A **2-regular graph**: every node has degree 2, which decomposes into **multiple disjoint cycles** (a “2-factor”).
- This is intentionally *not* a full TSP with subtour elimination.
  - **Problem solved:** the full TSP formulation is significantly harder at large scale; the degree-2 model is a structured relaxation that can be solved more reliably under time limits, then repaired/merged with heuristics.

---

### 3) Cycle Extraction and Geometric Merging
**Goal:** Convert multiple disjoint cycles into a single long tour order suitable for continuous drawing.

**Cycle extraction**
- Build adjacency lists from chosen edges (each node should have 2 neighbors).
- Traverse unvisited nodes to recover each cycle as an ordered list.

**Geometric merging heuristic**
- For each pair of cycles, compute the closest pair of nodes between them (minimum Euclidean distance).
- Pick the two closest cycles and merge them by:
  - rotating each cycle so the chosen “closest node” becomes the merge anchor,
  - considering multiple orientation combinations (keeping/reversing direction of each cycle),
  - selecting the merged ordering that yields the shortest resulting open-path length.

**Why this works**
- The nearest inter-cycle connection is a strong geometric signal for minimal added length.
- Testing directions prevents accidental long detours caused by inconsistent cycle orientations.

**Output:** A single “big tour” (node ordering) containing all points.

---

### 4) 2-opt Tour Improvement
**Goal:** Remove crossings and shorten the route with a classic local search method.

**2-opt move**
- Randomly select two non-adjacent edges `(a–b)` and `(c–d)`.
- Replace them with `(a–c)` and `(b–d)` if this reduces total length.
- Apply by reversing the tour segment between the selected indices.

**Stopping logic**
- Iterate up to a maximum number of iterations.
- Use a “patience” threshold: stop early after many consecutive non-improving trials.

**Problem solved**
- The merged tour may contain crossings and local inefficiencies.
- 2-opt tends to “straighten” the route, improving visual coherence and reducing unnecessary zig-zags.

---

### 5) Rendering / Export
**Goal:** Render a clean line drawing from the final tour.

- Close the tour by returning to the start point.
- Plot as a single thin black polyline.
- Hide axes and enforce equal aspect ratio.
- Export to a high DPI image to preserve fine strokes.

---

## Key Design Choices (Why This Works)
- **Multi-signal sampling** (edges + local darkness): captures both sharp contours and shading-driven structure.
- **Semantic region boosts** (face/features/hair): prevents the algorithm from “wasting” points on irrelevant background and improves portrait recognizability.
- **Degree-2 optimization (2-factor)**: provides a strong global structure under a time limit without the full complexity of exact TSP subtour constraints.
- **Geometric cycle merging**: transforms multiple cycles into one continuous path using nearest-neighbor geometry and orientation checks.
- **2-opt**: removes crossings and improves local geometry, which is crucial for aesthetic line quality.

---

## Parameters You Can Tune
### Sampling / Image
- `num_points`: more points → richer detail but much heavier optimization.
- `resize_width`: controls pixel grid size for weight computation.
- Edge weights: relative importance of Sobel vs Laplacian combination.
- Local-darkness kernel size (Gaussian blur): larger kernel emphasizes broader local contrast.
- Region boost coefficients:
  - face ellipse contribution
  - feature Gaussian strength
  - hair mask multiplier (often the most visually impactful)
- Background suppression thresholds: adjust for different background colors.

### Optimization
- `time_limit`: solver time budget.
- `verbose`: toggle solver logs.

### Heuristics
- Cycle merge strategy: nearest-cycle pair selection and orientation options.
- `two_opt`: `max_iter`, `patience` to trade runtime for improvement.

### Rendering
- `lw` (line width): thin lines for stroke style; thicker lines for bold style.
- `dpi`: high DPI for crisp output.

---

## Complexity & Practical Notes
- Pairwise distance computation and full edge variable creation scale as **O(n²)**.
  - Large `num_points` can become computationally and memory intensive.
- The degree-2 model is a relaxation: it may return multiple cycles, which are merged heuristically afterward.
- If face detection fails, the pipeline still works using edge + local darkness signals, but recognizability may drop depending on background complexity.

---

## Reproducibility
- Random seeds are set for both Python `random` and NumPy, ensuring consistent sampling and 2-opt behavior across runs (given the same environment and solver behavior).

---

## Requirements
- Python 3.x
- `numpy`
- `opencv-python`
- `matplotlib`
- `gurobipy` (requires a working Gurobi installation and license)

(Optional cleanup)
- `PIL` and `time` are not required unless you extend the project; they can be removed from imports if unused.

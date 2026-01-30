#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum
import random
import time

# 1. Sampling points from portrait image
def load_and_sample_points(image_path, num_points=2000, resize_width=800, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Read and resize image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    new_h = int(h * resize_width / w)
    img = cv2.resize(img, (resize_width, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Convert to grayscale
    bgr = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    H, W = gray.shape

    # Sobel and Laplacian edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_mag = np.abs(lap)

    sobel_norm = sobel_mag / (sobel_mag.max() + 1e-8)
    lap_norm = lap_mag / (lap_mag.max() + 1e-8)
    W_edges = 0.6 * sobel_norm + 0.4 * lap_norm

    # Local darkness detection
    darkness = 255 - gray
    darkness_blur = cv2.GaussianBlur(darkness, (51, 51), 0)
    local_dark = np.clip(darkness - darkness_blur, 0, None)
    local_dark_norm = local_dark / (local_dark.max() + 1e-8)

    W_local_dark = 0.30 * local_dark_norm

    # Base region boost
    region_boost = np.ones((H, W), dtype=np.float32)

    # Face region enhancement
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray.astype(np.uint8), 1.1, 5)

    if len(faces) > 0:
        x, y, fw, fh = sorted(faces, key=lambda x: x[2] * x[3])[-1]

        # Face ellipse mask
        face_mask = np.zeros((H, W), float)
        center = (x + fw // 2, y + fh // 2)
        axes = (int(fw * 0.5), int(fh * 0.65))
        cv2.ellipse(face_mask, center, axes, 0, 0, 360, 1, -1)

        # Facial feature Gaussian mask
        features_mask = np.zeros((H, W), float)
        cx = x + fw // 2
        cy = y + int(fh * 0.45)
        rx = int(fw * 0.28)
        ry = int(fh * 0.22)

        Y, X = np.ogrid[:H, :W]
        dist = ((X - cx) / (rx + 1e-8))**2 + ((Y - cy) / (ry + 1e-8))**2
        features_mask = np.exp(-dist * 2.5)

        # Hair enhancement rectangle
        hair_mask = np.zeros((H, W), float)
        hx1 = max(0, x - int(fw * 0.4))
        hx2 = min(W, x + int(fw * 1.4))
        hy1 = max(0, y - int(fh * 1.0))
        hy2 = min(H, y + int(fh * 0.2))
        hair_mask[hy1:hy2, hx1:hx2] = 1.0

        # Combine boosts
        region_boost = (
            0.6
            + 0.4 * face_mask
            + 0.3 * features_mask
            + 2.5 * hair_mask
        )

    # Background suppression (blue tone)
    B = bgr[:, :, 0]
    G = bgr[:, :, 1]
    R = bgr[:, :, 2]

    background_mask = (B > 120) & (R < 130) & (G < 130)
    region_boost[background_mask] *= 0.15

    # Weight aggregation and normalization
    W_total = 1.4 * W_edges + W_local_dark
    W_total = W_total * region_boost

    assert W_total.ndim == 2, f"W_total shape error: {W_total.shape}"

    W_total += 1e-8
    W_total /= W_total.sum()

    # Weighted sampling
    flat_idx = np.random.choice(H * W, num_points, p=W_total.ravel())
    ys, xs = np.unravel_index(flat_idx, (H, W))

    xs_norm = xs / (W - 1)
    ys_norm = 1 - ys / (H - 1)

    points = np.column_stack([xs_norm, ys_norm])

    return points, gray


# 2. Degree-2 Gurobi model (multiple cycles)
def compute_distance_matrix(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def extract_cycles(chosen_edges, n):
    neigh = [[] for _ in range(n)]
    for i, j in chosen_edges:
        neigh[i].append(j)
        neigh[j].append(i)

    visited = [False] * n
    cycles = []

    for start in range(n):
        if visited[start]:
            continue
        cycle, cur, prev = [], start, -1
        while True:
            cycle.append(cur)
            visited[cur] = True
            nxt = [x for x in neigh[cur] if x != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
        cycles.append(cycle)
    return cycles


def solve_degree_tsp(points, time_limit=300, verbose=True):
    n = points.shape[0]
    dist = compute_distance_matrix(points)

    m = Model("deg2")
    if not verbose:
        m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)

    x = {}
    for i in range(n):
        for j in range(i + 1, n):
            x[i, j] = m.addVar(vtype=GRB.BINARY, obj=dist[i, j])

    m.update()

    # Degree constraints
    for i in range(n):
        expr = quicksum(x[min(i, j), max(i, j)] for j in range(n) if j != i)
        m.addConstr(expr == 2)

    m.optimize()

    edges = [(i, j) for (i, j), v in x.items() if v.X > 0.5]
    return extract_cycles(edges, n), m


# 3. Merge cycles using geometric heuristic
def cycle_distance(c1, c2, points):
    best = 1e9
    best_pair = (0, 0)
    for i, a in enumerate(c1):
        for j, b in enumerate(c2):
            d = np.linalg.norm(points[a] - points[b])
            if d < best:
                best = d
                best_pair = (i, j)
    return best, best_pair


def merge_two_cycles_geo(c1, c2, points):
    _, (i1, i2) = cycle_distance(c1, c2, points)

    def rotate(c, idx):
        return c[idx:] + c[:idx]

    c1r, c2r = rotate(c1, i1), rotate(c2, i2)

    options = [
        c1r + c2r,
        c1r + list(reversed(c2r)),
        list(reversed(c1r)) + c2r,
        list(reversed(c1r)) + list(reversed(c2r)),
    ]

    def path_len(c):
        p = points[c]
        return np.sum(np.linalg.norm(p[1:] - p[:-1], axis=1))

    return min(options, key=path_len)


def merge_all_cycles(cycles, points):
    cycles = [list(c) for c in cycles]

    while len(cycles) > 1:
        best = 1e9
        pair = (0, 1)
        for i in range(len(cycles)):
            for j in range(i + 1, len(cycles)):
                d, _ = cycle_distance(cycles[i], cycles[j], points)
                if d < best:
                    best, pair = d, (i, j)

        i, j = pair
        merged = merge_two_cycles_geo(cycles[i], cycles[j], points)

        cycles_new = []
        for k in range(len(cycles)):
            if k not in (i, j):
                cycles_new.append(cycles[k])
        cycles_new.append(merged)
        cycles = cycles_new

        print(f"Merged one cycle, remaining cycles = {len(cycles)}")

    return cycles[0]


# 4. 2-opt improvement
def tour_length(tour, points):
    p = points[tour]
    return np.sum(np.linalg.norm(p[1:] - p[:-1], axis=1)) + np.linalg.norm(
        p[0] - p[-1]
    )


def two_opt_improve(tour, points, max_iter=100000, patience=10000):
    n = len(tour)
    tour = tour.copy()
    best_len = tour_length(tour, points)
    no_improve = 0

    for it in range(max_iter):
        i = random.randint(0, n - 3)
        j = random.randint(i + 2, n - 1)
        if i == 0 and j == n - 1:
            continue

        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n if j + 1 < n else 0]

        pa, pb, pc, pd = points[a], points[b], points[c], points[d]

        old = np.linalg.norm(pa - pb) + np.linalg.norm(pc - pd)
        new = np.linalg.norm(pa - pc) + np.linalg.norm(pb - pd)

        if new + 1e-9 < old:
            tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
            best_len -= (old - new)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > patience:
            print(f"2-opt stopped after {patience} non-improving steps")
            break

    print("2-opt improved length:", best_len)
    return tour


# 5. Drawing function
def plot_tsp(tour, points, save_path=None, lw=0.3, dpi=600):
    pts = points[tour + [tour[0]]]
    xs, ys = pts[:, 0], pts[:, 1]

    plt.figure(figsize=(7, 7))
    plt.plot(xs, ys, color="black", linewidth=lw)
    plt.axis("off")
    plt.gca().set_aspect("equal", "box")

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        print("Saved:", save_path)
    else:
        plt.show()


# Main execution
if __name__ == "__main__":
    image_path = "example.jpg"
    output_path = "tsp_art_result.png"

    num_points = 8000
    time_limit = 900

    points, _ = load_and_sample_points(image_path, num_points=num_points)
    print("Sampling completed")

    img_show = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(7, 7))
    plt.imshow(img_show)
    plt.scatter(points[:, 0] * img_show.shape[1],
                (1 - points[:, 1]) * img_show.shape[0],
                s=3, c='red', alpha=0.8)
    plt.axis("off")
    plt.title("Sampling Visualization")
    plt.show()

    cycles, model = solve_degree_tsp(points, time_limit=time_limit)
    print(f"Initial number of cycles = {len(cycles)}")

    big_tour = merge_all_cycles(cycles, points)
    print("Merged path length (number of nodes):", len(big_tour))

    big_tour_opt = two_opt_improve(big_tour, points,
                                   max_iter=300000,
                                   patience=15000)

    plot_tsp(big_tour_opt, points, save_path=output_path, lw=0.25, dpi=900)


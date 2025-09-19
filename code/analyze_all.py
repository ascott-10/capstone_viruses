#!/usr/bin/env python3
"""
Analyze all segmented virus images (wildtype + mutant).
Extracts:
  - spike angles
  - centroid→spike distances
  - spike–spike perimeter distances

Saves:
  - results/tables/all_distance_data.csv (raw measurements)
  - results/tables/summary_stats.csv (per-class summary)
  - results/figures/kde_comparisons.png (mutant vs wildtype KDE plots)

Author: Arielle Scott
"""

import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import regionprops, label
from scipy.stats import skew, ks_2samp

#### Paths ####
mutant_dir = "/home/ascott10/documents/projects/capstone_viruses/segmented_images/mutant_manual_correction"
wt_dir     = "/home/ascott10/documents/projects/capstone_viruses/segmented_images/wildtype_manual_correction"
results_dir = "/home/ascott10/documents/projects/capstone_viruses/results"
tables_dir  = os.path.join(results_dir, "tables")
figures_dir = os.path.join(results_dir, "figures")
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

#### Functions ####

def extract_morphology(segmented_image_path):
    """Extract body mask, spike coordinates, and centroid."""
    image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None, None, []
    body_mask = (image == 150).astype(np.uint8)
    spikes_mask = (image == 200).astype(np.uint8)
    body_labels = label(body_mask)
    spike_labels = label(spikes_mask)
    body_props = regionprops(body_labels)
    spike_props = regionprops(spike_labels)
    if not body_props:
        return None, None, None, None, []
    body = body_props[0]
    cy, cx = map(int, body.centroid)
    spike_coords = [(int(p.centroid[1]), int(p.centroid[0])) for p in spike_props]
    return body_mask, cx, cy, body, spike_coords

def get_perimeter_coords(body_mask):
    """Find perimeter pixels of the virus body."""
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = np.vstack(contours)[:, 0, :]
    return [(int(x), int(y)) for (x, y) in contour]

def compute_spike_pair_distances(cx, cy, spike_coords, perimeter_coords):
    """Compute perimeter-adjusted spike–spike distances (Frank’s method)."""
    P_len = len(perimeter_coords)
    if P_len == 0 or len(spike_coords) < 2:
        return []
    distances = []
    for i, (xi, yi) in enumerate(spike_coords):
        Ai = np.degrees(np.arctan2(yi - cy, xi - cx)) % 360
        Ri = np.sqrt((xi - cx)**2 + (yi - cy)**2)
        Pi = min(np.sqrt((px - cx)**2 + (py - cy)**2) for (px, py) in perimeter_coords)
        for j, (xj, yj) in enumerate(spike_coords):
            if j <= i: 
                continue
            Aj = np.degrees(np.arctan2(yj - cy, xj - cx)) % 360
            Rj = np.sqrt((xj - cx)**2 + (yj - cy)**2)
            Pj = min(np.sqrt((px - cx)**2 + (py - cy)**2) for (px, py) in perimeter_coords)
            diff = abs(Ai - Aj)
            ang_dist = min(diff, 360 - diff)
            M = int((ang_dist / 360.0) * P_len)
            Dij = 0.5 * ((M * Ri / Pi) + (M * Rj / Pj))
            distances.append(Dij)
    return distances

def analyze_dataset(image_paths, label, results):
    """Extract measurements for all images in a dataset."""
    for path in image_paths:
        body_mask, cx, cy, body, spike_coords = extract_morphology(path)
        if body_mask is None or len(spike_coords) < 2:
            continue
        perimeter_coords = get_perimeter_coords(body_mask)

        # angles
        spike_angles = [np.arctan2(y - cy, x - cx) for (x, y) in spike_coords]
        results.extend([{"sample": os.path.basename(path), "class": label, "metric": "angle", "value": a}
                        for a in spike_angles])

        # centroid→spike distances
        radial = [np.sqrt((x - cx)**2 + (y - cy)**2) for (x, y) in spike_coords]
        results.extend([{"sample": os.path.basename(path), "class": label, "metric": "radial", "value": r}
                        for r in radial])

        # pairwise spike–spike distances
        pair_d = compute_spike_pair_distances(cx, cy, spike_coords, perimeter_coords)
        results.extend([{"sample": os.path.basename(path), "class": label, "metric": "pairwise", "value": d}
                        for d in pair_d])

#### Main ####

if __name__ == "__main__":
    # collect all files
    mutant_files = [os.path.join(mutant_dir, f) for f in os.listdir(mutant_dir) if f.lower().endswith((".png", ".tif", ".jpg"))]
    wt_files     = [os.path.join(wt_dir, f) for f in os.listdir(wt_dir) if f.lower().endswith((".png", ".tif", ".jpg"))]

    results = []
    analyze_dataset(mutant_files, "mutant", results)
    analyze_dataset(wt_files, "wildtype", results)

    df = pd.DataFrame(results)
    out_csv = os.path.join(tables_dir, "all_distance_data.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved raw data to {out_csv}")

    # summary stats
    summary = (
        df.groupby(["class", "metric"])
          .agg(mean=("value","mean"),
               variance=("value","var"),
               skewness=("value", lambda x: skew(x, bias=False)),
               count=("value","size"))
          .reset_index()
    )
    out_sum = os.path.join(tables_dir, "summary_stats.csv")
    summary.to_csv(out_sum, index=False)
    print(f"Saved summary stats to {out_sum}")

    # plots: KDE comparisons
    sns.set_style("whitegrid")
    metrics = ["angle", "radial", "pairwise"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, metrics):
        subset = df[df["metric"] == metric]
        if not subset.empty:
            sns.kdeplot(data=subset, x="value", hue="class", ax=ax, common_norm=False)
            ax.set_title(metric)
    plt.tight_layout()
    out_fig = os.path.join(figures_dir, "kde_comparisons.png")
    plt.savefig(out_fig, dpi=300)
    print(f"Saved KDE comparison plot to {out_fig}")
    plt.show()

#!/usr/bin/env python3
"""
Coronavirus Spike Distance Analysis

This script runs across all segmented virus images (mutant + wildtype),
and computes distributions of:
    1. Spike angles
    2. Spike radial distances
    3. Perimeter radial profile
    4. Spike–spike distances along the virus perimeter

Outputs:
  - results/tables/all_distance_data.csv   (raw measurements)
  - results/tables/summary_stats.csv       (summary statistics per class/metric)
  - results/figures/kde_comparisons.png    (mutant vs wildtype KDE plots)

Implements Frank’s perimeter distance procedure:
  - Angular wrapping (e.g. 350° vs 10° → 20°)
  - Perimeter-based distance, not straight-line
  - Distance adjustment using (Ri/Pi), averaged across spike pairs
  - Smoothed histograms (post-processing)

Author: Arielle Scott (incorporating Frank’s notes, Prof. Hasan’s input)
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import regionprops, label
from scipy.stats import skew

#### Functions ####

def extract_morphology(segmented_image_path):
    """Extract body mask, spike coords, and centroid from segmented image."""
    image = cv2.imread(segmented_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None, []

    body_mask = (image == 150).astype(np.uint8)   # virus particle
    spikes_mask = (image == 200).astype(np.uint8) # spikes

    body_labels = label(body_mask)
    spike_labels = label(spikes_mask)

    body_props = regionprops(body_labels)
    spike_props = regionprops(spike_labels)

    if not body_props:
        return None, None, None, []

    body = body_props[0]
    cy, cx = map(int, body.centroid)

    spike_coords = [(int(p.centroid[1]), int(p.centroid[0])) for p in spike_props]
    return body_mask, cx, cy, spike_coords


def get_perimeter_coords(body_mask):
    """Find perimeter coordinates of virus particle."""
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
            ang_dist = min(diff, 360 - diff)  # wraparound
            M = int((ang_dist / 360.0) * P_len)

            Dij = 0.5 * ((M * Ri / Pi) + (M * Rj / Pj))
            distances.append(Dij)

    return distances


#### Main ####

if __name__ == "__main__":
    mutant_dir = "/home/ascott10/documents/projects/capstone_viruses/segmented_images/mutant_manual_correction"
    wt_dir     = "/home/ascott10/documents/projects/capstone_viruses/segmented_images/wildtype_manual_correction"

    results_dir = "/home/ascott10/documents/projects/capstone_viruses/results"
    tables_dir  = os.path.join(results_dir, "tables")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # collect all files
    mutant_files = [os.path.join(mutant_dir, f) for f in os.listdir(mutant_dir) if f.lower().endswith((".png", ".tif", ".jpg"))]
    wt_files     = [os.path.join(wt_dir, f) for f in os.listdir(wt_dir) if f.lower().endswith((".png", ".tif", ".jpg"))]

    results = []

    # analyze datasets
    for cls, file_list in [("mutant", mutant_files), ("wildtype", wt_files)]:
        for path in file_list:
            body_mask, cx, cy, spike_coords = extract_morphology(path)
            if body_mask is None or len(spike_coords) < 2:
                continue
            perim_coords = get_perimeter_coords(body_mask)

            # spike angles
            spike_angles = [np.arctan2(y - cy, x - cx) for (x, y) in spike_coords]
            results.extend([{"sample": os.path.basename(path), "class": cls, "metric": "angle", "value": a}
                            for a in spike_angles])

            # centroid→spike distances
            radial = [np.sqrt((x - cx)**2 + (y - cy)**2) for (x, y) in spike_coords]
            results.extend([{"sample": os.path.basename(path), "class": cls, "metric": "radial", "value": r}
                            for r in radial])

            # pairwise spike–spike distances
            pair_d = compute_spike_pair_distances(cx, cy, spike_coords, perim_coords)
            results.extend([{"sample": os.path.basename(path), "class": cls, "metric": "pairwise", "value": d}
                            for d in pair_d])

    # save raw data
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

    # class-level KDE plots
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw Fig. 1: Experiment and evaluation pipeline for CMS-LIME paper.
Output: pictures/fig1_experiment_pipeline.pdf (and .png)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Output dir: project root / pictures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_ROOT, "pictures")
os.makedirs(OUT_DIR, exist_ok=True)


def _arrow(ax, start, end, **kwargs):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", **kwargs))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Box style
    box_kw = dict(boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="black", linewidth=1.2)
    box_hl = dict(boxstyle="round,pad=0.25", facecolor="wheat", edgecolor="black", linewidth=1)

    # ---- Row 1: Data and model ----
    # CHB-MIT
    ax.text(1.5, 5.5, "CHB-MIT\nPre/Inter segments", ha="center", va="center", fontsize=9, bbox=box_kw)
    # Sliding window
    ax.text(3.5, 5.5, "Sliding window\n(5 s)", ha="center", va="center", fontsize=9, bbox=box_kw)
    # EEGNet
    ax.text(5.5, 5.5, "EEGNet\nper-window prob.", ha="center", va="center", fontsize=9, bbox=box_kw)

    _arrow(ax, (2.2, 5.5), (2.9, 5.5))
    _arrow(ax, (4.2, 5.5), (4.9, 5.5))

    # ---- Row 2: Three branches (Baseline / +Biomarker / +Full) ----
    ax.text(5.5, 4.2, "Window-level\npredictions", ha="center", va="center", fontsize=8)
    _arrow(ax, (5.5, 5.2), (5.5, 4.6))

    # Branch labels
    ax.text(4.0, 4.0, "Baseline\n(K-of-N only)", ha="center", va="center", fontsize=8, bbox=box_hl)
    ax.text(5.5, 4.0, "+ Biomarker\n(match + prob.)", ha="center", va="center", fontsize=8, bbox=box_hl)
    ax.text(7.0, 4.0, "+ Full\n(dynamic thresh.)", ha="center", va="center", fontsize=8, bbox=box_hl)
    _arrow(ax, (5.2, 4.35), (4.3, 4.1))
    _arrow(ax, (5.5, 4.35), (5.5, 4.15))
    _arrow(ax, (5.8, 4.35), (6.7, 4.1))

    # ---- Row 3: K-of-N and refractory ----
    ax.text(4.0, 3.0, "K-of-N\n+ refractory", ha="center", va="center", fontsize=9, bbox=box_kw)
    ax.text(5.5, 3.0, "K-of-N\n+ refractory", ha="center", va="center", fontsize=9, bbox=box_kw)
    ax.text(7.0, 3.0, "K-of-N\n+ refractory", ha="center", va="center", fontsize=9, bbox=box_kw)
    _arrow(ax, (4.0, 3.75), (4.0, 3.35))
    _arrow(ax, (5.5, 3.75), (5.5, 3.35))
    _arrow(ax, (7.0, 3.75), (7.0, 3.35))

    # ---- Row 4: Event matching and metrics ----
    ax.text(5.5, 1.8, "Match to seizure onsets\n→ Event sensitivity, FDR, time under alert", ha="center", va="center", fontsize=9, bbox=box_kw)
    _arrow(ax, (4.0, 2.65), (5.2, 2.0))
    _arrow(ax, (5.5, 2.65), (5.5, 2.15))
    _arrow(ax, (7.0, 2.65), (5.8, 2.0))

    # Settings label
    ax.text(1.5, 6.5, "Setting A: Patient-specific  |  Setting B: Leave-one-patient-out", ha="center", va="center", fontsize=9, style="italic")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        out_path = os.path.join(OUT_DIR, f"fig1_experiment_pipeline.{ext}")
        fig.savefig(out_path, bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

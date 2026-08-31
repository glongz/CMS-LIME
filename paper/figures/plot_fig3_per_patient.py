#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 3: Per-patient sensitivity — baseline vs full post-processing, by base model.
Small multiples: one scatter per model; diagonal = no change. Journal-friendly style.
Placeholder: list (model, patient_id, baseline_sens, full_sens). Replace with CSV.
Output: pictures/fig3_per_patient_sensitivity.pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_ROOT, "pictures")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["EEGNet", "ShallowConvNet", "DeepConvNet", "EEG-Inception", "Transformer"]

# Journal palette: one main point color (teal) + diagonal gray
COLOR_SCATTER = "#2a9d8f"   # teal (same as +Full in Fig. 2)
COLOR_DIAG = "#9e9e9e"
EDGE_COLOR = "white"
EDGE_WIDTH = 0.5

# Placeholder: (model, patient_id, baseline_sens%, full_sens%). Vary slightly by model.
def _make_placeholder_per_patient():
    base = [
        ("P01", 45, 62), ("P02", 72, 78), ("P03", 55, 70), ("P04", 88, 90),
        ("P05", 38, 58), ("P06", 65, 75), ("P07", 52, 68), ("P08", 80, 85),
        ("P09", 42, 60), ("P10", 70, 76),
    ]
    out = {}
    for i, m in enumerate(MODELS):
        # Slight offset per model so subplots differ a bit
        offset_b = -2 * i
        offset_f = 1 * i
        out[m] = [(pid, max(0, b + offset_b), min(100, f + offset_f)) for pid, b, f in base]
    return out

PER_PATIENT_BY_MODEL = _make_placeholder_per_patient()


def main():
    n_models = len(MODELS)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 4.5 * n_rows))
    axes = np.atleast_2d(axes)
    lims = [0, 100]

    for idx, model in enumerate(MODELS):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        data = PER_PATIENT_BY_MODEL[model]
        ids = [d[0] for d in data]
        baseline = np.array([d[1] for d in data])
        full = np.array([d[2] for d in data])

        ax.plot(lims, lims, color=COLOR_DIAG, ls="--", lw=1.2, alpha=0.8, label="No change")
        ax.scatter(baseline, full, s=42, c=COLOR_SCATTER, edgecolors=EDGE_COLOR, linewidths=EDGE_WIDTH, zorder=2)
        for i, pid in enumerate(ids):
            ax.annotate(pid, (baseline[i], full[i]), xytext=(4, 4), textcoords="offset points", fontsize=14, alpha=0.85)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("Baseline sensitivity (%)", fontsize=18)
        ax.set_ylabel("Full post-process. sensitivity (%)", fontsize=18)
        ax.set_title(model, fontsize=20, fontweight="medium")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=16)

    # Hide unused subplot
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig3_per_patient_sensitivity.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

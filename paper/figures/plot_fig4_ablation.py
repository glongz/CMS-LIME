#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 4: Ablation — 4 conditions × multiple base models.
Conditions: Baseline / +Unfiltered biomarkers / +Filtered biomarkers / +Full.
One row: sensitivity; one row: FDR. Journal-friendly sequential palette.
Output: pictures/fig4_ablation.pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_ROOT, "pictures")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["EEGNet", "ShallowConvNet", "DeepConvNet", "EEG-Inception", "Transformer"]
CONDITIONS = ["Baseline", "+ Unfiltered\nbiomarkers", "+ Filtered\nbiomarkers", "+ Full\npost-process."]

# Sequential palette: neutral → light teal → teal → dark teal (print-friendly)
COLORS_ABLATION = ["#7f7f7f", "#8cbcb4", "#3d8c84", "#1d6b5c"]

# Placeholder: per model, (sens[4], sens_std[4], fdr[4], fdr_std[4]). Replace with real ablation results.
def _placeholder_ablation():
    base_sens = [65, 68, 74, 78]
    base_std = [12, 11.5, 10.5, 10]
    base_fdr = [0.35, 0.38, 0.32, 0.30]
    fdr_std = [0.08, 0.09, 0.07, 0.06]
    out = {}
    for i, m in enumerate(MODELS):
        offset = 2 * i
        out[m] = (
            [s + offset for s in base_sens],
            base_std,
            [f + 0.02 * i for f in base_fdr],
            fdr_std,
        )
    return out

ABLATION_BY_MODEL = _placeholder_ablation()


def main():
    n_models = len(MODELS)
    n_cond = len(CONDITIONS)
    x = np.arange(n_cond)
    width = 0.6

    # 2 rows (Sensitivity, FDR) × 5 columns (one panel per model)
    fig, axes = plt.subplots(2, n_models, figsize=(12, 5), sharex="col")
    axes = np.atleast_2d(axes)
    # Row 0: sensitivity; Row 1: FDR
    for idx, model in enumerate(MODELS):
        sens, sens_std, fdr, fdr_std = ABLATION_BY_MODEL[model]
        ax_s = axes[0, idx]
        ax_f = axes[1, idx]
        ax_s.bar(x, sens, width, yerr=sens_std, color=COLORS_ABLATION, capsize=3, edgecolor="white", linewidth=0.6)
        ax_s.set_ylabel("Sensitivity (%)", fontsize=18)
        ax_s.set_ylim(0, 100)
        ax_s.set_xticks(x)
        ax_s.set_xticklabels(CONDITIONS, fontsize=14, rotation=12, ha="right")
        ax_s.set_title(model, fontsize=20, fontweight="medium")
        ax_s.grid(axis="y", alpha=0.25, linestyle="--")
        ax_s.spines["top"].set_visible(False)
        ax_s.spines["right"].set_visible(False)

        ax_f.bar(x, fdr, width, yerr=fdr_std, color=COLORS_ABLATION, capsize=3, edgecolor="white", linewidth=0.6)
        ax_f.set_ylabel("FDR (per h)", fontsize=18)
        ax_f.set_xticks(x)
        ax_f.set_xticklabels(CONDITIONS, fontsize=14, rotation=12, ha="right")
        ax_f.grid(axis="y", alpha=0.25, linestyle="--")
        ax_f.spines["top"].set_visible(False)
        ax_f.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=COLORS_ABLATION[i], edgecolor="white", label=CONDITIONS[i].replace("\n", " ")) for i in range(n_cond)]
    fig.legend(handles=legend_handles, labels=[c.replace("\n", " ") for c in CONDITIONS], loc="lower center", ncol=4, fontsize=16, frameon=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = os.path.join(OUT_DIR, "fig4_ablation.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

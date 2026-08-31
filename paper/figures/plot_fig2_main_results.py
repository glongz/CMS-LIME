#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 2: Main results — multiple base models × (Baseline / +Biomarker / +Full).
Two settings: Patient-specific and Cross-patient. Journal-friendly palette.
Data: placeholder mean ± std; replace with Table 1/2 or CSV.
Output: pictures/fig2_sensitivity_fdr_comparison.pdf
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(PROJECT_ROOT, "pictures")
os.makedirs(OUT_DIR, exist_ok=True)

# Base models to compare (extend or reduce as needed)
MODELS = ["EEGNet", "ShallowConvNet", "DeepConvNet", "EEG-Inception", "Transformer"]

# Journal-friendly, colorblind-safe: Baseline = neutral, +Biomarker = amber, +Full = teal
COLORS_METHOD = {
    "Baseline": "#6b6b6b",       # dark gray
    "+ Biomarker": "#d99a2e",    # amber
    "+ Full": "#2a9d8f",         # teal
}

# Placeholder: per (model, setting) → (sensitivity, sens_std, fdr, fdr_std) for 3 methods.
# Order per model: Baseline, +Biomarker, +Full. Replace with real data or CSV.
def _placeholder_sens_fdr(n_models, base_sens=62, step_sens=6, base_fdr=0.38, step_fdr=-0.02):
    sens = [base_sens + i * step_sens for i in range(3)]
    sens_std = [12 - i for i in range(3)]
    fdr = [base_fdr + i * step_fdr for i in range(3)]
    fdr_std = [0.08, 0.07, 0.06]
    return sens, sens_std, fdr, fdr_std

# Patient-specific: slightly higher sensitivity, lower FDR
PATIENT_SPECIFIC = {
    m: _placeholder_sens_fdr(3, base_sens=64 + i * 2, base_fdr=0.34 - i * 0.01)
    for i, m in enumerate(MODELS)
}
# Cross-patient: lower sensitivity, higher FDR
CROSS_PATIENT = {
    m: _placeholder_sens_fdr(3, base_sens=56 + i * 2, base_fdr=0.42 - i * 0.01)
    for i, m in enumerate(MODELS)
}


def main():
    n_models = len(MODELS)
    x = np.arange(n_models)
    width = 0.26  # 3 bars per model
    offsets = [-width, 0, width]
    method_order = ["Baseline", "+ Biomarker", "+ Full"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    # Row 0: Sensitivity (Patient-specific, Cross-patient)
    # Row 1: FDR (Patient-specific, Cross-patient)
    ax_sens_ps, ax_sens_cp = axes[0, 0], axes[0, 1]
    ax_fdr_ps, ax_fdr_cp = axes[1, 0], axes[1, 1]

    for j, method in enumerate(method_order):
        pos = x + offsets[j]
        c = COLORS_METHOD[method]
        sens_ps = [PATIENT_SPECIFIC[m][0][j] for m in MODELS]
        sens_ps_std = [PATIENT_SPECIFIC[m][1][j] for m in MODELS]
        sens_cp = [CROSS_PATIENT[m][0][j] for m in MODELS]
        sens_cp_std = [CROSS_PATIENT[m][1][j] for m in MODELS]
        fdr_ps = [PATIENT_SPECIFIC[m][2][j] for m in MODELS]
        fdr_ps_std = [PATIENT_SPECIFIC[m][3][j] for m in MODELS]
        fdr_cp = [CROSS_PATIENT[m][2][j] for m in MODELS]
        fdr_cp_std = [CROSS_PATIENT[m][3][j] for m in MODELS]

        ax_sens_ps.bar(pos, sens_ps, width * 0.88, yerr=sens_ps_std, label=method, color=c, capsize=2, edgecolor="white", linewidth=0.6)
        ax_sens_cp.bar(pos, sens_cp, width * 0.88, yerr=sens_cp_std, color=c, capsize=2, edgecolor="white", linewidth=0.6)
        ax_fdr_ps.bar(pos, fdr_ps, width * 0.88, yerr=fdr_ps_std, color=c, capsize=2, edgecolor="white", linewidth=0.6)
        ax_fdr_cp.bar(pos, fdr_cp, width * 0.88, yerr=fdr_cp_std, color=c, capsize=2, edgecolor="white", linewidth=0.6)

    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_sens_ps.set_ylabel("Event sensitivity (%)", fontsize=20)
    ax_sens_ps.set_ylim(0, 105)
    ax_sens_cp.set_ylim(0, 105)
    ax_fdr_ps.set_ylabel("FDR (per h)", fontsize=20)
    ax_fdr_ps.set_ylim(0, 0.65)
    ax_fdr_cp.set_ylim(0, 0.65)
    ax_sens_ps.set_xticks(x)
    ax_sens_ps.set_xticklabels(MODELS, rotation=22, ha="right", fontsize=18)
    ax_fdr_ps.set_xticks(x)
    ax_fdr_ps.set_xticklabels(MODELS, rotation=22, ha="right", fontsize=18)
    ax_sens_cp.set_xticks(x)
    ax_sens_cp.set_xticklabels(MODELS, rotation=22, ha="right", fontsize=18)
    ax_fdr_cp.set_xticks(x)
    ax_fdr_cp.set_xticklabels(MODELS, rotation=22, ha="right", fontsize=18)

    ax_sens_ps.set_title("Patient-specific", fontsize=20, fontweight="medium")
    ax_sens_cp.set_title("Cross-patient", fontsize=20, fontweight="medium")
    ax_sens_ps.legend(loc="upper right", frameon=True, fontsize=16)
    ax_sens_ps.grid(axis="y", alpha=0.25, linestyle="--")
    ax_sens_cp.grid(axis="y", alpha=0.25, linestyle="--")
    ax_fdr_ps.grid(axis="y", alpha=0.25, linestyle="--")
    ax_fdr_cp.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig2_sensitivity_fdr_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    main()

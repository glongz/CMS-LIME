#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-segment perturbation sanity: verify hybrid noise changes the input and model confidence.

Run from repo root (with this directory on path), e.g.:
  python 可解释归因示例/faithfulness_perturbation_debug.py --data_dir D:/.../chb01 --max_regions 3

Exits 0: prints L2, max |Δ|, P(preictal) before/after for union of first-k CMS masks.

For full deletion/insertion curves, moment-matching A/B, CSV + PNG, use instead:
  `faithfulness_deletion_curve_diagnostic.py`
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve().parent
_ROOT = _THIS.parent
for p in (str(_ROOT), str(_THIS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plot_cms_attribution_panels as pan  # noqa: E402
from chb_loocv_weights import apply_loocv_model_cli  # noqa: E402
from chb_paths import chb_patient_data_dir  # noqa: E402
from faithfulness_protocol_experiment import (  # noqa: E402
    conf_preictal,
    explain_cms_lime_regions,
    hybrid_noise_on_mask,
    union_masks,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=r"D:\public_data\CHBMIT\1_data_clean\chb01", help="Or set --chb_patient")
    p.add_argument("--chb_patient", type=int, default=0, help="1..99 -> .../1_data_clean/chb%%02d")
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--loocv_model_name", type=str, default="")
    p.add_argument("--loocv_select_loop", type=int, default=1)
    p.add_argument("--loocv_sync_data", action="store_true")
    p.add_argument("--loocv_weight_root", type=str, default="")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target_class", type=int, default=1)
    p.add_argument("--max_regions", type=int, default=3)
    p.add_argument("--hybrid_noise_scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if int(args.chb_patient or 0) > 0:
        pid = int(args.chb_patient)
        if not 1 <= pid <= 99:
            raise SystemExit("--chb_patient must be 1..99")
        args.data_dir = str(chb_patient_data_dir(pid))
    apply_loocv_model_cli(args)

    if not (args.data_dir or "").strip():
        raise SystemExit("Set --data_dir or --chb_patient")

    data_dir = Path(args.data_dir)
    npy_files = sorted(data_dir.glob("*.npy"), key=lambda x: x.name)
    if not npy_files:
        raise SystemExit("No .npy under data_dir")
    data = pan.load_long_npy(str(npy_files[0]))
    if data.ndim != 3 or data.shape[0] != 1:
        raise SystemExit("Need (1,C,T) npy")
    n_ch = int(data.shape[1])
    device = torch.device(args.device)
    mp, _ = pan.resolve_model_path((args.model_path or "").strip() or None)
    model = pan.load_model_robust(mp, n_chans=n_ch, n_classes=2, device=device)
    wrapper = pan.EEGModelWrapper(model, device, n_chans=n_ch)

    def predict_proba_np(x_bct: np.ndarray) -> np.ndarray:
        return wrapper.predict_proba(x_bct)

    from cms_lime_explainer import CMSLimeConfig, CMSLimeExplainer  # type: ignore

    X_list, y_list = [], []
    for w, s, e in pan.iter_windows(data, 1280, 20000):
        if len(X_list) >= 24:
            break
        w = np.array(w, dtype=np.float32, copy=True)
        pr = wrapper.predict_proba(w)[0]
        y_list.append(int(np.argmax(pr)))
        X_list.append(w[0])
    X_train = np.stack(X_list, axis=0)
    y_train = np.asarray(y_list, dtype=int)
    cfg = CMSLimeConfig(verbose=False, n_perturbations=200, n_features=10)
    ex = CMSLimeExplainer(cfg)
    ex.fit(X_train, y_train, wrapper)

    x0 = X_list[0]
    C, T = x0.shape
    rng = np.random.default_rng(args.seed)
    regions = explain_cms_lime_regions(
        ex, x0, int(args.target_class), int(args.max_regions), exp_out=None
    )
    if not regions:
        print("No CMS regions; cannot diagnose hybrid path.")
        return
    masks = [r.mask for r in regions]
    print("segment", npy_files[0].name, "C,T", C, T, "n_regions", len(masks))
    p0 = conf_preictal(x0, predict_proba_np, int(args.target_class))
    print(f"P(preictal) original: {p0:.6f}")
    hkw = float(args.hybrid_noise_scale)
    for k in (1, len(masks)):
        mk = union_masks(masks[:k])
        nnz = int(mk.sum())
        frac = float(mk.mean())
        print(f"\nk={k} mask nnz={nnz} frac={frac:.6f}")
        if nnz == 0:
            print("  empty mask: perturbation is a no-op by construction")
            continue
        x_pert = hybrid_noise_on_mask(x0, mk, rng, noise_scale=hkw)
        diff = x_pert.astype(np.float64) - x0.astype(np.float64)
        l2 = float(np.linalg.norm(diff.ravel()))
        mabs = float(np.max(np.abs(diff)))
        p1 = conf_preictal(x_pert, predict_proba_np, int(args.target_class))
        allclose = bool(np.allclose(x0, x_pert, rtol=0, atol=1e-12))
        print(f"  L2 ||X-X'||: {l2:.8g}")
        print(f"  max |Δ|:    {mabs:.8g}")
        print(f"  allclose:   {allclose}")
        print(f"  P(preictal): {p0:.6f} -> {p1:.6f} (delta {p1 - p0:+.6f})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Faithfulness protocol (Section: Explanation Faithfulness Protocol).

Computes per-segment × method:
  - Deletion AUC (lower better): trapz of conf while cumulatively masking top-ranked regions
  - Insertion AUC (higher better): trapz of conf while restoring from fully perturbed
  - Stability (higher better): mean Jaccard of top-K ranked ids over R repetitions
  - Localization IoU/Dice vs. weak proxy ROI (metadata: localization_mode=proxy)

Also compares baselines: lime_time, lime_segment, tf_lime, integrated_gradients, cms_lime.

Run from repo root:
  python 可解释归因示例/faithfulness_protocol_experiment.py --out_dir 可解释归因示例/faithfulness_out --fast

Progress: **tqdm** bar over segment×method jobs (disable with ``--no_progress`` or if tqdm is missing).

When auto-resolved weights mismatch your data channel count, add ``--simple_model`` to use the built-in
SimpleEEGModel (same fallback as ``load_model_robust`` when no valid checkpoint is found).

**CMS-LIME default** uses all three prior pools in ``explain_instance``; use ``--cms_unit_types microstate`` only
for a runtime-limited ablation and set the table label to ``CMS-LIME (microstate only)`` in the paper.
**Table outputs** are only as good as the protocol configuration and checkpoint match; re-run (smaller
``--max_segments`` e.g. 10) before treating numbers as final.

**Result tables** (under ``--out_dir`` after each run): ``faithfulness_table.md`` (Markdown 汇总表，含 CI),
``faithfulness_aggregate_table.csv`` (Excel 友好列式汇总), plus ``faithfulness_table_journal.tex`` and
``faithfulness_table.tex`` for LaTeX, and ``faithfulness_wilcoxon.json`` (paired Wilcoxon of CMS-LIME vs.\ baselines per segment).

**Minimal full-CMS, multi-patient pilot** (3 subjects × 2 windows each, 6 segments total), reduce runtime::

    python 可解释归因示例/faithfulness_protocol_experiment.py
      --out_dir 可解释归因示例/faithfulness_pilot_3p2w_fullcms
      --data_root D:/public_data/CHBMIT/1_data_clean
      --patient_cohorts chb01,chb05,chb10
      --max_npy_per_patient 1 --max_segments_per_patient 2
      --fast --cms_half_pert --cms_dpp_k 6
      --methods cms_lime
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import zlib
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[1]
_THIS_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sklearn.linear_model import Ridge

import plot_cms_attribution_panels as pan
from chb_loocv_weights import apply_loocv_model_cli
from chb_paths import CHB_SEGMENT_CLEAN_ROOT, chb_patient_data_dir


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RankedRegion:
    """One ranked interpretable region (channel-time mask)."""

    rid: str
    mask: np.ndarray  # (C, T) bool


@dataclass
class FaithfulnessRow:
    method: str
    stem: str
    cohort_patient: str
    win_start: int
    win_end: int
    target_class: int
    deletion_auc: float
    insertion_auc: float
    stability_jaccard: float
    localization_iou: float
    localization_dice: float
    n_regions: int
    topk_stability: int
    localization_mode: str
    notes: str = ""


def natural_keys(text: str):
    import re

    def atoi(t):
        return int(t) if t.isdigit() else t

    return [atoi(c) for c in re.split(r"(\d+)", text)]


def patient_cohort_id(file_stem: str) -> str:
    """
    Map a window's file stem to a study-level patient/cohort id.
    e.g. ``chb01_04`` -> ``chb01`` (CHB-MIT: one id per subject folder).
    """
    m = re.match(r"^(chb\d+)", str(file_stem), re.IGNORECASE)
    if m:
        return m.group(1).lower()
    if "_" in file_stem:
        return file_stem.split("_", 1)[0]
    return str(file_stem)


def trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal integral of y over x (x must be increasing)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float(y[0]) if len(y) else 0.0
    return float(np.trapz(y, x))


def channel_moment_match(orig: np.ndarray, pert: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Per-channel mean/std match (Method: stats-preserving post-processing)."""
    out = pert.astype(np.float64).copy()
    C, T = orig.shape
    for i in range(C):
        mu0, s0 = float(np.mean(orig[i])), float(np.std(orig[i]) + eps)
        mu1, s1 = float(np.mean(out[i])), float(np.std(out[i]) + eps)
        out[i] = (out[i] - mu1) / s1 * s0 + mu0
    return out.astype(np.float32)


def hybrid_noise_on_mask(
    base: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    beta_g: float = 0.7,
    noise_scale: float = 2.0,
    replacement: bool = True,
    preserve_stats: bool = False,
) -> np.ndarray:
    """Hybrid innovation on masked positions; replacement mode is stronger than additive."""
    x = base.astype(np.float64).copy()
    m = mask.astype(bool)
    C, T = x.shape
    for c in range(C):
        idx = np.where(m[c])[0]
        if idx.size == 0:
            continue
        seg = base[c, idx].astype(np.float64)
        sig = float(np.std(seg) + 1e-10)
        g = rng.normal(0.0, sig, size=idx.size)
        ar = np.zeros_like(g)
        if idx.size > 1:
            phi = float(np.corrcoef(seg[:-1], seg[1:])[0, 1]) if seg.size > 2 else 0.0
            if not np.isfinite(phi):
                phi = 0.0
            nu = rng.normal(0.0, sig, size=idx.size)
            ar[0] = nu[0]
            for t in range(1, idx.size):
                ar[t] = phi * ar[t - 1] + nu[t]
        eps = noise_scale * (beta_g * g + (1.0 - beta_g) * ar)
        if replacement:
            mu = float(np.mean(seg))
            x[c, idx] = mu + eps
        else:
            x[c, idx] = x[c, idx] + eps
    out = x.astype(np.float32)
    if preserve_stats:
        out = channel_moment_match(base, out).astype(np.float32)
    return out


def union_masks(masks: Sequence[np.ndarray]) -> np.ndarray:
    u = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        u |= m.astype(bool)
    return u


def conf_preictal(
    x_ct: np.ndarray,
    predict_proba: Callable[[np.ndarray], np.ndarray],
    target_class: int = 1,
) -> float:
    """x_ct: (C,T)"""
    w = x_ct.astype(np.float32)
    if w.ndim != 2:
        raise ValueError(x_ct.shape)
    p = predict_proba(w[np.newaxis, :, :])[0]
    return float(p[target_class])


def iou_dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> Tuple[float, float]:
    a = a.astype(bool).ravel()
    b = b.astype(bool).ravel()
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum()) + eps
    iou = inter / union
    dice = (2.0 * inter) / (float(a.sum()) + float(b.sum()) + eps)
    return iou, dice


def overlap_span(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def default_chb_segment_info_root() -> Path:
    """Default under ``segment_clean/30-1-240``; override with env ``CHB_SEGMENT_INFO_ROOT`` (see ``chb_paths``)."""
    return CHB_SEGMENT_CLEAN_ROOT


def resolve_segment_info_path(
    data_dir: Path, explicit: str = "", segment_info_root: Optional[Path] = None
) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    root = default_chb_segment_info_root() if segment_info_root is None else Path(segment_info_root)
    patient = data_dir.name
    cand = root / patient / "segment_info.json"
    return cand if cand.is_file() else None


def resolve_segment_info_path_for_patient_cohort(
    patient_cohort: str, segment_info_root: Path
) -> Optional[Path]:
    cand = Path(segment_info_root) / patient_cohort / "segment_info.json"
    return cand if cand.is_file() else None


def load_segment_info_merged(paths: Sequence[Optional[Path]]) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    out: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    for p in paths:
        if p is None:
            continue
        part = load_segment_info(p)
        out.update(part)
    return out


def collect_npy_for_patient_cohorts(
    data_root: Path,
    patient_cohorts: List[str],
    max_npy_per_patient: int,
) -> List[Path]:
    """List ``chb01_03.npy``-style files: per patient, sorted, truncated to ``max_npy_per_patient`` each."""
    out: List[Path] = []
    m = max(1, int(max_npy_per_patient))
    for p in patient_cohorts:
        pdir = data_root / p
        if not pdir.is_dir():
            print(f"[warn] patient data dir missing: {pdir}")
            continue
        got = sorted(pdir.glob(f"{p}_*.npy"), key=lambda fpath: natural_keys(fpath.name))[:m]
        if not got:
            print(f"[warn] no {p}_*.npy under {pdir}")
        out.extend(got)
    return out


def load_segment_info(path: Optional[Path]) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    """
    Returns:
      {file_stem: {"pre": [(s,e),...], "onset": [(s,e),...]}}
    """
    out: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
    if path is None or not path.is_file():
        return out
    try:
        arr = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(arr, list):
        return out
    for row in arr:
        if not isinstance(row, dict):
            continue
        f = str(row.get("File", "")).strip()
        lbl = str(row.get("Label", "")).strip().lower()
        span = row.get("Span", [])
        if not f or not isinstance(span, list) or len(span) != 2:
            continue
        stem = Path(f).stem
        s0, s1 = int(span[0]), int(span[1])
        if s1 <= s0:
            continue
        bucket = out.setdefault(stem, {"pre": [], "onset": []})
        if re.match(r"^pre", lbl):
            bucket["pre"].append((s0, s1))
        elif re.match(r"^onset", lbl):
            bucket["onset"].append((s0, s1))
    for v in out.values():
        v["pre"] = sorted(v["pre"])
        v["onset"] = sorted(v["onset"])
    return out


# ---------------------------------------------------------------------------
# Baseline explainers → RankedRegion list
# ---------------------------------------------------------------------------


def explain_lime_time(
    x: np.ndarray,
    predict_proba: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    n_samples: int,
    subsample_step: int,
    point_radius: int = 0,
    target_class: int = 1,
) -> List[RankedRegion]:
    """LIME on subsampled time indices (sparse binary presence)."""
    C, T = x.shape
    t_idx = list(range(0, T, max(1, subsample_step)))
    d = len(t_idx)
    if d == 0:
        return []
    Z = []
    y = []
    x0 = x.astype(np.float32)
    c0 = conf_preictal(x0, predict_proba, target_class)
    for _ in range(n_samples):
        z = rng.integers(0, 2, size=d).astype(np.float32)
        mcombo = np.zeros((C, T), dtype=bool)
        for j, ti in enumerate(t_idx):
            if z[j] < 0.5:
                lo, hi = max(0, int(ti) - int(point_radius)), min(T, int(ti) + int(point_radius) + 1)
                mcombo[:, lo:hi] = True
        xp = hybrid_noise_on_mask(x0, mcombo, rng) if mcombo.any() else x0.copy()
        Z.append(z)
        y.append(conf_preictal(xp, predict_proba, target_class) - c0)
    Zm = np.stack(Z, axis=0)
    yv = np.asarray(y)
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(Zm, yv)
    coef = np.abs(ridge.coef_.ravel())
    order = np.argsort(-coef)
    regions: List[RankedRegion] = []
    for rank, j in enumerate(order):
        ti = t_idx[int(j)]
        m = np.zeros((C, T), dtype=bool)
        lo, hi = max(0, int(ti) - int(point_radius)), min(T, int(ti) + int(point_radius) + 1)
        m[:, lo:hi] = True
        regions.append(RankedRegion(rid=f"t{ti}", mask=m))
    return regions


def explain_lime_segment(
    x: np.ndarray,
    predict_proba: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    n_samples: int,
    seg_len: int,
    target_class: int = 1,
) -> List[RankedRegion]:
    C, T = x.shape
    starts = list(range(0, max(1, T - seg_len + 1), seg_len))
    d = len(starts)
    if d == 0:
        return []
    Z, y = [], []
    x0 = x.astype(np.float32)
    c0 = conf_preictal(x0, predict_proba, target_class)
    for _ in range(n_samples):
        z = rng.integers(0, 2, size=d).astype(np.float32)
        mcombo = np.zeros((C, T), dtype=bool)
        for j, s in enumerate(starts):
            if z[j] < 0.5:
                mcombo[:, s : s + seg_len] = True
        xp = hybrid_noise_on_mask(x0, mcombo, rng) if mcombo.any() else x0.copy()
        Z.append(z)
        y.append(conf_preictal(xp, predict_proba, target_class) - c0)
    Zm = np.stack(Z, axis=0)
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(Zm, np.asarray(y))
    coef = np.abs(ridge.coef_.ravel())
    order = np.argsort(-coef)
    regions = []
    for j in order:
        s = starts[int(j)]
        m = np.zeros((C, T), dtype=bool)
        m[:, s : s + seg_len] = True
        regions.append(RankedRegion(rid=f"seg{s}", mask=m))
    return regions


def band_power_vector(x: np.ndarray, fs: float, n_bands: int = 5) -> np.ndarray:
    """Very coarse band power per channel (mean over time of bandpassed energy)."""
    C, T = x.shape
    out = []
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, min(45.0, fs / 2 - 1))]
    from scipy.signal import butter, filtfilt

    for c in range(C):
        sig = x[c].astype(np.float64)
        for lo, hi in bands[:n_bands]:
            if hi >= fs / 2:
                continue
            b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
            try:
                f = filtfilt(b, a, sig)
            except Exception:
                f = sig
            out.append(float(np.mean(f**2)))
    return np.asarray(out, dtype=np.float64)


def explain_tf_lime(
    x: np.ndarray,
    predict_proba: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    n_samples: int,
    fs: float,
    target_class: int = 1,
) -> List[RankedRegion]:
    """TF-LIME: linear surrogate on band-power vector; map dims back to full-time mask uniformly."""
    C, T = x.shape
    phi0 = band_power_vector(x, fs)
    d = int(phi0.size)
    Z, y = [], []
    x0 = x.astype(np.float32)
    c0 = conf_preictal(x0, predict_proba, target_class)
    for _ in range(n_samples):
        z = rng.normal(0, 1, size=d)
        # multiplicative perturbation in feature space -> approximate by scaling EEG per band chunk
        xp = x0.copy()
        # crude map: distribute each feature dim across time uniformly
        for j in range(d):
            scale = float(1.0 + 0.15 * z[j])
            c_idx = j % C
            xp[c_idx, :] = xp[c_idx, :] * scale
        xp = channel_moment_match(x0, xp)
        Z.append(z)
        y.append(conf_preictal(xp.astype(np.float32), predict_proba, target_class) - c0)
    ridge = Ridge(alpha=1.0)
    ridge.fit(np.stack(Z), np.asarray(y))
    coef = np.abs(ridge.coef_.ravel())
    order = np.argsort(-coef)
    regions: List[RankedRegion] = []
    bands_n = min(5, d // max(1, C))
    for j in order[: min(d, 32)]:
        c_idx = int(j) % C
        m = np.zeros((C, T), dtype=bool)
        m[c_idx, :] = True
        regions.append(RankedRegion(rid=f"tf{j}", mask=m))
    return regions


def explain_integrated_gradients(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    n_steps: int = 20,
    point_radius: int = 0,
    target_class: int = 1,
) -> List[RankedRegion]:
    """Integrated gradients (baseline 0): sum_k grad(f(alpha_k x)) * (x / n_steps)."""
    C, T = x.shape
    x_np = x.astype(np.float32)
    x0 = torch.from_numpy(x_np).to(device)
    accum = torch.zeros((C, T), device=device)
    model.eval()
    for k in range(1, n_steps + 1):
        alpha = float(k) / float(n_steps)
        xt = (alpha * x0).detach().clone().requires_grad_(True)
        inp = xt.unsqueeze(0)
        out = model(inp)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        if out.shape[-1] > 1 and (out.min() < 0 or out.max() > 1):
            prob = torch.softmax(out, dim=-1)[0, target_class]
        else:
            prob = out[0, target_class]
        g = torch.autograd.grad(prob, xt, retain_graph=False, create_graph=False)[0]
        accum = accum + g * (x0 / float(n_steps))
    ig = accum.abs().detach().cpu().numpy()
    score_t = ig.sum(axis=0)
    order = np.argsort(-score_t)
    regions: List[RankedRegion] = []
    for ti in order[: min(T, 64)]:
        m = np.zeros((C, T), dtype=bool)
        lo, hi = max(0, int(ti) - int(point_radius)), min(T, int(ti) + int(point_radius) + 1)
        m[:, lo:hi] = True
        regions.append(RankedRegion(rid=f"ig_t{int(ti)}", mask=m))
    return regions


# ---------------------------------------------------------------------------
# CMS-LIME adapter
# ---------------------------------------------------------------------------


def _cms_region_stable_id(unit: Dict[str, Any], C: int, T: int) -> str:
    """
    Identifier for a CMS region that does **not** depend on rank in the importance list
    (rank-dependent ids spuriously deflate stability Jaccard when only ordering jiggles).
    """
    u = unit
    if u.get("primitive_id") is not None and str(u.get("primitive_id")).strip():
        return str(u["primitive_id"])
    tr = u.get("time_range")
    if tr is not None and len(tr) == 2:
        t0, t1 = int(tr[0]), int(tr[1])
    else:
        t0 = int(u.get("start_time", 0))
        t1 = int(u.get("end_time", T - 1))
    t0, t1 = max(0, t0), max(0, t1)
    typ = str(u.get("type", "?"))
    if typ == "shapelet":
        return f"shapelet|sid={u.get('shapelet_id', '')}|t{t0}_{t1}"
    if typ == "timefreq":
        chs0 = u.get("channels", []) or [0]
        try:
            c0 = min(int(c) for c in chs0)
        except (TypeError, ValueError):
            c0 = 0
        return (
            f"timefreq|{u.get('freq_band', '')}|t{u.get('time_start', t0)}_{u.get('time_end', t1)}|c{c0}"
        )
    if typ == "microstate":
        st = u.get("state", u.get("microstate", ""))
        return f"microstate|s{st}|t{t0}_{t1}"
    return f"{typ}|t{t0}_{t1}"


def cms_unit_dict_to_mask(unit: Dict[str, Any], C: int, T: int) -> np.ndarray:
    m = np.zeros((C, T), dtype=bool)
    chs = unit.get("channels", list(range(C)))
    chs = [int(c) for c in chs if 0 <= int(c) < C]
    if "time_range" in unit:
        t0, t1 = int(unit["time_range"][0]), int(unit["time_range"][1])
    else:
        t0, t1 = int(unit.get("start_time", 0)), int(unit.get("end_time", T - 1))
    t0 = max(0, min(t0, T - 1))
    t1 = max(0, min(t1, T - 1))
    if t1 < t0:
        t0, t1 = t1, t0
    for c in chs:
        m[c, t0 : t1 + 1] = True
    return m


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(z) for z in x]
    if isinstance(x, (str, bool)) or x is None:
        return x
    return str(x)


def _faithfulness_deletion_perturb_sanity(
    x: np.ndarray,
    regions: List[RankedRegion],
    noise_scale: float = 2.0,
) -> Dict[str, Any]:
    """
    Same construction as ``deletion_insertion_curves`` (union of first-k masks, then
    ``hybrid_noise_on_mask``). If relative RMS on the masked voxels is ~0, the model
    can stay at conf ~1 even when Del.\,AUC looks pathological.
    """
    if not regions:
        return {"empty_regions": True}
    K = len(regions)
    masks = [r.mask for r in regions]
    out: Dict[str, Any] = {}
    for k in sorted(set([1, K])):
        if k > K:
            continue
        mk = union_masks(masks[:k])
        if not np.any(mk):
            out[f"step_k{k}_mask_frac"] = 0.0
            out[f"step_k{k}_relative_rms_on_mask"] = None
            out[f"step_k{k}_empty_mask"] = True
            continue
        g = np.random.default_rng(9000 + k)
        x_pert = hybrid_noise_on_mask(x, mk, g, noise_scale=float(noise_scale))
        diff = (x_pert.astype(np.float64) - x.astype(np.float64))[mk]
        rms = float(np.sqrt(np.mean(diff**2)))
        ref = float(np.sqrt(np.mean(x.astype(np.float64)[mk] ** 2)) + 1e-10)
        out[f"step_k{k}_mask_frac"] = float(mk.mean())
        out[f"step_k{k}_relative_rms_on_mask"] = rms / ref
    return out


def write_cms_diagnostic(
    path: Path,
    x_ct: np.ndarray,
    regions: List[RankedRegion],
    y_del: np.ndarray,
    explainer: Any,
    explanation_snapshot: Optional[Dict[str, Any]],
    cms_ut: List[str],
    hybrid_noise_scale: float = 2.0,
) -> None:
    """Debug microstate-only / CMS path: masks, regression weights, deletion conf curve."""
    path.parent.mkdir(parents=True, exist_ok=True)
    C, T = x_ct.shape
    rec: Dict[str, Any] = {
        "unit_types": cms_ut,
        "deletion_conf_curve": [float(v) for v in np.asarray(y_del).ravel()],
        "max_min_del": float(np.max(y_del) - np.min(y_del)) if y_del.size else 0.0,
        "deletion_perturb_vs_protocol": _faithfulness_deletion_perturb_sanity(
            x_ct.astype(np.float32), regions, noise_scale=hybrid_noise_scale
        ),
        "final_regions": [],
    }
    for r in regions:
        m = r.mask
        rec["final_regions"].append(
            {
                "rid": r.rid,
                "mask_fraction": float(m.mean()),
                "mask_nnz": int(m.sum()),
            }
        )
    if explanation_snapshot:
        ms = (explanation_snapshot.get("unit_explanations") or {}).get("microstate") or {}
        mus = ms.get("units") or []
        rec["microstate_units"] = []
        for u in mus:
            mk = cms_unit_dict_to_mask(u, C, T)
            rec["microstate_units"].append(
                {
                    "primitive_id": u.get("primitive_id"),
                    "type": u.get("type"),
                    "mask_fraction": float(mk.mean()),
                    "mask_nnz": int(mk.sum()),
                }
            )
    lr = getattr(explainer, "local_regressor", None)
    if lr is not None:
        reg = getattr(lr, "regressor", None)
        if reg is not None and hasattr(reg, "get_coefficients"):
            try:
                coef = reg.get_coefficients()
                rec["surrogate_coefficients"] = _jsonable(np.asarray(coef).ravel())
                rec["surrogate_nonzero"] = int(np.sum(np.abs(np.asarray(coef).ravel()) > 1e-8))
            except Exception as e:
                rec["surrogate_coefficients_error"] = str(e)
        sel = getattr(lr, "selected_features", None)
        if sel is not None:
            rec["dpp_selected_feature_indices"] = _jsonable(sel)
    path.write_text(json.dumps(_jsonable(rec), indent=2, ensure_ascii=False), encoding="utf-8")


def explain_cms_lime_regions(
    explainer: Any,
    x_ct: np.ndarray,
    target_class: int,
    max_regions: int,
    unit_types: Optional[List[str]] = None,
    exp_out: Optional[Dict[str, Any]] = None,
) -> List[RankedRegion]:
    ut = unit_types if unit_types else ["microstate", "shapelet", "timefreq"]
    exp = explainer.explain_instance(x_ct, target_class=target_class, unit_types=ut)
    if exp_out is not None:
        exp_out["explanation"] = exp
    units = exp.get("selected_units") or []
    imps = exp.get("selected_importances") or []
    pairs = sorted(zip(units, imps), key=lambda z: float(z[1]), reverse=True)
    if len(pairs) < min(2, max_regions):
        # Fallback: some CMS pipelines collapse selected_units to 1 item.
        # Expand from per-type unit_explanations to avoid stability degeneration.
        uexp = exp.get("unit_explanations") or {}
        all_pairs: List[Tuple[Dict[str, Any], float]] = []
        for typ in ("microstate", "shapelet", "timefreq"):
            blk = uexp.get(typ) or {}
            us = blk.get("units") or []
            ws = blk.get("importances") or [1.0] * len(us)
            for u, w in zip(us, ws):
                u2 = dict(u)
                if "primitive_id" not in u2:
                    u2["primitive_id"] = f"{typ}"
                all_pairs.append((u2, float(abs(w))))
        if all_pairs:
            pairs = sorted(all_pairs, key=lambda z: z[1], reverse=True)
    C, T = x_ct.shape
    out: List[RankedRegion] = []
    for i, (u, imp) in enumerate(pairs[:max_regions]):
        uid = _cms_region_stable_id(u, C, T)
        if any(r.rid == uid for r in out):
            uid = f"{uid}::alt{i}"
        out.append(RankedRegion(rid=uid, mask=cms_unit_dict_to_mask(u, C, T)))
    return out


# ---------------------------------------------------------------------------
# Faithfulness metrics
# ---------------------------------------------------------------------------


def deletion_insertion_curves(
    x: np.ndarray,
    regions: List[RankedRegion],
    predict_proba: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    target_class: int = 1,
    insertion_moment_match: bool = True,
    hybrid_noise_scale: float = 2.0,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deletion: start from original, progressively perturb union of top-k masks.
    Insertion: start from fully perturbed, progressively restore original in top-k order.

    AUCs are over normalized abscissa k / K in [0,1].

    ``insertion_moment_match``: if False, skip per-channel moment matching on the
    partially-restored path (diagnostic: aggressive matching can erase perturbation).
    """
    x = x.astype(np.float32)
    K = len(regions)
    if K == 0:
        return 0.0, 0.0, np.array([0.0]), np.array([0.0]), np.array([0.0])
    masks = [r.mask for r in regions]
    hkw = float(hybrid_noise_scale)

    # Deletion trajectory
    xs = [0.0]
    ys = [conf_preictal(x, predict_proba, target_class)]
    cur = x.copy()
    for k in range(1, K + 1):
        mk = union_masks(masks[:k])
        cur = hybrid_noise_on_mask(x, mk, rng, noise_scale=hkw)
        xs.append(k / float(K))
        ys.append(conf_preictal(cur, predict_proba, target_class))
    del_auc = trapz_auc(np.asarray(xs), np.asarray(ys))

    # Insertion: full perturb then restore
    full_m = union_masks(masks)
    x_pert = hybrid_noise_on_mask(x, full_m, rng, noise_scale=hkw)
    xs2 = [0.0]
    ys2 = [conf_preictal(x_pert, predict_proba, target_class)]
    cur = x_pert.copy()
    # restore by copying original values back for each region cumulatively
    for k in range(1, K + 1):
        mk = union_masks(masks[:k])
        cur = cur.copy()
        cur[mk] = x[mk]
        if insertion_moment_match:
            cur = channel_moment_match(x, cur)
        cur = cur.astype(np.float32)
        xs2.append(k / float(K))
        ys2.append(conf_preictal(cur, predict_proba, target_class))
    ins_auc = trapz_auc(np.asarray(xs2), np.asarray(ys2))
    return del_auc, ins_auc, np.asarray(xs), np.asarray(ys), np.asarray(ys2)


def stability_jaccard(
    rank_fn: Callable[[np.random.Generator], List[str]],
    rng_master: np.random.Generator,
    R: int,
    topk: int,
) -> float:
    if topk < 2:
        return float("nan")
    sets: List[set] = []
    for r in range(R):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        ids = rank_fn(rng)[:topk]
        sets.append(set(ids))
    if len(sets) < 2:
        return float("nan")
    acc = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            inter = len(a & b)
            union = len(a | b) + 1e-10
            acc.append(inter / union)
    return float(np.mean(acc)) if acc else float("nan")


def proxy_roi_mask(
    mode: str,
    C: int,
    T: int,
    win_start: int,
    win_end: int,
    file_meta: Optional[Dict[str, List[Tuple[int, int]]]],
    onset_before_sec: float,
    fs: float,
) -> np.ndarray:
    """Weak proxy ROI G from metadata spans (fallback to center window)."""
    m = np.zeros((C, T), dtype=bool)
    if file_meta is None:
        t0, t1 = int(0.3 * T), int(0.7 * T)
        m[:, t0:t1] = True
        return m

    if mode == "preictal_block":
        for s0, s1 in file_meta.get("pre", []):
            lo = max(win_start, s0)
            hi = min(win_end, s1)
            if hi > lo:
                m[:, lo - win_start : hi - win_start] = True
    elif mode == "onset_window":
        before = int(max(1.0, onset_before_sec) * fs)
        for os0, _ in file_meta.get("onset", []):
            s0, s1 = os0 - before, os0
            lo = max(win_start, s0)
            hi = min(win_end, s1)
            if hi > lo:
                m[:, lo - win_start : hi - win_start] = True
    if not m.any():
        t0, t1 = int(0.3 * T), int(0.7 * T)
        m[:, t0:t1] = True
    return m


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------


def window_passes_sampling_mode(
    sampling_mode: str,
    file_meta: Optional[Dict[str, List[Tuple[int, int]]]],
    start: int,
    end: int,
    args: argparse.Namespace,
) -> bool:
    """
    In ``preictal`` / ``onset_near`` mode, require **non-empty** metadata for that file stem.
    Otherwise missing ``segment_info`` would incorrectly admit *all* sliding windows.
    """
    if sampling_mode == "all":
        return True
    if sampling_mode == "preictal":
        if not file_meta or not file_meta.get("pre"):
            return False
        return any(overlap_span(start, end, s0, s1) for s0, s1 in file_meta["pre"])
    if sampling_mode == "onset_near":
        if not file_meta or not file_meta.get("onset"):
            return False
        before = int(max(1.0, float(args.onset_before_sec)) * float(args.fs))
        return any(overlap_span(start, end, o0 - before, o0) for o0, _ in file_meta["onset"])
    return True


def gather_faithfulness_windows(
    npy_files: List[Path],
    seg_info: Dict[str, Dict[str, List[Tuple[int, int]]]],
    args: argparse.Namespace,
) -> List[Tuple[np.ndarray, int, int, str]]:
    """
    Build (x, start, end, file_stem) window list. If max_segments_per_patient > 0, pool
    all windows per subject cohort and subsample; otherwise use max_segments per .npy file.
    """
    from collections import defaultdict

    wlen, stride = 1280, int(args.stride)
    mspp = int(getattr(args, "max_segments_per_patient", 0) or 0)
    if mspp > 0:
        by_cohort: Dict[str, List[Tuple[np.ndarray, int, int, str]]] = defaultdict(list)
        for fp in npy_files:
            stem = fp.stem
            file_meta = seg_info.get(stem)
            data = pan.load_long_npy(str(fp))
            for w, start, end in pan.iter_windows(data, wlen, stride):
                if not window_passes_sampling_mode(args.sampling_mode, file_meta, start, end, args):
                    continue
                ch = patient_cohort_id(stem)
                by_cohort[ch].append((np.array(w[0], dtype=np.float32, copy=True), start, end, stem))
        jobs: List[Tuple[np.ndarray, int, int, str]] = []
        for _coh, cands in sorted(by_cohort.items(), key=lambda z: z[0]):
            if not cands:
                continue
            if len(cands) > mspp:
                idxs = np.linspace(0, len(cands) - 1, num=mspp, dtype=int)
                cands = [cands[int(i)] for i in idxs]
            jobs.extend(cands)
        return jobs
    out: List[Tuple[np.ndarray, int, int, str]] = []
    for fp in npy_files:
        stem = fp.stem
        file_meta = seg_info.get(stem)
        data = pan.load_long_npy(str(fp))
        cand: List[Tuple[np.ndarray, int, int, str]] = []
        for w, start, end in pan.iter_windows(data, wlen, stride):
            if not window_passes_sampling_mode(args.sampling_mode, file_meta, start, end, args):
                continue
            cand.append((np.array(w[0], dtype=np.float32, copy=True), start, end, stem))
        if not cand:
            continue
        if len(cand) > int(args.max_segments):
            idxs = np.linspace(0, len(cand) - 1, num=int(args.max_segments), dtype=int)
            picked = [cand[int(i)] for i in idxs]
        else:
            picked = cand
        out.extend(picked)
    return out


METHODS = [
    "cms_lime",
    "cms_lime_random",
    "lime_time",
    "lime_segment",
    "tf_lime",
    "integrated_gradients",
]
# Jaccard stability uses region string IDs; full-channel or tied IDs → trivial 1.0, not cross-method comparable.
STABILITY_ID_PROTOCOL_NA = frozenset({"tf_lime", "integrated_gradients"})

METHOD_DISPLAY = {
    "cms_lime": "CMS-LIME",
    "cms_lime_random": "CMS (random del. order)",
    "lime_time": "Time-point LIME",
    "lime_segment": "LIME-Segment",
    "tf_lime": "TF-LIME",
    "integrated_gradients": "Int. Gradients",
}


def bootstrap_mean_ci(vals: np.ndarray, B: int, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap 95\\% CI for the sample mean."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2 or B < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = int(vals.size)
    means = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = float(np.mean(vals[idx]))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def cohort_stats_from_rows(rows: List[FaithfulnessRow]) -> Dict[str, Any]:
    """Unique segments, files, and patient ids (from file stem) for table footnotes."""
    segs = {(r.stem, r.win_start, r.win_end) for r in rows}
    stems = {r.stem for r in rows}
    patients = {patient_cohort_id(r.stem) for r in rows}
    return {
        "n_segments": len(segs),
        "n_files": len(stems),
        "n_patients": len(patients),
        "patient_ids": sorted(patients, key=natural_keys),
        "file_stems": sorted(stems, key=natural_keys),
    }


def paired_wilcoxon_cms_vs_baselines(
    rows: List[FaithfulnessRow],
    reference: str = "cms_lime",
    baselines: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank on segment-aligned pairs (same stem, start, end).
    Two-sided p-values on differences d = v_ref - v_other (per segment).
    """
    from collections import defaultdict

    if baselines is None:
        baselines = [m for m in METHODS if m not in (reference, "cms_lime_random")]

    by_seg: Dict[Tuple[Any, ...], Dict[str, FaithfulnessRow]] = defaultdict(dict)
    for r in rows:
        if r.method not in METHODS:
            continue
        by_seg[(r.stem, r.win_start, r.win_end)][r.method] = r

    metrics = [
        ("deletion_auc", "del"),
        ("insertion_auc", "ins"),
        ("stability_jaccard", "stab"),
        ("localization_iou", "iou"),
        ("localization_dice", "dice"),
    ]

    out: Dict[str, Any] = {
        "reference": reference,
        "alternative": "two-sided",
        "difference": "ref_minus_other",
        "pairs": {},
    }

    try:
        from scipy.stats import wilcoxon
    except Exception as e:  # pragma: no cover
        out["error"] = f"scipy.stats.wilcoxon unavailable: {e}"
        return out

    import inspect

    _w_kw = {}
    if "method" in inspect.signature(wilcoxon).parameters:
        _w_kw["method"] = "auto"

    for bline in baselines:
        if bline == reference:
            continue
        out["pairs"][bline] = {}
        for key, short in metrics:
            if key == "stability_jaccard" and (
                bline in STABILITY_ID_PROTOCOL_NA or reference in STABILITY_ID_PROTOCOL_NA
            ):
                out["pairs"][bline][short] = {
                    "n_paired": 0,
                    "p_value_two_sided": float("nan"),
                    "note": "stability_id_not_comparable",
                }
                continue

            diffs: List[float] = []
            for _k, dmap in by_seg.items():
                if reference not in dmap or bline not in dmap:
                    continue
                vr = float(getattr(dmap[reference], key))
                vo = float(getattr(dmap[bline], key))
                if not (np.isfinite(vr) and np.isfinite(vo)):
                    continue
                diffs.append(vr - vo)

            d = np.asarray(diffs, dtype=float)
            n = int(d.size)
            if n < 3:
                out["pairs"][bline][short] = {
                    "n_paired": n,
                    "p_value_two_sided": float("nan"),
                    "note": "too_few_pairs",
                }
                continue
            if np.allclose(d, 0.0):
                out["pairs"][bline][short] = {
                    "n_paired": n,
                    "p_value_two_sided": 1.0,
                    "median_diff_ref_minus_other": 0.0,
                    "note": "all_differences_zero",
                }
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided", **_w_kw)
                p = float(res.pvalue)
            except Exception as ex:
                out["pairs"][bline][short] = {
                    "n_paired": n,
                    "p_value_two_sided": float("nan"),
                    "error": str(ex),
                }
                continue

            out["pairs"][bline][short] = {
                "n_paired": n,
                "p_value_two_sided": p,
                "median_diff_ref_minus_other": float(np.median(d)),
            }
    return out


def _wilcoxon_notes_tex(w: Optional[Dict[str, Any]], cohort: Dict[str, Any]) -> str:
    """Short LaTeX for table footnote: N, P, F, pilot warning, and vs LIME-Segment p-values."""
    n0 = cohort.get("n_segments", 0)
    p0 = cohort.get("n_patients", 0)
    f0 = cohort.get("n_files", 0)
    parts: List[str] = [
        rf"\textbf{{Cohort:}} $N={n0}$ segments; $P={p0}$ patient id(s); $F={f0}$ recording file(s). "
    ]
    if int(p0) <= 1 and int(f0) <= 1:
        parts.append(
            r"\textbf{Pilot (single-patient and/or single recording):} not treated as population inference; "
            r"see Sec.~\ref{sec:experiments} for multi-patient results. "
        )
    elif int(p0) <= 1:
        parts.append(
            r"\textbf{Single-patient cohort:} interpret with care; see multi-patient table in Sec.~\ref{sec:experiments}. "
        )

    if not w or w.get("error"):
        return "".join(parts)
    ls = (w.get("pairs") or {}).get("lime_segment")
    if not ls:
        return "".join(parts)

    def _fmt_p(key: str) -> str:
        cell = ls.get(key) or {}
        pv = cell.get("p_value_two_sided")
        if pv is None or not np.isfinite(float(pv)):
            return r"\text{---}"
        pvf = float(pv)
        if pvf < 1e-4:
            return r"<10^{-4}"
        return f"{pvf:.4f}"

    _n_ls = (ls.get("del") or {}).get("n_paired", 0)
    parts.append(
        r"\textbf{Paired Wilcoxon (two-sided, CMS-LIME vs LIME-Segment):} "
        rf"Del.\,AUC $p={_fmt_p('del')}$; Ins.\,AUC $p={_fmt_p('ins')}$; Stab.\ $p={_fmt_p('stab')}$; "
        rf"IoU $p={_fmt_p('iou')}$; Dice $p={_fmt_p('dice')}$ "
        rf"(paired segment count for Del.: $n={int(_n_ls)}$). "
    )
    return "".join(parts)


def run_for_segment(
    method: str,
    x: np.ndarray,
    stem: str,
    start: int,
    end: int,
    predict_proba_np: Callable[[np.ndarray], np.ndarray],
    torch_model: Optional[torch.nn.Module],
    device: torch.device,
    cms_explainer: Optional[Any],
    file_meta: Optional[Dict[str, List[Tuple[int, int]]]],
    args: argparse.Namespace,
    rng: np.random.Generator,
    out_dir: Optional[Path] = None,
) -> Tuple[FaithfulnessRow, List[Dict[str, Any]]]:
    C, T = x.shape
    tc = int(args.target_class)
    cohort = patient_cohort_id(stem)

    cms_ut = [s.strip() for s in str(getattr(args, "cms_unit_types", "microstate,shapelet,timefreq")).split(",") if s.strip()]

    def rank_ids_for_stability(rng_inner: np.random.Generator) -> List[str]:
        if method in ("cms_lime", "cms_lime_random") and cms_explainer is not None:
            x_stab = x
            if float(args.cms_stability_jitter) > 0:
                sig = float(np.std(x) + 1e-10)
                x_stab = x + rng_inner.normal(0.0, float(args.cms_stability_jitter) * sig, size=x.shape).astype(np.float32)
            regs = list(
                explain_cms_lime_regions(
                    cms_explainer, x_stab, tc, args.max_regions, unit_types=cms_ut
                )
            )
            # Align with Deletion/Insertion: cms_lime_random shuffles explainer order (sanity baseline).
            if method == "cms_lime_random" and regs:
                rng_inner.shuffle(regs)
        elif method == "lime_time":
            regs = explain_lime_time(
                x, predict_proba_np, rng_inner, args.n_pert_samples, args.time_subsample, args.point_radius, tc
            )
        elif method == "lime_segment":
            regs = explain_lime_segment(x, predict_proba_np, rng_inner, args.n_pert_samples, args.seg_len, tc)
        elif method == "tf_lime":
            regs = explain_tf_lime(x, predict_proba_np, rng_inner, args.n_pert_samples, float(args.fs), tc)
        elif method == "integrated_gradients" and torch_model is not None:
            regs = explain_integrated_gradients(
                torch_model, x, device, n_steps=args.ig_steps, point_radius=args.point_radius, target_class=tc
            )
        else:
            regs = []
        return [r.rid for r in regs]

    if method in ("cms_lime", "cms_lime_random") and cms_explainer is None:
        return FaithfulnessRow(
            method=method,
            stem=stem,
            cohort_patient=cohort,
            win_start=start,
            win_end=end,
            target_class=tc,
            deletion_auc=np.nan,
            insertion_auc=np.nan,
            stability_jaccard=np.nan,
            localization_iou=np.nan,
            localization_dice=np.nan,
            n_regions=0,
            topk_stability=args.topk_stability,
            localization_mode=f"proxy:{args.proxy_mode}",
            notes="cms_lime_unavailable",
        ), []

    if method == "integrated_gradients" and torch_model is None:
        return FaithfulnessRow(
            method=method,
            stem=stem,
            cohort_patient=cohort,
            win_start=start,
            win_end=end,
            target_class=tc,
            deletion_auc=np.nan,
            insertion_auc=np.nan,
            stability_jaccard=np.nan,
            localization_iou=np.nan,
            localization_dice=np.nan,
            n_regions=0,
            topk_stability=args.topk_stability,
            localization_mode=f"proxy:{args.proxy_mode}",
            notes="no_torch_model",
        ), []

    exp_stash: Optional[Dict[str, Any]] = None
    if method == "cms_lime" and cms_explainer is not None:
        if (
            out_dir is not None
            and bool(getattr(args, "cms_debug", False))
            and int(getattr(args, "_cms_dbg_n", 0) or 0) < int(getattr(args, "cms_debug_max", 0) or 0)
        ):
            exp_stash = {}
    if method == "cms_lime":
        regions = explain_cms_lime_regions(
            cms_explainer, x, tc, args.max_regions, unit_types=cms_ut, exp_out=exp_stash
        )
    elif method == "cms_lime_random" and cms_explainer is not None:
        regions = list(
            explain_cms_lime_regions(
                cms_explainer, x, tc, args.max_regions, unit_types=cms_ut, exp_out=None
            )
        )
        if regions:
            rng.shuffle(regions)
    elif method == "lime_time":
        regions = explain_lime_time(
            x, predict_proba_np, rng, args.n_pert_samples, args.time_subsample, args.point_radius, tc
        )
    elif method == "lime_segment":
        regions = explain_lime_segment(x, predict_proba_np, rng, args.n_pert_samples, args.seg_len, tc)
    elif method == "tf_lime":
        regions = explain_tf_lime(x, predict_proba_np, rng, args.n_pert_samples, float(args.fs), tc)
    else:
        regions = explain_integrated_gradients(
            torch_model, x, device, n_steps=args.ig_steps, point_radius=args.point_radius, target_class=tc
        )

    regions = regions[: args.max_regions]
    del_auc, ins_auc, x_depth, y_del, y_ins = deletion_insertion_curves(
        x,
        regions,
        predict_proba_np,
        rng,
        tc,
        insertion_moment_match=not bool(getattr(args, "no_insertion_moment_match", False)),
        hybrid_noise_scale=float(getattr(args, "hybrid_noise_scale", 2.0)),
    )
    curve_rows: List[Dict[str, Any]] = []
    for i in range(len(x_depth)):
        curve_rows.append(
            {
                "method": method,
                "stem": stem,
                "win_start": start,
                "win_end": end,
                "target_class": tc,
                "k_norm": float(x_depth[i]),
                "deletion_conf": float(y_del[i]) if i < len(y_del) else np.nan,
                "insertion_conf": float(y_ins[i]) if i < len(y_ins) else np.nan,
            }
        )

    if (
        method == "cms_lime"
        and exp_stash
        and out_dir is not None
        and cms_explainer is not None
    ):
        try:
            p = out_dir / "cms_diagnostics" / f"{stem}_s{start}_e{end}.json"
            write_cms_diagnostic(
                p,
                x,
                regions,
                y_del,
                cms_explainer,
                (exp_stash or {}).get("explanation"),
                cms_ut,
                hybrid_noise_scale=float(getattr(args, "hybrid_noise_scale", 2.0)),
            )
            setattr(args, "_cms_dbg_n", int(getattr(args, "_cms_dbg_n", 0) or 0) + 1)
        except Exception as e:
            print(f"[warn] cms diagnostic write: {e}")

    notes: List[str] = []
    eff0 = min(args.topk_stability, len(regions))
    eff_topk = eff0
    # LIME-Segment: if K equals the number of disjoint segment tiles, top-K id sets
    # collapse to the full partition → Jaccard is trivially 1.0 (protocol artifact, not RNG).
    if method == "lime_segment" and len(regions) >= 2:
        eff_topk = min(eff_topk, len(regions) - 1)
    if eff_topk < eff0:
        notes.append("lime_segment_stab_topk_capped")
    stab = stability_jaccard(lambda r: rank_ids_for_stability(r), rng, args.stability_repeats, eff_topk)
    if method in STABILITY_ID_PROTOCOL_NA:
        stab = float("nan")
        notes.append("stability_id_protocol_not_comparable")
    if eff_topk < 2:
        notes.append("stability_insufficient_regions")
    if (np.max(y_del) - np.min(y_del)) < float(args.min_curve_delta):
        notes.append("deletion_curve_degenerate")
    if (np.max(y_ins) - np.min(y_ins)) < float(args.min_curve_delta):
        notes.append("insertion_curve_degenerate")

    g = proxy_roi_mask(
        args.proxy_mode,
        C,
        T,
        start,
        end,
        file_meta,
        onset_before_sec=float(args.onset_before_sec),
        fs=float(args.fs),
    )
    loc_k = getattr(args, "loc_topk", 8)
    e = union_masks([r.mask for r in regions[:loc_k]]) if regions else np.zeros((C, T), dtype=bool)
    iou, dice = iou_dice(e, g)
    if g.mean() > 0.95:
        notes.append("proxy_roi_almost_full")

    return FaithfulnessRow(
        method=method,
        stem=stem,
        cohort_patient=cohort,
        win_start=start,
        win_end=end,
        target_class=tc,
        deletion_auc=del_auc,
        insertion_auc=ins_auc,
        stability_jaccard=stab,
        localization_iou=iou,
        localization_dice=dice,
        n_regions=len(regions),
        topk_stability=args.topk_stability,
        localization_mode=f"proxy:{args.proxy_mode}",
        notes=";".join(notes),
    ), curve_rows


def aggregate(
    rows: List[FaithfulnessRow],
    out_dir: Path,
    bootstrap_B: int = 0,
    run_meta: Optional[Dict[str, Any]] = None,
) -> None:
    from collections import defaultdict

    by_m: Dict[str, List[FaithfulnessRow]] = defaultdict(list)
    for r in rows:
        by_m[r.method].append(r)

    lines = [
        "# Faithfulness summary (mean ± std over segments)",
        "",
        "**Localization (proxy)**: `localization_mode` encodes `proxy:preictal_block` (metadata-overlap preictal span) or "
        "`proxy:onset_window` (metadata onset-before horizon). This is **not** spike–wave ground truth; use only as a "
        "weak, reproducible sanity check. Deletion/Insertion AUCs integrate `conf(preictal)` over normalized "
        "abscissa `k/K` in `[0,1]`.",
        "",
    ]
    if bootstrap_B > 0:
        lines.append(f"**Bootstrap**: {bootstrap_B} resamples of segment-wise means → 95% CI for the mean.")
        lines.append("")

    tex_lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Del.\,AUC $\downarrow$ & Ins.\,AUC $\uparrow$ & Stab.\,$\uparrow$ & IoU (p.) & Dice (p.) \\",
        r"\midrule",
    ]

    journal_f_rows: List[str] = []
    journal_l_rows: List[str] = []
    journal_open: List[str] = [
        r"% --- Journal-ready fragment (requires \usepackage{booktabs} in preamble) ---",
        r"% Two blocks: (a) faithfulness, (b) proxy localization. Use table* in two-column layout (e.g. IEEE).",
        r"% Single-column: change \begin{table*} to \begin{table} and \end{table*} to \end{table}.",
        r"% Localization columns: weak proxy ROI only (not spike-wave GT).",
        r"\begin{table*}[!t]",
        r"\centering",
        r"\small",
        r"\caption{Explanation faithfulness on held EEG segments. "
        r"Deletion/Insertion: trapezoidal AUC of $P(\mathrm{preictal})$ vs.\ normalized mask depth $k/K$; "
        r"higher insertion and lower deletion are better. "
        r"Stability: mean pairwise Jaccard of top-$K$ region ids over stochastic runs. "
        r"IoU/Dice: overlap of explanation union vs.\ proxy ROI.}",
        r"\label{tab:eeg_faithfulness_main}",
        r"\noindent\textit{(a) Faithfulness (deletion/insertion AUC, stability).}",
        r"\par\smallskip",
        r"\begin{tabular}{@{}l*{3}{c}@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Del.\,AUC} $\downarrow$ & \textbf{Ins.\,AUC} $\uparrow$ & \textbf{Stab.} $\uparrow$ \\",
        r"\midrule",
    ]

    n_seg_total = len({(r.stem, r.win_start, r.win_end) for r in rows})

    cms_lime_rows = [r for r in rows if r.method == "cms_lime"]
    n_cms_fail = sum(1 for r in cms_lime_rows if "cms_lime_unavailable" in r.notes)
    n_cms_ok = sum(1 for r in cms_lime_rows if np.isfinite(r.deletion_auc))
    if len(cms_lime_rows) > 0 and n_cms_fail > 0:
        cms_status_tex = (
            rf" CMS-LIME aggregates: {n_cms_ok} of {len(cms_lime_rows)} segment rows enter the mean "
            rf"({n_cms_fail} omitted with \texttt{{cms\_lime\_unavailable}} in the segment CSV; see \texttt{{notes}} column). "
        )
    elif len(cms_lime_rows) > 0:
        cms_status_tex = (
            r" All CMS-LIME segment rows completed (no \texttt{cms\_lime\_unavailable} in CSV \texttt{notes}). "
        )
    else:
        cms_status_tex = ""

    md_table_rows: List[str] = []
    agg_csv_rows: List[Dict[str, Any]] = []

    def _md_cell_plain(amu: str, asd: str, aci: Tuple[float, float]) -> str:
        t = f"{amu}±{asd}"
        if bootstrap_B > 0 and np.isfinite(aci[0]) and np.isfinite(aci[1]):
            t += f" [{aci[0]:.3f}, {aci[1]:.3f}]"
        return t

    def _fnum_str(s: str) -> str:
        try:
            return f"{float(s):.10g}"
        except (TypeError, ValueError):
            return ""

    for m in METHODS:
        rs = by_m.get(m, [])
        if not rs:
            continue

        def mean_std_ci(key: str) -> Tuple[str, str, Tuple[float, float]]:
            vals = np.asarray([getattr(x, key) for x in rs], dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                return "—", "—", (float("nan"), float("nan"))
            mu, sd = float(np.mean(finite)), float(np.std(finite))
            lo, hi = (
                bootstrap_mean_ci(
                    finite,
                    bootstrap_B,
                    seed=int(zlib.adler32((m + key).encode("utf-8"))) % (2**31 - 1),
                )
                if bootstrap_B > 0
                else (float("nan"), float("nan"))
            )
            return f"{mu:.4f}", f"{sd:.4f}", (lo, hi)

        md, sd, ci_d = mean_std_ci("deletion_auc")
        mi, si, ci_i = mean_std_ci("insertion_auc")
        if m in STABILITY_ID_PROTOCOL_NA:
            ms, ss, ci_s = "—", "—", (float("nan"), float("nan"))
        else:
            ms, ss, ci_s = mean_std_ci("stability_jaccard")
        mo, so, ci_o = mean_std_ci("localization_iou")
        mdc, sdc, ci_c = mean_std_ci("localization_dice")

        disp = METHOD_DISPLAY.get(m, m)
        if m == "cms_lime" and run_meta and run_meta.get("cms_lime_paper_name"):
            disp = str(run_meta["cms_lime_paper_name"])
        lines.append(f"## {m} ({disp})")
        lines.append(f"- deletion_auc: {md} ± {sd}")
        if bootstrap_B > 0 and np.isfinite(ci_d[0]):
            lines.append(f"  - 95% CI (mean): [{ci_d[0]:.4f}, {ci_d[1]:.4f}]")
        lines.append(f"- insertion_auc: {mi} ± {si}")
        if bootstrap_B > 0 and np.isfinite(ci_i[0]):
            lines.append(f"  - 95% CI (mean): [{ci_i[0]:.4f}, {ci_i[1]:.4f}]")
        if m in STABILITY_ID_PROTOCOL_NA:
            lines.append(f"- stability_jaccard: — (N/A, ID protocol not comparable to other methods)")
        else:
            lines.append(f"- stability_jaccard: {ms} ± {ss}")
        lines.append(f"- localization_iou (proxy): {mo} ± {so}")
        lines.append(f"- localization_dice (proxy): {mdc} ± {sdc}")
        lines.append("")

        _stab = r"$\text{---}$" if m in STABILITY_ID_PROTOCOL_NA else f"${ms} \\pm {ss}$"
        tex_lines.append(
            f"{disp} & ${md} \\pm {sd}$ & ${mi} \\pm {si}$ & {_stab} & ${mo} \\pm {so}$ & ${mdc} \\pm {sdc}$ \\\\"
        )

        def fmt_ci_cell(lo: float, hi: float) -> str:
            if not (np.isfinite(lo) and np.isfinite(hi)):
                return ""
            return f"\\,{{\\scriptsize $[{lo:.3f},\\,{hi:.3f}]$}}"

        _stab_j = r"$\text{---}$" if m in STABILITY_ID_PROTOCOL_NA else f"${ms} \\pm {ss}$"
        _ci_s = "" if m in STABILITY_ID_PROTOCOL_NA else fmt_ci_cell(*ci_s)
        journal_f_rows.append(
            f"{disp} & ${md} \\pm {sd}${fmt_ci_cell(*ci_d)} & "
            f"${mi} \\pm {si}${fmt_ci_cell(*ci_i)} & {_stab_j}{_ci_s} \\\\"
        )
        journal_l_rows.append(
            f"{disp} & ${mo} \\pm {so}${fmt_ci_cell(*ci_o)} & ${mdc} \\pm {sdc}${fmt_ci_cell(*ci_c)} \\\\"
        )

        _st_md = "—" if m in STABILITY_ID_PROTOCOL_NA else _md_cell_plain(ms, ss, ci_s)
        md_table_rows.append(
            "| "
            + " | ".join(
                [
                    str(disp),
                    _md_cell_plain(md, sd, ci_d),
                    _md_cell_plain(mi, si, ci_i),
                    _st_md,
                    _md_cell_plain(mo, so, ci_o),
                    _md_cell_plain(mdc, sdc, ci_c),
                ]
            )
            + " |"
        )
        _row = {
            "method": m,
            "display_name": disp,
            "n_segment_rows": len(rs),
            "deletion_auc_mean": _fnum_str(md),
            "deletion_auc_std": _fnum_str(sd),
            "deletion_auc_ci95_lo": f"{ci_d[0]:.10g}" if np.isfinite(ci_d[0]) else "",
            "deletion_auc_ci95_hi": f"{ci_d[1]:.10g}" if np.isfinite(ci_d[1]) else "",
            "insertion_auc_mean": _fnum_str(mi),
            "insertion_auc_std": _fnum_str(si),
            "insertion_auc_ci95_lo": f"{ci_i[0]:.10g}" if np.isfinite(ci_i[0]) else "",
            "insertion_auc_ci95_hi": f"{ci_i[1]:.10g}" if np.isfinite(ci_i[1]) else "",
            "stability_jaccard_mean": _fnum_str(ms) if m not in STABILITY_ID_PROTOCOL_NA else "",
            "stability_jaccard_std": _fnum_str(ss) if m not in STABILITY_ID_PROTOCOL_NA else "",
            "stability_jaccard_ci95_lo": f"{ci_s[0]:.10g}"
            if m not in STABILITY_ID_PROTOCOL_NA and np.isfinite(ci_s[0])
            else "",
            "stability_jaccard_ci95_hi": f"{ci_s[1]:.10g}"
            if m not in STABILITY_ID_PROTOCOL_NA and np.isfinite(ci_s[1])
            else "",
            "stability_comparable": "no" if m in STABILITY_ID_PROTOCOL_NA else "yes",
            "localization_iou_mean": _fnum_str(mo),
            "localization_iou_std": _fnum_str(so),
            "localization_dice_mean": _fnum_str(mdc),
            "localization_dice_std": _fnum_str(sdc),
        }
        agg_csv_rows.append(_row)

    _best_summary_tex = ""
    stability_id_na_tex = ""
    if any(by_m.get(m) for m in STABILITY_ID_PROTOCOL_NA):
        stability_id_na_tex = (
            r" \textsf{TF-LIME} and Int.\,Gradients: Stab.\ column is \text{---} "
            r"(Jaccard over their region id scheme is not comparable to tile- or point-based ids in this table). "
        )
    if run_meta and run_meta.get("include_table_leader_text", False):
        def _dispname(mm: str) -> str:
            d0 = METHOD_DISPLAY.get(mm, mm)
            if mm == "cms_lime" and run_meta and run_meta.get("cms_lime_paper_name"):
                d0 = str(run_meta["cms_lime_paper_name"])
            return d0

        def _mmean(key: str) -> Dict[str, float]:
            o: Dict[str, float] = {}
            for mm in METHODS:
                if key == "stability_jaccard" and mm in STABILITY_ID_PROTOCOL_NA:
                    continue
                rs2 = by_m.get(mm) or []
                v = np.asarray([getattr(x, key) for x in rs2], dtype=float)
                v = v[np.isfinite(v)]
                if v.size:
                    o[mm] = float(np.mean(v))
            return o

        _metric_specs = [
            ("deletion_auc", "min", "Del.\\,AUC", "lower is better"),
            ("insertion_auc", "max", "Ins.\\,AUC", "higher is better"),
            ("stability_jaccard", "max", "Stab.", "higher is better"),
            ("localization_iou", "max", "IoU (proxy)", "higher is better"),
            ("localization_dice", "max", "Dice (proxy)", "higher is better"),
        ]
        _leaders: Dict[str, Any] = {}
        _leader_lines = ["# Segment-mean best method per column", ""]
        _leader_lines.append(
            "| Metric | Best method | Segment-mean on winning column |"
        )
        _leader_lines.append("|--------|------------|--------------------------------|")
        for mkey, mode, mlab, mnote in _metric_specs:
            means = _mmean(mkey)
            if not means:
                continue
            if mode == "min":
                best = min(means, key=means.get)
            else:
                best = max(means, key=means.get)
            _leaders[mkey] = {
                "best": best,
                "value": means[best],
                "all_segment_means": {k: means[k] for k in sorted(means)},
                "criterion": mnote,
            }
            _leader_lines.append(
                f"| {mlab} ({mnote}) | {_dispname(best)} | {means[best]:.4f} |"
            )
        if _leaders and run_meta is not None:
            run_meta["metric_leaders"] = _leaders
        if _leaders:
            (out_dir / "faithfulness_leaders.md").write_text(
                "\n".join(_leader_lines) + "\n", encoding="utf-8"
            )
            _frag: List[str] = []
            for mkey, mode, mlab, _mnote in _metric_specs:
                it = _leaders.get(mkey)
                if not it:
                    continue
                b = it["best"]
                v = it["value"]
                _dn = _dispname(b).replace("_", "\\_")
                _frag.append(
                    f"{mlab} $\\rightarrow$ "
                    r"\textsf{"
                    + _dn
                    + r"} "
                    + f"(mean$={v:.3f}$)"
                )
            if _frag:
                _best_summary_tex = (
                    r" \emph{Segment-mean per column:} " + ";\, ".join(_frag) + ". "
                    r"Proxy IoU/Dice are weak sanity checks, not clinical localization ground truth. "
                )

    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    cohort_stats = cohort_stats_from_rows(rows)
    wilcox_payload = paired_wilcoxon_cms_vs_baselines(rows)
    try:
        (out_dir / "faithfulness_wilcoxon.json").write_text(
            json.dumps({"cohort": cohort_stats, "wilcoxon": wilcox_payload}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
    if run_meta is not None:
        run_meta["cohort_stats"] = cohort_stats
        run_meta["wilcoxon_vs_baselines"] = wilcox_payload

    _wilcox_intro = _wilcoxon_notes_tex(wilcox_payload, cohort_stats)
    _notes_continued = (
        _wilcox_intro
        + rf"Proxy ROI encoded in CSV column \texttt{{localization\_mode}}. "
        + (
            rf"Brackets: 95\% bootstrap CI for the mean over segments ($B={bootstrap_B}$). "
            if bootstrap_B > 0
            else r"Reported as mean$\pm$std over segments. "
        )
        + _best_summary_tex
        + stability_id_na_tex
        + cms_status_tex
        + (
            str(run_meta.get("sampling_plan_tex", ""))
            if run_meta and run_meta.get("sampling_plan_tex")
            else ""
        )
        + (
            r" \textsuperscript{$\dagger$}LIME-Segment stability uses $K'=\min(K,|\mathcal{S}|-1)$ when $|\mathcal{S}|\ge 2$, "
            r"so top-$K$ id sets are not forced to equal the full tile partition (which would yield trivial Jaccard $=1$). "
            if run_meta and run_meta.get("lime_segment_stability_dagger", True)
            else ""
        )
    )
    journal_blocks: List[str] = (
        journal_open
        + journal_f_rows
        + [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\medskip",
            r"\noindent\textit{(b) Proxy localization (metadata ROI; not clinical GT).}",
            r"\par\smallskip",
            r"\begin{tabular}{@{}l*{2}{c}@{}}",
            r"\toprule",
            r"\textbf{Method} & \textbf{IoU} & \textbf{Dice} \\",
            r"\midrule",
        ]
        + journal_l_rows
        + [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\medskip",
            r"\begin{minipage}{0.95\linewidth}\footnotesize",
            r"\emph{Notes.} " + _notes_continued,
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )

    (out_dir / "faithfulness_summary.md").write_text("\n".join(lines), encoding="utf-8")

    _md_table_doc: List[str] = [
        "# Faithfulness — aggregate result table",
        "",
        f"- **N** = {n_seg_total} unique segments (file × window).",
    ]
    if bootstrap_B > 0:
        _md_table_doc.append(
            f"- **95% CI** of the segment-mean: bootstrap $B={bootstrap_B}$ (bracket in each cell)."
        )
    else:
        _md_table_doc.append(
            "- **95% CI** not computed; only mean±std (set e.g. `--bootstrap_B 1000`)."
        )
    _md_table_doc += [
        "",
        "| Method | Del. AUC ↓ | Ins. AUC ↑ | Stab. ↑ | IoU (proxy) | Dice (proxy) |",
        "|--------|------------|------------|---------|-------------|--------------|",
    ]
    if md_table_rows:
        _md_table_doc += md_table_rows
    else:
        _md_table_doc += ["| *No method aggregates (empty or missing rows)* | | | | | | |"]
    _md_table_doc += [
        "",
        "— *Stab.*: not comparable (N/A) for **TF-LIME** and **Int. Gradients** under this Jaccard protocol.",
    ]
    (out_dir / "faithfulness_table.md").write_text(
        "\n".join(_md_table_doc) + "\n", encoding="utf-8"
    )
    if agg_csv_rows:
        ap_csv = out_dir / "faithfulness_aggregate_table.csv"
        with ap_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_csv_rows[0].keys()))
            w.writeheader()
            for row in agg_csv_rows:
                w.writerow(row)

    (out_dir / "faithfulness_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    journal_tex = "\n".join(journal_blocks) + "\n"
    (out_dir / "faithfulness_table_journal.tex").write_text(journal_tex, encoding="utf-8")

    standalone = (
        "% twocolumn enables table* (double-width float). For IEEEtran, switch to that class and drop twocolumn here.\n"
        "\\documentclass[twocolumn]{article}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\begin{document}\n"
        "\\input{faithfulness_table_journal.tex}\n"
        "\\end{document}\n"
    )
    (out_dir / "faithfulness_standalone_compile.tex").write_text(standalone, encoding="utf-8")

    if run_meta is not None:
        meta_path = out_dir / "faithfulness_run_meta.json"
        meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"D:\public_data\CHBMIT\1_data_clean\chb01")
    parser.add_argument(
        "--chb_patient",
        type=int,
        default=0,
        help="If 1..99, set data_dir to .../1_data_clean/chb%%02d and default segment_info_root to segment_clean/30-1-240 (chb_paths). Mutually exclusive with --patient_cohorts.",
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument(
        "--loocv_model_name",
        type=str,
        default="",
        help="e.g. eeginception: resolve a .pth under 组合留一法/<name>/ (chb_loocv_weights; requires --chb_patient). Overrides --model_path when set.",
    )
    parser.add_argument(
        "--loocv_select_loop",
        type=int,
        default=1,
        help="Matches training select_loop_n_w; uses the (select_loop_n_w-1)-th .pth in the LOOCV subject subfolder.",
    )
    parser.add_argument(
        "--loocv_sync_data",
        action="store_true",
        help="With LOOCV, set data_dir to the remapped chb id (e.g. patient 12 -> chb13) as in the training script.",
    )
    parser.add_argument(
        "--loocv_weight_root",
        type=str,
        default="",
        help="Optional LOOCV root (default: env CHB_LOOCV_WEIGHT_ROOT or 组合留一法 on D:). E.g. E:\\...\\LOOCV",
    )
    parser.add_argument(
        "--simple_model",
        action="store_true",
        help="Force SimpleEEGModel (skip resolve_model_path auto-discovery when channels mismatch)",
    )
    parser.add_argument("--out_dir", type=str, default=str(_THIS_DIR / "faithfulness_out"))
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Cap on .npy files after filters (single-patient or pooled list). "
        "0 = use all files (recommended; previous default 1 caused a 1-recording cohort by accident).",
    )
    parser.add_argument("--max_segments", type=int, default=30)
    parser.add_argument(
        "--stride",
        type=int,
        default=20000,
        help="Sliding-window stride (smaller → more preictal candidates; 20000 matches prior paper runs).",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_class", type=int, default=1, help="Class index for conf (CHB: 1=preictal)")
    parser.add_argument(
        "--conf_mode",
        type=str,
        default="prob",
        choices=["prob", "target_logit"],
        help="Confidence scale for deletion/insertion and surrogate targets",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(m for m in METHODS if m != "cms_lime_random"),
        help="Comma-separated. Add cms_lime_random for shuffled-deletion-order sanity (extra CMS per segment).",
    )
    parser.add_argument("--fast", action="store_true", help="Fewer perturb samples / regions")
    parser.add_argument("--cms_fit_windows", type=int, default=24, help="Windows to fit CMS-LIME microstate/shapelet")
    parser.add_argument(
        "--cms_stability_jitter",
        type=float,
        default=1e-3,
        help="Relative input jitter (std-scaled) for CMS-LIME stability repeats",
    )
    parser.add_argument(
        "--cms_unit_types",
        type=str,
        default="microstate,shapelet,timefreq",
        help="Comma-separated unit_types for CMS-LIME explain_instance. Default is the full three-type pool; "
        "use e.g. microstate alone only for a runtime-limited ablation (label as microstate-only in the paper).",
    )
    parser.add_argument("--max_regions", type=int, default=12)
    parser.add_argument("--n_pert_samples", type=int, default=120)
    parser.add_argument("--time_subsample", type=int, default=16)
    parser.add_argument("--point_radius", type=int, default=16, help="Temporal half-width for point-based masks (lime_time, IG)")
    parser.add_argument("--seg_len", type=int, default=160)
    parser.add_argument("--fs", type=float, default=256.0)
    parser.add_argument("--ig_steps", type=int, default=16)
    parser.add_argument("--stability_repeats", type=int, default=50)
    parser.add_argument("--topk_stability", type=int, default=8)
    parser.add_argument(
        "--loc_topk",
        type=int,
        default=8,
        help="Top-K regions unioned for localization IoU/Dice vs proxy ROI",
    )
    parser.add_argument("--proxy_mode", type=str, default="onset_window", choices=["preictal_block", "onset_window"])
    parser.add_argument("--onset_before_sec", type=float, default=30.0, help="Seconds before onset for proxy onset_window ROI")
    parser.add_argument(
        "--segment_info_json",
        type=str,
        default="",
        help="Optional segment_info.json for preictal/onset spans; auto-resolved when empty",
    )
    parser.add_argument(
        "--segment_info_root",
        type=str,
        default="",
        help="Root containing {chb01,chb05,...}/segment_info.json; default env CHB_SEGMENT_INFO_ROOT or standard CHB path",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Parent directory of per-patient folders (e.g. .../1_data_clean). "
        "Use with --patient_cohorts to avoid single-patient 'sorted file order' bias.",
    )
    parser.add_argument(
        "--patient_cohorts",
        type=str,
        default="",
        help="Comma-separated subject ids (e.g. chb01,chb05,chb10). "
        "Empty: single-patient mode using --data_dir only.",
    )
    parser.add_argument(
        "--max_npy_per_patient",
        type=int,
        default=2,
        help="With --patient_cohorts: at most this many .npy files per subject (sorted stem order), not 'first 5 files' globally.",
    )
    parser.add_argument(
        "--max_segments_per_patient",
        type=int,
        default=0,
        help="If >0, pool preictal windows per patient cohort and keep this many (for e.g. 3 patients x 2 segments = 6). If 0, use --max_segments per file.",
    )
    parser.add_argument(
        "--cms_n_perturbations",
        type=int,
        default=0,
        help="CMS n_perturbations; 0=auto (400 full / 120 fast, then optional --cms_half_pert).",
    )
    parser.add_argument(
        "--cms_half_pert",
        action="store_true",
        help="Halve CMS n_perturbations (runtime trade-off; prefer this over microstate-only as 'full' CMS).",
    )
    parser.add_argument(
        "--cms_dpp_k",
        type=int,
        default=10,
        help="DPP / local regressor: max number of features to select (also caps n_features; lower is faster).",
    )
    parser.add_argument(
        "--cms_regression",
        type=str,
        default="ridge",
        choices=["ridge", "lasso", "elastic_net"],
        help="Local surrogate in CMS _combine (use lasso to inspect sparsity in --cms_debug dumps).",
    )
    parser.add_argument(
        "--cms_debug",
        action="store_true",
        help="Write cms_diagnostics/*.json: microstate mask cover, surrogate coefficients, P(preictal) along deletion depth.",
    )
    parser.add_argument(
        "--cms_debug_max",
        type=int,
        default=3,
        help="Max segments to dump CMS diagnostics (CSV notes still has all segments).",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="preictal",
        choices=["all", "preictal", "onset_near"],
        help="Window sampling strategy before explanation",
    )
    parser.add_argument(
        "--min_curve_delta",
        type=float,
        default=1e-6,
        help="Flag degenerate Deletion/Insertion curves when max-min < this value",
    )
    parser.add_argument(
        "--bootstrap_B",
        type=int,
        default=2000,
        help="Bootstrap resamples for 95%% CI of segment-mean (0 to skip CI in journal table)",
    )
    parser.add_argument(
        "--no_insertion_moment_match",
        action="store_true",
        help="Diagnosis: skip per-channel moment matching on the insertion path (if Del/Ins stay flat, matching may be erasing perturbation).",
    )
    parser.add_argument(
        "--hybrid_noise_scale",
        type=float,
        default=2.0,
        help="Scale for hybrid noise on the deletion and full-perturbation paths (see hybrid_noise_on_mask).",
    )
    parser.add_argument(
        "--include_table_leader_text",
        action="store_true",
        help="Add segment-mean 'best per column' text to the journal .tex notes (off by default).",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar on segment×method jobs (for logs/CI).",
    )
    args = parser.parse_args()

    if int(getattr(args, "chb_patient", 0) or 0) > 0:
        if (getattr(args, "patient_cohorts", "") or "").strip():
            raise SystemExit("Use either --chb_patient or --patient_cohorts, not both.")
        p = int(args.chb_patient)
        if p < 1 or p > 99:
            raise SystemExit("--chb_patient must be in 1..99 (maps to chb%02d).")
        args.data_dir = str(chb_patient_data_dir(p))
        if not (getattr(args, "segment_info_root", "") or "").strip():
            args.segment_info_root = str(CHB_SEGMENT_CLEAN_ROOT)

    apply_loocv_model_cli(args)

    if args.fast:
        args.n_pert_samples = min(args.n_pert_samples, 48)
        args.max_regions = min(args.max_regions, 8)
        args.stability_repeats = min(args.stability_repeats, 15)
        args.cms_fit_windows = min(args.cms_fit_windows, 12)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    seg_info_root = Path(args.segment_info_root) if (args.segment_info_root or "").strip() else default_chb_segment_info_root()
    patient_cohorts = [c.strip() for c in (args.patient_cohorts or "").split(",") if c.strip()]
    seg_info_path: Optional[Path] = None
    seg_info_paths_merged: List[str] = []
    if patient_cohorts:
        if not (args.data_root or "").strip():
            raise SystemExit("Multi-patient mode requires --data_root, e.g. D:\\...\\1_data_clean")
        data_dir = Path(args.data_root)
        npy_files = collect_npy_for_patient_cohorts(
            data_dir, patient_cohorts, int(args.max_npy_per_patient)
        )
        s_paths = [resolve_segment_info_path_for_patient_cohort(p, seg_info_root) for p in patient_cohorts]
        for p, sp in zip(patient_cohorts, s_paths):
            if sp is None or not sp.is_file():
                print(
                    f"[warn] missing segment_info: expected {seg_info_root / p / 'segment_info.json'}. "
                    f"In preictal/onset mode, .npy stems from this subject have no Pre*/onset spans in metadata."
                )
        seg_info = load_segment_info_merged(s_paths)
        seg_info_paths_merged = [str(s) for s in s_paths if s is not None]
    else:
        data_dir = Path(args.data_dir)
        seg_info_path = resolve_segment_info_path(data_dir, args.segment_info_json, seg_info_root)
        seg_info = load_segment_info(seg_info_path)
        all_npy = sorted(data_dir.glob("*.npy"), key=lambda p: natural_keys(p.name))
        if args.sampling_mode in ("preictal", "onset_near") and seg_info:
            selected_stems = set()
            for stem, meta in seg_info.items():
                if args.sampling_mode == "preictal" and meta.get("pre"):
                    selected_stems.add(stem)
                if args.sampling_mode == "onset_near" and meta.get("onset"):
                    selected_stems.add(stem)
            npy_files = [p for p in all_npy if p.stem in selected_stems]
        else:
            npy_files = all_npy
        if int(getattr(args, "max_files", 0) or 0) > 0:
            npy_files = npy_files[: int(args.max_files)]
    if not npy_files:
        raise SystemExit("No npy files")

    sample = pan.load_long_npy(str(npy_files[0]))
    if sample.ndim != 3 or sample.shape[0] != 1:
        raise SystemExit("Need (1,C,T) npy")
    n_ch = int(sample.shape[1])

    device = torch.device(args.device)
    mp, ok = pan.resolve_model_path(args.model_path.strip() or None)
    if getattr(args, "simple_model", False):
        mp = ""
    model = pan.load_model_robust(mp, n_chans=n_ch, n_classes=2, device=device)
    wrapper = pan.EEGModelWrapper(model, device, n_chans=n_ch)

    def predict_proba_np(x_bct: np.ndarray) -> np.ndarray:
        """Model confidence backend used by faithfulness metrics."""
        if args.conf_mode == "prob":
            return wrapper.predict_proba(x_bct)
        x_bct = np.array(x_bct, dtype=np.float32, copy=True, order="C")
        xt = torch.from_numpy(x_bct).to(device)
        if xt.dim() == 2:
            xt = xt.unsqueeze(0)
        with torch.no_grad():
            out = model(xt)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            # If model already emits probabilities, map to logits.
            if out.min() >= 0 and out.max() <= 1 and out.sum(dim=-1).min() > 0.99:
                out = torch.logit(torch.clamp(out, 1e-6, 1 - 1e-6))
        return out.cpu().numpy()

    # Optional CMS-LIME
    cms_explainer = None
    cms_n_pert_effective: Optional[int] = None
    cms_dpp_k_effective: Optional[int] = None
    if any(m in methods for m in ("cms_lime", "cms_lime_random")):
        try:
            from cms_lime_explainer import CMSLimeConfig, CMSLimeExplainer

            n_pert = int(getattr(args, "cms_n_perturbations", 0) or 0) or (120 if args.fast else 400)
            if bool(getattr(args, "cms_half_pert", False)):
                n_pert = max(32, n_pert // 2)
            n_feat = max(1, min(int(getattr(args, "cms_dpp_k", 10)), int(args.max_regions), 16))
            cfg = CMSLimeConfig(
                verbose=False,
                n_perturbations=n_pert,
                n_shapelets=12 if args.fast else 60,
                n_microstates=3 if args.fast else 4,
                n_features=n_feat,
                regression_method=str(getattr(args, "cms_regression", "ridge")),
            )
            cms_n_pert_effective, cms_dpp_k_effective = n_pert, n_feat
            cms_explainer = CMSLimeExplainer(cfg)
            # build tiny training set from first file
            X_list, y_list = [], []
            data = pan.load_long_npy(str(npy_files[0]))
            for w, s, e in pan.iter_windows(data, 1280, args.stride):
                if len(X_list) >= args.cms_fit_windows:
                    break
                w = np.array(w, dtype=np.float32, copy=True)
                pr = wrapper.predict_proba(w)[0]
                y_list.append(int(np.argmax(pr)))
                X_list.append(w[0])
            X_train = np.stack(X_list, axis=0)
            y_train = np.asarray(y_list, dtype=int)
            cms_explainer.fit(X_train, y_train, wrapper)
        except Exception as e:
            print(f"[warn] CMS-LIME init failed: {e}")
            cms_explainer = None

    rows: List[FaithfulnessRow] = []
    curve_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(42)
    setattr(args, "_cms_dbg_n", 0)

    window_jobs = gather_faithfulness_windows(npy_files, seg_info, args)
    if not window_jobs:
        raise SystemExit("No windows after sampling filters; check segment_info.json and preictal spans.")

    _n_jobs = max(1, len(window_jobs)) * max(1, len(methods))
    _use_pbar = (tqdm is not None) and (not bool(getattr(args, "no_progress", False)))
    _pbar = (
        tqdm(
            total=_n_jobs,
            desc="Faithfulness",
            unit="job",
            dynamic_ncols=True,
            mininterval=0.25,
            file=sys.stdout,
        )
        if _use_pbar
        else None
    )
    try:
        for nseg, (x, start, end, stem) in enumerate(window_jobs):
            file_meta = seg_info.get(stem)
            rng_seg = np.random.default_rng(1000 * nseg + 7)
            for method in methods:
                row, crv = run_for_segment(
                    method,
                    x,
                    stem,
                    start,
                    end,
                    predict_proba_np,
                    model,
                    device,
                    cms_explainer,
                    file_meta,
                    args,
                    rng_seg,
                    out_dir=out_dir,
                )
                rows.append(row)
                curve_rows.extend(crv)
                if _pbar is not None:
                    _pbar.update(1)
                    _pbar.set_postfix_str(
                        f"{method}:{stem}[{start}-{end}]",
                        refresh=False,
                    )
    finally:
        if _pbar is not None:
            _pbar.close()

    csv_path = out_dir / "faithfulness_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "stem",
                "cohort_patient",
                "win_start",
                "win_end",
                "target_class",
                "deletion_auc",
                "insertion_auc",
                "stability_jaccard",
                "localization_iou",
                "localization_dice",
                "n_regions",
                "topk_stability",
                "localization_mode",
                "notes",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)

    curve_csv = out_dir / "faithfulness_curves.csv"
    with curve_csv.open("w", newline="", encoding="utf-8-sig") as f:
        cw = csv.DictWriter(
            f,
            fieldnames=["method", "stem", "win_start", "win_end", "target_class", "k_norm", "deletion_conf", "insertion_conf"],
        )
        cw.writeheader()
        for r in curve_rows:
            cw.writerow(r)

    n_unique_seg = len({(r.stem, r.win_start, r.win_end) for r in rows})
    if patient_cohorts:
        seg_json = "merged:" + ";".join(seg_info_paths_merged)
    else:
        seg_json = str(seg_info_path) if seg_info_path is not None else ""
    cms_ut_list = [s.strip() for s in str(args.cms_unit_types).split(",") if s.strip()]
    ut_tex = (
        ", ".join([r"\texttt{" + u + "}" for u in cms_ut_list])
        if cms_ut_list
        else r"\texttt{microstate}, \texttt{shapelet}, \texttt{timefreq}"
    )
    if len(cms_ut_list) == 1 and cms_ut_list[0] == "microstate":
        cms_lime_paper_name = "CMS-LIME (microstate only)"
    elif len(cms_ut_list) == 1:
        cms_lime_paper_name = f"CMS-LIME ({cms_ut_list[0]} only)"
    else:
        cms_lime_paper_name = "CMS-LIME"
    sampling_plan_tex = (
        r"Sampling: $N=" + str(n_unique_seg) + r"$ counts unique $(\mathrm{file},\mathrm{start},\mathrm{end})$ tuples. "
        r"Candidates are $1280$-sample windows with stride " + str(int(args.stride))
        + r" that overlap metadata \texttt{Pre*} spans (from \texttt{segment\_info.json}). "
        r"If a file has more than " + str(int(args.max_segments))
        + r" candidates, we keep " + str(int(args.max_segments))
        + r" windows at indices spaced by \texttt{numpy.linspace} along the filtered candidate list. "
        + (
            r"We use " + str(len(npy_files)) + r" recording file(s) after filters"
            + (
                r" (entire list after metadata filters; no per-run file cap)."
                if int(getattr(args, "max_files", 0) or 0) <= 0
                else r" (first " + str(int(args.max_files)) + r" file(s) in sorted order after filters)."
            )
        )
        + r" CMS-LIME uses \texttt{explain\_instance} with unit\_types$=\{$" + ut_tex + r"$\}$."
    )
    _cms_meta = (
        r" CMS-LIME training: n\_perturbations $="
        + str(cms_n_pert_effective if cms_n_pert_effective is not None else "n/a")
        + r"$, selector cap $k \leq "
        + str(cms_dpp_k_effective if cms_dpp_k_effective is not None else "n/a")
        + r"$ (see run \texttt{faithfulness\_run\_meta.json})."
    )
    sampling_plan_tex += _cms_meta
    if rows:
        ucoh = sorted({r.cohort_patient for r in rows}, key=natural_keys)
        f_stems = sorted({r.stem for r in rows}, key=natural_keys)
        sampling_plan_tex += (
            r" Cohort: $P=" + str(len(ucoh)) + r"$ unique patient id(s) (\texttt{cohort\_patient} in CSV, from file \texttt{stem}): "
            + ", ".join(r"\texttt{" + c + "}" for c in ucoh) + "."
            r" $F=" + str(len(f_stems)) + r"$ file stems (recordings) include at least one window: "
            + ", ".join(r"\texttt{" + s + "}" for s in f_stems) + "."
        )
    if int(getattr(args, "max_segments_per_patient", 0) or 0) > 0:
        sampling_plan_tex += (
            r" Per-patient cap: we pool all filtered windows per \texttt{cohort\_patient} and keep at most "
            + str(int(args.max_segments_per_patient))
            + r" windows, spaced with \texttt{numpy.linspace} (see \texttt{--max\_segments\_per\_patient})."
        )
    if patient_cohorts:
        sampling_plan_tex += (
            r" Multi-patient \texttt{.npy} selection: for each of "
            + str(len(patient_cohorts))
            + r" patients, at most the first "
            + str(int(args.max_npy_per_patient))
            + r" files in sorted name order in that subject folder under \texttt{--data\_root} (not a single global \texttt{sorted} grab across all subjects)."
        )
    run_meta: Dict[str, Any] = {
        "data_dir": str(data_dir.resolve()),
        "data_root": str((args.data_root or "").strip()),
        "patient_cohorts": patient_cohorts,
        "max_npy_per_patient": int(args.max_npy_per_patient),
        "max_segments_per_patient": int(getattr(args, "max_segments_per_patient", 0) or 0),
        "segment_info_root": str(seg_info_root),
        "segment_info_json": seg_json,
        "segment_info_merged_files": seg_info_paths_merged,
        "model_path_resolved": mp,
        "loocv_model_name": (str(getattr(args, "loocv_model_name", "")).strip() or None),
        "loocv_select_loop": int(getattr(args, "loocv_select_loop", 1) or 1),
        "loocv_sync_data": bool(getattr(args, "loocv_sync_data", False)),
        "loocv_weight_root": (str(getattr(args, "loocv_weight_root", "")).strip() or None),
        "simple_model": bool(getattr(args, "simple_model", False)),
        "n_channels": n_ch,
        "npy_files": [str(p) for p in npy_files],
        "max_files": int(getattr(args, "max_files", 0) or 0),
        "max_segments": args.max_segments,
        "stride": args.stride,
        "methods": methods,
        "conf_mode": args.conf_mode,
        "point_radius": int(args.point_radius),
        "proxy_mode": args.proxy_mode,
        "sampling_mode": args.sampling_mode,
        "onset_before_sec": float(args.onset_before_sec),
        "loc_topk": args.loc_topk,
        "topk_stability": args.topk_stability,
        "stability_repeats": args.stability_repeats,
        "cms_stability_jitter": float(args.cms_stability_jitter),
        "cms_unit_types": cms_ut_list,
        "fast": args.fast,
        "bootstrap_B": int(args.bootstrap_B),
        "cms_lime_available": cms_explainer is not None,
        "n_result_rows": len(rows),
        "n_curve_rows": len(curve_rows),
        "n_unique_segments": n_unique_seg,
        "n_unique_patient_cohorts": (len({r.cohort_patient for r in rows}) if rows else 0),
        "sampling_plan_tex": sampling_plan_tex,
        "cms_lime_paper_name": cms_lime_paper_name,
        "cms_stability_uses_rank_free_region_ids": True,
        "cms_n_perturbations_effective": cms_n_pert_effective,
        "cms_dpp_k_effective": cms_dpp_k_effective,
        "cms_regression": str(getattr(args, "cms_regression", "ridge")),
        "cms_half_pert": bool(getattr(args, "cms_half_pert", False)),
        "manuscript_caution": "Do not use an older microstate-only run as a proxy for full-pool CMS-LIME; label ablations explicitly and re-run the protocol on the intended configuration.",
        "lime_segment_stability_dagger": True,
        "lime_segment_stability_rule": "K_eff = min(K_cfg, n_regions) then min(..., n_regions-1) for lime_segment when n_regions>=2",
        "include_table_leader_text": bool(getattr(args, "include_table_leader_text", False)),
        "no_insertion_moment_match": bool(getattr(args, "no_insertion_moment_match", False)),
        "hybrid_noise_scale": float(getattr(args, "hybrid_noise_scale", 2.0)),
        "stability_not_comparable_methods": sorted(STABILITY_ID_PROTOCOL_NA),
    }
    aggregate(rows, out_dir, bootstrap_B=max(0, int(args.bootstrap_B)), run_meta=run_meta)
    print(
        f"[done] wrote {csv_path}, {curve_csv.name}, "
        f"faithfulness_table.md, faithfulness_aggregate_table.csv, "
        f"faithfulness_table.tex, faithfulness_table_journal.tex, faithfulness_summary.md under {out_dir}"
    )


if __name__ == "__main__":
    main()

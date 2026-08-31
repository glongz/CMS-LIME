#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMS 归因子图：对每个 5s 窗口 (1, n_ch, 1280) 单独保存 PNG。

- Baseline：EEG 分类模型 softmax 概率 [P(Non-seizure), P(Preictal)]
- CMS：后处理过程改进/biomarker_fusion.apply_biomarker_fusion（与 BiomarkerEnhancedPredictor 一致）
- 依据：标志物与窗口的相关系数 confidence（Top-K 排名点线图）；右下角为 **Top-1 基元**
  模板与对齐窗口片段的热图 + 通道均值 z-score 曲线对比。
- 汇总表：默认在 out_dir 根下追加 UTF-8 BOM CSV（cms_attribution_records.csv），
  含每窗 ΔP(Pre)、相对提升%、Top3 基元 r，便于论文统计。

默认：仅当 baseline P(Preictal) >= 0.5 时保存；滑窗 stride=1280。

用法（在项目根目录执行）:
  python 可解释归因示例/plot_cms_attribution_panels.py
  python 可解释归因示例/plot_cms_attribution_panels.py --data_dir "D:\\public_data\\CHBMIT\\1_data_clean\\chb01"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)

# 仓库根目录 + 本目录（导入 model / 同目录 biomarker 模块）
_REPO_ROOT = Path(__file__).resolve().parents[1]
_THIS_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from model.EEGInception_SE import EEGInception

from biomarker_enhanced_prediction_v2 import BiomarkerMatcher

# 后处理融合（与训练管线一致）
import importlib.util

_fusion_path = _REPO_ROOT / "后处理过程改进" / "biomarker_fusion.py"
_spec = importlib.util.spec_from_file_location("biomarker_fusion", str(_fusion_path))
_biomarker_fusion = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_biomarker_fusion)
apply_biomarker_fusion = _biomarker_fusion.apply_biomarker_fusion


def natural_keys(text: str):
    def atoi(t):
        return int(t) if t.isdigit() else t

    return [atoi(c) for c in re.split(r"(\d+)", text)]


def load_biomarker_subset(biomarker_dir: str, max_files: int) -> List[Dict]:
    """加载至多 max_files 个 .npz 标志物（按文件名排序），供匹配与 Top-K 展示。"""
    pattern = os.path.join(biomarker_dir, "*.npz")
    files = sorted(glob.glob(pattern), key=lambda p: natural_keys(os.path.basename(p)))
    if max_files > 0:
        files = files[:max_files]
    out: List[Dict] = []
    for filepath in files:
        try:
            npz = np.load(filepath, allow_pickle=False)
            data = npz["data"]
            meta = json.loads(npz["meta_json"].item())
            out.append(
                {
                    "filepath": filepath,
                    "data": data,
                    "meta": meta,
                    "primitive_type": meta.get("primitive_type"),
                    "primitive_id": meta.get("primitive_id"),
                    "channels": meta.get("channels", []),
                    "time_range": meta.get("time_range", [0, 0]),
                    "sample_index": meta.get("sample_index"),
                    "shape": data.shape,
                }
            )
        except Exception:
            continue
    return out


def load_model_robust(model_path: str, n_chans: int, n_classes: int, device: torch.device):
    """与 cms_lime_chb_analysis_final 一致：EEGInception + strict=False；失败则 SimpleEEGModel。"""
    import torch.nn as nn

    class SimpleEEGModel(nn.Module):
        def __init__(self, n_channels: int, n_timepoints: int = 1280, n_classes: int = 2):
            super().__init__()
            self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(32)
            self.fc1 = nn.Linear(256 * 32, 512)
            self.fc2 = nn.Linear(512, n_classes)
            self.dropout = nn.Dropout(0.5)
            self.n_channels = n_channels
            self.n_timepoints = n_timepoints

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(0)
            if x.shape[1] != self.n_channels or x.shape[2] != self.n_timepoints:
                raise ValueError(f"Expected (batch,{self.n_channels},{self.n_timepoints}), got {x.shape}")
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = self.pool(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return torch.softmax(x, dim=1)

    if not model_path or not os.path.exists(model_path):
        m = SimpleEEGModel(n_channels=n_chans)
        m.to(device)
        m.eval()
        return m

    model = EEGInception(input_time=1280, fs=256, ncha=n_chans, n_classes=n_classes)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    except Exception:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


class EEGModelWrapper:
    """(batch, n_ch, 1280) -> softmax 概率。"""

    def __init__(self, model: torch.nn.Module, device: torch.device, n_chans: int):
        self.model = model
        self.device = device
        self.n_chans = n_chans

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                # mmap / 只读视图会导致 PyTorch 警告；强制可写副本
                X = np.array(X, dtype=np.float32, copy=True, order="C")
                if X.ndim == 2:
                    if X.shape[0] != self.n_chans or X.shape[1] != 1280:
                        raise ValueError(f"Expected ({self.n_chans},1280), got {X.shape}")
                    X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)
                elif X.ndim == 3:
                    if X.shape[-2] != self.n_chans or X.shape[-1] != 1280:
                        raise ValueError(f"Expected (batch,{self.n_chans},1280), got {X.shape}")
                    X_tensor = torch.from_numpy(X).to(self.device)
                else:
                    raise ValueError(f"Unexpected shape {X.shape}")
            else:
                X_tensor = X.to(self.device)
            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(0)
            outputs = self.model(X_tensor)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            o = outputs.float()
            if o.shape[-1] > 1:
                row_sums = o.sum(dim=-1, keepdim=True)
                looks_prob = bool(
                    (o >= 0).all()
                    and (o <= 1.0 + 1e-3).all()
                    and (row_sums - 1.0).abs().max() < 0.02
                )
            else:
                looks_prob = bool((o >= 0).all() and (o <= 1.0 + 1e-3).all())
            if looks_prob:
                probabilities = o / (row_sums + 1e-12)
            else:
                probabilities = torch.softmax(o, dim=1)
            # 数值下溢时 conf 会看似恒为 0；钳制不改变排序，但使曲线/AUC 可读
            eps = 1e-7
            probabilities = torch.clamp(probabilities, min=eps, max=1.0 - eps)
            s = probabilities.sum(dim=-1, keepdim=True)
            probabilities = probabilities / s
            return probabilities.cpu().numpy()


def resolve_model_path(explicit: Optional[str]) -> Tuple[str, bool]:
    if explicit and os.path.isfile(explicit):
        return explicit, True
    candidates = [
        Path(r"D:\public_data\CHBMIT\weight\eeginception+se"),
        Path(r"D:\public_data\CHBMIT\weight"),
    ]
    for d in candidates:
        if d.is_dir():
            pths = sorted([p for p in d.glob("*.pth")], key=lambda p: natural_keys(p.name))
            if pths:
                return str(pths[0]), True
        if d.suffix in (".pth", ".pt") and d.is_file():
            return str(d), True
    return "", False


def load_long_npy(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    return np.asarray(arr)


def iter_windows(
    data: np.ndarray, window: int, stride: int
) -> Tuple[np.ndarray, int, int]:
    """
    data: (1, C, T) 或 (C, T)
    yield (window_slice (1,C,window), start, end)
    """
    if data.ndim == 3 and data.shape[0] == 1:
        x = data[0]  # (C, T)
    elif data.ndim == 2:
        x = data
    else:
        raise ValueError(f"Expected (1,C,T) or (C,T), got {data.shape}")
    c, t = x.shape
    for start in range(0, t - window + 1, stride):
        end = start + window
        w = x[:, start:end]
        yield w[np.newaxis, :, :], start, end


def alignment_slices_for_row(
    window: np.ndarray, row: Dict[str, Any]
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    按 BiomarkerMatcher._check_biomarker_match 的通道/长度规则，用 row 中的 best_start、window_len
    取出与打分一致的基元模板 proto 与窗口对齐片段 patch，形状均为 (n_sub_ch, L)。
    """
    biomarker = row.get("biomarker")
    if biomarker is None or "data" not in biomarker:
        return None
    sample_data = np.asarray(window, dtype=np.float64)
    if sample_data.ndim == 2:
        sample_data = sample_data[np.newaxis, :, :]
    if sample_data.ndim != 3:
        return None

    biomarker_data = np.asarray(biomarker["data"], dtype=np.float64)
    biomarker_channels = biomarker.get("channels") or []
    time_range = biomarker.get("time_range", [0, 0])

    if time_range[1] - time_range[0] > 0:
        window_len = int(time_range[1] - time_range[0])
    else:
        window_len = int(biomarker_data.shape[1])

    if biomarker_channels:
        channel_indices = [i for i, ch in enumerate(biomarker_channels) if ch < sample_data.shape[0]]
        if not channel_indices:
            return None
        biomarker_slice = biomarker_data[channel_indices, :]
        if biomarker_slice.shape[1] != window_len:
            window_len = int(biomarker_slice.shape[1])
    else:
        channel_indices = list(np.arange(sample_data.shape[0]))
        biomarker_slice = biomarker_data
        if biomarker_slice.ndim != 2:
            return None
        window_len = int(biomarker_slice.shape[1])

    n_sample_time = int(sample_data.shape[2])
    if window_len > n_sample_time:
        return None

    try:
        bs = int(round(float(row.get("best_start", 0))))
    except (TypeError, ValueError):
        bs = 0
    bs = max(0, min(bs, max(0, n_sample_time - 1)))
    wl = min(window_len, n_sample_time - bs)
    if wl <= 0:
        return None

    biomarker_slice = np.asarray(biomarker_slice[:, :wl], dtype=np.float64)
    sample_slice = sample_data[:, channel_indices, bs : bs + wl]
    patch = np.asarray(sample_slice[0], dtype=np.float64)
    proto = biomarker_slice
    if proto.shape != patch.shape:
        mc = min(proto.shape[0], patch.shape[0])
        mt = min(proto.shape[1], patch.shape[1])
        proto = proto[:mc, :mt]
        patch = patch[:mc, :mt]
    meta = {
        "best_start": bs,
        "window_len": wl,
        "n_ch": proto.shape[0],
    }
    return proto, patch, meta


def _pooled_vlim(a: np.ndarray, b: np.ndarray, pct: Tuple[float, float] = (3.0, 97.0)) -> Tuple[float, float]:
    x = np.concatenate([a.ravel(), b.ravel()])
    x = x[np.isfinite(x)]
    if x.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(x, pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if lo >= hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    return float(lo), float(hi)


def plot_top1_primitive_panel(
    fig: plt.Figure,
    gs_cell,
    window: np.ndarray,
    top_rows: List[Dict],
    stem: str,
    start: int,
    end: int,
    matched_count: int,
    boost: Dict[str, Any],
    show_footer: bool = True,
) -> None:
    """右下角：相似度最高基元的模板 vs 对齐窗口片段（热图 + z-score 均值曲线）。"""
    br = gs_cell.subgridspec(2, 1, height_ratios=[1.15, 0.85], hspace=0.42)
    ax_hm = fig.add_subplot(br[0, 0])
    ax_ln = fig.add_subplot(br[1, 0])

    if not top_rows:
        ax_hm.axis("off")
        ax_ln.axis("off")
        ax_hm.text(0.5, 0.5, "No biomarkers", ha="center", va="center", transform=ax_hm.transAxes)
        return

    top = top_rows[0]
    pid = str(top.get("primitive_id", "?"))
    ptp = str(top.get("primitive_type", "?"))
    r = float(top.get("correlation", top.get("confidence", 0.0)))
    aligned = alignment_slices_for_row(window, top)

    if aligned is None:
        ax_hm.axis("off")
        ax_ln.axis("off")
        msg = f"Top-1: {pid} ({ptp})\nr = {r:.3f}\n(无法重建对齐切片，见左侧条形图)"
        ax_hm.text(0.5, 0.5, msg, ha="center", va="center", transform=ax_hm.transAxes, fontsize=9)
        return

    proto, patch, meta = aligned
    n_ch, L = proto.shape
    lo, hi = _pooled_vlim(proto, patch)

    gap = max(2, L // 64)
    sep = np.full((n_ch, gap), np.nan)
    combo = np.hstack([proto, sep, patch])
    im = ax_hm.imshow(combo, aspect="auto", cmap="RdBu_r", vmin=lo, vmax=hi, interpolation="nearest")
    ax_hm.axvline(L + gap / 2 - 0.5, color="white", linewidth=2.0)
    ax_hm.set_ylabel("sub-ch index", fontsize=8)
    ax_hm.set_xticks([L // 2, L + gap + L // 2])
    ax_hm.set_xticklabels(["Primitive\n(template)", "Window\n(aligned)"], fontsize=8)
    ax_hm.set_title(
        f"Top-1 primitive  |  {pid}  ({ptp})  |  r={r:.3f}  |  t=[{meta['best_start']},{meta['best_start']+meta['window_len']})",
        fontsize=9,
        fontweight="bold",
    )
    cb = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Amplitude (recording units)", fontsize=7)

    def _z(x: np.ndarray) -> np.ndarray:
        m = float(np.nanmean(x))
        s = float(np.nanstd(x))
        if s < 1e-9:
            return np.zeros_like(x)
        return (x - m) / s

    t_axis = np.arange(L)
    p1 = np.nanmean(_z(proto), axis=0)
    p2 = np.nanmean(_z(patch), axis=0)
    ax_ln.plot(t_axis, p1, color="#2980B9", linewidth=1.6, label="Template (z/mean over ch)")
    ax_ln.plot(t_axis, p2, color="#E67E22", linewidth=1.6, linestyle="--", label="Window patch (z/mean over ch)")
    ax_ln.fill_between(t_axis, p1, p2, color="gray", alpha=0.12)
    ax_ln.set_xlabel("Time (samples within patch)", fontsize=8)
    ax_ln.set_ylabel("z-scored mean", fontsize=8)
    ax_ln.set_title("Shape agreement (channel-mean z-score)", fontsize=8, style="italic")
    ax_ln.legend(loc="upper right", fontsize=7, framealpha=0.92)
    ax_ln.grid(True, alpha=0.25)

    if show_footer:
        foot = (
            f"{stem}  win [{start},{end})  |  matched={matched_count}  "
            f"n_fusion={boost.get('n_matches')}  max_conf={float(boost.get('max_confidence', 0)):.3f}"
        )
        ax_ln.text(0.5, -0.28, foot, transform=ax_ln.transAxes, ha="center", fontsize=7, color="#555")


def collect_match_scores(
    matcher: BiomarkerMatcher, window: np.ndarray, topk: int
) -> Tuple[List[Dict], List[Dict]]:
    """对所有已加载标志物跑 _check_biomarker_match；返回 (matched_only, topk_all_by_conf)。"""
    scored: List[Dict] = []
    for b in matcher.biomarkers:
        info = matcher._check_biomarker_match(b, window, channels=None)
        info = dict(info)
        info["primitive_id"] = b.get("primitive_id")
        info["primitive_type"] = b.get("primitive_type")
        scored.append(info)
    scored.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    top = scored[:topk]
    matched = [d for d in scored if d.get("matched")]
    return matched, top


RECORD_FIELDNAMES = [
    "stem",
    "win_start",
    "win_end",
    "png_relpath",
    "p_inter_base",
    "p_pre_base",
    "p_inter_cms",
    "p_pre_cms",
    "delta_pre",
    "rel_boost_pre_pct",
    "n_matches",
    "max_confidence",
    "boost_strength",
    "fusion_applied",
    "top1_primitive_id",
    "top1_correlation",
    "top2_primitive_id",
    "top2_correlation",
    "top3_primitive_id",
    "top3_correlation",
]


def _rel_boost_pct(p_base: float, p_cms: float) -> str:
    if p_base <= 1e-8:
        return ""
    return f"{100.0 * (p_cms - p_base) / p_base:.4f}"


def build_panel_record(
    stem: str,
    start: int,
    end: int,
    png_rel: str,
    base_probs: np.ndarray,
    cms_probs: np.ndarray,
    boost: Dict[str, Any],
    top_rows: List[Dict],
) -> Dict[str, Any]:
    pb0, pb1 = float(base_probs[0]), float(base_probs[1])
    pc0, pc1 = float(cms_probs[0]), float(cms_probs[1])
    d1 = pc1 - pb1
    fusion_on = float(boost.get("boost_strength", 0.0) or 0.0) > 0
    row: Dict[str, Any] = {
        "stem": stem,
        "win_start": start,
        "win_end": end,
        "png_relpath": png_rel.replace("\\", "/"),
        "p_inter_base": f"{pb0:.6f}",
        "p_pre_base": f"{pb1:.6f}",
        "p_inter_cms": f"{pc0:.6f}",
        "p_pre_cms": f"{pc1:.6f}",
        "delta_pre": f"{d1:.6f}",
        "rel_boost_pre_pct": _rel_boost_pct(pb1, pc1),
        "n_matches": int(boost.get("n_matches", 0)),
        "max_confidence": f"{float(boost.get('max_confidence', 0.0)):.6f}",
        "boost_strength": f"{float(boost.get('boost_strength', 0.0)):.6f}",
        "fusion_applied": int(fusion_on),
    }
    for i, key in enumerate(
        [("top1_primitive_id", "top1_correlation"), ("top2_primitive_id", "top2_correlation"), ("top3_primitive_id", "top3_correlation")]
    ):
        if i < len(top_rows):
            r = top_rows[i]
            row[key[0]] = str(r.get("primitive_id", ""))
            row[key[1]] = f"{float(r.get('correlation', r.get('confidence', 0.0))):.6f}"
        else:
            row[key[0]] = ""
            row[key[1]] = ""
    return row


def append_record_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=RECORD_FIELDNAMES, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RECORD_FIELDNAMES})


def write_effectiveness_summary(summary_path: Path, stats: Dict[str, float], pre_threshold: float) -> None:
    """输出总统计文件，用有效率百分比表达 CMS 提升效果。"""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    scanned = int(stats.get("scanned_windows", 0))
    considered = int(stats.get("considered_windows", 0))
    improved = int(stats.get("improved_windows", 0))
    degraded = int(stats.get("degraded_windows", 0))
    unchanged = int(stats.get("unchanged_windows", 0))
    fusion_active = int(stats.get("fusion_active_windows", 0))
    delta_sum = float(stats.get("delta_pre_sum", 0.0))

    def _pct(num: int, den: int) -> float:
        return (100.0 * num / den) if den > 0 else 0.0

    eff_pct = _pct(improved, considered)
    deg_pct = _pct(degraded, considered)
    fusion_pct = _pct(fusion_active, considered)
    mean_delta = (delta_sum / considered) if considered > 0 else 0.0

    lines = [
        "# CMS Effectiveness Summary",
        "",
        f"- baseline threshold (P(Preictal)): `{pre_threshold:.4f}`",
        f"- scanned windows: `{scanned}`",
        f"- considered windows (baseline >= threshold): `{considered}`",
        f"- improved windows (CMS P(Pre) > baseline): `{improved}`",
        f"- degraded windows (CMS P(Pre) < baseline): `{degraded}`",
        f"- unchanged windows: `{unchanged}`",
        f"- fusion active windows: `{fusion_active}`",
        "",
        "## Key Percentages",
        f"- **CMS effectiveness rate** = improved / considered = `{eff_pct:.2f}%`",
        f"- degradation rate = degraded / considered = `{deg_pct:.2f}%`",
        f"- fusion activation rate = fusion_active / considered = `{fusion_pct:.2f}%`",
        f"- mean ΔP(Preictal) on considered windows = `{mean_delta:+.6f}`",
        "",
        "Definition: effective window means CMS raises P(Preictal) compared with baseline.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_one_panel(
    out_path: Path,
    stem: str,
    start: int,
    end: int,
    window: np.ndarray,
    base_probs: np.ndarray,
    cms_probs: np.ndarray,
    boost: Dict[str, Any],
    top_rows: List[Dict],
    matched_count: int,
) -> None:
    """单窗 PNG：分组柱状 + 概率迁移 + Top-K 排名点线图 + Top-1 基元对齐可视化。"""
    base_probs = np.asarray(base_probs, dtype=float).reshape(2)
    cms_probs = np.asarray(cms_probs, dtype=float).reshape(2)
    pb0, pb1 = float(base_probs[0]), float(base_probs[1])
    pc0, pc1 = float(cms_probs[0]), float(cms_probs[1])
    delta_pre = pc1 - pb1
    rel_pct = (100.0 * delta_pre / pb1) if pb1 > 1e-8 else float("nan")

    fig = plt.figure(figsize=(14, 9), constrained_layout=False)
    fig.patch.set_facecolor("#FAFAFA")
    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.22],
        width_ratios=[1.15, 1.0],
        hspace=0.28,
        wspace=0.28,
        left=0.07,
        right=0.97,
        top=0.90,
        bottom=0.06,
    )

    classes = ["Non-seizure\n(Interictal)", "Preictal"]
    colors_ns, colors_pre = "#2E8B57", "#C0392B"
    x = np.arange(2)
    w = 0.36

    # ----- 上左：分组柱状（Baseline vs CMS） -----
    ax_bar = fig.add_subplot(gs[0, 0])
    bars1 = ax_bar.bar(
        x - w / 2,
        base_probs,
        width=w,
        label="Baseline",
        color=[colors_ns, colors_pre],
        alpha=0.88,
        edgecolor="#2C3E50",
        linewidth=0.7,
    )
    bars2 = ax_bar.bar(
        x + w / 2,
        cms_probs,
        width=w,
        label="CMS (fusion)",
        color=[colors_ns, colors_pre],
        alpha=0.55,
        edgecolor="#2C3E50",
        linewidth=0.7,
        hatch="//",
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(classes, fontsize=10)
    ax_bar.set_ylim(0, 1.08)
    ax_bar.set_ylabel("Probability", fontsize=10)
    ax_bar.set_title("Class probabilities: Baseline vs CMS", fontsize=11, fontweight="bold")
    ax_bar.legend(loc="upper right", framealpha=0.95)
    ax_bar.axhline(0.5, color="#7F8C8D", linestyle=":", linewidth=0.8, alpha=0.8)
    for bars in (bars1, bars2):
        for rect in bars:
            h = rect.get_height()
            ax_bar.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + 0.02,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # ----- 上右：Preictal 一维标尺 + 双类概率迁移（直观对比） -----
    ax_delta = fig.add_subplot(gs[0, 1])
    ax_delta.set_facecolor("#FAFAFA")
    ax_delta.set_xlim(0, 1)
    ax_delta.set_ylim(0, 1)
    ax_delta.axhspan(0.52, 0.68, facecolor="#E8EEF2", alpha=0.9, zorder=0)
    track_y = 0.60
    ax_delta.hlines(track_y, 0, 1, colors="#95A5A6", linewidth=1.5, zorder=1)
    ax_delta.scatter([pb1], [track_y], s=130, c="#27AE60", zorder=3, edgecolors="black", linewidths=0.85, label="Baseline")
    ax_delta.scatter([pc1], [track_y], s=130, c="#E74C3C", zorder=3, edgecolors="black", linewidths=0.85, label="CMS")
    ax_delta.annotate(
        "",
        xy=(pc1, track_y),
        xytext=(pb1, track_y),
        arrowprops=dict(arrowstyle="<->", color="#2C3E50", lw=1.6, shrinkA=6, shrinkB=6),
    )
    ax_delta.text(pb1, 0.72, f"{pb1:.3f}", ha="center", fontsize=9, fontweight="bold", color="#1E8449")
    ax_delta.text(pc1, 0.72, f"{pc1:.3f}", ha="center", fontsize=9, fontweight="bold", color="#C0392B")
    ax_delta.set_xlabel("P(Preictal)", fontsize=10)
    ax_delta.set_yticks([])
    ax_delta.set_title("Preictal shift (read on probability axis)", fontsize=11, fontweight="bold")
    ax_delta.legend(loc="upper center", ncol=2, fontsize=8, framealpha=0.95)
    rel_line = f"Rel. vs baseline: {rel_pct:+.2f}%" if np.isfinite(rel_pct) else "Rel. boost: — (baseline P(Pre)≈0)"
    ax_delta.text(
        0.5,
        0.88,
        f"ΔP(Pre) = {delta_pre:+.4f}   |   {rel_line}",
        transform=ax_delta.transAxes,
        ha="center",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#FCF3CF", alpha=0.95),
    )
    d0 = pc0 - pb0
    ax_delta.text(
        0.5,
        0.12,
        f"ΔP(Non-seizure) = {d0:+.4f}\nΔP(Preictal)   = {delta_pre:+.4f}",
        transform=ax_delta.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#BDC3C7", alpha=0.95),
    )
    ax_delta.set_xticks(np.linspace(0, 1, 11))

    # ----- 下左：Top-K 相似度（排名点线图；非柱状） -----
    ax2 = fig.add_subplot(gs[1, 0])
    if top_rows:
        labels = []
        for i, r in enumerate(top_rows, 1):
            pid = str(r.get("primitive_id", "?"))[:20]
            ptp = str(r.get("primitive_type", "?"))[:10]
            labels.append(f"{i:02d}. {pid} ({ptp})")
        corr = [float(r.get("correlation", r.get("confidence", 0.0))) for r in top_rows]
        matched_flags = [bool(r.get("matched")) for r in top_rows]
        y_pos = np.arange(len(labels))
        for yi, rv, m in zip(y_pos, corr, matched_flags):
            col = "#16A085" if m else "#A0A7AE"
            ax2.hlines(yi, 0.0, rv, color=col, linewidth=2.2, alpha=0.95)
            ax2.plot(rv, yi, marker="o", markersize=7, color=col, markeredgecolor="#2C3E50")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=7)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("Pearson correlation r (non-negative; match if r>0.5)", fontsize=9)
        ax2.set_title(f"Top-{len(top_rows)} primitive similarity ranking (dot-line)", fontsize=10, fontweight="bold")
        ax2.axvline(0.5, color="#E74C3C", linestyle="--", linewidth=1.1, alpha=0.85)
        ax2.grid(axis="x", alpha=0.20, linestyle=":")
        leg = [
            Line2D([0], [0], marker="o", color="#16A085", linestyle="-", label="matched (r>0.5)"),
            Line2D([0], [0], marker="o", color="#A0A7AE", linestyle="-", label="below thr"),
            Line2D([0], [0], color="#E74C3C", linestyle="--", label="thr=0.5"),
        ]
        ax2.legend(handles=leg, loc="lower right", fontsize=7, framealpha=0.92)
        for yi, rv in zip(y_pos, corr):
            ax2.text(min(rv + 0.02, 0.98), yi, f"{rv:.2f}", va="center", fontsize=7, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No biomarkers loaded", ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.set_axis_off()

    # ----- 下右：相似度最高基元 — 模板 vs 对齐片段（热图 + 曲线） -----
    plot_top1_primitive_panel(fig, gs[1, 1], window, top_rows, stem, start, end, matched_count, boost)

    fig.suptitle(
        f"CMS attribution  |  {stem}  |  P(Pre): {pb1:.3f} → {pc1:.3f}  |  Δ={delta_pre:+.4f}",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def process_one_npy(
    npy_path: str,
    matcher: BiomarkerMatcher,
    wrapper: EEGModelWrapper,
    out_root: Path,
    stride: int,
    pre_threshold: float,
    biomarker_weight: float,
    min_biomarker_confidence: float,
    topk: int,
    max_panels_per_file: int,
    records_csv: Optional[Path],
    effect_stats: Dict[str, float],
) -> int:
    stem = Path(npy_path).stem
    out_dir = out_root / stem
    data = load_long_npy(npy_path)
    if data.ndim != 3 or data.shape[0] != 1:
        print(f"[skip] {npy_path}: expected (1,C,T), got {data.shape}")
        return 0
    n_ch = data.shape[1]
    t = data.shape[2]
    if n_ch != wrapper.n_chans:
        print(f"[warn] {npy_path}: channels {n_ch} != model n_chans {wrapper.n_chans}, skip")
        return 0
    if t < 1280:
        print(f"[skip] {npy_path}: T={t} < 1280")
        return 0
    saved = 0
    for window, start, end in iter_windows(data, 1280, stride):
        if max_panels_per_file > 0 and saved >= max_panels_per_file:
            break
        effect_stats["scanned_windows"] = effect_stats.get("scanned_windows", 0) + 1
        window = np.array(window, dtype=np.float32, copy=True, order="C")
        probs = wrapper.predict_proba(window)[0]
        p_pre = float(probs[1])
        if p_pre < pre_threshold:
            continue
        effect_stats["considered_windows"] = effect_stats.get("considered_windows", 0) + 1
        matched, top_scored = collect_match_scores(matcher, window, topk)
        n_m = len(matched)
        max_c = max((float(m["confidence"]) for m in matched), default=0.0)
        cms_probs, boost = apply_biomarker_fusion(
            probs,
            n_m,
            max_c,
            biomarker_weight,
            min_biomarker_confidence=min_biomarker_confidence,
        )
        pb, pc = float(probs[1]), float(cms_probs[1])
        delta_pre = pc - pb
        effect_stats["delta_pre_sum"] = effect_stats.get("delta_pre_sum", 0.0) + delta_pre
        if delta_pre > 1e-10:
            effect_stats["improved_windows"] = effect_stats.get("improved_windows", 0) + 1
        elif delta_pre < -1e-10:
            effect_stats["degraded_windows"] = effect_stats.get("degraded_windows", 0) + 1
        else:
            effect_stats["unchanged_windows"] = effect_stats.get("unchanged_windows", 0) + 1
        if float(boost.get("boost_strength", 0.0) or 0.0) > 0:
            effect_stats["fusion_active_windows"] = effect_stats.get("fusion_active_windows", 0) + 1
        fname = f"win_{start:08d}_pre{pb:.3f}_cms{pc:.3f}.png"
        out_png = out_dir / fname
        plot_one_panel(
            out_png,
            stem,
            start,
            end,
            window,
            probs.astype(float),
            cms_probs.astype(float),
            boost,
            top_scored,
            n_m,
        )
        if records_csv is not None:
            try:
                png_rel = str(out_png.relative_to(out_root))
            except ValueError:
                png_rel = str(out_png)
            append_record_csv(
                records_csv,
                build_panel_record(
                    stem,
                    start,
                    end,
                    png_rel,
                    probs.astype(float),
                    cms_probs.astype(float),
                    boost,
                    top_scored,
                ),
            )
        saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="CMS attribution panels per 5s window")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\public_data\CHBMIT\1_data_clean\chb01",
        help="Directory containing .npy long recordings",
    )
    parser.add_argument(
        "--biomarker_dir",
        type=str,
        default=r"D:\2025_important_projects\data\chbmit_biomarkers\shared",
        help="Directory of biomarker .npz files",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Optional explicit .pth; empty = auto-detect under CHBMIT/weight",
    )
    parser.add_argument("--chb_patient", type=int, default=0, help="1..99: set data_dir to .../1_data_clean/chb%%02d")
    parser.add_argument(
        "--loocv_model_name",
        type=str,
        default="",
        help="e.g. eeginception: resolve .pth from LOOCV tree (requires --chb_patient); overrides --model_path",
    )
    parser.add_argument("--loocv_select_loop", type=int, default=1)
    parser.add_argument("--loocv_sync_data", action="store_true", help="LOOCV remapped chb for data (12->chb13)")
    parser.add_argument("--loocv_weight_root", type=str, default="", help="Optional LOOCV root (e.g. E:\\...\\LOOCV)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(_THIS_DIR / "cms_attribution_out"),
        help="Output directory root",
    )
    parser.add_argument("--stride", type=int, default=1280)
    parser.add_argument("--pre_threshold", type=float, default=0.5)
    parser.add_argument("--biomarker_weight", type=float, default=0.5)
    parser.add_argument("--min_biomarker_confidence", type=float, default=0.6)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument(
        "--max_biomarkers_scan",
        type=int,
        default=2000,
        help="Max number of .npz biomarker files to load (sorted by name); 0 = all",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Process at most this many .npy files (sorted by name); 0 = all",
    )
    parser.add_argument(
        "--max_panels_per_file",
        type=int,
        default=0,
        help="Max PNG panels per recording after pre_threshold filter; 0 = unlimited",
    )
    parser.add_argument(
        "--records_csv",
        type=str,
        default="",
        help="Append per-window metrics CSV (UTF-8 BOM). Empty = <out_dir>/cms_attribution_records.csv",
    )
    parser.add_argument(
        "--no_records",
        action="store_true",
        help="Do not write the summary CSV",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        default="",
        help="Overall effectiveness summary file. Empty = <out_dir>/cms_effectiveness_summary.md",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from chb_loocv_weights import apply_loocv_model_cli
    from chb_paths import chb_patient_data_dir

    if int(getattr(args, "chb_patient", 0) or 0) > 0:
        p = int(args.chb_patient)
        if not 1 <= p <= 99:
            raise SystemExit("--chb_patient must be 1..99")
        args.data_dir = str(chb_patient_data_dir(p))
    apply_loocv_model_cli(args)

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir not found: {data_dir}")

    npy_files = sorted(data_dir.glob("*.npy"), key=lambda p: natural_keys(p.name))
    if not npy_files:
        raise SystemExit(f"No .npy under {data_dir}")
    if args.max_files > 0:
        npy_files = npy_files[: args.max_files]

    # 用第一个文件推断通道数
    sample = load_long_npy(str(npy_files[0]))
    if sample.ndim != 3 or sample.shape[0] != 1:
        raise SystemExit(f"First file shape must be (1,C,T), got {sample.shape}")
    n_chans = int(sample.shape[1])

    mp, ok = resolve_model_path(args.model_path.strip() or None)
    if not ok:
        print("[warn] No model weights found; using untrained SimpleEEGModel fallback.")
    model = load_model_robust(mp, n_chans=n_chans, n_classes=2, device=device)
    wrapper = EEGModelWrapper(model, device, n_chans=n_chans)

    max_bm = args.max_biomarkers_scan
    if max_bm <= 0:
        matcher = BiomarkerMatcher(args.biomarker_dir, filtered_biomarkers=None)
        print(f"[info] Loaded all biomarkers: {len(matcher.biomarkers)}")
    else:
        subset = load_biomarker_subset(args.biomarker_dir, max_bm)
        matcher = BiomarkerMatcher(args.biomarker_dir, filtered_biomarkers=subset)
        print(f"[info] Loaded biomarker subset: {len(matcher.biomarkers)} (cap={max_bm})")

    out_root = Path(args.out_dir)
    if args.no_records:
        records_csv: Optional[Path] = None
    else:
        rc = (args.records_csv or "").strip()
        records_csv = Path(rc) if rc else (out_root / "cms_attribution_records.csv")
    summary_file = Path(args.summary_file) if (args.summary_file or "").strip() else (out_root / "cms_effectiveness_summary.md")
    effect_stats: Dict[str, float] = {
        "scanned_windows": 0,
        "considered_windows": 0,
        "improved_windows": 0,
        "degraded_windows": 0,
        "unchanged_windows": 0,
        "fusion_active_windows": 0,
        "delta_pre_sum": 0.0,
    }
    total_saved = 0
    for npy in npy_files:
        n = process_one_npy(
            str(npy),
            matcher,
            wrapper,
            out_root,
            stride=args.stride,
            pre_threshold=args.pre_threshold,
            biomarker_weight=args.biomarker_weight,
            min_biomarker_confidence=args.min_biomarker_confidence,
            topk=args.topk,
            max_panels_per_file=args.max_panels_per_file,
            records_csv=records_csv,
            effect_stats=effect_stats,
        )
        print(f"[done] {npy.name}: saved {n} panels -> {out_root / npy.stem}")
        total_saved += n
    print(f"[summary] total PNG saved: {total_saved} -> {out_root}")
    if records_csv is not None:
        print(f"[summary] records CSV (append): {records_csv}")
    write_effectiveness_summary(summary_file, effect_stats, pre_threshold=args.pre_threshold)
    print(f"[summary] effectiveness summary: {summary_file}")


if __name__ == "__main__":
    main()

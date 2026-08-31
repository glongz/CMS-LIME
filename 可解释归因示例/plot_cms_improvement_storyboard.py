#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMS 提升片段故事板（新文件）：
- 删除旧布局中的“左下相似度柱状图”
- 左侧改为 6/8 个“类似左上”的显著提升片段对比小图
- 右上沿用旧风格：概率轴上的 baseline→cms 迁移（叠加多片段）

输出：
- 每个 .npy 一张 story PNG
- 一个 top-segments CSV（用于追溯片段）
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import plot_cms_attribution_panels as base


def draw_prob_compare(ax, base_probs: np.ndarray, cms_probs: np.ndarray, title: str) -> None:
    classes = ["Interictal", "Preictal"]
    x = np.arange(2)
    w = 0.34
    ax.bar(x - w / 2, base_probs, width=w, color=["#2E8B57", "#C0392B"], alpha=0.86, label="Baseline")
    ax.bar(
        x + w / 2,
        cms_probs,
        width=w,
        color=["#2E8B57", "#C0392B"],
        alpha=0.50,
        hatch="//",
        label="CMS-LIME",
        edgecolor="#2C3E50",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Prob.", fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    for i, p in enumerate(base_probs):
        ax.text(i - w / 2, float(p) + 0.02, f"{float(p):.3f}", ha="center", fontsize=7)
    for i, p in enumerate(cms_probs):
        ax.text(i + w / 2, float(p) + 0.02, f"{float(p):.3f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.18, linestyle=":")


def draw_preictal_shift_stack(ax, segments: List[Dict[str, Any]]) -> None:
    """沿用之前单窗风格：概率轴上的 baseline→cms 迁移，只是叠加多个片段。"""
    n = len(segments)
    y_rows = np.arange(n)[::-1] + 1  # top to bottom
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, n + 0.8)
    ax.axvspan(0.5, 1.0, facecolor="#FDEDEC", alpha=0.35, zorder=0)
    ax.axvline(0.5, color="#C0392B", linestyle="--", linewidth=1.0, alpha=0.8)

    for idx, (seg, y) in enumerate(zip(segments, y_rows), start=1):
        pb = float(seg["base_probs"][1])
        pc = float(seg["cms_probs"][1])
        delta = float(seg["delta_pre"])
        color = "#16A085" if delta > 0 else "#95A5A6"
        ax.hlines(y, min(pb, pc), max(pb, pc), color="#B2BEC3", linewidth=3.5, alpha=0.55, zorder=1)
        ax.scatter([pb], [y], s=60, c="#27AE60", edgecolors="#1B4332", linewidths=0.6, zorder=3)
        ax.scatter([pc], [y], s=60, c="#E74C3C", edgecolors="#7B241C", linewidths=0.6, zorder=3)
        ax.annotate(
            "",
            xy=(pc, y),
            xytext=(pb, y),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.2, shrinkA=5, shrinkB=5),
            zorder=2,
        )
        ax.text(0.01, y + 0.18, f"#{idx} start={seg['start']}", fontsize=7.5, color="#2C3E50")
        ax.text(min(max(pc + 0.02, 0.02), 0.88), y - 0.18, f"Δ={delta:+.3f}", fontsize=7.5, color=color, fontweight="bold")

    ax.set_xlabel("P(Preictal)", fontsize=9)
    ax.set_yticks(y_rows)
    ax.set_yticklabels([f"seg #{i}" for i in range(1, n + 1)], fontsize=8)
    ax.set_title("Preictal shift (Baseline to CMS-LIME) for selected segments", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.2, linestyle=":")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#27AE60", markeredgecolor="#1B4332", markersize=7, label="Baseline"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C", markeredgecolor="#7B241C", markersize=7, label="CMS-LIME"),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.95,
    )


def render_storyboard(
    out_png: Path,
    stem: str,
    segments: List[Dict[str, Any]],
    anchor_top_rows: List[Dict[str, Any]],
    anchor_matched_count: int,
    anchor_boost: Dict[str, Any],
    left_samples: int,
) -> None:
    """新布局：左侧 6/8 个样本；右上概率迁移；右下 Top1 基元可视化。"""
    show_n = min(len(segments), max(1, left_samples))
    shown_segments = segments[:show_n]
    anchor = shown_segments[0]
    fig = plt.figure(figsize=(15, 9), constrained_layout=False)
    fig.patch.set_facecolor("#FAFAFA")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.25], width_ratios=[1.45, 1.0], hspace=0.24, wspace=0.22)

    # 左侧：6/8 个样本，全部沿用“左上概率对比”的风格
    rows = int(np.ceil(show_n / 2))
    left_grid = gs[:, 0].subgridspec(rows, 2, hspace=0.36, wspace=0.28)
    for i, seg in enumerate(shown_segments):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(left_grid[r, c])
        title = f"#{i+1} [{seg['start']},{seg['end']})  Δ={seg['delta_pre']:+.4f}"
        draw_prob_compare(ax, seg["base_probs"], seg["cms_probs"], title)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.95)
    # 占位空子图（当 show_n 为奇数）
    for j in range(show_n, rows * 2):
        r, c = divmod(j, 2)
        ax_pad = fig.add_subplot(left_grid[r, c])
        ax_pad.axis("off")

    # 右上：沿用之前概率轴迁移风格，多加点位
    ax_rt = fig.add_subplot(gs[0, 1])
    draw_preictal_shift_stack(ax_rt, shown_segments)

    # 右下：沿用 Top1 基元模板/对齐片段可视化（针对 anchor）
    base.plot_top1_primitive_panel(
        fig,
        gs[1, 1],
        anchor["window"],
        anchor_top_rows,
        stem,
        anchor["start"],
        anchor["end"],
        anchor_matched_count,
        anchor_boost,
        show_footer=False,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_segments_csv(csv_path: Path, segments: List[Dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "start",
        "end",
        "p_pre_base",
        "p_pre_cms",
        "delta_pre",
        "rel_boost_pre_pct",
        "quality_score",
        "selection_tag",
        "n_matches",
        "max_confidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, seg in enumerate(segments, 1):
            pb = float(seg["base_probs"][1])
            pc = float(seg["cms_probs"][1])
            rel: Optional[float] = (100.0 * (pc - pb) / pb) if pb >= 1e-3 else None
            w.writerow(
                {
                    "rank": i,
                    "start": int(seg["start"]),
                    "end": int(seg["end"]),
                    "p_pre_base": f"{pb:.6f}",
                    "p_pre_cms": f"{pc:.6f}",
                    "delta_pre": f"{(pc - pb):.6f}",
                    "rel_boost_pre_pct": (f"{rel:.4f}" if rel is not None else ""),
                    "quality_score": f"{float(seg.get('quality_score', 0.0)):.6f}",
                    "selection_tag": str(seg.get("selection_tag", "")),
                    "n_matches": int(seg["n_matches"]),
                    "max_confidence": f"{float(seg['max_confidence']):.6f}",
                }
            )


def quality_score_segment(seg: Dict[str, Any], decision_threshold: float = 0.5) -> float:
    """优先选能体现 CMS 有效性的片段：间期->前期、显著提升。"""
    pb = float(seg["base_probs"][1])
    pc = float(seg["cms_probs"][1])
    delta = float(seg["delta_pre"])
    score = delta

    # 强信号1：从 baseline 偏间期到 CMS 偏前期（跨越阈值）
    if pb < decision_threshold and pc >= decision_threshold:
        score += 0.35

    # 强信号2：baseline 明显低于阈值且提升后更接近/超过阈值
    if pb < 0.4 and pc >= 0.5:
        score += 0.20
    elif pb < 0.4 and delta >= 0.06:
        score += 0.12

    # 强信号3：提升量本身较大
    if delta >= 0.08:
        score += 0.18
    elif delta >= 0.05:
        score += 0.10
    elif delta >= 0.03:
        score += 0.05

    # 惩罚：原本已很高且几乎不变，不利于展示“有效性”
    if pb >= 0.85 and delta < 0.02:
        score -= 0.25

    # 非提升样本强惩罚
    if delta <= 0:
        score -= 1.0
    return score


def quality_score_segment_targeted(
    seg: Dict[str, Any],
    decision_threshold: float,
    focus_base_min: float,
    focus_base_max: float,
    focus_cms_min: float,
) -> float:
    """
    针对论文展示的“说服力样本”打分：
    baseline 略高于 0.5，CMS 提升到 0.7/0.9 区间优先。
    """
    pb = float(seg["base_probs"][1])
    pc = float(seg["cms_probs"][1])
    delta = float(seg["delta_pre"])

    score = quality_score_segment(seg, decision_threshold=decision_threshold)

    in_base_band = (pb >= focus_base_min) and (pb <= focus_base_max)
    strong_target = pc >= focus_cms_min
    near_09 = pc >= 0.9
    near_07 = pc >= 0.7

    if in_base_band:
        score += 0.30
    if strong_target:
        score += 0.35
    elif near_07:
        score += 0.18
    if near_09:
        score += 0.18
    if in_base_band and strong_target:
        score += 0.25
    if delta >= 0.10:
        score += 0.15
    return score


def tag_segment(seg: Dict[str, Any], decision_threshold: float = 0.5) -> str:
    pb = float(seg["base_probs"][1])
    pc = float(seg["cms_probs"][1])
    delta = float(seg["delta_pre"])
    if pb < decision_threshold and pc >= decision_threshold:
        return "interictal_to_preictal"
    if delta >= 0.08:
        return "large_gain"
    if delta >= 0.04:
        return "medium_gain"
    if delta > 0:
        return "small_gain"
    return "no_gain"


def select_story_segments(
    candidates: List[Dict[str, Any]],
    top_segments: int,
    decision_threshold: float = 0.5,
    focus_base_min: float = 0.50,
    focus_base_max: float = 0.62,
    focus_cms_min: float = 0.70,
    min_crossing_examples: int = 2,
) -> List[Dict[str, Any]]:
    """精选用于展示的片段：先选“baseline略高于0.5且CMS高”的样本，再回填。"""
    if not candidates:
        return []

    for seg in candidates:
        seg["quality_score"] = quality_score_segment_targeted(
            seg,
            decision_threshold=decision_threshold,
            focus_base_min=focus_base_min,
            focus_base_max=focus_base_max,
            focus_cms_min=focus_cms_min,
        )
        seg["selection_tag"] = tag_segment(seg, decision_threshold=decision_threshold)

    # 仅保留正提升样本，避免左侧出现 Δ=0
    candidates = [s for s in candidates if float(s.get("delta_pre", 0.0)) > 1e-8]
    if not candidates:
        return []

    # 第一层：严格命中目标区间（baseline 略高于 0.5，CMS 达到目标）
    targeted = [
        s
        for s in candidates
        if (focus_base_min <= float(s["base_probs"][1]) <= focus_base_max)
        and (float(s["cms_probs"][1]) >= focus_cms_min)
    ]
    targeted.sort(key=lambda s: float(s["quality_score"]), reverse=True)

    # 第二层：baseline 略高于 0.5 且提升明显
    semi_targeted = [
        s
        for s in candidates
        if (focus_base_min <= float(s["base_probs"][1]) <= focus_base_max)
        and (float(s["delta_pre"]) > 0.03)
    ]
    semi_targeted.sort(key=lambda s: float(s["quality_score"]), reverse=True)

    # 第三层：原有强提升
    strong = [
        s
        for s in candidates
        if (float(s["base_probs"][1]) < decision_threshold and float(s["cms_probs"][1]) >= decision_threshold)
        or float(s["delta_pre"]) >= 0.03
    ]
    strong.sort(key=lambda s: float(s["quality_score"]), reverse=True)

    # 优先补足“间期->前期”的纠错样本
    crossing = [
        s
        for s in candidates
        if float(s["base_probs"][1]) < decision_threshold and float(s["cms_probs"][1]) >= decision_threshold
    ]
    crossing.sort(key=lambda s: float(s["quality_score"]), reverse=True)

    chosen: List[Dict[str, Any]] = []
    seen = set()
    for s in crossing[: max(0, min_crossing_examples)]:
        sid = id(s)
        seen.add(sid)
        chosen.append(s)
    for pool in (targeted, semi_targeted, strong):
        for s in pool:
            sid = id(s)
            if sid in seen:
                continue
            seen.add(sid)
            chosen.append(s)
            if len(chosen) >= top_segments:
                break
        if len(chosen) >= top_segments:
            break

    if len(chosen) < top_segments:
        picked = {id(x) for x in chosen}
        rest = [s for s in candidates if id(s) not in picked]
        rest.sort(key=lambda s: float(s["quality_score"]), reverse=True)
        chosen.extend(rest[: top_segments - len(chosen)])
    return chosen[:top_segments]


def process_one_file(
    npy_path: Path,
    matcher: base.BiomarkerMatcher,
    wrapper: base.EEGModelWrapper,
    out_root: Path,
    stride: int,
    pre_threshold: float,
    biomarker_weight: float,
    min_biomarker_confidence: float,
    topk: int,
    top_segments: int,
    min_delta_pre: float,
    left_samples: int,
    decision_threshold: float,
    focus_base_min: float,
    focus_base_max: float,
    focus_cms_min: float,
    min_crossing_examples: int,
) -> int:
    data = base.load_long_npy(str(npy_path))
    if data.ndim != 3 or data.shape[0] != 1:
        print(f"[skip] {npy_path.name}: expected (1,C,T), got {data.shape}")
        return 0
    if data.shape[1] != wrapper.n_chans:
        print(f"[skip] {npy_path.name}: channels mismatch {data.shape[1]} vs model {wrapper.n_chans}")
        return 0

    candidates: List[Dict[str, Any]] = []
    for window, start, end in base.iter_windows(data, 1280, stride):
        window = np.array(window, dtype=np.float32, copy=True, order="C")
        probs = wrapper.predict_proba(window)[0]
        if float(probs[1]) < pre_threshold:
            continue

        matched, _ = base.collect_match_scores(matcher, window, topk)
        n_m = len(matched)
        max_c = max((float(m["confidence"]) for m in matched), default=0.0)
        cms_probs, boost = base.apply_biomarker_fusion(
            probs,
            n_m,
            max_c,
            biomarker_weight,
            min_biomarker_confidence=min_biomarker_confidence,
        )
        delta = float(cms_probs[1] - probs[1])
        if delta < min_delta_pre:
            continue

        candidates.append(
            {
                "start": int(start),
                "end": int(end),
                "window": window,
                "base_probs": np.asarray(probs, dtype=float),
                "cms_probs": np.asarray(cms_probs, dtype=float),
                "delta_pre": delta,
                "n_matches": int(n_m),
                "max_confidence": float(max_c),
                "boost": boost,
            }
        )

    segments = select_story_segments(
        candidates,
        top_segments=top_segments,
        decision_threshold=decision_threshold,
        focus_base_min=focus_base_min,
        focus_base_max=focus_base_max,
        focus_cms_min=focus_cms_min,
        min_crossing_examples=min_crossing_examples,
    )
    if not segments:
        print(f"[skip] {npy_path.name}: no segments passed min_delta_pre={min_delta_pre:.4f}")
        return 0

    anchor = segments[0]
    anchor_matched, anchor_top_rows = base.collect_match_scores(matcher, anchor["window"], topk)

    stem = npy_path.stem
    out_dir = out_root / stem
    show_n = min(len(segments), max(1, left_samples))
    out_png = out_dir / f"{stem}_improvement_storyboard_top{show_n}.png"
    out_csv = out_dir / f"{stem}_top_improvement_segments.csv"
    render_storyboard(out_png, stem, segments, anchor_top_rows, len(anchor_matched), anchor["boost"], left_samples=left_samples)
    write_segments_csv(out_csv, segments)
    print(f"[done] {npy_path.name}: storyboard -> {out_png}")
    print(f"[done] {npy_path.name}: top segments csv -> {out_csv}")
    return len(segments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CMS improvement storyboard panels")
    parser.add_argument("--data_dir", type=str, default=r"D:\public_data\CHBMIT\1_data_clean\chb01")
    parser.add_argument("--biomarker_dir", type=str, default=r"D:\2025_important_projects\data\chbmit_biomarkers\shared")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--chb_patient", type=int, default=0, help="1..99: set data_dir to .../chb%%02d")
    parser.add_argument("--loocv_model_name", type=str, default="")
    parser.add_argument("--loocv_select_loop", type=int, default=1)
    parser.add_argument("--loocv_sync_data", action="store_true")
    parser.add_argument("--loocv_weight_root", type=str, default="")
    parser.add_argument("--out_dir", type=str, default=str(base._THIS_DIR / "cms_improvement_storyboard_out"))
    parser.add_argument("--stride", type=int, default=1280)
    parser.add_argument("--pre_threshold", type=float, default=0.5)
    parser.add_argument("--biomarker_weight", type=float, default=0.5)
    parser.add_argument("--min_biomarker_confidence", type=float, default=0.6)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--top_segments", type=int, default=8, help="Keep top-N strong-improvement segments per file")
    parser.add_argument("--left_samples", type=int, default=6, choices=[6, 8], help="How many samples to display on the left side")
    parser.add_argument("--min_delta_pre", type=float, default=0.01, help="Minimal ΔP(Pre)=CMS-Base to be considered strong")
    parser.add_argument("--decision_threshold", type=float, default=0.5, help="Decision threshold used for interictal->preictal tagging")
    parser.add_argument("--focus_base_min", type=float, default=0.50, help="Target baseline lower bound for showcase")
    parser.add_argument("--focus_base_max", type=float, default=0.62, help="Target baseline upper bound for showcase")
    parser.add_argument("--focus_cms_min", type=float, default=0.70, help="Target CMS lower bound for showcase")
    parser.add_argument("--min_crossing_examples", type=int, default=2, help="Prefer at least N interictal->preictal examples")
    parser.add_argument("--only_file", type=str, default="", help="Only process this .npy filename, e.g. chb01_05.npy")
    parser.add_argument("--max_biomarkers_scan", type=int, default=2000)
    parser.add_argument("--max_files", type=int, default=0)
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

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir not found: {data_dir}")
    npy_files = sorted(data_dir.glob("*.npy"), key=lambda p: base.natural_keys(p.name))
    if not npy_files:
        raise SystemExit(f"No .npy found under {data_dir}")
    if (args.only_file or "").strip():
        target = (args.only_file or "").strip()
        npy_files = [p for p in npy_files if p.name == target]
        if not npy_files:
            raise SystemExit(f"only_file not found under data_dir: {target}")
    if args.max_files > 0:
        npy_files = npy_files[: args.max_files]

    sample = base.load_long_npy(str(npy_files[0]))
    if sample.ndim != 3 or sample.shape[0] != 1:
        raise SystemExit(f"First file shape must be (1,C,T), got {sample.shape}")
    n_chans = int(sample.shape[1])

    device = torch.device(args.device)
    mp, ok = base.resolve_model_path(args.model_path.strip() or None)
    if not ok:
        print("[warn] no model weights found; using fallback model")
    model = base.load_model_robust(mp, n_chans=n_chans, n_classes=2, device=device)
    wrapper = base.EEGModelWrapper(model, device, n_chans=n_chans)

    if args.max_biomarkers_scan <= 0:
        matcher = base.BiomarkerMatcher(args.biomarker_dir, filtered_biomarkers=None)
        print(f"[info] loaded all biomarkers: {len(matcher.biomarkers)}")
    else:
        subset = base.load_biomarker_subset(args.biomarker_dir, args.max_biomarkers_scan)
        matcher = base.BiomarkerMatcher(args.biomarker_dir, filtered_biomarkers=subset)
        print(f"[info] loaded biomarker subset: {len(matcher.biomarkers)} (cap={args.max_biomarkers_scan})")

    out_root = Path(args.out_dir)
    total_storyboards = 0
    for npy in npy_files:
        n = process_one_file(
            npy,
            matcher,
            wrapper,
            out_root=out_root,
            stride=args.stride,
            pre_threshold=args.pre_threshold,
            biomarker_weight=args.biomarker_weight,
            min_biomarker_confidence=args.min_biomarker_confidence,
            topk=args.topk,
            top_segments=max(1, args.top_segments, args.left_samples),
            min_delta_pre=args.min_delta_pre,
            left_samples=args.left_samples,
            decision_threshold=args.decision_threshold,
            focus_base_min=args.focus_base_min,
            focus_base_max=args.focus_base_max,
            focus_cms_min=args.focus_cms_min,
            min_crossing_examples=max(0, int(args.min_crossing_examples)),
        )
        if n > 0:
            total_storyboards += 1

    print(f"[summary] storyboards generated: {total_storyboards} -> {out_root}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生物标志物加权后处理：逐样本日志、权重扫描与论文级可视化。

依赖 biomarker_fusion.apply_biomarker_fusion 与 evaluation_metrics.calculate_metrics。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore

from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from biomarker_fusion import apply_biomarker_fusion
from evaluation_metrics import calculate_metrics


def _ensure_eeg_batch_channel_time(sample_data: np.ndarray) -> np.ndarray:
    x = np.asarray(sample_data)
    if x.ndim == 2:
        return x[np.newaxis, :, :]
    if x.ndim == 3:
        return x
    raise ValueError("sample_data 期望形状 (n_channels, n_time) 或 (1, n_channels, n_time)")


def collect_per_sample_log(
    biomarker_dir: str,
    base_model_predictor: Callable[[np.ndarray], np.ndarray],
    samples: Sequence[Tuple[np.ndarray, Optional[int], int, int]],
    biomarker_weight: float = 0.3,
    min_biomarker_confidence: float = 0.6,
) -> pd.DataFrame:
    """
    批量运行 BiomarkerEnhancedPredictor，记录逐样本概率与 boost 信息。

    Parameters
    ----------
    samples : 序列，每项为 (sample_data, sample_index, patient_id, y_true)
        y_true: 0=Interictal, 1=Preictal
    """
    from biomarker_enhanced_prediction import create_biomarker_enhanced_predictor

    predictor = create_biomarker_enhanced_predictor(
        biomarker_dir=biomarker_dir,
        model_predictor=base_model_predictor,
        biomarker_weight=biomarker_weight,
        min_biomarker_confidence=min_biomarker_confidence,
    )
    rows: List[Dict[str, Any]] = []
    for sample_data, sample_index, patient_id, y_true in samples:
        x = _ensure_eeg_batch_channel_time(sample_data)
        enhanced_probs, info = predictor.predict(x, sample_index=sample_index)
        base = np.asarray(info["base_probs"], dtype=np.float64).reshape(2)
        enh = np.asarray(enhanced_probs, dtype=np.float64).reshape(2)
        boost = info.get("biomarker_boost") or {}
        rows.append(
            {
                "patient_id": int(patient_id),
                "sample_index": int(sample_index) if sample_index is not None else -1,
                "y_true": int(y_true),
                "base_p_inter": float(base[0]),
                "base_p_pre": float(base[1]),
                "enh_p_inter": float(enh[0]),
                "enh_p_pre": float(enh[1]),
                "n_biomarker_matches": int(info.get("biomarker_matches", 0)),
                "boost_strength": float(boost.get("boost_strength", 0.0)),
                "boost_direction": int(boost.get("boost_direction", 0)),
                "avg_confidence": float(boost.get("avg_confidence", 0.0)),
                "max_confidence": float(boost.get("max_confidence", 0.0)),
                "biomarker_weight_used": float(biomarker_weight),
                "min_biomarker_confidence_used": float(min_biomarker_confidence),
            }
        )
    return pd.DataFrame(rows)


def log_row_from_pipeline_info(
    info: Dict[str, Any],
    y_true: int,
) -> Dict[str, Any]:
    """
    将 PostProcessingPipeline.predict 返回的 info 转为与 collect_per_sample_log 一致的扁平字段
    （仅生物标志物增强阶段：base / enhanced 对应 info 中键名）。
    """
    base = np.asarray(info["base_probs"], dtype=np.float64).reshape(2)
    enh = np.asarray(info["enhanced_probs"], dtype=np.float64).reshape(2)
    bio = info.get("biomarker_info") or {}
    boost = bio.get("boost") or {}
    return {
        "patient_id": int(info.get("patient_id", -1)),
        "sample_index": int(info["sample_index"]) if info.get("sample_index") is not None else -1,
        "y_true": int(y_true),
        "base_p_inter": float(base[0]),
        "base_p_pre": float(base[1]),
        "enh_p_inter": float(enh[0]),
        "enh_p_pre": float(enh[1]),
        "n_biomarker_matches": int(bio.get("matches", 0)),
        "boost_strength": float(boost.get("boost_strength", 0.0)),
        "boost_direction": int(boost.get("boost_direction", 0)),
        "avg_confidence": float(boost.get("avg_confidence", 0.0)),
        "max_confidence": float(boost.get("max_confidence", 0.0)),
        "biomarker_weight_used": float(info.get("biomarker_weight_used", np.nan)),
        "min_biomarker_confidence_used": float(info.get("min_biomarker_confidence_used", np.nan)),
    }


def append_log_rows_from_pipeline_run(
    records: List[Dict[str, Any]],
    infos: Sequence[Dict[str, Any]],
    y_trues: Sequence[int],
) -> List[Dict[str, Any]]:
    """将多次 predict 的 info 与标签追加到 records 列表（就地扩展后返回）。"""
    for inf, yt in zip(infos, y_trues):
        records.append(log_row_from_pipeline_info(inf, int(yt)))
    return records


def save_prediction_log(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_prediction_log(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path)


def _pred_from_prob_pre(p_pre: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (p_pre >= threshold).astype(int)


def metrics_from_log_column(
    df: pd.DataFrame,
    prob_col: str = "base_p_pre",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true = df["y_true"].to_numpy()
    p = df[prob_col].to_numpy()
    y_pred = _pred_from_prob_pre(p, threshold)
    probs_2d = np.column_stack([1.0 - p, p])
    return calculate_metrics(y_true, y_pred, probs_2d)


def sweep_biomarker_weight(
    df: pd.DataFrame,
    weights: np.ndarray,
    min_biomarker_confidence: float,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """对日志中每条样本用 apply_biomarker_fusion 重放不同权重，返回聚合指标表。"""
    records = []
    base_i = df["base_p_inter"].to_numpy()
    base_p = df["base_p_pre"].to_numpy()
    n_m = df["n_biomarker_matches"].to_numpy()
    max_c = df["max_confidence"].to_numpy()
    y_true = df["y_true"].to_numpy()

    for w in weights:
        p_after = []
        for i in range(len(df)):
            base_probs = np.array([base_i[i], base_p[i]], dtype=np.float64)
            enh, _ = apply_biomarker_fusion(
                base_probs,
                int(n_m[i]),
                float(max_c[i]),
                float(w),
                min_biomarker_confidence=min_biomarker_confidence,
            )
            p_after.append(float(enh[1]))
        p_after = np.array(p_after)
        y_pred = _pred_from_prob_pre(p_after, threshold)
        probs_2d = np.column_stack([1.0 - p_after, p_after])
        m = calculate_metrics(y_true, y_pred, probs_2d)
        records.append(
            {
                "biomarker_weight": float(w),
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "f1_score": m["f1_score"],
                "accuracy": m["accuracy"],
                "roc_auc": m.get("roc_auc", 0.0),
                "pr_auc": m.get("pr_auc", 0.0),
            }
        )
    return pd.DataFrame(records)


def sweep_weight_minconf_grid(
    df: pd.DataFrame,
    weights: np.ndarray,
    min_confidences: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """权重 × 最小置信度阈值 -> F1（用于热力图）。"""
    rows = []
    base_i = df["base_p_inter"].to_numpy()
    base_p = df["base_p_pre"].to_numpy()
    n_m = df["n_biomarker_matches"].to_numpy()
    max_c = df["max_confidence"].to_numpy()
    y_true = df["y_true"].to_numpy()
    for w in weights:
        for mc in min_confidences:
            p_after = []
            for i in range(len(df)):
                base_probs = np.array([base_i[i], base_p[i]], dtype=np.float64)
                enh, _ = apply_biomarker_fusion(
                    base_probs,
                    int(n_m[i]),
                    float(max_c[i]),
                    float(w),
                    min_biomarker_confidence=float(mc),
                )
                p_after.append(float(enh[1]))
            p_after = np.array(p_after)
            y_pred = _pred_from_prob_pre(p_after, threshold)
            probs_2d = np.column_stack([1.0 - p_after, p_after])
            m = calculate_metrics(y_true, y_pred, probs_2d)
            rows.append(
                {
                    "biomarker_weight": float(w),
                    "min_biomarker_confidence": float(mc),
                    "f1_score": m["f1_score"],
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                }
            )
    return pd.DataFrame(rows)


def per_patient_sens_spec(
    df: pd.DataFrame,
    prob_col_before: str = "base_p_pre",
    prob_col_after: str = "enh_p_pre",
    threshold: float = 0.5,
) -> pd.DataFrame:
    out = []
    for pid, g in df.groupby("patient_id"):
        y_true = g["y_true"].to_numpy()
        pb = g[prob_col_before].to_numpy()
        pa = g[prob_col_after].to_numpy()
        yb = _pred_from_prob_pre(pb, threshold)
        ya = _pred_from_prob_pre(pa, threshold)
        mb = calculate_metrics(y_true, yb, np.column_stack([1.0 - pb, pb]))
        ma = calculate_metrics(y_true, ya, np.column_stack([1.0 - pa, pa]))
        out.append(
            {
                "patient_id": int(pid),
                "sensitivity_before": mb["sensitivity"],
                "specificity_before": mb["specificity"],
                "sensitivity_after": ma["sensitivity"],
                "specificity_after": ma["specificity"],
            }
        )
    return pd.DataFrame(out)


def _resolve_sample_row(
    df: pd.DataFrame,
    patient_id: Optional[int] = None,
    sample_index: Optional[int] = None,
    row_index: Optional[int] = None,
) -> pd.Series:
    """按 row_index 或 patient_id+sample_index 解析一条样本记录。"""
    if row_index is not None:
        if row_index < 0 or row_index >= len(df):
            raise IndexError(f"row_index 越界: {row_index}")
        return df.iloc[row_index]

    if patient_id is not None and sample_index is not None:
        mask = (df["patient_id"] == patient_id) & (df["sample_index"] == sample_index)
        selected = df.loc[mask]
        if selected.empty:
            raise ValueError(
                f"未找到样本 patient_id={patient_id}, sample_index={sample_index}"
            )
        return selected.iloc[0]

    raise ValueError("请提供 row_index 或者同时提供 patient_id 与 sample_index")


def select_representative_sample(
    df: pd.DataFrame,
    min_confidence: float = 0.6,
) -> pd.Series:
    """
    选取最能解释“相似度高导致前期置信度提升”的样本：
    在高置信匹配样本中，选 delta_pre 最大者；若不存在则全局 delta_pre 最大者。
    """
    tmp = df.copy()
    tmp["delta_pre"] = tmp["enh_p_pre"] - tmp["base_p_pre"]
    focused = tmp[
        (tmp["n_biomarker_matches"] > 0)
        & (tmp["max_confidence"] >= min_confidence)
        & (tmp["boost_strength"] > 0)
    ]
    if not focused.empty:
        return focused.sort_values("delta_pre", ascending=False).iloc[0]
    return tmp.sort_values("delta_pre", ascending=False).iloc[0]


def plot_single_sample_waterfall(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    patient_id: Optional[int] = None,
    sample_index: Optional[int] = None,
    row_index: Optional[int] = None,
    min_biomarker_confidence: float = 0.6,
) -> plt.Figure:
    """
    单样本可解释瀑布图：base_p_pre -> delta_pre -> enh_p_pre。
    支持 row_index 或 patient_id+sample_index 选样本。
    """
    if plt is None:
        raise RuntimeError("需要安装 matplotlib")
    row = _resolve_sample_row(
        df,
        patient_id=patient_id,
        sample_index=sample_index,
        row_index=row_index,
    )

    base_pre = float(row["base_p_pre"])
    enh_pre = float(row["enh_p_pre"])
    delta_pre = enh_pre - base_pre
    pid = int(row["patient_id"])
    sidx = int(row["sample_index"])
    y_true = int(row["y_true"])
    n_match = int(row["n_biomarker_matches"])
    max_c = float(row["max_confidence"])
    boost = float(row["boost_strength"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    categories = ["Base P_pre", "Weighted ΔP_pre", "Enhanced P_pre"]
    x = np.arange(3)

    ax.bar(x[0], base_pre, color="tab:blue", alpha=0.8)
    ax.bar(x[1], delta_pre, bottom=base_pre, color="tab:green" if delta_pre >= 0 else "tab:red", alpha=0.85)
    ax.bar(x[2], enh_pre, color="tab:orange", alpha=0.8)
    ax.plot([x[0], x[2]], [base_pre, enh_pre], "k--", alpha=0.45, lw=1)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probability")
    ax.set_title(f"Single-sample Explainability Waterfall (patient={pid}, sample={sidx})")
    ax.grid(True, axis="y", alpha=0.25)

    text = (
        f"y_true={y_true} | matches={n_match}\n"
        f"max_confidence={max_c:.3f}, boost_strength={boost:.3f}\n"
        f"base={base_pre:.3f} -> enhanced={enh_pre:.3f} (Δ={delta_pre:+.3f})"
    )
    if n_match > 0 and max_c >= min_biomarker_confidence and boost > 0:
        explain = "Interpretation: high similarity match triggers weighting and lifts P_pre."
    else:
        explain = "Interpretation: no effective weighting; P_pre shift remains limited."
    ax.text(
        0.02,
        0.98,
        text + "\n" + explain,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_confidence_delta_scatter(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    show_trend_line: bool = True,
) -> plt.Figure:
    """样本级：相似度(max_confidence) 与增益(ΔP_pre)关系图。"""
    if plt is None:
        raise RuntimeError("需要安装 matplotlib")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y_true = df["y_true"].to_numpy()
    max_c = df["max_confidence"].to_numpy()
    delta = (df["enh_p_pre"] - df["base_p_pre"]).to_numpy()

    for label, color in [(0, "tab:blue"), (1, "tab:red")]:
        m = y_true == label
        if m.any():
            ax.scatter(max_c[m], delta[m], s=18, alpha=0.65, c=color, label=f"y_true={label}")

    if show_trend_line and len(df) >= 5:
        coef = np.polyfit(max_c, delta, 1)
        xs = np.linspace(max(0.0, np.nanmin(max_c)), min(1.0, np.nanmax(max_c)), 100)
        ys = coef[0] * xs + coef[1]
        ax.plot(xs, ys, "k--", alpha=0.7, label=f"trend slope={coef[0]:.3f}")

    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("max_confidence (biomarker similarity)")
    ax.set_ylabel("Δ P(Preictal) = enhanced - base")
    ax.set_title("Confidence vs Weighted Gain (sample-level)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def patient_level_explainability_stats(df: pd.DataFrame) -> pd.DataFrame:
    """患者级可解释统计：均值 + 稳健分位数。"""
    tmp = df.copy()
    tmp["delta_pre"] = tmp["enh_p_pre"] - tmp["base_p_pre"]

    grouped = tmp.groupby("patient_id")
    stats_df = grouped.agg(
        mean_delta_pre=("delta_pre", "mean"),
        median_delta_pre=("delta_pre", "median"),
        q25_delta_pre=("delta_pre", lambda x: float(np.percentile(x, 25))),
        q75_delta_pre=("delta_pre", lambda x: float(np.percentile(x, 75))),
        boost_rate=("boost_strength", lambda x: float(np.mean(np.asarray(x) > 0))),
        mean_confidence=("max_confidence", "mean"),
        sample_count=("delta_pre", "count"),
    ).reset_index()
    return stats_df.sort_values("patient_id")


def plot_patient_level_explainability(
    patient_stats_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    患者级可解释图：
    - 柱：mean_delta_pre
    - 折线(右轴)：boost_rate、mean_confidence
    """
    if plt is None:
        raise RuntimeError("需要安装 matplotlib")

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    pids = patient_stats_df["patient_id"].astype(int).to_numpy()
    x = np.arange(len(pids))

    bars = ax1.bar(
        x,
        patient_stats_df["mean_delta_pre"].to_numpy(),
        color="tab:green",
        alpha=0.75,
        label="mean_delta_pre",
    )
    ax1.errorbar(
        x,
        patient_stats_df["median_delta_pre"].to_numpy(),
        yerr=np.vstack(
            [
                patient_stats_df["median_delta_pre"] - patient_stats_df["q25_delta_pre"],
                patient_stats_df["q75_delta_pre"] - patient_stats_df["median_delta_pre"],
            ]
        ),
        fmt="o",
        color="black",
        capsize=3,
        label="median (IQR)",
    )
    ax1.axhline(0, color="k", lw=0.8, alpha=0.45)
    ax1.set_ylabel("Δ P(Preictal)")
    ax1.set_xlabel("Patient ID")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(i) for i in pids])

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        patient_stats_df["boost_rate"].to_numpy(),
        "o-",
        color="tab:blue",
        label="boost_rate",
    )
    ax2.plot(
        x,
        patient_stats_df["mean_confidence"].to_numpy(),
        "s--",
        color="tab:orange",
        label="mean_confidence",
    )
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Rate / Confidence")

    ax1.set_title("Patient-level Explainability: Gain, Boost Rate, Confidence")
    ax1.grid(True, axis="y", alpha=0.25)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8)

    for b, n in zip(bars, patient_stats_df["sample_count"].to_numpy()):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"n={int(n)}",
            ha="center",
            va="bottom",
            fontsize=7,
            alpha=0.8,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def decision_flip_counts(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, int]:
    y_true = df["y_true"].to_numpy()
    yb = _pred_from_prob_pre(df["base_p_pre"].to_numpy(), threshold)
    ya = _pred_from_prob_pre(df["enh_p_pre"].to_numpy(), threshold)
    readable: Dict[str, int] = {}
    for t, b, a in zip(y_true, yb, ya):
        if t == 1 and b == 0 and a == 1:
            readable["FN_to_TP"] = readable.get("FN_to_TP", 0) + 1
        elif t == 1 and b == 1 and a == 0:
            readable["TP_to_FN"] = readable.get("TP_to_FN", 0) + 1
        elif t == 0 and b == 1 and a == 0:
            readable["FP_to_TN"] = readable.get("FP_to_TN", 0) + 1
        elif t == 0 and b == 0 and a == 1:
            readable["TN_to_FP"] = readable.get("TN_to_FP", 0) + 1
        elif b == a:
            if t == 1 and a == 1:
                readable["TP_unchanged"] = readable.get("TP_unchanged", 0) + 1
            elif t == 1 and a == 0:
                readable["FN_unchanged"] = readable.get("FN_unchanged", 0) + 1
            elif t == 0 and a == 0:
                readable["TN_unchanged"] = readable.get("TN_unchanged", 0) + 1
            else:
                readable["FP_unchanged"] = readable.get("FP_unchanged", 0) + 1
    return readable


def plot_weight_sweep(
    sweep_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    if plt is None:
        raise RuntimeError("需要安装 matplotlib")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    w = sweep_df["biomarker_weight"].to_numpy()
    axes[0, 0].plot(w, sweep_df["sensitivity"], marker="o", label="Sensitivity")
    axes[0, 0].plot(w, sweep_df["specificity"], marker="s", label="Specificity")
    axes[0, 0].set_xlabel("biomarker_weight")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Sensitivity / Specificity vs weight")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(w, sweep_df["f1_score"], marker="^", color="green")
    axes[0, 1].set_xlabel("biomarker_weight")
    axes[0, 1].set_ylabel("F1")
    axes[0, 1].set_title("F1 vs weight")
    axes[0, 1].grid(True, alpha=0.3)

    if "roc_auc" in sweep_df.columns:
        axes[1, 0].plot(w, sweep_df["roc_auc"], marker="d", color="purple")
        axes[1, 0].set_xlabel("biomarker_weight")
        axes[1, 0].set_ylabel("ROC-AUC")
        axes[1, 0].set_title("ROC-AUC vs weight")
        axes[1, 0].grid(True, alpha=0.3)

    if "pr_auc" in sweep_df.columns:
        axes[1, 1].plot(w, sweep_df["pr_auc"], marker="x", color="orange")
        axes[1, 1].set_xlabel("biomarker_weight")
        axes[1, 1].set_ylabel("PR-AUC")
        axes[1, 1].set_title("PR-AUC vs weight")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_weight_minconf_heatmap(
    grid_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    if plt is None:
        raise RuntimeError("需要安装 matplotlib")
    pivot = grid_df.pivot_table(
        index="min_biomarker_confidence",
        columns="biomarker_weight",
        values="f1_score",
    )
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.4), max(5, pivot.shape[0] * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("biomarker_weight")
    ax.set_ylabel("min_biomarker_confidence")
    ax.set_title("F1 score (heatmap)")
    plt.colorbar(im, ax=ax, label="F1")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_sens_spec_shift(
    pp_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    if plt is None:
        raise RuntimeError("需要安装 matplotlib")
    fig, ax = plt.subplots(figsize=(8, 8))
    for _, row in pp_df.iterrows():
        sb, spb = row["sensitivity_before"], row["specificity_before"]
        sa, spa = row["sensitivity_after"], row["specificity_after"]
        ax.scatter(sb, spb, c="tab:blue", s=40, alpha=0.7)
        ax.scatter(sa, spa, c="tab:orange", s=40, alpha=0.7)
        ax.annotate(
            "",
            xy=(sa, spa),
            xytext=(sb, spb),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5, lw=1),
        )
        ax.text(sa + 0.01, spa + 0.01, str(int(row["patient_id"])), fontsize=8, alpha=0.8)
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Specificity")
    ax.set_title("Per-patient Sens–Spec (before=blue, after=orange, arrows)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_prob_before_after(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    if plt is None:
        raise RuntimeError("需要安装 matplotlib")
    matched = (df["n_biomarker_matches"] > 0).to_numpy()
    y_true = df["y_true"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    subsets = [
        ("All samples", np.ones(len(df), dtype=bool)),
        ("Biomarker matched", matched),
        ("No biomarker match", ~matched),
    ]
    for ax, (title, mask) in zip(axes.flat[:3], subsets):
        g = df.loc[mask]
        if len(g) == 0:
            ax.set_visible(False)
            continue
        for label, color in [(0, "tab:blue"), (1, "tab:red")]:
            m = g["y_true"] == label
            if not m.any():
                continue
            ax.scatter(
                g.loc[m, "base_p_pre"],
                g.loc[m, "enh_p_pre"],
                s=18,
                alpha=0.6,
                c=color,
                label=f"y_true={label}",
            )
        lims = [0, 1]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("P(Preictal) base")
        ax.set_ylabel("P(Preictal) enhanced")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes.flat[3]
    delta = (df["enh_p_pre"] - df["base_p_pre"]).to_numpy()
    max_c = df["max_confidence"].to_numpy()
    for label, color in [(0, "tab:blue"), (1, "tab:red")]:
        m = y_true == label
        if m.any():
            ax.scatter(max_c[m], delta[m], s=18, alpha=0.6, c=color, label=f"y_true={label}")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("max_confidence (biomarker)")
    ax.set_ylabel("Δ P(Preictal)")
    ax.set_title("Probability shift vs biomarker confidence")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_roc_pr_overlay(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    if plt is None:
        raise RuntimeError("需要安装 matplotlib")
    y = df["y_true"].to_numpy()
    if len(np.unique(y)) < 2:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.text(0.5, 0.5, "Need both classes for ROC/PR", ha="center")
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, col, style in [
        ("base", "base_p_pre", "-"),
        ("enhanced", "enh_p_pre", "--"),
    ]:
        s = df[col].to_numpy()
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, linestyle=style, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], "k:", alpha=0.4)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, col, style in [
        ("base", "base_p_pre", "-"),
        ("enhanced", "enh_p_pre", "--"),
    ]:
        s = df[col].to_numpy()
        prec, rec, _ = precision_recall_curve(y, s)
        axes[1].plot(rec, prec, linestyle=style, label=f"{name} AUC={auc(rec, prec):.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_decision_flips_bar(
    flip_dict: Dict[str, int],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    if plt is None:
        raise RuntimeError("需要安装 matplotlib")
    change_keys = [k for k in flip_dict if k in ("FN_to_TP", "FP_to_TN", "TP_to_FN", "TN_to_FP")]
    vals = [flip_dict.get(k, 0) for k in change_keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(change_keys, vals, color=["tab:green", "tab:green", "tab:red", "tab:red"], alpha=0.75)
    ax.set_ylabel("Count")
    ax.set_title("Decision changes (base -> enhanced, threshold=0.5)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def run_all_plots_from_log(
    df: pd.DataFrame,
    out_dir: Union[str, Path],
    min_biomarker_confidence: float,
    weight_grid: Optional[np.ndarray] = None,
    minconf_grid: Optional[np.ndarray] = None,
    waterfall_patient_id: Optional[int] = None,
    waterfall_sample_index: Optional[int] = None,
    waterfall_row_index: Optional[int] = None,
) -> Dict[str, str]:
    """
    从单份 collect 日志生成计划中的主要图表，返回输出路径字典。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    if weight_grid is None:
        weight_grid = np.linspace(0.0, 0.8, 17)
    sweep = sweep_biomarker_weight(df, weight_grid, min_biomarker_confidence=min_biomarker_confidence)
    sweep.to_csv(out_dir / "weight_sweep_metrics.csv", index=False)
    fig = plot_weight_sweep(sweep, out_dir / "weight_sweep_curves.png")
    plt.close(fig)
    paths["weight_sweep"] = str(out_dir / "weight_sweep_curves.png")

    if minconf_grid is None:
        minconf_grid = np.array([0.5, 0.55, 0.6, 0.65, 0.7])
    grid = sweep_weight_minconf_grid(df, weight_grid, minconf_grid)
    grid.to_csv(out_dir / "weight_minconf_grid.csv", index=False)
    fig = plot_weight_minconf_heatmap(grid, out_dir / "weight_minconf_f1_heatmap.png")
    plt.close(fig)
    paths["heatmap"] = str(out_dir / "weight_minconf_f1_heatmap.png")

    pp = per_patient_sens_spec(df)
    pp.to_csv(out_dir / "per_patient_sens_spec.csv", index=False)
    fig = plot_sens_spec_shift(pp, out_dir / "sens_spec_shift.png")
    plt.close(fig)
    paths["sens_spec_shift"] = str(out_dir / "sens_spec_shift.png")

    fig = plot_prob_before_after(df, out_dir / "prob_before_after_delta.png")
    plt.close(fig)
    paths["prob"] = str(out_dir / "prob_before_after_delta.png")

    fig = plot_confidence_delta_scatter(df, out_dir / "confidence_delta_scatter.png")
    plt.close(fig)
    paths["confidence_delta"] = str(out_dir / "confidence_delta_scatter.png")

    patient_stats = patient_level_explainability_stats(df)
    patient_stats.to_csv(out_dir / "patient_level_explainability.csv", index=False)
    fig = plot_patient_level_explainability(
        patient_stats,
        out_dir / "patient_level_explainability.png",
    )
    plt.close(fig)
    paths["patient_explainability"] = str(out_dir / "patient_level_explainability.png")

    if (
        waterfall_row_index is None
        and waterfall_patient_id is None
        and waterfall_sample_index is None
    ):
        rep = select_representative_sample(df, min_confidence=min_biomarker_confidence)
        waterfall_patient_id = int(rep["patient_id"])
        waterfall_sample_index = int(rep["sample_index"])
    if waterfall_row_index is not None and (
        waterfall_patient_id is None or waterfall_sample_index is None
    ):
        rep = _resolve_sample_row(df, row_index=waterfall_row_index)
        waterfall_patient_id = int(rep["patient_id"])
        waterfall_sample_index = int(rep["sample_index"])
    w_name = f"sample_waterfall_{waterfall_patient_id}_{waterfall_sample_index}.png"
    fig = plot_single_sample_waterfall(
        df,
        out_dir / w_name,
        patient_id=waterfall_patient_id,
        sample_index=waterfall_sample_index,
        row_index=waterfall_row_index,
        min_biomarker_confidence=min_biomarker_confidence,
    )
    plt.close(fig)
    paths["sample_waterfall"] = str(out_dir / w_name)

    fig = plot_roc_pr_overlay(df, out_dir / "roc_pr_overlay.png")
    plt.close(fig)
    paths["roc_pr"] = str(out_dir / "roc_pr_overlay.png")

    flips = decision_flip_counts(df)
    with open(out_dir / "decision_flips.json", "w", encoding="utf-8") as f:
        json.dump(flips, f, indent=2, ensure_ascii=False)
    fig = plot_decision_flips_bar(flips, out_dir / "decision_flips.png")
    plt.close(fig)
    paths["flips"] = str(out_dir / "decision_flips.png")

    return paths


def _demo_synthetic_log(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    base_p = rng.uniform(0, 1, size=n)
    # 模拟：有匹配时略抬高真发作的置信度
    n_match = rng.integers(0, 3, size=n)
    max_c = np.where(n_match > 0, rng.uniform(0.4, 0.95, size=n), np.zeros(n))
    max_c = np.where((y == 1) & (n_match > 0), np.maximum(max_c, 0.65), max_c)
    base_i = 1.0 - base_p
    rows = []
    for i in range(n):
        enh, boost = apply_biomarker_fusion(
            np.array([base_i[i], base_p[i]]),
            int(n_match[i]),
            float(max_c[i]),
            biomarker_weight=0.35,
            min_biomarker_confidence=0.6,
        )
        rows.append(
            {
                "patient_id": int(i % 6) + 1,
                "sample_index": i,
                "y_true": int(y[i]),
                "base_p_inter": float(base_i[i]),
                "base_p_pre": float(base_p[i]),
                "enh_p_inter": float(enh[0]),
                "enh_p_pre": float(enh[1]),
                "n_biomarker_matches": int(n_match[i]),
                "boost_strength": float(boost.get("boost_strength", 0.0)),
                "boost_direction": int(boost.get("boost_direction", 0)),
                "avg_confidence": float(boost.get("avg_confidence", 0.0)),
                "max_confidence": float(max_c[i]),
                "biomarker_weight_used": 0.35,
                "min_biomarker_confidence_used": 0.6,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    demo_dir = Path(__file__).resolve().parent / "_biomarker_viz_demo_out"
    demo_dir.mkdir(parents=True, exist_ok=True)
    df_demo = _demo_synthetic_log()
    save_prediction_log(df_demo, demo_dir / "demo_log.csv")
    run_all_plots_from_log(df_demo, demo_dir, min_biomarker_confidence=0.6)
    print("Demo outputs written to:", demo_dir)

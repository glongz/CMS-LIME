#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨患者生物标志物统计分析

对 01-06 六名患者的 filtered_biomarkers 进行跨患者生物标志物分析，
参考 biomarker_analysis.py 的分析方式，并增加患者间重叠、一致性等分析。

Author: CMS-LIME Framework
Date: 2025
"""

import numpy as np
import json
import os
import glob
import re
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import pandas as pd
from scipy import stats
import gc
import sys
import shutil
from pathlib import Path

# 保证同目录下的 biomarker_analysis 可被导入（运行于项目根或本目录均可）
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from biomarker_analysis import (
    BiomarkerData,
    setup_english_font,
    basic_statistics,
    temporal_analysis,
    spatial_analysis,
    quality_assessment,
    pattern_analysis,
    validation_analysis,
    plot_overview,
    plot_comparison,
    plot_top_biomarkers,
    generate_report,
)


class BiomarkerDataWithPatient(BiomarkerData):
    """带患者ID的生物标志物数据，用于跨患者分析"""

    def __init__(self, filepath: str, patient_id: str):
        self.patient_id = str(patient_id)
        super().__init__(filepath)

    def _parse_filename(self):
        """解析文件名：支持 s000207_b0000_ 与 unique_0025_shapelet_73 两种格式"""
        # 原格式: s000207_b0000_shapelet_shapelet_63.npz
        pattern1 = r's(\d+)_b(\d+)_(\w+)_'
        match = re.search(pattern1, self.filename)
        if match:
            self.sample_index = int(match.group(1))
            self.batch_index = int(match.group(2))
            return
        # filtered_biomarkers 格式: unique_0025_shapelet_73.npz
        pattern2 = r'unique_(\d+)_(.+)\.npz'
        match2 = re.search(pattern2, self.filename)
        if match2:
            self.batch_index = int(match2.group(1))
            if self.sample_index is None:
                self.sample_index = int(match2.group(1))


def load_patient_biomarkers(patient_dir: str, patient_id: str) -> List[BiomarkerDataWithPatient]:
    """
    加载单名患者 filtered_biomarkers 目录下的所有 npz 生物标志物。

    Parameters:
    -----------
    patient_dir : str
        该患者的 filtered_biomarkers 目录，如 D:\\...\\chbmit_biomarkers\\01\\filtered_biomarkers
    patient_id : str
        患者编号，如 "01"

    Returns:
    --------
    biomarkers : List[BiomarkerDataWithPatient]
    """
    pattern = os.path.join(patient_dir, "*.npz")
    files = glob.glob(pattern)
    biomarkers = []
    for filepath in files:
        try:
            b = BiomarkerDataWithPatient(filepath, patient_id)
            biomarkers.append(b)
        except Exception as e:
            print(f"  加载失败 {os.path.basename(filepath)}: {e}")
    return biomarkers


def load_cross_patient_biomarkers(
    base_dir: str,
    patient_ids: List[str],
) -> Tuple[List[BiomarkerDataWithPatient], Dict[str, List[BiomarkerDataWithPatient]]]:
    """
    加载多名患者的生物标志物。

    Parameters:
    -----------
    base_dir : str
        chbmit_biomarkers 根目录，如 D:\\2025_important_projects\\data\\chbmit_biomarkers
    patient_ids : List[str]
        患者编号列表，如 ["01","02",...,"06"]

    Returns:
    --------
    all_biomarkers : List[BiomarkerDataWithPatient]
        所有患者的生物标志物合并列表
    by_patient : Dict[str, List[BiomarkerDataWithPatient]]
        按患者ID分组的列表
    """
    by_patient = {}
    all_biomarkers = []
    for pid in patient_ids:
        patient_dir = os.path.join(base_dir, pid, "filtered_biomarkers")
        if not os.path.isdir(patient_dir):
            print(f"警告: 目录不存在，跳过患者 {pid}: {patient_dir}")
            continue
        biomarkers = load_patient_biomarkers(patient_dir, pid)
        by_patient[pid] = biomarkers
        all_biomarkers.extend(biomarkers)
        print(f"  患者 {pid}: 加载 {len(biomarkers)} 个生物标志物")
    return all_biomarkers, by_patient


def load_filtering_stats(base_dir: str, patient_ids: List[str]) -> Dict[str, Dict]:
    """加载各患者的 filtering_stats.json"""
    stats_per_patient = {}
    for pid in patient_ids:
        path = os.path.join(base_dir, pid, "filtered_biomarkers", "filtering_stats.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                stats_per_patient[pid] = json.load(f)
    return stats_per_patient


def cross_patient_basic_stats(
    all_biomarkers: List[BiomarkerDataWithPatient],
    by_patient: Dict[str, List[BiomarkerDataWithPatient]],
) -> Dict[str, Any]:
    """跨患者基本统计：总体 + 分患者"""
    total = basic_statistics(all_biomarkers)
    per_patient = {}
    for pid, blist in by_patient.items():
        per_patient[pid] = basic_statistics(blist)
    return {
        "total": total,
        "per_patient": per_patient,
        "n_patients": len(by_patient),
        "total_count": len(all_biomarkers),
    }


def cross_patient_primitive_id_overlap(
    by_patient: Dict[str, List[BiomarkerDataWithPatient]],
) -> Dict[str, Any]:
    """
    跨患者 primitive_id 重叠分析。
    - 每个患者有哪些 primitive_id
    - 哪些 primitive_id 在多个患者中出现（共同生物标志物）
    """
    patient_to_ids = {}
    for pid, blist in by_patient.items():
        ids = set()
        for b in blist:
            if getattr(b, "primitive_id", None):
                ids.add(b.primitive_id)
        patient_to_ids[pid] = ids

    all_ids = set()
    for ids in patient_to_ids.values():
        all_ids |= ids

    id_to_patients = defaultdict(set)
    for pid, ids in patient_to_ids.items():
        for primitive_id in ids:
            id_to_patients[primitive_id].add(pid)

    # 共同标志物：在至少 2 个患者中出现
    common_ids = {pid: len(patients) for pid, patients in id_to_patients.items() if len(patients) >= 2}
    shared_biomarkers = sorted(common_ids.items(), key=lambda x: -x[1])

    return {
        "patient_to_primitive_ids": {k: list(v) for k, v in patient_to_ids.items()},
        "primitive_id_to_patients": {k: list(v) for k, v in id_to_patients.items()},
        "shared_primitive_ids": dict(shared_biomarkers),
        "n_shared": len(shared_biomarkers),
        "n_unique_all": len(all_ids),
    }


def cross_patient_type_distribution(
    by_patient: Dict[str, List[BiomarkerDataWithPatient]],
) -> Dict[str, Any]:
    """各患者按 primitive_type 的分布"""
    per_patient = {}
    for pid, blist in by_patient.items():
        type_counts = Counter([getattr(b, "primitive_type", None) or "unknown" for b in blist])
        per_patient[pid] = dict(type_counts)
    return {"per_patient": per_patient}


def plot_cross_patient_overview(
    all_biomarkers: List[BiomarkerDataWithPatient],
    by_patient: Dict[str, List[BiomarkerDataWithPatient]],
    overlap_stats: Dict[str, Any],
    save_dir: str,
):
    """生成跨患者总览图"""
    setup_english_font()
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    fig.suptitle("Cross-Patient Biomarker Analysis (Patients 01-06)", fontsize=18, fontweight="bold", y=0.98)

    # 1. 各患者生物标志物数量
    ax1 = fig.add_subplot(gs[0, 0])
    patients = sorted(by_patient.keys())
    counts = [len(by_patient[p]) for p in patients]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(patients)))
    bars = ax1.bar(patients, counts, color=colors, edgecolor="black")
    ax1.set_xlabel("Patient ID", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Biomarkers", fontsize=12, fontweight="bold")
    ax1.set_title("Biomarker Count per Patient", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(c), ha="center", va="bottom", fontweight="bold")

    # 2. 各患者类型分布（堆叠柱状图）
    ax2 = fig.add_subplot(gs[0, 1])
    type_dist = cross_patient_type_distribution(by_patient)
    per_patient = type_dist["per_patient"]
    types_seen = set()
    for tdict in per_patient.values():
        types_seen.update(tdict.keys())
    types_order = sorted(types_seen)
    x = np.arange(len(patients))
    width = 0.6
    bottom = np.zeros(len(patients))
    color_map = {"shapelet": "#4ECDC4", "timefreq": "#45B7D1", "microstate": "#FF6B6B", "unknown": "#96CEB4"}
    for t in types_order:
        vals = [per_patient.get(p, {}).get(t, 0) for p in patients]
        ax2.bar(x, vals, width, label=t, bottom=bottom, color=color_map.get(t, "#cccccc"))
        bottom += np.array(vals)
    ax2.set_xticks(x)
    ax2.set_xticklabels(patients)
    ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax2.set_title("Type Distribution per Patient", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. 跨患者共同 primitive_id 数量（在 ≥2 个患者中出现的数量分布）
    ax3 = fig.add_subplot(gs[1, 0])
    shared = overlap_stats.get("shared_primitive_ids", {})
    if shared:
        n_patients_list = list(shared.values())
        n_patients_counts = Counter(n_patients_list)
        xs = sorted(n_patients_counts.keys())
        ys = [n_patients_counts[k] for k in xs]
        ax3.bar(xs, ys, color="#45B7D1", alpha=0.8, edgecolor="black")
        ax3.set_xlabel("Number of Patients Sharing the Biomarker", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Number of Such Biomarkers", fontsize=12, fontweight="bold")
        ax3.set_title("Shared Biomarkers (by primitive_id)", fontsize=14, fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "No shared primitive_id across patients", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Shared Biomarkers (by primitive_id)", fontsize=14, fontweight="bold")

    # 4. Top 15 跨患者出现最多的 primitive_id
    ax4 = fig.add_subplot(gs[1, 1])
    shared_sorted = sorted(overlap_stats.get("shared_primitive_ids", {}).items(), key=lambda x: -x[1])[:15]
    if shared_sorted:
        ids_label = [x[0][:20] + ("..." if len(x[0]) > 20 else "") for x in shared_sorted]
        n_pats = [x[1] for x in shared_sorted]
        ax4.barh(range(len(ids_label)), n_pats, color="#4ECDC4", alpha=0.8, edgecolor="black")
        ax4.set_yticks(range(len(ids_label)))
        ax4.set_yticklabels(ids_label, fontsize=9)
        ax4.set_xlabel("Number of Patients", fontsize=12, fontweight="bold")
        ax4.set_title("Top 15 Shared Biomarkers (primitive_id)", fontsize=14, fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="x")
    else:
        ax4.text(0.5, 0.5, "No shared biomarkers", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Top 15 Shared Biomarkers", fontsize=14, fontweight="bold")

    # 5. 统计摘要表格
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    total_count = len(all_biomarkers)
    n_patients = len(by_patient)
    n_shared = overlap_stats.get("n_shared", 0)
    n_unique_all = overlap_stats.get("n_unique_all", 0)
    summary_data = [
        ["Total biomarkers (all patients)", str(total_count)],
        ["Number of patients", str(n_patients)],
        ["Unique primitive_id (union)", str(n_unique_all)],
        ["Shared primitive_id (appear in ≥2 patients)", str(n_shared)],
        ["Average biomarkers per patient", f"{total_count / n_patients:.1f}" if n_patients else "0"],
    ]
    table = ax5.table(
        cellText=summary_data,
        colLabels=["Statistic", "Value"],
        cellLoc="center",
        loc="center",
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    for i in range(len(summary_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor("#4CAF50")
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "white")
    ax5.set_title("Cross-Patient Summary", fontsize=14, fontweight="bold", pad=20)

    plt.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.06, hspace=0.35, wspace=0.3)
    save_path = os.path.join(save_dir, "cross_patient_overview.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png", facecolor="white")
    except Exception as e:
        print(f"保存跨患者总览图时出错: {e}")
    plt.close("all")
    gc.collect()
    print(f"跨患者总览图已保存: {save_path}")


def plot_patient_type_heatmap(
    by_patient: Dict[str, List[BiomarkerDataWithPatient]],
    save_dir: str,
):
    """各患者 × 类型的数量热图"""
    setup_english_font()
    type_dist = cross_patient_type_distribution(by_patient)
    patients = sorted(by_patient.keys())
    types_seen = set()
    for tdict in type_dist["per_patient"].values():
        types_seen.update(tdict.keys())
    types_order = sorted(types_seen)
    data = np.zeros((len(types_order), len(patients)))
    for j, p in enumerate(patients):
        for i, t in enumerate(types_order):
            data[i, j] = type_dist["per_patient"].get(p, {}).get(t, 0)
    fig, ax = plt.subplots(figsize=(10, max(4, len(types_order) * 0.5)))
    sns.heatmap(data, xticklabels=patients, yticklabels=types_order, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("Biomarker Type Count per Patient", fontsize=14, fontweight="bold")
    ax.set_xlabel("Patient ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("Primitive Type", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "cross_patient_type_heatmap.png")
    try:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png", facecolor="white")
    except Exception as e:
        print(f"保存类型热图时出错: {e}")
    plt.close("all")
    gc.collect()
    print(f"类型热图已保存: {save_path}")


def generate_cross_patient_report(
    all_biomarkers: List[BiomarkerDataWithPatient],
    by_patient: Dict[str, List[BiomarkerDataWithPatient]],
    overlap_stats: Dict[str, Any],
    filtering_stats_per_patient: Dict[str, Dict],
    save_dir: str,
):
    """生成跨患者分析文本报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("跨患者生物标志物统计分析报告 (患者 01-06)")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("1. 总体与分患者基本统计")
    report_lines.append("-" * 80)
    report_lines.append(f"  总生物标志物数量: {len(all_biomarkers)}")
    report_lines.append(f"  患者数: {len(by_patient)}")
    for pid in sorted(by_patient.keys()):
        n = len(by_patient[pid])
        report_lines.append(f"    患者 {pid}: {n} 个")
    if by_patient:
        report_lines.append(f"  平均每患者: {len(all_biomarkers) / len(by_patient):.1f} 个")
    report_lines.append("")

    report_lines.append("2. 跨患者 primitive_id 重叠")
    report_lines.append("-" * 80)
    report_lines.append(f"  所有患者中出现的不同 primitive_id 总数: {overlap_stats.get('n_unique_all', 0)}")
    report_lines.append(f"  在 ≥2 个患者中均出现的 primitive_id 数量: {overlap_stats.get('n_shared', 0)}")
    report_lines.append("  出现患者数最多的前 20 个 primitive_id:")
    for pid, n in list(overlap_stats.get("shared_primitive_ids", {}).items())[:20]:
        report_lines.append(f"    {pid}: {n} 个患者")
    report_lines.append("")

    report_lines.append("3. 各患者类型分布")
    report_lines.append("-" * 80)
    type_dist = cross_patient_type_distribution(by_patient)
    for pid in sorted(type_dist["per_patient"].keys()):
        report_lines.append(f"  患者 {pid}:")
        for t, c in type_dist["per_patient"][pid].items():
            report_lines.append(f"    {t}: {c}")
    report_lines.append("")

    report_lines.append("4. 各患者筛选统计 (filtering_stats.json)")
    report_lines.append("-" * 80)
    for pid in sorted(filtering_stats_per_patient.keys()):
        s = filtering_stats_per_patient[pid]
        report_lines.append(f"  患者 {pid}: total_filtered = {s.get('total_filtered', 'N/A')}")
    report_lines.append("")

    report_lines.append("5. 整体统计（合并所有患者后的单患者视角分析）")
    report_lines.append("-" * 80)
    basic = basic_statistics(all_biomarkers)
    report_lines.append(f"  类型分布: {basic.get('type_distribution', {})}")
    report_lines.append(f"  唯一样本数: {basic.get('unique_samples', 'N/A')}")
    report_lines.append(f"  唯一批次数: {basic.get('unique_batches', 'N/A')}")
    report_lines.append("")

    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    report_path = os.path.join(save_dir, "cross_patient_biomarker_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\n" + report_text)
    print(f"\n跨患者报告已保存: {report_path}")

    # CSV 摘要
    rows = []
    for pid in sorted(by_patient.keys()):
        n = len(by_patient[pid])
        type_dist_p = type_dist["per_patient"].get(pid, {})
        row = {"patient_id": pid, "n_biomarkers": n}
        for t, c in type_dist_p.items():
            row[f"n_{t}"] = c
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "cross_patient_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 摘要已保存: {csv_path}")


def _sanitize_primitive_id_for_path(primitive_id: str) -> str:
    """将 primitive_id 转为可作文件夹名的安全字符串"""
    if not primitive_id:
        return "unknown"
    s = str(primitive_id)
    for c in r'\/:*?"<>|':
        s = s.replace(c, "_")
    return s.strip() or "unknown"


def copy_shared_biomarkers_to_data(
    all_biomarkers: List[BiomarkerDataWithPatient],
    overlap_stats: Dict[str, Any],
    target_base_dir: str,
) -> Dict[str, Any]:
    """
    将「在 ≥2 个患者中均出现的 primitive_id」对应的所有生物标志物 .npz 文件
    直接复制保存到 target_base_dir 下（即 D:\\2025_important_projects\\data），不分子文件夹。

    文件名: patient_<id>_<原文件名>.npz，保证唯一。
    并生成 manifest 清单 JSON。

    Returns:
    --------
    result : Dict
        copied_count, skipped_count, save_dir, manifest_path, manifest 内容摘要
    """
    save_dir = target_base_dir
    os.makedirs(save_dir, exist_ok=True)

    shared_ids = set(overlap_stats.get("shared_primitive_ids", {}).keys())
    if not shared_ids:
        print("未发现任何在 ≥2 个患者中均出现的 primitive_id，跳过复制。")
        return {"copied_count": 0, "save_dir": save_dir}

    copied_count = 0
    skipped_count = 0
    manifest = []  # list of { primitive_id, patient_id, source, dest, filename }

    for b in all_biomarkers:
        pid = getattr(b, "primitive_id", None)
        if pid not in shared_ids:
            continue
        src = getattr(b, "filepath", None)
        if not src or not os.path.isfile(src):
            skipped_count += 1
            continue
        patient_id = getattr(b, "patient_id", "unknown")
        basename = os.path.basename(src)
        dest_name = f"patient_{patient_id}_{basename}"
        dest_path = os.path.join(save_dir, dest_name)
        try:
            shutil.copy2(src, dest_path)
            copied_count += 1
            manifest.append({
                "primitive_id": pid,
                "patient_id": patient_id,
                "source": src,
                "dest": dest_path,
                "filename": dest_name,
            })
        except Exception as e:
            print(f"  复制失败 {src} -> {dest_path}: {e}")
            skipped_count += 1

    manifest_path = os.path.join(save_dir, "shared_biomarkers_manifest.json")
    manifest_content = {
        "description": "Biomarkers with primitive_id appearing in >= 2 patients (01-06)",
        "total_copied_files": copied_count,
        "skipped": skipped_count,
        "n_shared_primitive_ids": len(shared_ids),
        "save_dir": save_dir,
        "entries": manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_content, f, ensure_ascii=False, indent=2)

    print(f"\n已在 ≥2 个患者中出现的 primitive_id 对应生物标志物已复制到: {save_dir}")
    print(f"  复制文件数: {copied_count}, 跳过/失败: {skipped_count}")
    print(f"  涉及 shared primitive_id 数: {len(shared_ids)}")
    print(f"  清单已保存: {manifest_path}")

    return {
        "copied_count": copied_count,
        "skipped_count": skipped_count,
        "save_dir": save_dir,
        "manifest_path": manifest_path,
        "n_shared_primitive_ids": len(shared_ids),
    }


def main():
    # 数据根目录与患者列表
    base_dir = r"D:\2025_important_projects\data\chbmit_biomarkers"
    patient_ids = ["01", "02", "03", "04", "05", "06"]
    output_dir = r"D:\2025_important_projects\paper-main\biomarker_analysis_results\cross_patient_01_06"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("跨患者生物标志物统计分析 (患者 01-06)")
    print("=" * 80)

    # 1. 加载多患者数据
    print("\n加载各患者 filtered_biomarkers...")
    all_biomarkers, by_patient = load_cross_patient_biomarkers(base_dir, patient_ids)
    if not all_biomarkers:
        print("错误: 未加载到任何生物标志物")
        return

    # 2. 加载筛选统计（可选）
    filtering_stats_per_patient = load_filtering_stats(base_dir, patient_ids)

    # 3. 跨患者重叠与类型分布
    overlap_stats = cross_patient_primitive_id_overlap(by_patient)
    cross_basic = cross_patient_basic_stats(all_biomarkers, by_patient)

    # 4. 跨患者总览图与类型热图
    plot_cross_patient_overview(all_biomarkers, by_patient, overlap_stats, output_dir)
    plot_patient_type_heatmap(by_patient, output_dir)

    # 5. 使用原分析模块对「合并后的全体」做总览、对比、Top 标志物、报告（与单患者分析方式一致）
    print("\n对合并数据做整体统计与可视化...")
    plot_overview(all_biomarkers, output_dir)
    plot_comparison(all_biomarkers, output_dir)
    plot_top_biomarkers(all_biomarkers, output_dir, top_n=10)
    generate_report(all_biomarkers, output_dir)

    # 6. 生成跨患者专用报告与 CSV
    generate_cross_patient_report(
        all_biomarkers,
        by_patient,
        overlap_stats,
        filtering_stats_per_patient,
        output_dir,
    )

    # 7. 将「在 ≥2 个患者中均出现的 primitive_id」对应的 .npz 复制到 data 文件夹
    data_folder = r"D:\2025_important_projects\data"
    copy_shared_biomarkers_to_data(all_biomarkers, overlap_stats, data_folder)

    print("\n" + "=" * 80)
    print("跨患者分析完成，结果已保存到:", output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()

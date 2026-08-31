#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUT_DIR = PROJECT_ROOT / "pictures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TYPE_ORDER = ["microstate", "shapelet", "timefreq"]
TYPE_LABEL = {
    "microstate": "Microstate",
    "shapelet": "Shapelet",
    "timefreq": "Time-frequency",
}
TYPE_COLOR = {
    "microstate": ("#e8f4ff", "#3f7fb3"),
    "shapelet": ("#fff1dc", "#b06b13"),
    "timefreq": ("#efe7ff", "#6b4da9"),
}
FALLBACK_COUNTS = {"microstate": 39, "shapelet": 46, "timefreq": 536}


def add_box(ax, x, y, w, h, fc="#ffffff", ec="#222222", lw=1.2, radius=0.05):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
    )


def add_arrow(ax, p1, p2, color="#3a3a3a", lw=1.3, ms=11, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle="--" if dashed else "-",
        )
    )


def find_latest_stats_report(root: Path) -> Path | None:
    reports = [p for p in root.rglob("biomarker_statistics_report.txt") if ".venv" not in {x.lower() for x in p.parts}]
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime)


def parse_counts(report_path: Path) -> dict[str, int]:
    txt = report_path.read_text(encoding="utf-8", errors="ignore")
    counts = {k: 0 for k in TYPE_ORDER}
    pairs = re.findall(r"^\s*(shapelet|timefreq|microstate)\s*:\s*(\d+)", txt, flags=re.MULTILINE)
    for k, v in pairs:
        counts[k] = int(v)
    if sum(counts.values()) > 0:
        return counts
    m = re.search(r"类型分布:\s*(\{.*?\})", txt, flags=re.DOTALL)
    if m:
        try:
            d = ast.literal_eval(m.group(1))
            for k in TYPE_ORDER:
                if isinstance(d.get(k), int):
                    counts[k] = d[k]
        except (ValueError, SyntaxError):
            pass
    return counts if sum(counts.values()) > 0 else FALLBACK_COUNTS.copy()


def draw_left_count_panel(ax, counts: dict[str, int], source_name: str):
    total = sum(counts.values())
    add_box(ax, 0.4, 0.9, 4.2, 5.9, fc="#fbfdff", ec="#264f79", lw=1.5, radius=0.08)
    ax.text(2.5, 6.42, "Primitive Library", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#193d64")
    ax.text(2.5, 6.08, f"N={total}", ha="center", va="center", fontsize=10.5)

    # donut
    cx, cy = 2.5, 5.1
    r, w = 0.85, 0.28
    ang = 90.0
    for k in TYPE_ORDER:
        frac = counts[k] / total if total else 0.0
        nxt = ang - 360.0 * frac
        ax.add_patch(Wedge((cx, cy), r, nxt, ang, width=w, facecolor=TYPE_COLOR[k][0], edgecolor=TYPE_COLOR[k][1], linewidth=1.0))
        ang = nxt
    ax.text(cx, cy, str(total), ha="center", va="center", fontsize=12.5, fontweight="bold", color="#24374b")

    start_y = 4.0
    for i, k in enumerate(TYPE_ORDER):
        y = start_y - i * 0.9
        v = counts[k]
        p = v / total * 100 if total else 0
        fc, ec = TYPE_COLOR[k]
        add_box(ax, 0.7, y, 3.6, 0.66, fc=fc, ec=ec, lw=1.0, radius=0.04)
        ax.text(0.9, y + 0.33, TYPE_LABEL[k], ha="left", va="center", fontsize=8.8, color="#1f2f40")
        ax.text(3.2, y + 0.33, f"{v}", ha="right", va="center", fontsize=8.8, fontweight="bold", color="#1f2f40")
        ax.text(4.1, y + 0.33, f"{p:.1f}%", ha="right", va="center", fontsize=8.8, color="#1f2f40")

    # ratio strip
    x0, y0, ww, hh = 0.7, 1.45, 3.6, 0.3
    ax.add_patch(Rectangle((x0, y0), ww, hh, linewidth=1.0, edgecolor="#333333", facecolor="#ffffff"))
    cur = x0
    for k in TYPE_ORDER:
        seg = ww * (counts[k] / total if total else 0)
        ax.add_patch(Rectangle((cur, y0), seg, hh, linewidth=0, facecolor=TYPE_COLOR[k][0]))
        cur += seg

    ax.text(2.5, 1.12, f"Source: {Path(source_name).name}", ha="center", va="center", fontsize=7.2, color="#5a6776")


def icon_wave(ax, x, y, s=1.0):
    ax.add_patch(Rectangle((x - 0.5 * s, y - 0.15 * s), 0.32 * s, 0.18 * s, edgecolor="#9b6112", facecolor="#ffdcae", linewidth=0.8))
    ax.add_patch(Rectangle((x - 0.12 * s, y - 0.07 * s), 0.32 * s, 0.18 * s, edgecolor="#9b6112", facecolor="#ffdcae", linewidth=0.8))
    ax.plot([x + 0.28 * s, x + 0.5 * s, x + 0.72 * s, x + 0.94 * s], [y - 0.06 * s, y + 0.2 * s, y - 0.02 * s, y + 0.15 * s], color="#9b6112", linewidth=1.4)


def icon_graph(ax, x, y, s=1.0):
    pts = [(x - 0.5 * s, y + 0.12 * s), (x - 0.15 * s, y + 0.25 * s), (x + 0.2 * s, y + 0.05 * s), (x + 0.55 * s, y + 0.2 * s)]
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.06 * s, facecolor="#72cb81", edgecolor="#2b7d39", linewidth=0.8))
    add_arrow(ax, (x - 0.44 * s, y + 0.13 * s), (x - 0.21 * s, y + 0.23 * s), color="#2e7c3d", lw=1.0, ms=8)
    add_arrow(ax, (x - 0.08 * s, y + 0.22 * s), (x + 0.14 * s, y + 0.08 * s), color="#2e7c3d", lw=1.0, ms=8)
    add_arrow(ax, (x + 0.27 * s, y + 0.08 * s), (x + 0.49 * s, y + 0.18 * s), color="#2e7c3d", lw=1.0, ms=8)
    add_arrow(ax, (x - 0.5 * s, y - 0.22 * s), (x + 0.55 * s, y - 0.22 * s), color="#2e7c3d", lw=0.9, ms=8, dashed=True)


def icon_filter(ax, x, y, s=1.0):
    funnel = Polygon(
        [(x - 0.55 * s, y + 0.3 * s), (x + 0.55 * s, y + 0.3 * s), (x + 0.2 * s, y - 0.02 * s), (x + 0.2 * s, y - 0.25 * s), (x - 0.2 * s, y - 0.35 * s), (x - 0.2 * s, y - 0.02 * s)],
        closed=True,
        edgecolor="#6a4aa6",
        facecolor="#e8dcff",
        linewidth=0.9,
    )
    ax.add_patch(funnel)
    for px, py, c in [(x - 0.78 * s, y + 0.26 * s, "#8360bb"), (x - 0.62 * s, y + 0.08 * s, "#c2b2e3"), (x - 0.46 * s, y + 0.24 * s, "#8360bb")]:
        ax.add_patch(Circle((px, py), 0.04 * s, facecolor=c, edgecolor="#5e3f98", linewidth=0.6))
        add_arrow(ax, (px + 0.08 * s, py - 0.01 * s), (x - 0.3 * s, y + 0.12 * s), color="#6b4ba8", lw=0.7, ms=7)
    for px in [x + 0.56 * s, x + 0.72 * s]:
        ax.add_patch(Circle((px, y - 0.35 * s), 0.04 * s, facecolor="#8360bb", edgecolor="#5e3f98", linewidth=0.6))


def icon_threshold(ax, x, y, s=1.0):
    ax.plot([x - 0.6 * s, x + 0.15 * s], [y - 0.3 * s, y - 0.3 * s], color="#a43f51", linewidth=0.9)
    ax.plot([x - 0.5 * s, x - 0.5 * s], [y - 0.35 * s, y + 0.32 * s], color="#a43f51", linewidth=0.9)
    ax.plot([x - 0.42 * s, x - 0.2 * s, x + 0.02 * s, x + 0.14 * s], [y - 0.2 * s, y - 0.1 * s, y - 0.14 * s, y + 0.1 * s], color="#a43f51", linewidth=1.2)
    ax.plot([x - 0.6 * s, x + 0.15 * s], [y - 0.02 * s, y - 0.02 * s], color="#a43f51", linewidth=0.8, linestyle="--")
    ax.plot([x + 0.35 * s, x + 0.9 * s], [y + 0.15 * s, y + 0.15 * s], color="#a43f51", linewidth=0.9)
    ax.plot([x + 0.62 * s, x + 0.62 * s], [y + 0.15 * s, y - 0.15 * s], color="#a43f51", linewidth=0.9)
    ax.add_patch(Circle((x + 0.48 * s, y - 0.18 * s), 0.08 * s, facecolor="#ffc1c9", edgecolor="#a43f51", linewidth=0.7))
    ax.add_patch(Circle((x + 0.76 * s, y - 0.05 * s), 0.13 * s, facecolor="#ff94a2", edgecolor="#a43f51", linewidth=0.7))


def draw_linear_mechanism_flow(ax):
    # base rail
    y = 4.4
    x_nodes = [6.1, 8.0, 9.9, 11.8]
    bg = ["#fff5e9", "#effaf0", "#f6efff", "#fff0f2"]
    ec = ["#b8771a", "#2f8240", "#6b4ba8", "#b24557"]

    for i, x in enumerate(x_nodes):
        ax.add_patch(Circle((x, y), 0.72, facecolor=bg[i], edgecolor=ec[i], linewidth=1.3))
    add_arrow(ax, (6.86, y), (7.24, y), lw=1.2)
    add_arrow(ax, (8.76, y), (9.14, y), lw=1.2)
    add_arrow(ax, (10.66, y), (11.04, y), lw=1.2)

    icon_wave(ax, 6.1, y, s=0.9)
    icon_graph(ax, 8.0, y, s=0.95)
    icon_filter(ax, 9.9, y, s=0.95)
    icon_threshold(ax, 11.8, y, s=0.95)

    return x_nodes, y


def draw_outcome_panel(ax, x_nodes, y):
    add_box(ax, 7.2, 1.25, 5.6, 1.8, fc="#f4f6f8", ec="#2c2c2c", lw=1.2, radius=0.06)
    ax.plot([7.6, 8.25, 8.9], [1.55, 2.0, 2.5], color="#207d30", linewidth=1.9)
    add_arrow(ax, (8.86, 2.45), (9.18, 2.62), color="#207d30", lw=1.5, ms=10)

    ax.add_patch(Circle((10.25, 2.06), 0.34, facecolor="#ffecec", edgecolor="#b23f3f", linewidth=1.0))
    ax.add_patch(Rectangle((10.12, 1.82), 0.25, 0.14, linewidth=0.7, edgecolor="#b23f3f", facecolor="#ffcaca"))
    ax.plot([9.98, 10.52], [1.77, 2.33], color="#b23f3f", linewidth=1.2)

    ax.add_patch(Rectangle((11.15, 1.72), 0.82, 1.05, linewidth=0.8, edgecolor="#606060", facecolor="#ffffff"))
    ax.plot([11.28, 11.5, 11.79], [2.32, 2.15, 2.52], color="#2f6b2f", linewidth=1.0)
    ax.plot([11.28, 11.5, 11.79], [2.05, 1.88, 2.24], color="#2f6b2f", linewidth=1.0)

    # merge arrows from mechanism
    for x in x_nodes[1:]:
        add_arrow(ax, (x, y - 0.76), (9.95, 3.08), lw=1.0)


def main():
    report = find_latest_stats_report(PROJECT_ROOT)
    if report is None:
        counts = FALLBACK_COUNTS.copy()
        source = "fallback"
        print("Warning: no biomarker_statistics_report.txt found, using fallback counts.")
    else:
        counts = parse_counts(report)
        source = report.relative_to(PROJECT_ROOT).as_posix()
        print(f"Using report: {report}")
        print(f"Parsed counts: {counts}")

    fig, ax = plt.subplots(figsize=(14.8, 7.6))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    ax.text(7.4, 7.25, "Primitive Library and Mechanism Schematic", ha="center", va="center", fontsize=13, fontweight="bold")

    draw_left_count_panel(ax, counts, source)
    x_nodes, y = draw_linear_mechanism_flow(ax)
    draw_outcome_panel(ax, x_nodes, y)

    # input arrows
    add_arrow(ax, (4.7, 5.4), (5.35, 4.72), lw=1.2)
    add_arrow(ax, (4.7, 3.35), (5.35, 4.08), lw=1.2, dashed=True)
    ax.text(7.4, 0.22, "minimal visual flow: quantity -> mechanism -> outcome", ha="center", va="bottom", fontsize=8.0, color="#5c6775")

    plt.tight_layout()
    out_base = OUT_DIR / "fig5_primitive_library_interpretability"
    saved = []
    for ext in ["png", "pdf"]:
        path = out_base.with_suffix(f".{ext}")
        try:
            fig.savefig(path, bbox_inches="tight", dpi=280 if ext == "png" else None)
            saved.append(path)
            print(f"Saved {path}")
        except PermissionError:
            alt = out_base.with_name(f"{out_base.name}_{datetime.now().strftime('%H%M%S')}.{ext}")
            fig.savefig(alt, bbox_inches="tight", dpi=280 if ext == "png" else None)
            saved.append(alt)
            print(f"Locked target, saved alternate {alt}")
    plt.close(fig)
    print("Outputs:", [str(p) for p in saved])


if __name__ == "__main__":
    main()

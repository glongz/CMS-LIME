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
    "microstate": ("#dff2ff", "#2f80b9"),
    "shapelet": ("#ffe9cc", "#b06a12"),
    "timefreq": ("#ece2ff", "#6a4aa6"),
}
FALLBACK_COUNTS = {"microstate": 39, "shapelet": 46, "timefreq": 536}


def add_box(ax, x, y, w, h, fc="#ffffff", ec="#222222", lw=1.2, r=0.06):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
    )


def add_arrow(ax, p1, p2, color="#333333", lw=1.2, ms=10, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=ms,
            color=color,
            linewidth=lw,
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
        except (SyntaxError, ValueError):
            pass
    return counts if sum(counts.values()) > 0 else FALLBACK_COUNTS.copy()


def draw_count_panel(ax, counts: dict[str, int], source_name: str):
    total = sum(counts.values())
    add_box(ax, 0.35, 0.7, 4.35, 6.55, fc="#fbfdff", ec="#2b5b86", lw=1.45, r=0.07)
    ax.text(2.52, 6.9, "Primitive Library", ha="center", va="center", fontsize=11.7, fontweight="bold", color="#173a61")
    ax.text(2.52, 6.52, f"Total N={total}", ha="center", va="center", fontsize=10)

    center = (2.52, 5.25)
    r, width = 0.95, 0.3
    angle = 90
    for k in TYPE_ORDER:
        frac = counts[k] / total if total else 0.0
        nxt = angle - 360 * frac
        ax.add_patch(Wedge(center, r, nxt, angle, width=width, facecolor=TYPE_COLOR[k][0], edgecolor=TYPE_COLOR[k][1], linewidth=1.0))
        angle = nxt
    ax.text(center[0], center[1], f"{total}", ha="center", va="center", fontsize=13, fontweight="bold", color="#1d2f44")

    base_y = 4.1
    for i, k in enumerate(TYPE_ORDER):
        y = base_y - i * 0.95
        v = counts[k]
        p = (v / total * 100) if total else 0
        fc, ec = TYPE_COLOR[k]
        add_box(ax, 0.62, y, 3.95, 0.72, fc=fc, ec=ec, lw=1.0, r=0.04)
        ax.text(0.82, y + 0.36, TYPE_LABEL[k], ha="left", va="center", fontsize=9.0, color="#22313f")
        ax.text(3.45, y + 0.36, f"{v}", ha="right", va="center", fontsize=9.0, fontweight="bold", color="#22313f")
        ax.text(4.38, y + 0.36, f"{p:.1f}%", ha="right", va="center", fontsize=9.0, color="#22313f")

    bar_x, bar_y, bar_w, bar_h = 0.62, 1.5, 3.95, 0.33
    ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, linewidth=1.0, edgecolor="#2e2e2e", facecolor="#ffffff"))
    cur = bar_x
    for k in TYPE_ORDER:
        frac = counts[k] / total if total else 0.0
        w = bar_w * frac
        ax.add_patch(Rectangle((cur, bar_y), w, bar_h, linewidth=0, facecolor=TYPE_COLOR[k][0]))
        cur += w

    ax.text(2.52, 1.14, f"Source: {Path(source_name).name}", ha="center", va="center", fontsize=7.5, color="#566271")


def draw_icon_m1(ax, x, y, w, h):
    add_box(ax, x, y, w, h, fc="#fff5e9", ec="#b8771a", lw=1.2, r=0.05)
    for i in range(3):
        ax.add_patch(Rectangle((x + 0.2 + i * 0.42, y + h - 0.55 - i * 0.06), 0.3, 0.17, linewidth=0.8, edgecolor="#986012", facecolor="#ffddb0"))
    ax.plot([x + 1.6, x + 1.85, x + 2.1, x + 2.35], [y + 0.45, y + 0.78, y + 0.42, y + 0.72], color="#986012", linewidth=1.4)


def draw_icon_m2(ax, x, y, w, h):
    add_box(ax, x, y, w, h, fc="#effaf0", ec="#2f8240", lw=1.2, r=0.05)
    pts = [(x + 0.32, y + 1.0), (x + 0.85, y + 1.15), (x + 1.32, y + 0.86), (x + 1.88, y + 1.07), (x + 2.35, y + 0.88)]
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.065, facecolor="#73cb82", edgecolor="#2a7a3a", linewidth=0.9))
    for a, b in [((x + 0.38, y + 1.01), (x + 0.8, y + 1.13)), ((x + 0.92, y + 1.1), (x + 1.27, y + 0.9)), ((x + 1.38, y + 0.88), (x + 1.83, y + 1.04)), ((x + 1.95, y + 1.05), (x + 2.3, y + 0.9))]:
        add_arrow(ax, a, b, color="#2e7c3d", lw=1.0, ms=8)
    add_arrow(ax, (x + 0.3, y + 0.35), (x + 2.3, y + 0.35), color="#2e7c3d", lw=0.9, ms=8, dashed=True)


def draw_icon_m3(ax, x, y, w, h):
    add_box(ax, x, y, w, h, fc="#f6efff", ec="#6b4ba8", lw=1.2, r=0.05)
    funnel = Polygon([(x + 0.5, y + 1.25), (x + 2.45, y + 1.25), (x + 1.9, y + 0.78), (x + 1.9, y + 0.45), (x + 1.35, y + 0.28), (x + 1.35, y + 0.78)], closed=True, edgecolor="#6b4ba8", facecolor="#e8dbff", linewidth=1.0)
    ax.add_patch(funnel)
    for px, py, c in [(x + 0.2, y + 1.17, "#8462bb"), (x + 0.36, y + 1.0, "#c2b2e3"), (x + 0.54, y + 1.18, "#8462bb")]:
        ax.add_patch(Circle((px, py), 0.045, facecolor=c, edgecolor="#5e3f98", linewidth=0.6))
        add_arrow(ax, (px + 0.08, py - 0.02), (x + 0.85, y + 1.0), color="#6b4ba8", lw=0.75, ms=7)
    for px in [x + 2.25, x + 2.45]:
        ax.add_patch(Circle((px, y + 0.28), 0.045, facecolor="#8462bb", edgecolor="#5e3f98", linewidth=0.6))


def draw_icon_m4(ax, x, y, w, h):
    add_box(ax, x, y, w, h, fc="#fff0f2", ec="#b24557", lw=1.2, r=0.05)
    ax.plot([x + 0.22, x + 1.35], [y + 0.35, y + 0.35], color="#a43f51", linewidth=0.9)
    ax.plot([x + 0.32, x + 0.32], [y + 0.28, y + 1.2], color="#a43f51", linewidth=0.9)
    ax.plot([x + 0.42, x + 0.7, x + 0.98, x + 1.25], [y + 0.52, y + 0.6, y + 0.58, y + 0.95], color="#a43f51", linewidth=1.35)
    ax.plot([x + 0.22, x + 1.35], [y + 0.7, y + 0.7], color="#a43f51", linewidth=0.9, linestyle="--")
    ax.plot([x + 1.65, x + 2.45], [y + 1.0, y + 1.0], color="#a43f51", linewidth=1.0)
    ax.plot([x + 2.05, x + 2.05], [y + 1.0, y + 0.62], color="#a43f51", linewidth=1.0)
    ax.add_patch(Circle((x + 1.8, y + 0.56), 0.11, facecolor="#ffc2ca", edgecolor="#a43f51", linewidth=0.8))
    ax.add_patch(Circle((x + 2.28, y + 0.76), 0.17, facecolor="#ff94a2", edgecolor="#a43f51", linewidth=0.8))


def draw_mechanism_radial(ax):
    # center hub
    hub = (9.3, 4.2)
    ax.add_patch(Circle(hub, 0.45, facecolor="#e9eef5", edgecolor="#4b5f75", linewidth=1.2))

    m1 = (8.0, 5.75, 2.8, 1.5)
    m2 = (5.95, 3.95, 2.8, 1.5)
    m3 = (10.45, 3.95, 2.8, 1.5)
    m4 = (8.0, 2.15, 2.8, 1.5)

    draw_icon_m1(ax, *m1)
    draw_icon_m2(ax, *m2)
    draw_icon_m3(ax, *m3)
    draw_icon_m4(ax, *m4)

    add_arrow(ax, (8.0 + 2.2, 5.75 + 0.2), (9.05, 4.62), lw=1.1)
    add_arrow(ax, (5.95 + 2.65, 3.95 + 0.72), (8.82, 4.1), lw=1.1)
    add_arrow(ax, (10.45 + 0.15, 3.95 + 0.72), (9.78, 4.1), lw=1.1)
    add_arrow(ax, (8.0 + 2.2, 2.15 + 1.25), (9.05, 3.75), lw=1.1)

    return hub


def draw_outcome(ax, hub):
    add_box(ax, 10.2, 0.62, 4.15, 1.55, fc="#f4f6f8", ec="#2e2e2e", lw=1.2, r=0.05)
    ax.plot([10.5, 11.1, 11.65], [0.95, 1.28, 1.76], color="#207d30", linewidth=1.8)
    add_arrow(ax, (11.6, 1.72), (11.92, 1.84), color="#207d30", lw=1.5, ms=10)
    ax.add_patch(Circle((12.7, 1.3), 0.35, facecolor="#ffecec", edgecolor="#b23f3f", linewidth=1.1))
    ax.add_patch(Rectangle((12.56, 1.02), 0.27, 0.16, linewidth=0.8, edgecolor="#b23f3f", facecolor="#ffcaca"))
    ax.plot([12.43, 12.97], [0.98, 1.58], color="#b23f3f", linewidth=1.3)
    ax.add_patch(Rectangle((13.4, 0.93), 0.72, 1.0, linewidth=0.9, edgecolor="#606060", facecolor="#ffffff"))
    ax.plot([13.52, 13.71, 13.96], [1.52, 1.35, 1.74], color="#2f6b2f", linewidth=1.0)
    ax.plot([13.52, 13.71, 13.96], [1.24, 1.08, 1.46], color="#2f6b2f", linewidth=1.0)
    add_arrow(ax, (hub[0] + 0.4, hub[1] - 0.2), (10.15, 1.56), lw=1.2)


def render():
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

    fig, ax = plt.subplots(figsize=(14.8, 8.0))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    ax.text(7.4, 7.72, "Primitive Library and Mechanism Schematic (Style B)", ha="center", va="center", fontsize=13.2, fontweight="bold")

    draw_count_panel(ax, counts, source)
    hub = draw_mechanism_radial(ax)
    draw_outcome(ax, hub)

    add_arrow(ax, (4.78, 5.35), (8.72, 4.35), lw=1.25)
    add_arrow(ax, (4.78, 3.75), (8.65, 4.02), lw=1.25, dashed=True)
    ax.text(7.4, 0.16, "style B: radial mechanism composition", ha="center", va="bottom", fontsize=8.0, color="#596474")

    plt.tight_layout()
    base = OUT_DIR / "fig5_primitive_library_interpretability_B"
    saved = []
    for ext in ["png", "pdf"]:
        path = base.with_suffix(f".{ext}")
        try:
            fig.savefig(path, bbox_inches="tight", dpi=260 if ext == "png" else None)
            saved.append(path)
            print(f"Saved {path}")
        except PermissionError:
            alt = base.with_name(f"{base.name}_{datetime.now().strftime('%H%M%S')}.{ext}")
            fig.savefig(alt, bbox_inches="tight", dpi=260 if ext == "png" else None)
            saved.append(alt)
            print(f"Locked target, saved alternate {alt}")
    plt.close(fig)
    print("Outputs:", [str(p) for p in saved])


if __name__ == "__main__":
    render()

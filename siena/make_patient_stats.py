# -*- coding: utf-8 -*-
"""Build per-patient Siena stats from segment_info.json (protocol 30-1-240).

Usage (from ``paper-main`` root)::

  python -m siena.make_patient_stats
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .common import FS_OUT, SUBJECTS_IN_RELEASE
from .paths import SIENA_CLEAN_ROOT, SIENA_REPORTS_DIR, SIENA_SEG_ROOT


def samples_to_hours(n: int) -> float:
    return n / FS_OUT / 3600.0


def summarize(pid: str) -> dict:
    seg_path = SIENA_SEG_ROOT / pid / "segment_info.json"
    clean_dir = SIENA_CLEAN_ROOT / pid
    row = {
        "patient": pid,
        "has_segment_info": seg_path.is_file(),
        "retained_npy": 0,
        "n_onset": 0,
        "n_preictal_events": 0,
        "preictal_hours": 0.0,
        "interictal_hours": 0.0,
        "n_pre_blocks": 0,
        "n_inter_blocks": 0,
    }
    if clean_dir.is_dir():
        row["retained_npy"] = len(list(clean_dir.glob("*.npy")))
    if not seg_path.is_file():
        return row
    segs = json.loads(seg_path.read_text(encoding="utf-8"))
    pre_ids = set()
    for s in segs:
        lab = s["Label"]
        a, b = s["Span"]
        dur = max(0, b - a)
        if lab.startswith("Pre"):
            row["n_pre_blocks"] += 1
            row["preictal_hours"] += samples_to_hours(dur)
            digits = "".join(ch for ch in lab[3:] if ch.isdigit())
            if digits:
                pre_ids.add(digits)
        elif lab.startswith("Inter"):
            row["n_inter_blocks"] += 1
            row["interictal_hours"] += samples_to_hours(dur)
        elif lab.startswith("Onset"):
            row["n_onset"] += 1
    row["n_preictal_events"] = len(pre_ids)
    row["preictal_hours"] = round(row["preictal_hours"], 3)
    row["interictal_hours"] = round(row["interictal_hours"], 3)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=SIENA_REPORTS_DIR)
    args = ap.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize(pid) for pid in SUBJECTS_IN_RELEASE]
    out = out_dir / "siena_patient_stats_30-1-240.csv"
    keys = list(rows[0].keys()) if rows else []
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
    eligible = [r for r in rows if r["n_preictal_events"] > 0]
    print(f"patients with ≥1 preictal event: {len(eligible)}")
    for r in eligible:
        print(
            f"  {r['patient']}: onset={r['n_onset']} pre_events={r['n_preictal_events']} "
            f"pre_h={r['preictal_hours']} inter_h={r['interictal_hours']}"
        )


if __name__ == "__main__":
    main()

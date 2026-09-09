# -*- coding: utf-8 -*-
"""
Step A — Siena Scalp EEG inventory / C18 feasibility.

Reads the PhysioNet release at ``SIENA_RAW_ROOT``, reports per-EDF channel
coverage against the CHB-matched 18 bipolar montage, and summarises seizure lists.

Usage (from ``paper-main`` root)::

  python -m siena.inventory
  python -m siena.inventory --raw-root D:\\public_data\\siena-scalp-eeg-database-1.0.0
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import mne

from .common import (
    SUBJECTS_IN_RELEASE,
    build_electrode_index,
    can_build_c18,
    missing_for_c18,
    resolve_edf_name,
)
from .paths import SIENA_RAW_ROOT, SIENA_REPORTS_DIR


def parse_seizure_list(path: Path) -> list[dict]:
    """Parse Seizures-list-PNXX.txt into per-seizure dicts.

    PN00-style: each seizure block carries File name + Registration times.
    PN01-style: File name / Registration appear once in the header and are
    inherited by subsequent seizures; seizure clocks may be labelled
    ``Start time`` / ``End time`` instead of ``Seizure start/end time``.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    seizures: list[dict] = []
    headers = list(re.finditer(r"Seizure\s+n\s+(\d+)([^\n]*)", text, flags=re.IGNORECASE))
    parts = re.split(r"Seizure\s+n\s+\d+[^\n]*\n", text, flags=re.IGNORECASE)
    preamble = parts[0] if parts else ""
    body_parts = parts[1:]

    def _field(blob: str, *keys: str) -> str | None:
        for key in keys:
            m = re.search(rf"{key}:\s*([^\n]+)", blob, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    cur_file = _field(preamble, "File name")
    cur_reg_start = _field(preamble, "Registration start time")
    cur_reg_end = _field(preamble, "Registration end time")

    for hdr, body in zip(headers, body_parts):
        sid = hdr.group(1)
        note = hdr.group(2).strip()
        f = _field(body, "File name")
        rs = _field(body, "Registration start time")
        re_ = _field(body, "Registration end time")
        if f:
            cur_file = f
        if rs:
            cur_reg_start = rs
        if re_:
            cur_reg_end = re_
        seizures.append(
            {
                "seizure_id": sid,
                "note": note,
                "file": cur_file,
                "reg_start": cur_reg_start,
                "reg_end": cur_reg_end,
                "sz_start": _field(body, "Seizure start time", "Start time"),
                "sz_end": _field(body, "Seizure end time", "End time"),
            }
        )
    return seizures


def inventory_edf(edf_path: Path) -> dict:
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    ch_names = list(raw.ch_names)
    elec = build_electrode_index(ch_names)
    miss = missing_for_c18(elec)
    return {
        "file": edf_path.name,
        "sfreq": float(raw.info["sfreq"]),
        "n_channels_raw": len(ch_names),
        "n_unipolar_mapped": len(elec),
        "can_c18": can_build_c18(elec),
        "missing_electrodes": ";".join(miss),
        "mapped_electrodes": ";".join(sorted(elec.keys())),
        "duration_sec": float(raw.n_times / raw.info["sfreq"]),
        "raw_channel_names": "|".join(ch_names),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=SIENA_RAW_ROOT)
    ap.add_argument("--out-dir", type=Path, default=SIENA_REPORTS_DIR)
    args = ap.parse_args()
    raw_root: Path = args.raw_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    edf_rows: list[dict] = []
    patient_rows: list[dict] = []
    seizure_rows: list[dict] = []

    for pid in SUBJECTS_IN_RELEASE:
        pdir = raw_root / pid
        if not pdir.is_dir():
            patient_rows.append(
                {
                    "patient": pid,
                    "exists": False,
                    "n_edf": 0,
                    "n_edf_c18": 0,
                    "n_seizures_listed": 0,
                    "eligible": False,
                    "notes": "missing_folder",
                }
            )
            continue

        edfs = sorted(pdir.glob("*.edf"))
        list_path = pdir / f"Seizures-list-{pid}.txt"
        seizures = parse_seizure_list(list_path) if list_path.is_file() else []

        n_c18 = 0
        notes: list[str] = []
        for edf in edfs:
            try:
                row = inventory_edf(edf)
            except Exception as exc:  # noqa: BLE001
                row = {
                    "file": edf.name,
                    "sfreq": "",
                    "n_channels_raw": "",
                    "n_unipolar_mapped": "",
                    "can_c18": False,
                    "missing_electrodes": "",
                    "mapped_electrodes": "",
                    "duration_sec": "",
                    "raw_channel_names": "",
                    "error": str(exc),
                }
                notes.append(f"{edf.name}:read_fail")
            row["patient"] = pid
            edf_rows.append(row)
            if row.get("can_c18"):
                n_c18 += 1

        edf_names = [e.name for e in edfs]
        for sz in seizures:
            resolved = resolve_edf_name(sz.get("file"), edf_names)
            row = {**sz, "file_resolved": resolved}
            seizure_rows.append({"patient": pid, **row})
            if not sz.get("file"):
                notes.append(f"sz{sz['seizure_id']}:no_file")
            elif resolved is None:
                notes.append(f"sz{sz['seizure_id']}:unresolved:{sz.get('file')}")

        sz_with_file = sum(
            1 for s in seizures if resolve_edf_name(s.get("file"), edf_names)
        )
        eligible = n_c18 > 0 and sz_with_file > 0
        if n_c18 == 0:
            notes.append("no_c18_edf")
        if sz_with_file == 0:
            notes.append("no_seizure_files")

        patient_rows.append(
            {
                "patient": pid,
                "exists": True,
                "n_edf": len(edfs),
                "n_edf_c18": n_c18,
                "n_seizures_listed": len(seizures),
                "n_seizures_with_file": sz_with_file,
                "eligible_preprocess": eligible,
                "notes": ";".join(notes),
            }
        )
        print(
            f"{pid}: edf={len(edfs)} c18={n_c18} seizures={len(seizures)} "
            f"eligible={eligible} notes={';'.join(notes) or '-'}"
        )

    def _write(name: str, rows: list[dict]) -> Path:
        path = out_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        return path

    p_csv = _write("siena_inventory_patients.csv", patient_rows)
    e_csv = _write("siena_inventory_edfs.csv", edf_rows)
    s_csv = _write("siena_inventory_seizures.csv", seizure_rows)

    eligible = [r["patient"] for r in patient_rows if r.get("eligible_preprocess")]
    ineligible = [r for r in patient_rows if not r.get("eligible_preprocess")]
    md = out_dir / "siena_c18_feasibility.md"
    lines = [
        "# Siena C18 feasibility (inventory)",
        "",
        f"- Raw root: `{raw_root}`",
        f"- Patients in release: {len(SUBJECTS_IN_RELEASE)}",
        f"- Eligible for preprocess ( ≥1 C18 EDF and ≥1 seizure file ): **{len(eligible)}**",
        f"- Eligible IDs: {', '.join(eligible) if eligible else '(none)'}",
        "",
        "## Ineligible / flagged",
        "",
    ]
    for r in ineligible:
        lines.append(
            f"- `{r['patient']}`: edf_c18={r.get('n_edf_c18')} "
            f"seizures={r.get('n_seizures_listed')} notes={r.get('notes')}"
        )
    lines += [
        "",
        "## Outputs",
        "",
        f"- `{p_csv.name}`",
        f"- `{e_csv.name}`",
        f"- `{s_csv.name}`",
        "",
        "## Channel normalisation notes",
        "",
        "- EDF names like `EEG Fp1` / `EEG FP2` / `EEG T3` are mapped to `FP1`/`FP2`/`T7`.",
        "- Legacy T3/T4/T5/T6 → T7/T8/P7/P8.",
        "- Seizure-list channel legend may mark O1 as `1`; EDF usually has `EEG O1`.",
        "",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", p_csv)
    print("wrote", e_csv)
    print("wrote", s_csv)
    print("wrote", md)


if __name__ == "__main__":
    main()

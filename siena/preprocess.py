# -*- coding: utf-8 -*-
"""
Step B — Preprocess Siena Scalp EEG to the CHB-MIT matched protocol.

For each eligible recording:
  1) map unipolar labels (T3→T7, …); build 18 bipolar channels in CHB maj order
  2) anti-alias / resample 512 → 256 Hz; 50 Hz notch
  3) write float32 (18, T) .npy + channel_info.json + datetime_info.json
  4) build segment_info.json under 30-1-240 (Span in samples @ 256 Hz)

Protocol (manuscript Sec.4.1):
  SOP=30 min, SPH=1 min
  preictal = [onset-31 min, onset-1 min)
  lead seizure requires ≥31 min seizure-free preamble in-file
  interictal = 2 h windows before that preictal / after offset (with postictal guard)

Usage (from ``paper-main`` root)::

  python -m siena.preprocess --dry-run
  python -m siena.preprocess --patients PN00 PN01
  python -m siena.preprocess
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from .common import (
    BIPOLAR_PAIRS,
    CHB_MAJ_BIPOLAR_ORDER,
    FS_OUT,
    INTERICTAL_H,
    POSTICTAL_GUARD_MIN,
    PREICTAL_MIN,
    SPH_MIN,
    SUBJECTS_IN_RELEASE,
    build_electrode_index,
    can_build_c18,
    clock_delta_seconds,
    missing_for_c18,
    resolve_edf_name,
)
from .inventory import parse_seizure_list
from .paths import (
    SIENA_CLEAN_ROOT,
    SIENA_RAW_ROOT,
    SIENA_REPORTS_DIR,
    SIENA_SEG_ROOT,
)


@dataclass
class SeizureEvent:
    seizure_id: str
    edf_name: str
    onset_sec: float  # relative to file start (registration)
    offset_sec: float
    reg_start: str
    reg_end: str


def _stem_npy(edf_name: str) -> str:
    return Path(edf_name).stem + ".npy"


def extract_bipolar_array(raw: mne.io.BaseRaw) -> tuple[np.ndarray, list[str]]:
    """Return (C=18, T_native) float64 bipolar data in CHB maj order."""
    elec = build_electrode_index(list(raw.ch_names))
    miss = missing_for_c18(elec)
    if miss:
        raise RuntimeError(f"cannot build C18, missing {miss}")
    data = raw.get_data()  # (n_ch, T)
    rows = []
    for a, b in BIPOLAR_PAIRS:
        rows.append(data[elec[a]] - data[elec[b]])
    bip = np.stack(rows, axis=0)
    return bip, list(CHB_MAJ_BIPOLAR_ORDER)


def resample_notch(bip: np.ndarray, sfreq: float) -> np.ndarray:
    """bip (18, T) → (18, T_out) at FS_OUT with 50 Hz notch."""
    info = mne.create_info(CHB_MAJ_BIPOLAR_ORDER, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(bip, info, verbose=False)
    notches = [50.0]
    if sfreq / 2 > 100:
        notches.append(100.0)
    if sfreq / 2 > 150:
        notches.append(150.0)
    raw.notch_filter(notches, verbose=False)
    if abs(sfreq - FS_OUT) > 1e-6:
        raw.resample(FS_OUT, verbose=False)
    return raw.get_data().astype(np.float32)


def seizures_for_file(
    seizures: list[dict], edf_name: str, available_edfs: list[str]
) -> list[SeizureEvent]:
    out: list[SeizureEvent] = []
    for sz in seizures:
        resolved = resolve_edf_name(sz.get("file"), available_edfs)
        if resolved is None or resolved.lower() != edf_name.lower():
            continue
        if not sz.get("reg_start") or not sz.get("sz_start") or not sz.get("sz_end"):
            print(f"  skip seizure incomplete fields {edf_name} #{sz.get('seizure_id')}")
            continue
        try:
            onset = clock_delta_seconds(sz["reg_start"], sz["sz_start"])
            offset = clock_delta_seconds(sz["reg_start"], sz["sz_end"])
        except Exception as exc:  # noqa: BLE001
            print(f"  skip seizure parse {edf_name} #{sz.get('seizure_id')}: {exc}")
            continue
        if offset < onset:
            offset = onset + 1.0
        out.append(
            SeizureEvent(
                seizure_id=str(sz["seizure_id"]),
                edf_name=edf_name,
                onset_sec=float(onset),
                offset_sec=float(offset),
                reg_start=sz["reg_start"],
                reg_end=sz["reg_end"] or "",
            )
        )
    out.sort(key=lambda e: e.onset_sec)
    return out


def build_segments_for_file(
    edf_name: str,
    duration_sec: float,
    events: list[SeizureEvent],
    global_onset_counter_start: int,
) -> tuple[list[dict], int]:
    """
    Build Pre / Onset / Inter entries for one file (Span in samples @ FS_OUT).
    Returns (segments, next_global_onset_index).
    """
    segs: list[dict] = []
    npy = _stem_npy(edf_name)
    n_samples = int(round(duration_sec * FS_OUT))
    onset_idx = global_onset_counter_start
    noninter: list[tuple[float, float]] = []

    for ev in events:
        onset = min(max(ev.onset_sec, 0.0), duration_sec)
        offset = min(max(ev.offset_sec, onset), duration_sec)

        prev_clear = 0.0
        if noninter:
            prev_clear = noninter[-1][1]
        free_before = onset - prev_clear
        if free_before < PREICTAL_MIN * 60:
            onset_idx += 1
            segs.append(
                {
                    "Label": f"Onset{onset_idx}",
                    "File": npy,
                    "Span": [
                        int(round(onset * FS_OUT)),
                        int(round(offset * FS_OUT)),
                    ],
                    "Note": "no_preictal_insufficient_preamble",
                }
            )
            noninter.append(
                (
                    onset - PREICTAL_MIN * 60,
                    offset + POSTICTAL_GUARD_MIN * 60,
                )
            )
            continue

        onset_idx += 1
        pre_start = onset - PREICTAL_MIN * 60
        pre_end = onset - SPH_MIN * 60
        pre_start = max(pre_start, 0.0)
        pre_end = max(pre_end, pre_start)
        pre_start = max(pre_start, prev_clear)

        if pre_end > pre_start + 1e-6:
            segs.append(
                {
                    "Label": f"Pre{onset_idx}",
                    "File": npy,
                    "Span": [
                        int(round(pre_start * FS_OUT)),
                        int(round(pre_end * FS_OUT)),
                    ],
                }
            )

        segs.append(
            {
                "Label": f"Onset{onset_idx}",
                "File": npy,
                "Span": [
                    int(round(onset * FS_OUT)),
                    int(round(offset * FS_OUT)),
                ],
            }
        )

        excl0 = pre_start
        excl1 = offset + POSTICTAL_GUARD_MIN * 60
        noninter.append((excl0, excl1))

        inter_end = pre_start
        inter_start = max(0.0, inter_end - INTERICTAL_H * 3600)
        if inter_end > inter_start + 1e-6:
            segs.append(
                {
                    "Label": f"Inter{onset_idx}-pre",
                    "File": npy,
                    "Span": [
                        int(round(inter_start * FS_OUT)),
                        int(round(inter_end * FS_OUT)),
                    ],
                }
            )

        inter2_start = offset + POSTICTAL_GUARD_MIN * 60
        inter2_end = min(duration_sec, inter2_start + INTERICTAL_H * 3600)
        if inter2_end > inter2_start + 1e-6:
            segs.append(
                {
                    "Label": f"Inter{onset_idx}-post",
                    "File": npy,
                    "Span": [
                        int(round(inter2_start * FS_OUT)),
                        int(round(inter2_end * FS_OUT)),
                    ],
                }
            )

    clipped: list[dict] = []
    for s in segs:
        a, b = s["Span"]
        a = max(0, min(a, n_samples))
        b = max(0, min(b, n_samples))
        if b > a:
            s = dict(s)
            s["Span"] = [a, b]
            clipped.append(s)
    return clipped, onset_idx


def process_patient(pid: str, raw_root: Path, dry_run: bool = False) -> dict:
    pdir = raw_root / pid
    list_path = pdir / f"Seizures-list-{pid}.txt"
    seizures_meta = parse_seizure_list(list_path) if list_path.is_file() else []
    edfs = sorted(pdir.glob("*.edf"))
    edf_names = [e.name for e in edfs]

    clean_dir = SIENA_CLEAN_ROOT / pid
    seg_dir = SIENA_SEG_ROOT / pid
    if not dry_run:
        clean_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)

    channel_info = {"Channels": list(CHB_MAJ_BIPOLAR_ORDER), "EDF Files": []}
    datetime_info: list[dict] = []
    all_segments: list[dict] = []
    onset_counter = 0
    kept_edf = 0
    skipped: list[str] = []

    for edf_path in edfs:
        print(f"  {edf_path.name} ...", flush=True)
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{edf_path.name}:read_fail:{exc}")
            continue

        elec = build_electrode_index(list(raw.ch_names))
        if not can_build_c18(elec):
            skipped.append(
                f"{edf_path.name}:no_c18:missing={missing_for_c18(elec)}"
            )
            continue

        sfreq = float(raw.info["sfreq"])
        duration_sec = float(raw.n_times / sfreq)
        events = seizures_for_file(seizures_meta, edf_path.name, edf_names)

        if dry_run:
            kept_edf += 1
            segs, onset_counter = build_segments_for_file(
                edf_path.name, duration_sec, events, onset_counter
            )
            all_segments.extend(segs)
            continue

        bip_native, _ = extract_bipolar_array(raw)
        bip = resample_notch(bip_native, sfreq)
        npy_name = _stem_npy(edf_path.name)
        np.save(clean_dir / npy_name, bip)

        channel_info["EDF Files"].append(
            {
                "File Name": edf_path.name,
                "Npy Name": npy_name,
                "Native Sfreq": sfreq,
                "Duration Sec": duration_sec,
                "Shape": list(bip.shape),
            }
        )
        datetime_info.append(
            {
                "File Name": edf_path.name,
                "Record Datetimes": [
                    events[0].reg_start if events else "",
                    events[0].reg_end if events else "",
                ],
                "Seizures": [
                    [float(e.onset_sec), float(e.offset_sec)] for e in events
                ],
            }
        )

        segs, onset_counter = build_segments_for_file(
            edf_path.name, duration_sec, events, onset_counter
        )
        all_segments.extend(segs)
        kept_edf += 1
        print(f"    saved {npy_name} {tuple(bip.shape)}; segments+={len(segs)}")

    summary = {
        "patient": pid,
        "n_edf_total": len(edfs),
        "n_edf_kept": kept_edf,
        "n_segments": len(all_segments),
        "n_pre": sum(1 for s in all_segments if s["Label"].startswith("Pre")),
        "n_onset": sum(1 for s in all_segments if s["Label"].startswith("Onset")),
        "n_inter": sum(1 for s in all_segments if s["Label"].startswith("Inter")),
        "skipped": skipped,
    }

    if dry_run:
        print(f"  DRY {pid}: {summary}")
        return summary

    if kept_edf == 0:
        print(f"  SKIP patient {pid}: no usable EDF")
        return summary

    (clean_dir / "channel_info.json").write_text(
        json.dumps(channel_info, indent=2), encoding="utf-8"
    )
    (clean_dir / "datetime_info.json").write_text(
        json.dumps(datetime_info, indent=2), encoding="utf-8"
    )
    (seg_dir / "segment_info.json").write_text(
        json.dumps(all_segments, indent=2), encoding="utf-8"
    )
    (seg_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=SIENA_RAW_ROOT)
    ap.add_argument("--patients", nargs="*", default=None, help="e.g. PN00 PN01")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    patients = args.patients or list(SUBJECTS_IN_RELEASE)
    summaries = []
    for pid in patients:
        print(f"=== {pid} ===", flush=True)
        if not (args.raw_root / pid).is_dir():
            print("  missing folder")
            continue
        summaries.append(process_patient(pid, args.raw_root, dry_run=args.dry_run))

    SIENA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = SIENA_REPORTS_DIR / "siena_preprocess_summary.json"
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

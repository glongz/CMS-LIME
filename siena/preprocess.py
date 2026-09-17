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
  retain an onset if a valid Pre span contains at least one complete 60 s block
  interictal = all valid recording outside every onset-60 min / offset+60 min guard

Usage (from ``paper-main`` root)::

  python -m siena.preprocess --dry-run
  python -m siena.preprocess --patients PN00 PN01
  python -m siena.preprocess
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import mne
import numpy as np

from .common import (
    BIPOLAR_PAIRS,
    CHB_MAJ_BIPOLAR_ORDER,
    FS_OUT,
    PROTOCOL_VERSION,
    SUBJECTS_IN_RELEASE,
    build_electrode_index,
    can_build_c18,
    clock_delta_seconds,
    missing_for_c18,
    resolve_edf_name,
)
from .inventory import parse_seizure_list
from .segment_rules import SeizureEvent, build_segments_for_file
from .paths import (
    SIENA_CLEAN_ROOT,
    SIENA_RAW_ROOT,
    SIENA_REPORTS_DIR,
    SIENA_SEG_ROOT,
)



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
            raise ValueError(f"incomplete seizure fields {edf_name} #{sz.get('seizure_id')}")
        try:
            onset = clock_delta_seconds(sz["reg_start"], sz["sz_start"])
            offset = clock_delta_seconds(sz["reg_start"], sz["sz_end"])
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"seizure parse {edf_name} #{sz.get('seizure_id')}: {exc}") from exc
        if offset <= onset:
            raise ValueError(f"non-positive seizure duration {edf_name} #{sz.get('seizure_id')}")
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


def process_patient(pid: str, raw_root: Path, dry_run: bool = False,
                    *, index_only: bool = False, annotation_overrides: dict | None = None) -> dict:
    pdir = raw_root / pid
    list_path = pdir / f"Seizures-list-{pid}.txt"
    if not list_path.is_file():
        raise FileNotFoundError(f"missing seizure annotation list: {list_path}")
    seizures_meta = parse_seizure_list(list_path)
    overrides = annotation_overrides or {}
    ids = {str(sz['seizure_id']) for sz in seizures_meta}
    if set(overrides) - ids:
        raise ValueError(f"unknown seizure IDs in overrides for {pid}: {set(overrides) - ids}")
    for sz in seizures_meta:
        fields = overrides.get(str(sz['seizure_id']), {})
        if set(fields) - {'file', 'reg_start', 'reg_end', 'sz_start', 'sz_end'}:
            raise ValueError(f"unknown annotation override field for {pid}")
        sz.update(fields)
    edfs = sorted(pdir.glob("*.edf"))
    edf_names = [e.name for e in edfs]

    for sz in seizures_meta:
        if resolve_edf_name(sz.get('file'), edf_names) is None:
            raise ValueError(f"unresolved EDF for {pid} seizure {sz['seizure_id']}: {sz.get('file')}")

    # Validate every annotation before writing arrays/indices. The release EDF
    # dates are deidentified and do not establish chronology across files.
    # Cross-file guards require the actual experiment's recording timeline.
    headers = {}
    for path in edfs:
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        try:
            events = seizures_for_file(seizures_meta, path.name, edf_names)
            duration = raw.n_times / raw.info['sfreq']
            gaps = [(int(round(a['onset'] * FS_OUT)),
                     int(round((a['onset'] + a['duration']) * FS_OUT)))
                    for a in raw.annotations if a['description'].upper() == 'BAD_ACQ_SKIP']
            build_segments_for_file(path.name, duration, events, 0, invalid_spans=gaps)
            headers[path.name] = (events, gaps)
        finally:
            raw.close()

    clean_dir = SIENA_CLEAN_ROOT / pid
    seg_dir = SIENA_SEG_ROOT / pid
    if not dry_run:
        if not index_only:
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
            raw = mne.io.read_raw_edf(edf_path, preload=not (dry_run or index_only), verbose=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"cannot read {edf_path.name}: {exc}") from exc

        elec = build_electrode_index(list(raw.ch_names))
        if not can_build_c18(elec):
            skipped.append(
                f"{edf_path.name}:no_c18:missing={missing_for_c18(elec)}"
            )
            raw.close()
            continue

        sfreq = float(raw.info["sfreq"])
        duration_sec = float(raw.n_times / sfreq)
        events, gaps = headers[edf_path.name]

        if dry_run or index_only:
            kept_edf += 1
            segs, onset_counter = build_segments_for_file(
                edf_path.name, duration_sec, events, onset_counter,
                invalid_spans=gaps,
            )
            all_segments.extend(segs)
            raw.close()
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
            edf_path.name, duration_sec, events, onset_counter,
            invalid_spans=gaps,
        )
        all_segments.extend(segs)
        raw.close()
        kept_edf += 1
        print(f"    saved {npy_name} {tuple(bip.shape)}; segments+={len(segs)}")

    summary = {
        "patient": pid,
        "protocol": PROTOCOL_VERSION,
        "mode": "dry_run" if dry_run else "index_only" if index_only else "preprocess",
        "cross_file_guards_verified": False,
        "annotation_sha256": hashlib.sha256(list_path.read_bytes()).hexdigest(),
        "annotation_overrides": overrides,
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ('common.py', 'intervals.py', 'segment_rules.py', 'preprocess.py', 'inventory.py')},
        "n_annotated_in_list": len(seizures_meta),
        "n_eligible_events": len({s['Label'] for s in all_segments if s['Label'].startswith('Pre')}),
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

    if not index_only:
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
    ap.add_argument("--index-only", action="store_true", help="Write metadata labels without reading/writing EEG arrays; not a training run")
    ap.add_argument("--annotation-overrides", type=Path, help="JSON: patient -> seizure ID -> explicit field replacements from the actual experiment")
    args = ap.parse_args()
    if args.dry_run and args.index_only:
        ap.error('choose either --dry-run or --index-only')
    overrides = json.loads(args.annotation_overrides.read_text(encoding='utf-8')) if args.annotation_overrides else {}

    patients = args.patients or list(SUBJECTS_IN_RELEASE)
    summaries = []
    for pid in patients:
        print(f"=== {pid} ===", flush=True)
        if not (args.raw_root / pid).is_dir():
            print("  missing folder")
            continue
        summaries.append(process_patient(pid, args.raw_root, dry_run=args.dry_run,
                                         index_only=args.index_only, annotation_overrides=overrides.get(pid)))

    SIENA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = SIENA_REPORTS_DIR / "siena_preprocess_summary.json"
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

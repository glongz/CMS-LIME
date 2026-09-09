# -*- coding: utf-8 -*-
"""Shared constants / helpers for Siena -> CHB-MIT matched protocol."""
from __future__ import annotations

from pathlib import Path

# Protocol (match CHB Tables 1--2).
FS_OUT = 256
FS_IN_DEFAULT = 512
SOP_MIN = 30
SPH_MIN = 1
PREICTAL_MIN = SOP_MIN + SPH_MIN  # 31
INTERICTAL_H = 2.0  # hours, as stated in the manuscript
POSTICTAL_GUARD_MIN = 60  # exclude residue after offset before interictal

# CHB maj channel order from
# D:\public_data\CHBMIT\1_data_clean_18channels\chb01\channel_info.json
# (same set as the paper list; order must match CHB checkpoints for transfer).
CHB_MAJ_BIPOLAR_ORDER: list[str] = [
    "F8-T8",
    "T8-P8",
    "P7-O1",
    "CZ-PZ",
    "C4-P4",
    "F4-C4",
    "FP2-F4",
    "FP2-F8",
    "T7-P7",
    "P3-O1",
    "C3-P3",
    "P8-O2",
    "P4-O2",
    "F3-C3",
    "FZ-CZ",
    "F7-T7",
    "FP1-F3",
    "FP1-F7",
]

# Bipolar = (anode, cathode) in CHB_MAJ_BIPOLAR_ORDER.
BIPOLAR_PAIRS: list[tuple[str, str]] = [
    tuple(name.split("-", 1)) for name in CHB_MAJ_BIPOLAR_ORDER  # type: ignore[misc]
]

# Unipolar electrodes required to build C_18.
REQUIRED_UNIPOLAR: set[str] = set()
for a, b in BIPOLAR_PAIRS:
    REQUIRED_UNIPOLAR.add(a)
    REQUIRED_UNIPOLAR.add(b)

# Legacy 10--20 aliases used in Siena docs / older labels.
ELECTRODE_ALIASES: dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
    # Seizure-list PN00 marks O1 as literal "1".
    "1": "O1",
}


def normalize_electrode(raw_name: str) -> str | None:
    """Map an EDF / list channel string to a canonical unipolar label, or None."""
    n = raw_name.strip().upper()
    for prefix in ("EEG", "EEG "):
        if n.startswith(prefix):
            n = n[len(prefix) :].strip()
    n = n.replace(" ", "").replace(".", "").replace("_", "")
    # Drop non-EEG auxiliaries early.
    if n in {"EKG", "EKG1", "EKG2", "ECG", "SPO2", "HR", "MK", "PLET", "B", "C", "D"}:
        return None
    if n.isdigit() and n not in ELECTRODE_ALIASES:
        # Keep "1" via alias; other numeric junk (61..) discarded.
        return None
    n = ELECTRODE_ALIASES.get(n, n)
    # Canonicalise Fp / FP.
    if n in {"FP1", "FP2", "FZ", "CZ", "PZ", "OZ"}:
        return n
    if n.startswith("FP") and len(n) == 3:
        return n
    return n


def build_electrode_index(ch_names: list[str]) -> dict[str, int]:
    """First-occurrence index of each normalised unipolar electrode."""
    out: dict[str, int] = {}
    for i, name in enumerate(ch_names):
        lab = normalize_electrode(name)
        if lab is None:
            continue
        out.setdefault(lab, i)
    return out


def missing_for_c18(electrode_index: dict[str, int]) -> list[str]:
    return sorted(e for e in REQUIRED_UNIPOLAR if e not in electrode_index)


def can_build_c18(electrode_index: dict[str, int]) -> bool:
    return len(missing_for_c18(electrode_index)) == 0


def hms_to_seconds(token: str) -> int:
    """Parse '19.39.33' or '19:39:33' into seconds from midnight."""
    t = token.strip().replace(":", ".")
    parts = [int(x) for x in t.split(".")]
    if len(parts) != 3:
        raise ValueError(f"bad HMS token: {token!r}")
    h, m, s = parts
    return h * 3600 + m * 60 + s


def clock_delta_seconds(start_hms: str, event_hms: str) -> int:
    """Seconds from registration start to event; +24h if clock wrapped."""
    a = hms_to_seconds(start_hms)
    b = hms_to_seconds(event_hms)
    if b < a:
        b += 24 * 3600
    return b - a


SUBJECTS_IN_RELEASE: list[str] = [
    "PN00",
    "PN01",
    "PN03",
    "PN05",
    "PN06",
    "PN07",
    "PN09",
    "PN10",
    "PN11",
    "PN12",
    "PN13",
    "PN14",
    "PN16",
    "PN17",
]


def _norm_edf_key(name: str) -> str:
    """Lowercase stem with O/0 confusions collapsed for fuzzy match."""
    stem = Path(name).stem.lower().replace(" ", "")
    # Seizure lists occasionally write PNO6 instead of PN06.
    stem = stem.replace("pno", "pn0")
    return stem


def resolve_edf_name(listed: str | None, available: list[str]) -> str | None:
    """Map a seizure-list file name onto an on-disk ``*.edf`` name.

    Handles exact match, PN01.edf→PN01-1.edf (single-file patients),
    trailing-hyphen stems (PN11-.), and O/0 typos (PNO6).
    """
    if not listed or not available:
        return None
    avail_map = {a.lower(): a for a in available}
    key = listed.strip()
    if key.lower() in avail_map:
        return avail_map[key.lower()]

    listed_stem = _norm_edf_key(key)
    by_stem = {_norm_edf_key(a): a for a in available}
    if listed_stem in by_stem:
        return by_stem[listed_stem]

    # Prefix: listed PN10-4 vs disk PN10-4.5.6.edf
    prefix_hits = [
        a for a in available if _norm_edf_key(a).startswith(listed_stem)
    ]
    if len(prefix_hits) == 1:
        return prefix_hits[0]

    # Listed shorter / patient-only stem (PN01.edf) with one file on disk.
    patient_token = listed_stem.split("-")[0]
    patient_hits = [
        a for a in available if _norm_edf_key(a).startswith(patient_token)
    ]
    if len(patient_hits) == 1:
        return patient_hits[0]

    contain_hits = [
        a
        for a in available
        if listed_stem in _norm_edf_key(a) or _norm_edf_key(a) in listed_stem
    ]
    if len(contain_hits) == 1:
        return contain_hits[0]
    return None

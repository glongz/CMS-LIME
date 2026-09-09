# -*- coding: utf-8 -*-
"""
Siena default data / report paths (override with environment variables).

Mirrors ``可解释归因示例/chb_paths.py`` so CHB and Siena share the same
configuration style for later GitHub / multi-machine runs::

    set SIENA_RAW_ROOT=D:\\public_data\\siena-scalp-eeg-database-1.0.0
    set SIENA_OUT_ROOT=D:\\public_data\\SIENA
    set SIENA_REPORTS_DIR=D:\\path\\to\\reports
"""
from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent

# PhysioNet release root (user-local; not committed).
SIENA_RAW_ROOT: Path = Path(
    os.environ.get(
        "SIENA_RAW_ROOT",
        r"D:\public_data\siena-scalp-eeg-database-1.0.0",
    )
)

# Processed layout mirroring CHB ``1_data_clean_18channels`` + ``30-1-240``.
SIENA_OUT_ROOT: Path = Path(
    os.environ.get("SIENA_OUT_ROOT", r"D:\public_data\SIENA")
)
SIENA_CLEAN_ROOT: Path = Path(
    os.environ.get(
        "SIENA_DATA_CLEAN_ROOT",
        str(SIENA_OUT_ROOT / "1_data_clean_18channels"),
    )
)
SIENA_SEG_ROOT: Path = Path(
    os.environ.get(
        "SIENA_SEGMENT_INFO_ROOT",
        str(SIENA_OUT_ROOT / "segment_clean_18channels" / "30-1-240"),
    )
)

# Inventory / stats CSVs (small; safe to commit selectively under reports/).
SIENA_REPORTS_DIR: Path = Path(
    os.environ.get("SIENA_REPORTS_DIR", str(_PKG / "reports"))
)

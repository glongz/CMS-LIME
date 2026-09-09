# -*- coding: utf-8 -*-
"""Siena Scalp EEG preprocessing aligned to the CHB-MIT CMS-LIME protocol (30-1-240)."""

from .paths import (
    SIENA_CLEAN_ROOT,
    SIENA_OUT_ROOT,
    SIENA_RAW_ROOT,
    SIENA_REPORTS_DIR,
    SIENA_SEG_ROOT,
)

__all__ = [
    "SIENA_RAW_ROOT",
    "SIENA_OUT_ROOT",
    "SIENA_CLEAN_ROOT",
    "SIENA_SEG_ROOT",
    "SIENA_REPORTS_DIR",
]

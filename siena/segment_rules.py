"""Proposed Siena labels under the manuscript's retained-minute-block rule."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .common import (
    EVENT_BLOCK_SEC, FS_OUT, POSTICTAL_GUARD_MIN, PREICTAL_MIN,
    PRE_SEIZURE_GUARD_MIN, PROTOCOL_VERSION, SPH_MIN,
)
from .intervals import subtract_intervals


@dataclass
class SeizureEvent:
    seizure_id: str
    edf_name: str
    onset_sec: float
    offset_sec: float
    reg_start: str
    reg_end: str


def build_segments_for_file(
    edf_name: str,
    duration_sec: float,
    events: list[SeizureEvent],
    global_onset_counter_start: int,
    *,
    context_events: list[SeizureEvent] | None = None,
    invalid_spans: list[tuple[int, int]] | None = None,
) -> tuple[list[dict], int]:
    """Return disjoint labels with half-open sample spans at FS_OUT.

    ``events`` are seizures in this EDF. ``context_events`` are other seizures
    from the same patient, expressed relative to this EDF's start; they guard
    nearby recordings but do not receive new onset IDs in this file.
    ``invalid_spans`` covers known recording gaps/artifacts in sample units.

    Preictal eligibility requires a full 60 s block within an individual valid
    span. Disconnected fragments are never concatenated to invent a minute.
    Spans preserve their residual samples; MinuteBlocks counts complete packs
    and does not silently redefine the FPR/h denominator used by an evaluator.
    """
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("recording duration must be finite and positive")
    n_samples = int(round(duration_sec * FS_OUT))
    if n_samples <= 0:
        raise ValueError("recording has no samples at the output sampling rate")
    ordered = sorted(events, key=lambda e: e.onset_sec)
    for ev in ordered:
        if (ev.edf_name != edf_name or not math.isfinite(ev.onset_sec)
                or not math.isfinite(ev.offset_sec)
                or not 0 <= ev.onset_sec < ev.offset_sec):
            raise ValueError(f"invalid seizure bounds: {edf_name} #{ev.seizure_id}")
    all_events = ordered + list(context_events or [])
    for ev in all_events:
        if (not math.isfinite(ev.onset_sec) or not math.isfinite(ev.offset_sec)
                or ev.offset_sec <= ev.onset_sec):
            raise ValueError(f"invalid context event: {ev.seizure_id}")
    available = subtract_intervals([(0, n_samples)], list(invalid_spans or []))
    minute = EVENT_BLOCK_SEC * FS_OUT
    segments: list[dict] = []
    counter = global_onset_counter_start
    npy = Path(edf_name).stem + '.npy'

    def sample(seconds: float) -> int:
        return int(round(seconds * FS_OUT))

    def segment(label: str, span: tuple[int, int], **extra) -> dict:
        return {"Label": label, "File": npy, "Span": list(span),
                "MinuteBlocks": (span[1] - span[0]) // minute,
                "Protocol": PROTOCOL_VERSION, **extra}

    for ev in ordered:
        counter += 1
        # Only earlier seizures' ictal/postictal guards truncate Pre. The
        # current event's pre-onset interictal exclusion must not erase Pre.
        prior_guards = [
            (sample(other.onset_sec), sample(other.offset_sec + POSTICTAL_GUARD_MIN * 60))
            for other in all_events if other.onset_sec < ev.onset_sec
        ]
        nominal = (sample(ev.onset_sec - PREICTAL_MIN * 60),
                   sample(ev.onset_sec - SPH_MIN * 60))
        candidates = [(max(a, nominal[0]), min(b, nominal[1]))
                      for a, b in available if min(b, nominal[1]) > max(a, nominal[0])]
        in_prior_guard = any(left < sample(ev.onset_sec) < right
                             for left, right in prior_guards)
        valid_pre = [] if in_prior_guard else subtract_intervals(candidates, prior_guards)
        retained = [(a, b) for a, b in valid_pre if b - a >= minute]
        for a, b in retained:
            segments.append(segment(f'Pre{counter}', (a, b), SeizureID=ev.seizure_id))
        onset_span = (sample(ev.onset_sec), min(sample(ev.offset_sec), n_samples))
        segments.append(segment(
            f'Onset{counter}', onset_span,
            SeizureID=ev.seizure_id,
            PreictalEligible=bool(retained),
            OffsetClippedByFileBoundary=sample(ev.offset_sec) > n_samples,
            Note='retained_minute_block' if retained else 'no_complete_preictal_minute',
        ))

    # All seizures participate, including clustered/Pre-ineligible events and
    # seizures in adjacent files. This also retains seizure-free recordings.
    guards = [(sample(ev.onset_sec - PRE_SEIZURE_GUARD_MIN * 60),
               sample(ev.offset_sec + POSTICTAL_GUARD_MIN * 60)) for ev in all_events]
    for index, span in enumerate(subtract_intervals(available, guards), start=1):
        segments.append(segment(f'Inter-{Path(edf_name).stem}-{index}', span))
    return segments, counter

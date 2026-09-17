"""Half-open sample intervals used by the proposed retained-block protocol.

This implementation is not evidence that a historical experiment used it.
Run provenance and actual evaluation pairs must be supplied separately.
"""
from __future__ import annotations


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, stop in sorted((a, b) for a, b in intervals if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def subtract_intervals(
    available: list[tuple[int, int]], excluded: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Subtract the union of exclusions; never duplicate exposure."""
    result: list[tuple[int, int]] = []
    exclusions = merge_intervals(excluded)
    for start, stop in merge_intervals(available):
        cursor = start
        for left, right in exclusions:
            if right <= cursor:
                continue
            if left >= stop:
                break
            if left > cursor:
                result.append((cursor, min(left, stop)))
            cursor = max(cursor, right)
            if cursor >= stop:
                break
        if cursor < stop:
            result.append((cursor, stop))
    return result

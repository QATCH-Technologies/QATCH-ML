"""
rendering/_common.py
=====================

Shared helpers for the detector and fill-classifier renderers
(``detector_render.py`` / ``fill_render.py``). ``_strip_points`` and
``_robust_mad`` used to be copy-pasted, near-identically, into both modules
(and, for ``_robust_mad``, a third time within the fill-render module
itself); this is the one canonical implementation both now import.
"""

from __future__ import annotations

import numpy as np

from .legacy_dataprocessor import QModelV6YOLO_DataProcessor as DP

PADDING = DP.PADDING


def _robust_mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _strip_points(
    values: np.ndarray,
    img_w: int,
    strip_h: int,
    strip_idx: int,
    p_lower: float = 1.0,
    p_upper: float = 99.0,
) -> np.ndarray:
    """Same normalization/geometry contract as the v1 renderer's
    _get_signal_points (percentile clip -> strip pixel band)."""
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return None
    lo, hi = np.percentile(finite, [p_lower, p_upper])
    diff = hi - lo
    if diff <= 0:
        diff = 1.0
    norm = np.clip((values - lo) / diff, 0, 1)
    x = np.linspace(0, img_w - 1, len(values)).astype(np.int32)
    draw_h = strip_h - 2 * PADDING
    y_off = strip_idx * strip_h + PADDING
    y = (y_off + (1.0 - norm) * draw_h).astype(np.int32)
    return np.stack((x, y), axis=1)

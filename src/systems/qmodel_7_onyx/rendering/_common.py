"""Shared helper functions for detector and fill-classifier renderers.

Provides canonical implementations of common signal-processing and spatial mapping
utilities used by both `detector_render.py` and `fill_render.py`.

Attributes:
    PADDING (int): Standard pixel padding offset imported from
        `QModelOnyx_DataProcessor`.
"""

from __future__ import annotations

import numpy as np

from .legacy_dataprocessor import QModelOnyx_DataProcessor as DP

PADDING = DP.PADDING


def _robust_mad(x: np.ndarray) -> float:
    """Computes the Median Absolute Deviation (MAD) of a 1D array.

    Calculates a scaled median absolute deviation estimate for robust scale
    estimation, ignoring non-finite values (NaNs and Infinities).

    Args:
        x (np.ndarray): Input 1D numeric array.

    Returns:
        float: The robust median absolute deviation estimate (scaled by 1.4826
        for normal consistency), or 0.0 if fewer than 8 finite values exist.
    """
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
    """Maps signal values to pixel coordinates within a designated strip band.

    Applies percentile clipping and normalization to map floating-point signal
    values onto a 2D grid of pixel coordinates `(x, y)` bound within a
    specific vertical strip offset.

    Args:
        values (np.ndarray): 1D array of signal values to map.
        img_w (int): Total image width in pixels.
        strip_h (int): Height of an individual strip in pixels.
        strip_idx (int): Zero-based vertical index of the target strip band.
        p_lower (float, optional): Lower percentile for intensity clipping.
            Defaults to 1.0.
        p_upper (float, optional): Upper percentile for intensity clipping.
            Defaults to 99.0.

    Returns:
        np.ndarray | None: An `(N, 2)` array of integer pixel coordinates `[x, y]`
        ready for rendering, or `None` if fewer than 2 finite values are present.
    """
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

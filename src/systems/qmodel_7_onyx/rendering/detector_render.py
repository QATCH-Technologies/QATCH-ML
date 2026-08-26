"""
Version-2 detection-image renderer, addressing the two representation
failures the onyx training run exposed:

  1. LATE-EVENT FLATTENING. Per-strip global percentile normalization makes
     the early fill step own the entire dynamic range; late POIs in long
     viscous runs become featureless flat plateaus - the regions where ch2
     trained to recall 0.63 and ch3 collapsed. The fix is not a cleverer
     amplitude normalization (any global value scaling keeps the step
     dominant): it is to render what the detector is actually supposed to
     find. POIs are TRANSITIONS, so the third strip is replaced with a
     DERIVATIVE-ENERGY trace: the combined, robustly-scaled, log-compressed
     |d/dt| of dissipation and resonance frequency. Events appear as bright
     vertical ridges with near-uniform salience regardless of where in the
     run they occur or how large the absolute amplitude change is. The
     Difference curve it replaces is a linear combination of the two value
     strips and carried little independent information.

  2. The dissipation (R) and resonance (G) value strips are kept exactly as
     v1 renders them (same percentile normalization, fill + white outline),
     so global fill-context cues the detectors already exploit are
     preserved.

Train/inference contract: this module is used by BOTH build_dataset.py and
the production predictor (QModelOnyxConfig.RENDER_VERSION). The render the
weights were trained on MUST be the render they see at inference; the
version flag exists precisely so old (v1-trained) weights keep working
while v2-trained weights roll out.

Attributes:
    COL_TIME (str): DataFrame column name for relative time.
    COL_DISS (str): DataFrame column name for dissipation.
    COL_FREQ (str): DataFrame column name for resonance frequency.
    IMG_CHANNELS (int): Number of image channels (3 for BGR).
    COLOR_WHITE (tuple[int, int, int]): BGR color tuple for white outlines.
    DERIV_SCALES_S (tuple[float, ...]): Half-window timescales (in seconds)
        for transition detection.
    DERIV_SMOOTH_S (float): Post-smoothing window size (in seconds) for the
        salience trace.
    DERIV_UPPER_PCT (float): Robust percentile ceiling for ridge normalization.
    DERIV_EPS (float): Epsilon value to prevent division by zero in calculations.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from ._common import PADDING, _robust_mad, _strip_points
from .dataprocessor import QModelOnyxDataProcessor as DP

COL_TIME = "Relative_time"
COL_DISS = "Dissipation"
COL_FREQ = "Resonance_Frequency"

IMG_CHANNELS = 3
COLOR_WHITE = (255, 255, 255)

# Derivative-energy parameters.
DERIV_SCALES_S = (0.25, 1.0, 4.0)  # half-window timescales for transitions
DERIV_SMOOTH_S = 0.15  # post-smoothing of the salience trace
DERIV_UPPER_PCT = 99.8  # robust ceiling for ridge normalization
DERIV_EPS = 1e-12


def _scaled_curv(x: np.ndarray, w: int) -> np.ndarray:
    """Computes the windowed second difference of a 1D array.

    Calculates `|x[i+w] - 2x[i] + x[i-w]|` for each point, clamping the edges.
    This filter fires on both step changes and slope changes, while remaining
    inherently insensitive to linear trends. This property distinguishes late-fill
    points of interest (bends on a monotone background) from the background itself.

    Args:
        x (np.ndarray): 1D input array of signal values.
        w (int): Half-window size in samples.

    Returns:
        np.ndarray: A 1D array of the same length as `x` containing the
        computed windowed second difference.
    """
    n = len(x)
    out = np.zeros(n)
    if n <= 2 * w:
        return out
    out[w:-w] = np.abs(x[2 * w :] - 2.0 * x[w:-w] + x[: -2 * w])
    out[:w] = out[w]
    out[-w:] = out[-w - 1]
    return out


def derivative_energy(df: pd.DataFrame) -> np.ndarray:
    """Calculates multi-scale transition salience across signal channels.

    For each valid signal (Dissipation and Resonance Frequency) and each
    configured timescale, this computes a windowed second difference normalized
    by its own median absolute deviation (MAD). The per-sample maximum across
    signals and scales is then log-compressed and lightly smoothed.

    This ensures fast initial events and slow viscous transitions both appear
    as ridges of comparable salience, as each scale is normalized against its
    own localized noise floor.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data. Expected
            to contain columns for relative time, dissipation, and frequency.

    Returns:
        np.ndarray: A 1D array of the computed derivative-energy salience trace.
    """
    n = len(df)
    if n < 16:
        return np.zeros(n)
    t = pd.to_numeric(df[COL_TIME], errors="coerce").to_numpy(dtype=float)
    dt = float(np.nanmedian(np.diff(t))) or 0.005
    sal = np.zeros(n)
    for col in (COL_DISS, COL_FREQ):
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
        for scale_s in DERIV_SCALES_S:
            w = int(round(scale_s / dt))
            w = max(2, min(w, (n - 1) // 2 - 1))
            if w < 2:
                continue
            d = _scaled_curv(x, w)
            s = _robust_mad(d)
            if s <= 0:
                s = float(np.std(d)) or 1.0
            sal = np.maximum(sal, d / (s + DERIV_EPS))
    e = np.log1p(sal)
    k = max(3, int(round(DERIV_SMOOTH_S / dt)) | 1)
    if n > k:
        pad_w = k // 2
        e_padded = np.pad(e, (pad_w, pad_w), mode="edge")
        e = np.convolve(e_padded, np.ones(k) / k, mode="valid")
    return e


def generate_channel_det_v2(df: pd.DataFrame, img_w: int, img_h: int) -> np.ndarray:
    """Generates a version-2 detection render from signal data.

    Produces an RGB (BGR in OpenCV format) image where the channels correspond to:
    - Red (Channel 2 in BGR): Dissipation.
    - Green (Channel 1 in BGR): Resonance frequency.
    - Blue (Channel 0 in BGR): Derivative-energy ridge trace.

    Values are percentile-normalized and rendered as filled polygons with
    white outlines, preserving global fill-context cues used by detectors.

    Args:
        df (pd.DataFrame): Time series data containing dissipation, frequency,
            and time columns.
        img_w (int): Target width of the generated image in pixels.
        img_h (int): Target height of the generated image in pixels.

    Returns:
        np.ndarray: A 3D numpy array of shape `(img_h, img_w, 3)` containing
        the generated BGR image.
    """
    img = np.zeros((img_h, img_w, IMG_CHANNELS), dtype=np.uint8)
    if df is None or df.empty or len(df) < 2:
        return img
    strip_h = img_h // 3

    traces = [
        (
            (
                pd.to_numeric(df[COL_DISS], errors="coerce").to_numpy(dtype=float)
                if COL_DISS in df.columns
                else None
            ),
            0,
            2,
        ),  # red channel (BGR idx 2)
        (
            (
                pd.to_numeric(df[COL_FREQ], errors="coerce").to_numpy(dtype=float)
                if COL_FREQ in df.columns
                else None
            ),
            1,
            1,
        ),  # green
        (derivative_energy(df), 2, 0),  # blue: event ridges
    ]
    for values, strip_idx, ch_idx in traces:
        if values is None:
            continue
        p_hi = DERIV_UPPER_PCT if strip_idx == 2 else 99.0
        pts = _strip_points(values, img_w, strip_h, strip_idx, p_upper=p_hi)
        if pts is None:
            continue
        strip_bottom = (strip_idx + 1) * strip_h - PADDING
        poly = np.concatenate([pts, [[pts[-1][0], strip_bottom]], [[pts[0][0], strip_bottom]]])
        color = [0, 0, 0]
        color[ch_idx] = 255
        cv2.fillPoly(img, [poly], tuple(color))
        cv2.polylines(
            img,
            [pts.reshape((-1, 1, 2))],
            isClosed=False,
            color=COLOR_WHITE,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return img


def generate_det_image(df: pd.DataFrame, img_w: int, img_h: int, version: int = 2) -> np.ndarray:
    """Dispatches detection image generation to the specified render version.

    Used by both the training dataset builder (`build_dataset.py`) and the
    production inference predictor (`v6_yolo` via `QModelOnyxConfig.RENDER_VERSION`).
    Ensures that the render representation seen during inference matches
    the one used during training.

    Args:
        df (pd.DataFrame): Time series data to render.
        img_w (int): Target width of the generated image in pixels.
        img_h (int): Target height of the generated image in pixels.
        version (int, optional): Render version to use. Version 1 dispatches
            to the legacy `DP.generate_channel_det`, while Version 2 (default)
            dispatches to `generate_channel_det_v2`.

    Returns:
        np.ndarray: A 3D numpy array containing the generated BGR image.
    """
    if version == 1:
        return DP.generate_channel_det(df, img_w=img_w, img_h=img_h)
    return generate_channel_det_v2(df, img_w=img_w, img_h=img_h)

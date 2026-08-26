"""Fill-classification renderer.

Applies detector render insights to the type classifier and enforces a single
shared train/deploy input contract.

The fill classifier counts transitions (the class is the number of fill events
visible so far). Under plain percentile-value normalization, long viscous runs
flattened late transitions into featureless plateaus, obscuring the boundary
between 2-channel and 3-channel fills. This render addresses that by:
  1. Preserving strips 0 and 1 (dissipation and resonance) with plain
     percentile normalization, to maintain global fill-context cues.
  2. Rendering strip 2 as a step-coincidence energy trace: a transition-aware
     salience trace built from a matched step filter and cross-signal
     geometric means, which stays sensitive to extremely slow, late
     transitions that a simple derivative would wash out.

Train/deploy contract: `prepare_cls_input` is the sole function that turns a
preprocessed dataframe into the 224x224 tensor-ready image, ensuring that
training images and inference images are bit-identical by construction.

Attributes:
    COL_TIME (str): DataFrame column name for relative time.
    COL_DISS (str): DataFrame column name for dissipation.
    COL_FREQ (str): DataFrame column name for resonance frequency.
    IMG_CHANNELS (int): Number of image channels (3 for BGR).
    FILL_GEN_W (int): Base generation image width (640).
    FILL_GEN_H (int): Base generation image height (640).
    FILL_INFERENCE_W (int): Final inference image width (224).
    FILL_INFERENCE_H (int): Final inference image height (224).
    STRIP_SPEC (tuple): Configuration for rendering each horizontal strip band.
    STEP_ABS_SCALES_S (tuple[float, ...]): Absolute time scales for step detection.
    STEP_REL_SCALES (tuple[float, ...]): Span-relative scales for step detection.
    STEP_SMOOTH_S (float): Post-smoothing window size (in seconds) for the traces.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from ._common import PADDING, _robust_mad, _strip_points
from .detector_render import DERIV_UPPER_PCT

COL_TIME = "Relative_time"
COL_DISS = "Dissipation"
COL_FREQ = "Resonance_Frequency"

IMG_CHANNELS = 3

# Generation and inference geometry - identical to the current classifier
# path (QModelOnyxConfig.FILL_GEN_* / FILL_INFERENCE_*), restated here so the
# contract is self-contained.
FILL_GEN_W = 640
FILL_GEN_H = 640
FILL_INFERENCE_W = 224
FILL_INFERENCE_H = 224

# Visual language of the classification render: colored fills with a +50
# edge highlight, NOT the detector's channel masks.
STRIP_SPEC = (
    # (values_fn, BGR fill color, upper percentile)
    ("diss", (0, 0, 255), 99.0),
    ("freq", (0, 255, 0), 99.0),
    ("energy", (255, 0, 0), DERIV_UPPER_PCT),
)


def prepare_cls_input(df: pd.DataFrame) -> np.ndarray:
    """Converts a preprocessed dataframe into the final classifier tensor input.

    This function represents the strict train/deploy contract. It generates the
    classification image and performs the exact INTER_AREA resize down to the
    final inference resolution (224x224). This exact output is saved during
    dataset building and fed directly into the predictor during inference.

    Args:
        df (pd.DataFrame): Time series data to process.

    Returns:
        np.ndarray: A 224x224x3 uint8 BGR image ready for the classifier.
    """
    img = generate_fill_cls(df, FILL_GEN_W, FILL_GEN_H)
    return cv2.resize(img, (FILL_INFERENCE_W, FILL_INFERENCE_H), interpolation=cv2.INTER_AREA)


STEP_ABS_SCALES_S = (0.5, 2.0, 8.0)
STEP_REL_SCALES = (1.0 / 32.0, 1.0 / 12.0)
STEP_SMOOTH_S = 0.15


def _step_response(x: np.ndarray, w: int) -> np.ndarray:
    """Computes a normalized matched filter for level shifts at a specific scale.

    Calculates the absolute difference between the means of adjacent windows
    (`|mean(x[i+1..i+w]) - mean(x[i-w..i])|`) efficiently using cumulative sums.
    The result is normalized by the interior median absolute deviation (MAD) of
    its own response. This acts as a matched filter for step responses while
    suppressing noise `~sqrt(w/dt)`.

    Args:
        x (np.ndarray): 1D input array of signal values.
        w (int): Window size in samples.

    Returns:
        np.ndarray: A 1D array of the normalized step response.
    """
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=float(np.nanmedian(x))))])
    i = np.arange(n)
    hi = np.minimum(i + w, n - 1)
    lo = np.maximum(i - w, 0)
    mean_r = (cs[hi + 1] - cs[i + 1]) / np.maximum(hi - i, 1)
    mean_l = (cs[i + 1] - cs[lo]) / np.maximum(i - lo + 1, 1)
    d = np.abs(mean_r - mean_l)
    interior = d[w : n - w] if n > 2 * w else d
    s = _robust_mad(interior)
    if s <= 0:
        s = float(np.std(d)) or 1.0
    return d / s


def step_coincidence_energy(df: pd.DataFrame) -> np.ndarray:
    """Calculates multi-scale, cross-signal-coincident step salience.

    Stays sensitive to slow, late transitions (where a simple derivative
    washes out) by implementing a step filter combined with a cross-signal
    geometric mean. It evaluates multiple absolute scales (0.5s, 2s, 8s) and
    span-relative scales
    (span/32, span/12) to catch both fast initialized events and slow viscous fills.
    By requiring coordination across dissipation and resonance, it suppresses
    single-channel noise excursions.

    Args:
        df (pd.DataFrame): Time series data containing relative time, dissipation,
            and resonance frequency columns.

    Returns:
        np.ndarray: A 1D array representing the per-sample step-coincidence
        energy trace.
    """
    n = len(df)
    if n < 16:
        return np.zeros(n)
    t = pd.to_numeric(df[COL_TIME], errors="coerce").to_numpy(dtype=float)
    dt = float(np.nanmedian(np.diff(t))) or 0.005
    span = float(t[-1] - t[0])
    scales = sorted(set(list(STEP_ABS_SCALES_S) + [max(0.5, span * r) for r in STEP_REL_SCALES]))
    have = [c for c in (COL_DISS, COL_FREQ) if c in df.columns]
    if not have:
        return np.zeros(n)
    sal = np.zeros(n)
    for scale_s in scales:
        w = int(round(scale_s / dt))
        w = max(3, min(w, (n - 1) // 2 - 1))
        resp = [
            _step_response(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float), w)
            for c in have
        ]
        if len(resp) == 2:
            per_scale = np.sqrt(resp[0] * resp[1])  # coincidence
        else:
            per_scale = resp[0]  # degraded single-signal fallback
        sal = np.maximum(sal, per_scale)
    e = np.log1p(sal)
    k = max(3, int(round(STEP_SMOOTH_S / dt)) | 1)
    if n > k:
        e = np.convolve(e, np.ones(k) / k, mode="same")
    return e


def generate_fill_cls(
    df: pd.DataFrame, img_w: int = FILL_GEN_W, img_h: int = FILL_GEN_H
) -> np.ndarray:
    """Generates the fill-classification render image.

    Dissipation (red) and resonance (green) value strips use plain percentile
    normalization; the third strip (blue) is drawn using the step-coincidence
    energy trace.

    Args:
        df (pd.DataFrame): Time series data containing signal columns.
        img_w (int, optional): Total image width. Defaults to FILL_GEN_W.
        img_h (int, optional): Total image height. Defaults to FILL_GEN_H.

    Returns:
        np.ndarray: A 3D numpy array representing the generated BGR image.
    """
    img = np.zeros((img_h, img_w, IMG_CHANNELS), dtype=np.uint8)
    if df is None or df.empty or len(df) < 2:
        return img
    strip_h = img_h // 3
    series = {
        "diss": (
            pd.to_numeric(df[COL_DISS], errors="coerce").to_numpy(dtype=float)
            if COL_DISS in df.columns
            else None
        ),
        "freq": (
            pd.to_numeric(df[COL_FREQ], errors="coerce").to_numpy(dtype=float)
            if COL_FREQ in df.columns
            else None
        ),
        "energy": step_coincidence_energy(df),
    }
    for strip_idx, (key, color, p_hi) in enumerate(STRIP_SPEC):
        values = series[key]
        if values is None:
            continue
        pts = _strip_points(values, img_w, strip_h, strip_idx, p_upper=p_hi)
        if pts is None:
            continue
        strip_bottom = (strip_idx + 1) * strip_h - PADDING
        poly = np.concatenate([pts, [[pts[-1][0], strip_bottom]], [[pts[0][0], strip_bottom]]])
        cv2.fillPoly(img, [poly], color)
        edge_color = tuple(min(c + 50, 255) for c in color)
        cv2.polylines(
            img,
            [pts.reshape((-1, 1, 2))],
            isClosed=False,
            color=edge_color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return img

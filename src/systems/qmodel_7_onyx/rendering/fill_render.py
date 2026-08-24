"""
rendering/fill_render.py
=========================

Version-2 fill-CLASSIFICATION renderer, applying the v7 detector render
insights to the type classifier, plus the single shared train/deploy input
contract.

Why the classifier needs the derivative-energy strip even more than the
detectors did
-------------------------------------------------------------------------
The fill classifier's job is literally COUNTING transitions: the class IS
the number of fill events visible so far. Its current third strip is the
Difference curve - a linear combination of the two value strips that
carries almost no independent information - and all three strips share the
v1 percentile-value normalization whose failure mode the detector work
exposed: in a long viscous run the early fill step owns the entire dynamic
range, so the late transitions that distinguish 2ch from 3ch flatten into
featureless plateaus. That is exactly the confusion boundary a channel
counter cannot afford to lose.

The v2 classification render therefore mirrors detector_render:

  * Strips 0/1 (dissipation red, resonance green) are kept EXACTLY as the
    v1 classifier render draws them - same percentile normalization, color
    fill, +50 edge highlight - preserving the global fill-context cues
    (step position, plateau levels, fill fraction of frame) the current
    model already exploits. Those cues are what separate no_fill /
    initial_fill, where value shape matters more than transition count.
  * Strip 2 replaces the Difference curve with the DERIVATIVE-ENERGY trace
    from detector_render: multi-scale, per-scale-MAD-normalized,
    log-compressed curvature salience. Every transition - millisecond init
    events and minute-scale viscous channel fills alike - appears as a
    ridge of comparable height regardless of where in the run it occurs.
    Counting ridges is amplitude- and position-invariant in precisely the
    way the class label is.

Train/deploy contract (the detector lesson, applied)
----------------------------------------------------
``prepare_cls_input`` is the ONE function that turns a preprocessed
dataframe into the 224x224 tensor-ready image. build_fill_dataset.py saves
its exact output; QModelV7 inference feeds its exact output. The 640->224
INTER_AREA resize lives inside it, so training images and inference images
are bit-identical by construction - no "same-ish render" drift.

``FILL_RENDER_VERSION`` dispatches v1 (legacy weights) vs v2, the same
roll-out mechanism as QModelOnyxConfig.RENDER_VERSION on the detector side.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from ._common import PADDING, _robust_mad, _strip_points
from .detector_render import DERIV_UPPER_PCT, derivative_energy
from .legacy_dataprocessor import QModelOnyx_DataProcessor as DP

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

# Visual language of the classification render (matches v1 generate_fill_cls:
# colored fills with a +50 edge highlight, NOT the detector's channel masks).
STRIP_SPEC = (
    # (values_fn, BGR fill color, upper percentile)
    ("diss", (0, 0, 255), 99.0),
    ("freq", (0, 255, 0), 99.0),
    ("energy", (255, 0, 0), DERIV_UPPER_PCT),
)


def generate_fill_cls_v2(
    df: pd.DataFrame, img_w: int = FILL_GEN_W, img_h: int = FILL_GEN_H
) -> np.ndarray:
    """v2 classification render: dissipation (red) + resonance (green) value
    strips exactly as v1, derivative-energy ridge strip (blue) replacing the
    Difference curve. Takes TOTAL image dimensions."""
    img = np.zeros((img_h, img_w, IMG_CHANNELS), dtype=np.uint8)
    if df is None or df.empty or len(df) < 2:
        return img
    strip_h = img_h // 3

    series = {
        "diss": (
            pd.to_numeric(df.get(COL_DISS), errors="coerce").to_numpy(dtype=float)
            if COL_DISS in df.columns
            else None
        ),
        "freq": (
            pd.to_numeric(df.get(COL_FREQ), errors="coerce").to_numpy(dtype=float)
            if COL_FREQ in df.columns
            else None
        ),
        "energy": derivative_energy(df),
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


def generate_fill_image(df: pd.DataFrame, version: int = 2) -> np.ndarray:
    """Version dispatch. version=1 reproduces the legacy classifier render
    (diss/freq/Difference at FILL_GEN geometry) so old type_cls weights keep
    working; version=2 is the derivative-energy render; version=3 swaps in
    the step-coincidence energy (retrain required - weights and version
    travel together)."""
    if version == 1:
        # v1 generate_fill_cls takes PER-STRIP height.
        return DP.generate_fill_cls(df, img_h=FILL_GEN_H // 3, img_w=FILL_GEN_W)
    if version >= 3:
        return generate_fill_cls_v3(df, FILL_GEN_W, FILL_GEN_H)
    return generate_fill_cls_v2(df, FILL_GEN_W, FILL_GEN_H)


def prepare_cls_input(df: pd.DataFrame, version: int = 2) -> np.ndarray:
    """THE train/deploy contract: preprocessed dataframe -> the exact
    224x224 BGR uint8 image the classifier consumes. build_fill_dataset.py
    saves this image; the predictor feeds this image. INTER_AREA matches the
    existing inference path."""
    img = generate_fill_image(df, version=version)
    return cv2.resize(img, (FILL_INFERENCE_W, FILL_INFERENCE_H), interpolation=cv2.INTER_AREA)


# ===========================================================================
#  Version 3: step-coincidence energy
# ===========================================================================
#
# What the first v7 training run's offender triage established: the v2
# derivative-energy strip fails on slow LATE transitions, and not for lack
# of trying harder - for two structural reasons the triage quantified
# (POI1/2 salience ~4.2x trace median vs POI4/5 at ~1.6x, with measured
# transition extents of 12-67 s against a longest curvature scale of 4 s):
#
#   1. WRONG MATCHED FILTER. The windowed second difference samples three
#      POINTS, so per-sample noise never averages down as the window grows,
#      and at long windows the drifting background's own curvature (random
#      walk: ~sqrt(w)) inflates the per-scale MAD normalizer - the scales
#      that should catch slow transitions normalize themselves away.
#      Synthetic verification: adding long scales + pre-smoothing to the
#      curvature moved worst-event/phantom separation 0.97 -> 0.96 (nothing).
#      A STEP filter - difference of adjacent window MEANS - carries the
#      full step amplitude as signal while noise shrinks ~sqrt(w/dt):
#      separation 0.97 -> 1.18, POI5 salience 1.29x -> 2.43x median.
#
#   2. WRONG COMBINE. Taking the max across signals lets a single-channel
#      drift excursion or noise burst masquerade as an event - the phantom
#      fuel behind the 2ch->3ch over-counts. Physics says a genuine fill
#      transition moves dissipation AND resonance frequency together; the
#      per-scale GEOMETRIC MEAN of the two normalized step responses keeps
#      coordinated events and suppresses single-channel excursions:
#      separation 1.18 -> 1.22, POI5 salience 2.43x -> 2.70x median
#      (all figures median over 10 noise seeds, monotone improvement in
#      every seed).
#
# Scales are absolute (0.5/2/8 s) plus SPAN-RELATIVE (span/32, span/12):
# the fill classifier sees whole runs from 25 s to 750 s; fixed absolute
# scales cannot serve both ends, and the offenders' transition extents ran
# ~6-9% of span (the dynamic-box cap), which span/12..span/32 brackets.
#
# Version 3 = v2's value strips unchanged, B strip = this energy.
# v3-trained weights ship with FILL_RENDER_VERSION=3; the shared detector
# render (detector_render) is NOT touched - its weights contract stands.

STEP_ABS_SCALES_S = (0.5, 2.0, 8.0)
STEP_REL_SCALES = (1.0 / 32.0, 1.0 / 12.0)
STEP_SMOOTH_S = 0.15


def _step_response(x: np.ndarray, w: int) -> np.ndarray:
    """|mean(x[i+1..i+w]) - mean(x[i-w..i])|, normalized by the interior
    MAD of its own response - a matched filter for level shifts at scale
    ~w samples. O(n) via cumulative sums."""
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
    """Multi-scale, cross-signal-coincident step salience (see block
    comment above). Returns a per-sample trace; drawn as strip 3 by the
    version-3 render."""
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
        if w < 3:
            continue
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


def generate_fill_cls_v3(
    df: pd.DataFrame, img_w: int = FILL_GEN_W, img_h: int = FILL_GEN_H
) -> np.ndarray:
    """v3 render: identical to v2 except strip 3 is the step-coincidence
    energy instead of the curvature-based derivative energy."""
    img = np.zeros((img_h, img_w, IMG_CHANNELS), dtype=np.uint8)
    if df is None or df.empty or len(df) < 2:
        return img
    strip_h = img_h // 3
    series = {
        "diss": (
            pd.to_numeric(df.get(COL_DISS), errors="coerce").to_numpy(dtype=float)
            if COL_DISS in df.columns
            else None
        ),
        "freq": (
            pd.to_numeric(df.get(COL_FREQ), errors="coerce").to_numpy(dtype=float)
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

"""
crosscheck.py
=============

Analysis-time fill-verdict cross-check using the v7 ZOOM detectors —
the no-retrain lever against the audit's costliest error population.

Migrated verbatim (as ``inference/crosscheck.py``) from the top-level
``fill_crosscheck.py`` module as part of the ``inference/`` subpackage
split. Logic, thresholds and docstrings are unchanged from the original;
this module has no relative imports to fix (numpy/pandas + stdlib only).

As of this migration, this module IS wired into
``QModelV6YOLO.predict()`` — see ``controller.py``'s ``_crosscheck_fill``
method and the ``crosscheck`` keyword argument on ``predict()``.

Why this exists
---------------
The offender triage established that the analysis-time under-counts
(~13 val runs whose entire post-POI5 region reads 2ch) are runs where the
late transition is nearly invisible AT FULL-RUN SCALE: measured extents of
12-67 s render as a sub-percent slope change across a 640 px frame. That
is exactly the failure mode the detector side already solved with zoom
refinement — "at full-run scale a slow transition is a faint smear; in an
8-40 s window it fills a large fraction of the frame." The zoom detectors
(ch1_zoom/ch2_zoom/ch3_zoom) are trained, shipped, and idle at the moment
the fill classifier renders its verdict.

So: treat the classifier's channel count as a PRIOR, not a verdict — the
same philosophy as the configuration-prior decode layer. Two checks:

  * UNDER-COUNT RESCUE (``verify_fill_count``): if the classifier says k
    channels (k < 3), slide the ch(k+1)_zoom detector over the tail after
    the last confirmed POI. A confident detection means the (k+1)-th
    transition exists — upgrade the verdict, hand the detected time to the
    cascade as a candidate anchor, and repeat (a 1ch verdict on a true 3ch
    run can climb twice).
  * OVER-COUNT VETO (``verify_claimed_poi``): the mirror check for the
    smaller over-count population (2ch->3ch full-run misses). The cascade
    already produced a POI time for the claimed last channel; re-render a
    zoom window around it and ask the zoom detector whether anything is
    there. Silence at zoom scale is evidence the fill classifier
    hallucinated the count. This function only REPORTS (verdict + zoom
    confidence); downgrading is a controller decision because a missing
    zoom detection can also mean a zoom-recall failure.

Asymmetry is deliberate: rescue is safe-by-construction (a confident
positive detection is strong evidence; finding nothing changes nothing),
veto is advisory.

Cost: rescue runs only when the verdict is < 3ch. A worst case — long tail,
three window widths, half-width stride, a two-step climb — measured ~70
zoom inferences; typical single-step checks over a normal tail are ~10-30.
At batch-1 YOLO-s latency this is well under a second of analysis time,
and it is exactly zero on the 3ch verdicts that dominate. Trim windows_s
or stride_frac if it matters.

Integration point: QModelV6YOLO controller, immediately after the reverse
cascade completes and before configuration-prior decode — see
``controller.py``'s ``_crosscheck_fill``. The zoom detectors, thresholds,
and window geometry all come from the existing QModelV6Config REFINE
settings unless overridden.

Usage sketch
------------
    from .crosscheck import verify_fill_count

    result = verify_fill_count(
        df_p,
        fill_channels=fill_pred,          # classifier verdict (0..3)
        poi_times=cascade_poi_times,      # whatever the cascade has so far
        zoom_detectors={2: ch2_zoom_det, 3: ch3_zoom_det},
    )
    if result.upgraded:
        Log.i(TAG, f"fill verdict {fill_pred} -> {result.channels} "
                   f"(zoom conf {result.evidence[-1].conf:.2f})")
        fill_pred = result.channels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

COL_TIME = "Relative_time"

# Channel index -> the POI that transition creates, and the POI that
# anchors the start of its search tail.
CHANNEL_POI = {1: "POI3", 2: "POI4", 3: "POI5"}
ANCHOR_POI = {1: "POI2", 2: "POI3", 3: "POI4"}

# Window geometry mirrors the zoom training distribution (ZOOM_W_RANGE_S
# 8-40 s, POI landing at 0.25-0.75 of the window): sweep a few widths so
# both fast and slow transitions land near their trained appearance.
DEFAULT_WINDOWS_S = (12.0, 24.0, 40.0)
DEFAULT_STRIDE_FRAC = 0.5  # window overlap: stride = frac * width
# Confidence bar for UPGRADING a verdict. Deliberately far above
# REFINE_MIN_CONF (0.20, which refines an already-decoded POI): here the
# detection must overturn a classifier verdict on its own. Sweep against
# the val offender/clean split, not by hand.
DEFAULT_UPGRADE_CONF = 0.55
# Margin after the anchor POI before the search tail begins (skip the
# anchor's own transition extent so it cannot be re-detected as "next").
DEFAULT_ANCHOR_MARGIN_S = 3.0
MIN_WINDOW_POINTS = 64


@dataclass
class ZoomEvidence:
    channel: int
    time: float
    conf: float
    window: tuple


@dataclass
class CrosscheckResult:
    channels: int
    upgraded: bool = False
    evidence: List[ZoomEvidence] = field(default_factory=list)
    windows_scanned: int = 0


def _best_zoom_hit(
    df_p: pd.DataFrame,
    detector,
    t_start: float,
    t_end: float,
    windows_s=DEFAULT_WINDOWS_S,
    stride_frac: float = DEFAULT_STRIDE_FRAC,
) -> Optional[dict]:
    """Slides zoom windows over [t_start, t_end], returns the single best
    detection {time, conf, window} across all windows/widths, or None.
    Detector is a QModelV6YOLO_Detector wrapping a *_zoom model (single
    class -> predict_single returns {0: {time, conf}} when it fires)."""
    t = pd.to_numeric(df_p[COL_TIME], errors="coerce").to_numpy(dtype=float)
    best: Optional[dict] = None
    scanned = 0
    for W in windows_s:
        if t_end - t_start < max(4.0, 0.5 * W):
            continue
        stride = max(1.0, stride_frac * W)
        w0 = t_start
        while w0 < t_end - 2.0:
            w1 = min(w0 + W, t_end)
            m = (t >= w0) & (t < w1)
            if m.sum() >= MIN_WINDOW_POINTS:
                dets = detector.predict_single(df_p.loc[m])
                scanned += 1
                for _cls, d in (dets or {}).items():
                    if best is None or d["conf"] > best["conf"]:
                        best = {
                            "time": float(d["time"]),
                            "conf": float(d["conf"]),
                            "window": (float(w0), float(w1)),
                        }
            w0 += stride
    if best is not None:
        best["scanned"] = scanned
    return best if best else {"scanned": scanned} if scanned else None


def verify_fill_count(
    df_p: pd.DataFrame,
    fill_channels: int,
    poi_times: Dict[str, float],
    zoom_detectors: Dict[int, object],
    upgrade_conf: float = DEFAULT_UPGRADE_CONF,
    windows_s=DEFAULT_WINDOWS_S,
    anchor_margin_s: float = DEFAULT_ANCHOR_MARGIN_S,
) -> CrosscheckResult:
    """Under-count rescue: climbs the channel count while the next zoom
    detector confidently finds the next transition in the tail.

    Args:
        df_p: full preprocessed run (the analysis-time frame).
        fill_channels: classifier verdict, -1..3 (values < 1 are returned
            unchanged — POI1/2 existence is the init stage's business).
        poi_times: POI times known so far (cascade output or labels);
            detected upgrades are ADDED to a copy in the result evidence,
            not mutated in place.
        zoom_detectors: {channel_index: QModelV6YOLO_Detector} for the
            available zoom stages, e.g. {1: ch1_zoom, 2: ch2_zoom,
            3: ch3_zoom}. Missing entries stop the climb (no-op beyond).
    """
    result = CrosscheckResult(channels=fill_channels)
    if fill_channels < 1 or df_p is None or df_p.empty:
        return result

    t = pd.to_numeric(df_p[COL_TIME], errors="coerce").to_numpy(dtype=float)
    t_end = float(np.nanmax(t))
    known = dict(poi_times)

    k = fill_channels
    while k < 3:
        nxt = k + 1
        det = zoom_detectors.get(nxt)
        if det is None:
            break
        anchor = known.get(CHANNEL_POI[k]) or known.get(ANCHOR_POI[nxt])
        if anchor is None:
            break
        hit = _best_zoom_hit(df_p, det, float(anchor) + anchor_margin_s, t_end, windows_s=windows_s)
        if hit:
            result.windows_scanned += hit.get("scanned", 0)
        if not hit or "conf" not in hit or hit["conf"] < upgrade_conf:
            break
        known[CHANNEL_POI[nxt]] = hit["time"]
        result.evidence.append(
            ZoomEvidence(channel=nxt, time=hit["time"], conf=hit["conf"], window=hit["window"])
        )
        k = nxt
        result.channels = k
        result.upgraded = True
    return result


def verify_claimed_poi(
    df_p: pd.DataFrame,
    channel: int,
    poi_time: float,
    zoom_detector,
    window_s: float = 24.0,
    tolerance_s: Optional[float] = None,
) -> ZoomEvidence:
    """Over-count advisory: zoom-inspect the claimed last channel's POI.
    Returns the best zoom confidence for a detection NEAR the claimed time
    (within tolerance, default 30% of the window); conf 0.0 means the zoom
    detector saw nothing where the count claims a transition. Advisory
    only — the controller decides whether silence downgrades."""
    tol = tolerance_s if tolerance_s is not None else 0.3 * window_s
    t = pd.to_numeric(df_p[COL_TIME], errors="coerce").to_numpy(dtype=float)
    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
    best_conf, best_t = 0.0, poi_time
    for frac in (0.35, 0.5, 0.65):  # POI at several window positions
        w0 = max(t0, poi_time - frac * window_s)
        w1 = min(t1, w0 + window_s)
        m = (t >= w0) & (t < w1)
        if m.sum() < MIN_WINDOW_POINTS:
            continue
        dets = zoom_detector.predict_single(df_p.loc[m])
        for _cls, d in (dets or {}).items():
            if abs(d["time"] - poi_time) <= tol and d["conf"] > best_conf:
                best_conf, best_t = float(d["conf"]), float(d["time"])
    return ZoomEvidence(
        channel=channel, time=best_t, conf=best_conf, window=(poi_time - tol, poi_time + tol)
    )

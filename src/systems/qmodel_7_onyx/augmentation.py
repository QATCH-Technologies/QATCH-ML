"""
augmentation.py
================

Signal-domain augmentation for QModel detector training, plus dynamic
(event-extent-calibrated) bounding-box sizing.

Why signal-domain
-----------------
All geometric pixel-space augmentation (mosaic, flips, translation, scale)
is DISABLED in the trainer, because on these renders the x-axis IS time:
fliplr is time reversal, mosaic destroys global fill context, translation
breaks the time<->pixel map the cascade relies on. Instead we augment the
raw signal before rendering, where every transform has a physical meaning
and POI labels can be warped exactly along with the data:

  * time_warp        - global log-uniform stretch x [0.5 .. 4] composed with
                       a smooth piecewise-linear monotone jitter. This is the
                       high-viscosity synthesizer: stretching a fast run's
                       time axis manufactures the slow-fill geometry the
                       corpus barely contains (15 runs above 150 cP), instead
                       of merely re-weighting them. Monotonicity preserves
                       event order and shape topology; only the time scale -
                       which the decode layer, not the detector, is
                       responsible for - is changed.
  * inject_noise     - white noise scaled to a robust (MAD) signal sigma,
                       low-frequency baseline drift, and sparse spikes.
  * amplitude_jitter - per-signal gain/offset jitter, breaking any
                       memorization of absolute signal levels.

All transforms take and return (df_raw, poi_times) so labels stay exact.

Dynamic boxes
-------------
``dynamic_box_width_sec`` measures the actual temporal extent of the fill
transition around each POI from the dissipation derivative, so slow
(high-viscosity) events get proportionally wider boxes instead of the fixed
pixel size the current model was trained with. Width is clamped to sane
pixel bounds at render time by the dataset builder.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

COL_TIME = "Relative_time"
COL_DISS = "Dissipation"
COL_FREQ = "Resonance_Frequency"


# ===========================================================================
#  Time warping
# ===========================================================================


def make_monotone_warp(
    t_min: float,
    t_max: float,
    rng: np.random.Generator,
    log_stretch_range: Tuple[float, float] = (np.log(0.5), np.log(4.0)),
    n_knots: int = 5,
    slope_sigma: float = 0.35,
):
    """Returns a strictly monotone map w(t) on [t_min, t_max].

    w is a global stretch S = exp(U(log_stretch_range)) composed with a
    piecewise-linear jitter whose per-segment slopes are S * exp(N(0,
    slope_sigma)) - i.e. locally faster/slower filling, globally scaled.
    """
    S = float(np.exp(rng.uniform(*log_stretch_range)))
    knots = np.linspace(t_min, t_max, n_knots + 1)
    slopes = S * np.exp(rng.normal(0.0, slope_sigma, size=n_knots))
    seg = np.diff(knots) * slopes
    w_knots = np.concatenate([[t_min], t_min + np.cumsum(seg)])

    def w(t: np.ndarray) -> np.ndarray:
        return np.interp(t, knots, w_knots)

    return w, S


def time_warp(
    df: pd.DataFrame,
    poi_times: Dict[str, float],
    rng: np.random.Generator,
    **warp_kwargs,
) -> Tuple[pd.DataFrame, Dict[str, float], float]:
    """Apply a monotone time warp to the raw frame and its POI labels.
    Returns (warped_df, warped_poi_times, global_stretch)."""
    t = pd.to_numeric(df[COL_TIME], errors="coerce").to_numpy(dtype=float)
    w, S = make_monotone_warp(float(np.nanmin(t)), float(np.nanmax(t)), rng, **warp_kwargs)
    out = df.copy()
    out[COL_TIME] = w(t)
    poi_out = {k: float(w(np.array([v]))[0]) for k, v in poi_times.items()}
    return out, poi_out, S


# ===========================================================================
#  Noise / amplitude
# ===========================================================================


def _robust_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return 0.0
    d = np.diff(x)  # detrend by differencing: noise sigma, not signal range
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def inject_noise(
    df: pd.DataFrame,
    rng: np.random.Generator,
    white_frac: Tuple[float, float] = (0.2, 1.0),
    drift_frac: Tuple[float, float] = (0.0, 3.0),
    spike_prob: float = 0.15,
    cols: Tuple[str, ...] = (COL_DISS, COL_FREQ),
) -> pd.DataFrame:
    """Additive white noise + low-frequency baseline drift + sparse spikes,
    all scaled to each signal's own robust noise floor so the augmentation
    never overwhelms event shape."""
    out = df.copy()
    n = len(out)
    for col in cols:
        if col not in out.columns:
            continue
        x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        s = _robust_sigma(x)
        if s <= 0:
            continue
        x = x + rng.normal(0.0, s * rng.uniform(*white_frac), size=n)
        # drift: sum of 1-3 slow sinusoids with random phase
        amp = s * rng.uniform(*drift_frac)
        if amp > 0:
            tt = np.linspace(0, 1, n)
            drift = np.zeros(n)
            for _ in range(rng.integers(1, 4)):
                drift += np.sin(2 * np.pi * (rng.uniform(0.3, 2.0) * tt + rng.uniform()))
            drift *= amp / max(1e-9, np.abs(drift).max())
            x = x + drift
        if rng.random() < spike_prob:
            for _ in range(rng.integers(1, 4)):
                i = int(rng.integers(0, n))
                x[i : i + int(rng.integers(1, 4))] += s * rng.uniform(5, 20) * rng.choice([-1, 1])
        out[col] = x
    return out


def amplitude_jitter(
    df: pd.DataFrame,
    rng: np.random.Generator,
    gain_sigma: float = 0.08,
    cols: Tuple[str, ...] = (COL_DISS, COL_FREQ),
) -> pd.DataFrame:
    """Per-signal multiplicative gain jitter about the signal's own baseline
    (its early-run median), so event amplitude varies but the baseline does
    not run away."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        base = float(np.nanmedian(x[: max(8, len(x) // 50)]))
        out[col] = base + (x - base) * float(np.exp(rng.normal(0.0, gain_sigma)))
    return out


def augment_run(
    df: pd.DataFrame,
    poi_times: Dict[str, float],
    rng: np.random.Generator,
    p_warp: float = 0.9,
    p_noise: float = 0.6,
    p_amp: float = 0.6,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """Compose the augmentations. Returns (df, poi_times, info)."""
    info: Dict[str, float] = {"stretch": 1.0}
    if rng.random() < p_warp:
        df, poi_times, S = time_warp(df, poi_times, rng)
        info["stretch"] = S
    if rng.random() < p_amp:
        df = amplitude_jitter(df, rng)
    if rng.random() < p_noise:
        df = inject_noise(df, rng)
    return df, poi_times, info


# ===========================================================================
#  Dynamic box sizing
# ===========================================================================


def dynamic_box_width_sec(
    df_p: pd.DataFrame,
    poi_t: float,
    diss_col: str = COL_DISS,
    time_col: str = COL_TIME,
    rel_window: float = 0.04,
    min_window_s: float = 0.75,
    active_frac: float = 0.2,
    min_width_s: float = 0.05,
    max_width_frac: float = 0.06,
) -> float:
    """Temporal extent of the fill transition around poi_t, in seconds.

    Measures where the smoothed |d(dissipation)/dt| within a local window
    exceeds ``active_frac`` of its local peak - i.e. the duration of the
    transition itself. Slow (high-viscosity) events therefore get wider
    boxes automatically; the fixed-pixel-size assumption the current model
    was trained with is what starved it of gradient on stretched events.

    Falls back to min_width_s when the local derivative is degenerate.
    """
    t = pd.to_numeric(df_p[time_col], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(df_p[diss_col], errors="coerce").to_numpy(dtype=float)
    if len(t) < 8:
        return min_width_s
    span = float(t[-1] - t[0])
    half_w = max(min_window_s, rel_window * span)
    m = (t >= poi_t - half_w) & (t <= poi_t + half_w)
    if m.sum() < 8:
        return min_width_s
    tw, xw = t[m], x[m]
    d = np.gradient(xw, tw)
    # light smoothing (moving average over ~5 samples)
    k = 5
    d = np.convolve(np.abs(d), np.ones(k) / k, mode="same")
    peak = float(d.max())
    if not np.isfinite(peak) or peak <= 0:
        return min_width_s
    active = d >= active_frac * peak
    if not active.any():
        return min_width_s
    width = float(tw[active][-1] - tw[active][0])
    return float(np.clip(width, min_width_s, max_width_frac * span))

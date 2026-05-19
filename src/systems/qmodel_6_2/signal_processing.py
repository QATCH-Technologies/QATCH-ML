"""
QModel v6 — Signal processing and image rendering
==================================================

Three responsibilities, in order of execution:

  1. Standardised-dt preprocessing
       ``preprocess_dataframe`` interpolates the raw run onto a uniform
       time grid (TARGET_DT_SEC) and computes the Difference curve.
       After this step every downstream operation can treat sample index
       and physical time as proportional — no more sampling-rate
       surprises buried in gradient computations.

  2. Engineered feature channels
       ``compute_feature_channels`` builds three multiscale signed-
       gradient bands from the Difference curve:
           - diff_pos          : loading events (POI3 / POI4 / POI5)
           - diff_neg_fine     : sharp baseline events (POI1/POI2 low-cP)
           - diff_neg_coarse   : slow baseline events (POI1/POI2 high-cP)
       Locally-normalised via rolling p90 so each channel is calibrated
       against local activity rather than the run's global maximum.

  3. Image rendering
       ``render_detection_image`` composes the five-strip canvas:
           [ Diss | Freq | Diff | EngFeatures | TimeGradient ]
       Per-channel resolution and smoothing come from the channel's
       ``ResolutionPreset`` (config.py).

The renderer is shared by every cascade channel. The only thing that
varies between channels is the ResolutionPreset and a handful of bool /
window-size toggles, all packaged in :class:`ChannelConfig`.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.0.0
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import medfilt, savgol_filter

from config import (
    ALL_SCALES_PX,
    BASELINE_END_SEC,
    BASELINE_OFFSET_SEC,
    BASELINE_START_SEC,
    COARSE_SCALES_PX,
    DIFF_FACTOR,
    FINE_SCALES_PX,
    LOCAL_NORM_WIN_PX,
    RENDER_ENGINEERED_STRIP,
    RENDER_TIME_STRIP,
    RENDER_X_TICKS,
    RENDER_Y_TICKS,
    TARGET_DT_SEC,
    TICK_COLOR_BGR,
    TICK_THICKNESS,
    X_TICK_INTERVAL_SEC,
    Y_TICK_FRACTIONS,
    ChannelConfig,
)

LOG = logging.getLogger("v6.signal")


# ===========================================================================
#  Column names — kept identical to QModelV6YOLO_DataProcessor for drop-in
#  compatibility with existing run discovery code.
# ===========================================================================

COL_TIME = "Relative_time"
COL_DISS = "Dissipation"
COL_FREQ = "Resonance_Frequency"
COL_DIFF = "Difference"

DROP_COLS = ("Date", "Time", "Ambient", "Peak Magnitude (RAW)", "Temperature")

# BGR channel mapping for the three signal strips. Difference → Blue,
# Frequency → Green, Dissipation → Red. Matches v6 and the analysis
# scripts so downstream tooling renders the same.
SIGNAL_BGR_CHANNEL: Dict[str, int] = {
    COL_DIFF: 0,  # B
    COL_FREQ: 1,  # G
    COL_DISS: 2,  # R
}

EPSILON = 1e-9
PADDING = 5
WHITE = (255, 255, 255)


# ===========================================================================
#  Stage 1 — Preprocessing (uniform dt + Difference curve + median smooth)
# ===========================================================================


def preprocess_dataframe(
    df_raw: pd.DataFrame,
    target_dt: float = TARGET_DT_SEC,
    median_kernel: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Interpolate a raw run onto a uniform time grid and compute Difference.

    Args:
        df_raw: Raw sensor dataframe. Must contain ``Relative_time``,
            ``Resonance_Frequency``, and ``Dissipation``.
        target_dt: Uniform sample spacing in seconds. Defaults to the
            package-level :data:`TARGET_DT_SEC`.
        median_kernel: Kernel size for per-column median smoothing. Set
            to 1 to disable.

    Returns:
        A new dataframe with uniform ``Relative_time`` spacing, all
        numeric columns interpolated, the ``Difference`` column added,
        and median smoothing applied — or ``None`` if the input lacks a
        time column or has fewer than 50 rows.
    """
    if df_raw is None or df_raw.empty:
        return None

    df = df_raw.copy()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    if COL_TIME not in df.columns:
        return None

    df.drop_duplicates(subset=[COL_TIME], keep="first", inplace=True)
    if len(df) < 50:
        return None

    t_min = float(df[COL_TIME].min())
    t_max = float(df[COL_TIME].max())
    if (t_max - t_min) < 1e-6:
        return None

    # Uniform time grid.
    new_grid = np.arange(t_min, t_max, target_dt)
    if len(new_grid) < 50:
        return None

    df = df.set_index(COL_TIME)
    combined = df.index.union(new_grid).sort_values()
    df = df.reindex(combined).interpolate(method="index").loc[new_grid]
    df = df.reset_index().rename(columns={"index": COL_TIME})

    # Difference curve (post-resampling so the baseline window is
    # commensurable across runs).
    diff = _compute_difference_curve(df)
    df[COL_DIFF] = diff if diff is not None else 0.0

    if median_kernel and median_kernel >= 3:
        for col in df.columns:
            if col == COL_TIME or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            df[col] = medfilt(df[col].to_numpy(dtype=float), kernel_size=median_kernel)

    return df


def _compute_difference_curve(
    df: pd.DataFrame,
    diff_factor: float = DIFF_FACTOR,
) -> Optional[pd.Series]:
    """Return the Difference signal derived from baseline-normalised Diss and Freq."""
    if not all(c in df.columns for c in (COL_FREQ, COL_DISS, COL_TIME)):
        return None

    xs = df[COL_TIME].to_numpy(dtype=float)
    if len(xs) == 0:
        return None

    i = int(np.searchsorted(xs, BASELINE_START_SEC))
    j = int(np.searchsorted(xs, BASELINE_END_SEC))
    if i == j and j < len(xs):
        j = int(np.searchsorted(xs, xs[j] + BASELINE_OFFSET_SEC))
    if i >= len(df) or j > len(df) or i == j:
        i, j = 0, min(100, len(df))

    avg_f = float(df[COL_FREQ].iloc[i:j].mean())
    avg_d = float(df[COL_DISS].iloc[i:j].mean())

    ys_diss = (df[COL_DISS].to_numpy(dtype=float) - avg_d) * avg_f / 2.0
    ys_freq = avg_f - df[COL_FREQ].to_numpy(dtype=float)
    return pd.Series(ys_freq - diff_factor * ys_diss, index=df.index)


# ===========================================================================
#  Stage 2 — Engineered feature channels
# ===========================================================================
# All gradients here are taken on the uniform-dt pixel grid, which after
# resampling is exactly proportional to Relative_time. This is the
# explicit answer to bucket-list item #2: "ensure all features are
# computed w.r.t. relative time as the X-axis, not the indices of the
# data."


def _multiscale_signed(
    values_px: np.ndarray,
    scales_px: tuple,
) -> tuple:
    """Return (pos_envelope, neg_envelope) of signed-gradient event rate.

    For each smoothing window, compute the signed gradient, normalise by
    its own p99 magnitude, clip to [-1, 1], split into positive and
    negative envelopes, and take the per-pixel max across scales.
    """
    pos = np.zeros_like(values_px)
    neg = np.zeros_like(values_px)
    for win in scales_px:
        w = win if win % 2 == 1 else win + 1
        sm = uniform_filter1d(values_px, size=w, mode="reflect")
        g = np.gradient(sm)
        p99 = float(np.percentile(np.abs(g), 99)) + EPSILON
        pos = np.maximum(pos, np.clip(g / p99, 0.0, 1.0))
        neg = np.maximum(neg, np.clip(-g / p99, 0.0, 1.0))
    return pos, neg


def _adaptive_norm(rate: np.ndarray, win: int = LOCAL_NORM_WIN_PX) -> np.ndarray:
    """Locally-normalise a [0, 1] rate signal by its rolling 90th percentile."""
    if len(rate) < 2 * win + 1:
        # Fall back to global p90 on very short slices.
        p90 = float(np.percentile(rate, 90)) + EPSILON
        return np.clip(rate / p90, 0.0, 1.0)

    from numpy.lib.stride_tricks import sliding_window_view

    padded = np.pad(rate, win, mode="reflect")
    windows = sliding_window_view(padded, 2 * win + 1)
    local_p90 = np.percentile(windows, 90, axis=1) + EPSILON
    return np.clip(rate / local_p90, 0.0, 1.0)


def compute_feature_channels(diff_px: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Build the three engineered event-rate channels from a Difference
    signal resampled onto the pixel grid.

    Returns a dict with keys ``diff_pos``, ``diff_neg_fine``,
    ``diff_neg_coarse``, each a [0, 1] array of length ``len(diff_px)``.
    """
    pos_all, _ = _multiscale_signed(diff_px, ALL_SCALES_PX)
    _, neg_fine = _multiscale_signed(diff_px, FINE_SCALES_PX)
    _, neg_coarse = _multiscale_signed(diff_px, COARSE_SCALES_PX)

    return {
        "diff_pos": _adaptive_norm(pos_all),
        "diff_neg_fine": _adaptive_norm(neg_fine),
        "diff_neg_coarse": _adaptive_norm(neg_coarse),
    }


# ===========================================================================
#  Stage 3 — Image rendering
# ===========================================================================
# Layout (top → bottom):
#
#     ┌──────────────────────────────────┐  ┐
#     │  strip 0 : Dissipation (R)       │  │
#     ├──────────────────────────────────┤  │ 4 × strip_h
#     │  strip 1 : Frequency   (G)       │  │
#     ├──────────────────────────────────┤  │
#     │  strip 2 : Difference  (B)       │  │
#     ├──────────────────────────────────┤  │
#     │  strip 3 : Engineered (R/G/B)    │  │
#     ├──────────────────────────────────┤  ┘
#     │  strip 4 : Time-position ramp    │  ← time_strip_h
#     └──────────────────────────────────┘
#


def _signal_polyline(
    values: np.ndarray,
    img_w: int,
    strip_h: int,
    strip_y_off: int,
) -> Optional[np.ndarray]:
    """Map a 1-D signal to (x, y) integer pixel coordinates for one strip."""
    if len(values) < 2:
        return None

    v_min, v_max = np.nanpercentile(values, [1.0, 99.0])
    diff = v_max - v_min
    if diff == 0:
        diff = EPSILON
        v_min -= EPSILON
    norm = np.clip((values - v_min) / diff, 0.0, 1.0)

    xs = np.linspace(0, img_w - 1, len(values)).astype(np.int32)
    draw_h = strip_h - (2 * PADDING)
    y_rel = (strip_h - PADDING) - (norm * draw_h)
    ys = (strip_y_off + y_rel).astype(np.int32)
    return np.stack((xs, ys), axis=1)


def _resample_to_pixel_grid(df: pd.DataFrame, img_w: int) -> Dict[str, np.ndarray]:
    """Sample each signal column onto a uniform img_w-pixel grid.

    All gradients downstream operate on this grid, which after dt
    resampling is exactly proportional to Relative_time. So pixel-space
    gradients == physical-time gradients up to a constant scale.
    """
    if COL_TIME not in df.columns or len(df) < 2:
        return {}

    t = df[COL_TIME].to_numpy(dtype=float)
    t_min, t_max = float(t[0]), float(t[-1])
    if (t_max - t_min) < 1e-9:
        return {}

    pixel_t = np.linspace(t_min, t_max, img_w)
    out: Dict[str, np.ndarray] = {"time": pixel_t}
    for col in (COL_DISS, COL_FREQ, COL_DIFF):
        if col in df.columns:
            out[col] = np.interp(pixel_t, t, df[col].to_numpy(dtype=float))
    return out


def render_detection_image(
    df: pd.DataFrame,
    cfg: ChannelConfig,
) -> Optional[np.ndarray]:
    """
    Render the five-strip detection image for one channel.

    Args:
        df: A *preprocessed* dataframe (uniform dt, with Difference
            column). Run :func:`preprocess_dataframe` first.
        cfg: Channel configuration. Determines resolution, smoothing,
            and whether the engineered-feature strip is rendered.

    Returns:
        ``np.ndarray`` of shape ``(cfg.resolution.img_h, cfg.resolution.img_w, 3)``,
        dtype ``uint8``, BGR. ``None`` on degenerate input.
    """
    if df is None or df.empty or len(df) < 32:
        return None

    res = cfg.resolution
    img_w = res.img_w
    strip_h = res.strip_h
    time_strip_h = res.time_strip_h
    img_h = res.img_h

    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # ── Optional signal smoothing ──────────────────────────────────────
    if cfg.smooth_signal_window >= 5:
        df = df.copy()
        win = cfg.smooth_signal_window
        win = win if win % 2 == 1 else win + 1
        poly = min(3, win - 1)
        for col in (COL_DISS, COL_FREQ, COL_DIFF):
            if col in df.columns:
                arr = df[col].to_numpy(dtype=float)
                if len(arr) >= win:
                    try:
                        df[col] = savgol_filter(arr, win, poly)
                    except Exception:
                        pass

    # ── Sample each signal onto the rendering pixel grid ──────────────
    grid = _resample_to_pixel_grid(df, img_w)
    if not grid:
        return img

    # ── Strips 0-2: raw signals as polylines with filled silhouette ───
    strip_specs = (
        (0, COL_DISS),  # R
        (1, COL_FREQ),  # G
        (2, COL_DIFF),  # B
    )
    for strip_idx, col in strip_specs:
        if col not in grid:
            continue
        y_off = strip_idx * strip_h
        pts = _signal_polyline(grid[col], img_w, strip_h, y_off)
        if pts is None:
            continue

        ch = SIGNAL_BGR_CHANNEL[col]
        fill = [0, 0, 0]
        fill[ch] = 255

        bottom_y = y_off + strip_h - PADDING
        poly = np.concatenate([pts, [[pts[-1, 0], bottom_y]], [[pts[0, 0], bottom_y]]])
        cv2.fillPoly(img, [poly], tuple(fill))
        cv2.polylines(
            img,
            [pts.reshape((-1, 1, 2))],
            isClosed=False,
            color=WHITE,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # ── Strip 3: engineered feature channels as heatmap (optional) ────
    next_y = 3 * strip_h  # running Y cursor below the three signal strips
    if RENDER_ENGINEERED_STRIP and cfg.include_engineered_features and COL_DIFF in grid:
        features = compute_feature_channels(grid[COL_DIFF])
        # BGR layout: B = diff_neg_coarse, G = diff_neg_fine, R = diff_pos
        b = (features["diff_neg_coarse"] * 255).astype(np.uint8)
        g = (features["diff_neg_fine"] * 255).astype(np.uint8)
        r = (features["diff_pos"] * 255).astype(np.uint8)
        strip = np.stack(
            [
                np.tile(b, (strip_h, 1)),
                np.tile(g, (strip_h, 1)),
                np.tile(r, (strip_h, 1)),
            ],
            axis=-1,
        )
        img[next_y : next_y + strip_h, :, :] = strip
        next_y += strip_h

    # ── Strip 4: time-position gradient (optional) ─────────────────────
    if RENDER_TIME_STRIP:
        ramp = np.linspace(0, 255, img_w, dtype=np.uint8)
        img[next_y : next_y + time_strip_h, :, :] = ramp[np.newaxis, :, np.newaxis]

    # ── Tick-mark overlays ─────────────────────────────────────────────
    _draw_ticks(img, grid, strip_h, img_w, img_h)

    return img


def _draw_ticks(
    img: np.ndarray,
    grid: dict,
    strip_h: int,
    img_w: int,
    img_h: int,
) -> None:
    """Burn optional X and Y tick marks into *img* in-place.

    Ticks are drawn as thin lines in TICK_COLOR_BGR across all active
    strips. They are applied after all signal rendering so they always
    sit on top.

    X ticks (vertical lines)
        One line every X_TICK_INTERVAL_SEC physical seconds. The pixel
        position is derived from the pixel-grid time array so it is
        exact to the resampled grid.

    Y ticks (horizontal lines)
        Drawn at Y_TICK_FRACTIONS relative positions within each of the
        three raw-signal strips (0 = strip top, 1 = strip bottom).
        The engineered and time strips are deliberately excluded — they
        encode discrete information already.
    """
    if not (RENDER_X_TICKS or RENDER_Y_TICKS):
        return

    tick_col = tuple(int(c) for c in TICK_COLOR_BGR)

    # ── X ticks: vertical lines at fixed time intervals ────────────────
    if RENDER_X_TICKS and "time" in grid:
        t_arr = grid["time"]
        t_min = float(t_arr[0])
        t_max = float(t_arr[-1])
        duration = t_max - t_min
        if duration > 0 and X_TICK_INTERVAL_SEC > 0:
            n_ticks = int(duration / X_TICK_INTERVAL_SEC)
            for k in range(1, n_ticks + 1):
                t_tick = t_min + k * X_TICK_INTERVAL_SEC
                if t_tick >= t_max:
                    break
                # Map physical time → pixel column via the grid.
                frac = (t_tick - t_min) / duration
                x_px = int(round(frac * (img_w - 1)))
                cv2.line(img, (x_px, 0), (x_px, img_h - 1), tick_col, TICK_THICKNESS)

    # ── Y ticks: horizontal lines at fractional positions in signal strips
    if RENDER_Y_TICKS:
        # Only draw on the three raw-signal strips (strips 0-2); the
        # engineered heatmap and time ramp have their own implicit scale.
        for s in range(3):
            y_top = s * strip_h
            for frac in Y_TICK_FRACTIONS:
                y_px = int(round(y_top + frac * strip_h))
                y_px = max(0, min(img_h - 1, y_px))
                cv2.line(img, (0, y_px), (img_w - 1, y_px), tick_col, TICK_THICKNESS)

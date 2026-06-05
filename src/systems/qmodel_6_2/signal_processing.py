"""
QModel v6 — Signal processing and image rendering  (v8 grayscale single-signal)
===============================================================================

Two responsibilities, in order of execution:

  1. Standardised-dt preprocessing
       ``preprocess_dataframe`` interpolates the raw run onto a uniform
       time grid (TARGET_DT_SEC) and computes the Difference curve.
       After this step every downstream operation can treat sample index
       and physical time as proportional — no more sampling-rate
       surprises buried in gradient computations.

  2. Single-signal image rendering
       ``render_detection_image`` renders ONE signal — the channel's
       ``signal_column`` (Dissipation, Resonance_Frequency, or the derived
       Difference) — filling the whole canvas. There is one detector per
       signal per POI, so each rendered image contains exactly one curve.

What changed in v8 (grayscale)
------------------------------
v7 rendered a 3-channel BGR image and colour-coded the lone signal by its
BGR channel purely as a visual aid. Since each image contains exactly one
signal, that colour carried NO information YOLO needs. v8 renders a
single-channel grayscale image instead:

  * Fill and outline are white (255) on a black (0) canvas.
  * Tick marks use a single gray value.
  * The returned array is 2-D ``(img_h, img_w)`` uint8, written to disk as
    grayscale PNG by the renderer — ~3-5x smaller than the old JPEG and
    lossless. Ultralytics promotes grayscale back to 3-channel at load,
    so no model change is required and pretrained transfer still works.

What changed in v7
------------------
The old multi-strip layout stacked all three raw signals plus an
engineered multiscale-gradient heatmap (``diff_pos`` / ``diff_neg_fine`` /
``diff_neg_coarse``) and a time-position ramp into a single tall image.
v7 dropped the stacked layout entirely: one signal per image, and NO
slow-trend / gradient heatmap of any kind is rendered.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    8.0.0
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter

from config import (
    BASELINE_END_SEC,
    BASELINE_OFFSET_SEC,
    BASELINE_START_SEC,
    DIFF_FACTOR,
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

# Retained for back-compat with any external tooling that imports it. In v8
# the renderer no longer uses per-signal BGR colour — every image is a
# single grayscale channel — so this mapping is informational only.
SIGNAL_BGR_CHANNEL: Dict[str, int] = {
    COL_DIFF: 0,  # B
    COL_FREQ: 1,  # G
    COL_DISS: 2,  # R
}

EPSILON = 1e-9
PADDING = 5

# v8: single-channel intensities.
FILL_VALUE = 255  # signal fill + outline (white on black)
SIGNAL_OUTLINE = 255


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
#  Stage 2 — Single-signal image rendering (v8: grayscale, single channel)
# ===========================================================================
#     ┌──────────────────────────────────────┐
#     │                                       │
#     │   single signal (Diss | Freq | Diff)  │   img_h = strip_h
#     │                                       │
#     └──────────────────────────────────────┘
#
# The image is a 2-D uint8 array (white signal on black). No colour, no
# stacked strips, no engineered heatmap, no time-position ramp.


def _signal_polyline(
    values: np.ndarray,
    img_w: int,
    img_h: int,
) -> Optional[np.ndarray]:
    """Map a 1-D signal to (x, y) integer pixel coordinates spanning the canvas."""
    if len(values) < 2:
        return None

    v_min = np.nanmin(values)
    v_max = np.nanmax(values)
    pad = 0.02 * (v_max - v_min)
    v_min -= pad
    v_max += pad
    diff = v_max - v_min
    if diff == 0:
        diff = EPSILON
        v_min -= EPSILON
    norm = np.clip((values - v_min) / diff, 0.0, 1.0)

    xs = np.linspace(0, img_w - 1, len(values)).astype(np.int32)
    draw_h = img_h - (2 * PADDING)
    ys = ((img_h - PADDING) - (norm * draw_h)).astype(np.int32)
    return np.stack((xs, ys), axis=1)


def _resample_signal_to_pixel_grid(
    df: pd.DataFrame,
    col: str,
    img_w: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Sample ONE signal column onto a uniform img_w-pixel grid.

    The grid is uniform in Relative_time, so pixel column == physical time
    up to a constant scale. Returns ``{"time": ..., "signal": ...}`` or
    ``None`` if the column is missing or the slice is degenerate.
    """
    if COL_TIME not in df.columns or col not in df.columns or len(df) < 2:
        return None

    t = df[COL_TIME].to_numpy(dtype=float)
    t_min, t_max = float(t[0]), float(t[-1])
    if (t_max - t_min) < 1e-9:
        return None

    pixel_t = np.linspace(t_min, t_max, img_w)
    sig = np.interp(pixel_t, t, df[col].to_numpy(dtype=float))
    return {"time": pixel_t, "signal": sig}


def render_detection_image(
    df: pd.DataFrame,
    cfg: ChannelConfig,
) -> Optional[np.ndarray]:
    """
    Render the single-signal detection image for one channel (grayscale).

    Exactly one signal — ``cfg.signal_column`` — is drawn in white on a
    black canvas, filling the whole image. The returned array is 2-D
    (single channel); the renderer writes it as grayscale PNG.

    Args:
        df: A *preprocessed* dataframe (uniform dt, with Difference
            column). Run :func:`preprocess_dataframe` first.
        cfg: Channel configuration. Determines resolution, which signal to
            render, and per-signal smoothing.

    Returns:
        ``np.ndarray`` of shape ``(cfg.resolution.img_h, cfg.resolution.img_w)``,
        dtype ``uint8``, single channel. ``None`` on degenerate input.
    """
    if df is None or df.empty or len(df) < 32:
        return None

    res = cfg.resolution
    img_w = res.img_w
    img_h = res.img_h

    # v8: single channel.
    img = np.zeros((img_h, img_w), dtype=np.uint8)

    col = cfg.signal_column
    if col not in df.columns:
        LOG.warning("render: signal column %s missing for channel %s", col, cfg.name)
        return img

    # ── Optional per-signal smoothing ──────────────────────────────────
    if cfg.smooth_signal_window >= 5:
        df = df.copy()
        win = cfg.smooth_signal_window
        win = win if win % 2 == 1 else win + 1
        poly = min(3, win - 1)
        arr = df[col].to_numpy(dtype=float)
        if len(arr) >= win:
            try:
                df[col] = savgol_filter(arr, win, poly)
            except Exception:
                pass

    # ── Sample the single signal onto the pixel grid ──────────────────
    grid = _resample_signal_to_pixel_grid(df, col, img_w)
    if grid is None:
        return img

    pts = _signal_polyline(grid["signal"], img_w, img_h)
    if pts is None:
        return img

    # Fill under the curve + outline, both white. No BGR channel selection.
    bottom_y = img_h - PADDING
    poly = np.concatenate([pts, [[pts[-1, 0], bottom_y]], [[pts[0, 0], bottom_y]]])
    cv2.fillPoly(img, [poly], FILL_VALUE)
    cv2.polylines(
        img,
        [pts.reshape((-1, 1, 2))],
        isClosed=False,
        color=SIGNAL_OUTLINE,
        thickness=1,
        # LINE_8 (hard edge) compresses better than LINE_AA for PNG. If
        # sub-pixel POI refinement is re-enabled, switch back to cv2.LINE_AA
        # to keep the anti-aliased sub-pixel edge information.
        lineType=cv2.LINE_8,
    )

    # ── Tick-mark overlays ─────────────────────────────────────────────
    _draw_ticks(img, grid, img_w, img_h)

    return img


def _draw_ticks(
    img: np.ndarray,
    grid: dict,
    img_w: int,
    img_h: int,
) -> None:
    """Burn optional X and Y tick marks into *img* in-place (grayscale).

    X ticks (vertical lines)
        One line every X_TICK_INTERVAL_SEC physical seconds, positioned via
        the pixel-grid time array so they are exact to the resampled grid.

    Y ticks (horizontal lines)
        Drawn at Y_TICK_FRACTIONS relative positions across the full image
        height (0 = top edge, 1 = bottom edge).
    """
    if not (RENDER_X_TICKS or RENDER_Y_TICKS):
        return

    # Collapse the (legacy) BGR tuple to a single gray intensity.
    tick_val = int(TICK_COLOR_BGR[0])

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
                frac = (t_tick - t_min) / duration
                x_px = int(round(frac * (img_w - 1)))
                cv2.line(img, (x_px, 0), (x_px, img_h - 1), tick_val, TICK_THICKNESS)

    # ── Y ticks: horizontal lines at fractional positions ──────────────
    if RENDER_Y_TICKS:
        for frac in Y_TICK_FRACTIONS:
            y_px = int(round(frac * img_h))
            y_px = max(0, min(img_h - 1, y_px))
            cv2.line(img, (0, y_px), (img_w - 1, y_px), tick_val, TICK_THICKNESS)

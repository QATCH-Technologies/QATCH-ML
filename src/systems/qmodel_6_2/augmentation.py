"""
QModel V7 — Augmentation and variant generation
================================================

This module bundles three augmentation concerns:

  Time-axis stretching
      ``time_stretch_df`` resamples a run along the index axis to make
      events look slower or faster. Tier-biased factor sampling
      (``sample_stretch_factor``) pushes low-cP runs toward stretching
      and high-cP runs toward gentle stretching / no compression — the
      latter being the lesson from the zoom-ch3 experience where
      compressing already-slow runs crowded POIs into a distribution the
      model never sees in production.

  Dynamic bounding boxes
      ``compute_box_width_norm`` returns a normalised box width that
      derives from a PHYSICAL second target (from
      :data:`BOX_SIZE_PROFILES`). Wide boxes for slow high-cP events,
      tight boxes for sharp low-cP events, all in the same normalised
      [0, 1] x-axis the YOLO label format requires.

  High-cP variant boost
      ``effective_variant_count`` scales the per-run base variant count
      by both the tier-balance multiplier and the per-channel
      ``high_cp_boost`` factor. This is the dataset-side equivalent of
      the 10k approach's oversampling — but applied AFTER the
      train/val split, so it cannot leak between splits the way the
      10k randomised-after-upsample split did.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.0.0
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config import (
    BOX_SIZE_PROFILES,
    TIER_EDGES,
    TIER_LABELS,
    BoxSizeProfile,
    ChannelConfig,
)
from signal_processing import COL_DIFF, COL_DISS, COL_FREQ, COL_TIME

# ===========================================================================
#  Viscosity tier helpers
# ===========================================================================


def viscosity_tier(visc_cP: float, edges: Sequence[float] = TIER_EDGES) -> int:
    """0-indexed tier bucket. NaN viscosity → tier 0."""
    if visc_cP is None or (isinstance(visc_cP, float) and math.isnan(visc_cP)):
        return 0
    if visc_cP <= 0:
        return 0
    for i in range(len(edges) - 1):
        if edges[i] <= visc_cP < edges[i + 1]:
            return i
    return len(edges) - 2


def compute_tier_multipliers(
    viscosities: Sequence[float],
    cap: int = 12,
    floor: int = 1,
    target_method: str = "median",
) -> Dict[int, int]:
    """Tier-balance multiplier: round(target_count / actual_count).

    Used by the dataset builder to bring rare tiers closer to the median
    tier count in the variant-multiplied dataset, without exploding
    storage on the densely-populated low-cP tiers.
    """
    counts: Dict[int, int] = {}
    for v in viscosities:
        t = viscosity_tier(v)
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        return {}

    vals = list(counts.values())
    if target_method == "max":
        target = max(vals)
    elif target_method == "mean":
        target = sum(vals) / len(vals)
    else:
        target = float(np.median(vals))

    out: Dict[int, int] = {}
    for tier, cnt in counts.items():
        out[tier] = max(floor, min(cap, int(round(target / max(cnt, 1)))))
    return out


# ===========================================================================
#  Time-axis stretching
# ===========================================================================


def time_stretch_df(
    df: pd.DataFrame,
    factor: float,
    smooth_when_stretching: bool = True,
    signal_columns: Sequence[str] = (COL_DISS, COL_FREQ, COL_DIFF),
) -> pd.DataFrame:
    """Stretch (>1) or compress (<1) event morphology by index-axis resampling.

    Time values are linearly interpolated so the run's duration
    (t_min, t_max) is preserved — only sample density and event
    sharpness change. The rendered image therefore differs even though
    the run's TIME range and POI position are unchanged, so the YOLO
    label (x_center) does NOT need adjustment.

    Args:
        df: Preprocessed dataframe (uniform dt, with COL_DIFF).
        factor: Stretch factor. ``factor > 1`` => slower-looking events;
            ``factor < 1`` => faster-looking events; ``≈ 1`` is a no-op.
        smooth_when_stretching: When ``factor > 1`` apply a width-scaled
            Savitzky-Golay smooth before resampling, otherwise upsampling
            just creates piecewise-linear staircases.
        signal_columns: Columns to smooth before resampling.

    Returns:
        A new dataframe with ``round(len(df) * factor)`` rows. Falls
        back to the original frame on degenerate input.
    """
    if abs(factor - 1.0) < 1e-3 or len(df) < 64:
        return df

    work = df.copy()

    # ── Pre-smoothing (only when stretching) ──────────────────────────
    if factor > 1.0 and smooth_when_stretching:
        win = max(5, int(2.0 * factor + 1.0))
        if win % 2 == 0:
            win += 1
        if len(work) >= win:
            poly = min(3, win - 1)
            for col in signal_columns:
                if col in work.columns:
                    arr = work[col].to_numpy(dtype=float)
                    try:
                        work[col] = savgol_filter(arr, win, poly)
                    except Exception:
                        pass

    # ── Index-axis resampling ─────────────────────────────────────────
    n_old = len(work)
    n_new = max(64, int(round(n_old * factor)))
    if n_new == n_old:
        return work

    old_idx = np.arange(n_old)
    new_idx = np.linspace(0, n_old - 1, n_new)

    out = pd.DataFrame()
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            out[col] = np.interp(new_idx, old_idx, work[col].to_numpy(dtype=float))
        else:
            ii = np.round(new_idx).astype(int).clip(0, n_old - 1)
            out[col] = work[col].iloc[ii].reset_index(drop=True)
    return out


def sample_stretch_factor(tier: int, rng: np.random.Generator) -> float:
    """
    Tier-biased stretch factor sampler.

    Design intent per tier (calibrated from the zoom-ch3 retrospective):

      Low-cP (tier 0-2)
          Stretch 1.4-3.5×. Synthesise slower-looking morphology so the
          model sees high-cP-like events even when the natural dataset
          is dominated by sharp low-cP fills.
      Mid (tier 3)
          Bidirectional: modest stretch 1.2-2.2× OR compression
          0.65-0.9×. Tier-3 runs span a useful natural morphology range;
          both directions are plausible.
      High-cP (tier 4-5)
          Light stretch 1.1-1.6× only. These runs are already slow;
          compressing them crowds POIs into a distribution the model
          never sees in production (the zoom-ch3 negative-bias result).
          Stretching slightly further is still realistic.
    """
    if tier <= 2:
        return float(rng.uniform(1.4, 3.5))
    if tier == 3:
        if rng.random() < 0.5:
            return float(rng.uniform(1.2, 2.2))
        return float(rng.uniform(0.65, 0.90))
    return float(rng.uniform(1.1, 1.6))


# ===========================================================================
#  Dynamic bounding-box sizing
# ===========================================================================


def compute_box_width_norm(
    poi_name: str,
    slice_duration_s: float,
    viscosity_cP: float,
    profile: Optional[BoxSizeProfile] = None,
) -> float:
    """
    Return the normalised ([0, 1]) bounding-box width for one YOLO label.

    The width is computed from a PHYSICAL second target via
    :class:`BoxSizeProfile`, then divided by the slice duration to
    normalise. A min/max floor/ceiling (also on the profile) prevents
    degenerate widths on extreme slice durations.

    Args:
        poi_name: One of POI1..POI5.
        slice_duration_s: Duration of the slice the YOLO box will sit
            inside (after any forward / backward / windowed cropping).
        viscosity_cP: Mean viscosity of the run, used for tier lookup.
        profile: Override for the default profile (handy for ablations).

    Returns:
        Normalised box width in [profile.min_norm_width,
        profile.max_norm_width].
    """
    if profile is None:
        profile = BOX_SIZE_PROFILES.get(poi_name)
    if profile is None:
        return 0.025  # safe default

    tier = viscosity_tier(viscosity_cP)
    width_s = profile.width_seconds(tier)

    if slice_duration_s <= 0:
        return profile.min_norm_width

    raw = width_s / slice_duration_s
    return float(np.clip(raw, profile.min_norm_width, profile.max_norm_width))


# ===========================================================================
#  Per-run variant count
# ===========================================================================


def effective_variant_count(
    base: int,
    tier_multiplier: int,
    viscosity_tier_idx: int,
    high_cp_boost: float,
    high_cp_tiers: Sequence[int] = (4, 5),
) -> int:
    """
    Number of rendered variants for one run, with high-cP boost applied.

    The base count comes from :class:`ChannelConfig.base_truncations`.
    The tier multiplier comes from :func:`compute_tier_multipliers`. The
    high-cP boost is the per-channel knob from :class:`ChannelConfig`
    and is only applied for tiers in ``high_cp_tiers``.

    Args:
        base: Base truncation count (channel-specific).
        tier_multiplier: Stratification balance multiplier for this
            run's tier.
        viscosity_tier_idx: 0-indexed tier of the run.
        high_cp_boost: Extra multiplier applied iff the run is in
            ``high_cp_tiers``. 1.0 disables the boost.
        high_cp_tiers: Tier indices considered "high cP" (default 4, 5).

    Returns:
        Final variant count for this run. Always ≥ 1.
    """
    n = base * max(1, tier_multiplier)
    if viscosity_tier_idx in high_cp_tiers and high_cp_boost > 1.0:
        n = int(round(n * high_cp_boost))
    return max(1, n)

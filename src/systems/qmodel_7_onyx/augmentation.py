"""Signal-domain augmentation and dynamic bounding-box sizing for detector training.

Implements physical signal-domain transformations (time warping, noise
injection, amplitude jitter) for training datasets where the x-axis
represents time. Because geometric pixel-space augmentations (e.g., flips,
translation, mosaic) distort the time-to-pixel mapping or break global
fill context, signal augmentation is applied directly to raw dataframes
to keep point-of-interest (POI) labels precisely aligned.

Also provides dynamic bounding-box width calculation (`dynamic_box_width_sec`)
to size detector target boxes based on measured temporal event extents.

Attributes:
    COL_TIME (str): DataFrame column name for relative time.
    COL_DISS (str): DataFrame column name for dissipation.
    COL_FREQ (str): DataFrame column name for resonance frequency.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

COL_TIME = "Relative_time"
COL_DISS = "Dissipation"
COL_FREQ = "Resonance_Frequency"


def make_monotone_warp(
    t_min: float,
    t_max: float,
    rng: np.random.Generator,
    log_stretch_range: Tuple[float, float] = (np.log(0.5), np.log(4.0)),
    n_knots: int = 5,
    slope_sigma: float = 0.35,
):
    """Constructs a strictly monotone time-warping function w(t) on [t_min, t_max].

    Composes a global stretch factor S = exp(U(log_stretch_range)) with a
    piecewise-linear jitter whose segment slopes follow S * exp(N(0, slope_sigma)).
    This simulates locally faster or slower filling while scaling the time axis globally.

    Args:
        t_min (float): Start of the time interval in seconds.
        t_max (float): End of the time interval in seconds.
        rng (np.random.Generator): NumPy random number generator instance.
        log_stretch_range (Tuple[float, float], optional): Log-uniform bounds
            for the global stretch factor. Defaults to (log(0.5), log(4.0)).
        n_knots (int, optional): Number of interior segments for piecewise
            linear jitter. Defaults to 5.
        slope_sigma (float, optional): Standard deviation of the log-normal
            slope jitter. Defaults to 0.35.

    Returns:
        Tuple[Callable[[np.ndarray], np.ndarray], float]: A tuple containing:
            - w: Callable warping function mapping original time array to
              warped time array.
            - S: The global stretch factor applied.
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
    """Applies a monotone time warp to a raw dataframe and its POI timestamps.

    Args:
        df (pd.DataFrame): Time series DataFrame containing a relative time column.
        poi_times (Dict[str, float]): Dictionary mapping POI names to
            timestamps (in seconds).
        rng (np.random.Generator): NumPy random number generator instance.
        **warp_kwargs: Additional arguments passed to `make_monotone_warp`.

    Returns:
        Tuple[pd.DataFrame, Dict[str, float], float]: A tuple containing:
            - warped_df: Copy of the DataFrame with warped relative timestamps.
            - warped_poi_times: Dictionary of transformed POI timestamps.
            - S: Global stretch factor applied during warping.
    """
    t = pd.to_numeric(df[COL_TIME], errors="coerce").to_numpy(dtype=float)
    w, S = make_monotone_warp(float(np.nanmin(t)), float(np.nanmax(t)), rng, **warp_kwargs)
    out = df.copy()
    out[COL_TIME] = w(t)
    poi_out = {k: float(w(np.array([v]))[0]) for k, v in poi_times.items()}
    return out, poi_out, S


def _robust_sigma(x: np.ndarray) -> float:
    """Estimates signal noise standard deviation using detrended median absolute deviation.

    Detrends the input array via first-order differencing and calculates a
    scaled MAD to isolate high-frequency noise from global signal trends.

    Args:
        x (np.ndarray): 1D array of signal values.

    Returns:
        float: Estimated noise standard deviation, or 0.0 if fewer than
        8 finite values exist.
    """
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
    """Injects white noise, low-frequency baseline drift, and sparse spikes into signals.

    All noise components are scaled relative to each signal's robust noise floor
    (`_robust_sigma`) to prevent noise from corrupting event shapes.

    Args:
        df (pd.DataFrame): Input DataFrame with target signal columns.
        rng (np.random.Generator): NumPy random number generator instance.
        white_frac (Tuple[float, float], optional): Min/max fraction of noise
            sigma for additive Gaussian noise. Defaults to (0.2, 1.0).
        drift_frac (Tuple[float, float], optional): Min/max fraction of noise
            sigma for sinusoidal drift amplitude. Defaults to (0.0, 3.0).
        spike_prob (float, optional): Probability of introducing sparse spike
            artifacts. Defaults to 0.15.
        cols (Tuple[str, ...], optional): Target column names to augment.
            Defaults to (COL_DISS, COL_FREQ).

    Returns:
        pd.DataFrame: A copy of the input DataFrame with noise injected into
        the specified signal columns.
    """
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
    """Applies per-signal multiplicative gain jitter around baseline levels.

    Jitters event amplitude about each signal's early-run median baseline,
    breaking memorization of absolute amplitude levels without causing
    baseline drift.

    Args:
        df (pd.DataFrame): Input DataFrame with target signal columns.
        rng (np.random.Generator): NumPy random number generator instance.
        gain_sigma (float, optional): Standard deviation of log-normal gain
            multiplier. Defaults to 0.08.
        cols (Tuple[str, ...], optional): Target column names to augment.
            Defaults to (COL_DISS, COL_FREQ).

    Returns:
        pd.DataFrame: A copy of the input DataFrame with jittered signal amplitudes.
    """
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
    """Composes time warping, amplitude jitter, and noise injection augmentations.

    Args:
        df (pd.DataFrame): Input DataFrame of raw run signals.
        poi_times (Dict[str, float]): Point-of-interest labels mapping names
            to timestamps (in seconds).
        rng (np.random.Generator): NumPy random number generator instance.
        p_warp (float, optional): Probability of applying time warping.
            Defaults to 0.9.
        p_noise (float, optional): Probability of applying noise injection.
            Defaults to 0.6.
        p_amp (float, optional): Probability of applying amplitude jitter.
            Defaults to 0.6.

    Returns:
        Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]: A tuple containing:
            - augmented_df: Transformed signal DataFrame.
            - augmented_poi_times: Dictionary of updated POI timestamps.
            - info: Metadata dictionary containing augmentation details (e.g., "stretch").
    """
    info: Dict[str, float] = {"stretch": 1.0}
    if rng.random() < p_warp:
        df, poi_times, S = time_warp(df, poi_times, rng)
        info["stretch"] = S
    if rng.random() < p_amp:
        df = amplitude_jitter(df, rng)
    if rng.random() < p_noise:
        df = inject_noise(df, rng)
    return df, poi_times, info


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
    """Calculates the temporal duration of a fill transition around a POI.

    Measures the duration where the smoothed derivative |d(dissipation)/dt|
    exceeds `active_frac` of its local peak within a localized window around
    `poi_t`. High-viscosity (slow) events automatically yield wider boxes,
    preventing bounding-box gradient starvation on temporally stretched events.

    Args:
        df_p (pd.DataFrame): DataFrame containing signal and time series data.
        poi_t (float): Point-of-interest timestamp in seconds.
        diss_col (str, optional): Dissipation column name. Defaults to COL_DISS.
        time_col (str, optional): Relative time column name. Defaults to COL_TIME.
        rel_window (float, optional): Window size relative to total run duration.
            Defaults to 0.04.
        min_window_s (float, optional): Minimum half-window size in seconds.
            Defaults to 0.75.
        active_frac (float, optional): Fraction of peak derivative defining active
            transition extent. Defaults to 0.2.
        min_width_s (float, optional): Lower bound fallback width in seconds.
            Defaults to 0.05.
        max_width_frac (float, optional): Maximum allowed width as a fraction
            of total run duration. Defaults to 0.06.

    Returns:
        float: Measured transition duration in seconds, clamped between `min_width_s`
        and `max_width_frac * total_span`.
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

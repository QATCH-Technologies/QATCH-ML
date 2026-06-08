"""
QModel-MOIRAI pilot — Data layer
================================

Adapts the QModel v6 run-discovery + preprocessing contract (from
``dataset_builder.py`` / ``signal_processing.py``) to produce, for each run:

    sequence : float32 (SEQ_LEN, N_INPUT_CHANNELS)   # Diss, Freq, Difference
    targets  : float32 (N_POI, SEQ_LEN)              # soft Gaussian heatmaps
    poi_pos  : float32 (N_POI,)                       # normalised [0,1] truth
    poi_mask : bool    (N_POI,)                       # which POIs are present
    viscosity_cP, run_id, tier

What is preserved verbatim from v6
----------------------------------
  * Run discovery: ``*_poi.csv`` (headerless sample indices) + a data CSV
    containing ``Relative_time``, ``Resonance_Frequency``, ``Dissipation``.
  * Viscosity read from ``analyze-N.zip/analyze_out.csv`` (mean viscosity_raw).
  * POI_ROW_MAP row→name mapping (row 2 skipped).
  * Uniform-dt resampling, the Difference curve, median smoothing.
  * Stratified-by-tier split on PHYSICAL runs (no variant/leakage path here,
    but the split key is the same so results are comparable to YOLO).

What is NEW (the MOIRAI reframing)
----------------------------------
  * After uniform-dt preprocessing, each channel is resampled onto a FIXED
    normalised time grid of length SEQ_LEN. Run duration is factored out, so
    a 50 s run and a 1200 s viscous run both become length-SEQ_LEN sequences.
    POIs are expressed as fractions in [0,1] of that grid.
  * Per-channel standardisation (z-score) using the run's own stats. MOIRAI
    expects roughly standardised inputs; this also removes the absolute-scale
    differences between low- and high-viscosity runs.
  * POI targets are dense soft Gaussian bumps over the SEQ_LEN axis rather
    than bounding boxes.
"""

from __future__ import annotations

import io
import logging
import math
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import medfilt

import config as C

LOG = logging.getLogger("moirai.data")


# ===========================================================================
#  RunSpec  (mirrors dataset_builder.RunSpec)
# ===========================================================================


@dataclass
class RunSpec:
    csv_path: Path
    viscosity_cP: float
    poi_sample_idx: Dict[str, int]  # POI name -> raw sample index
    run_id: str = ""


# ===========================================================================
#  Viscosity tier helpers (identical edges to v6)
# ===========================================================================


def viscosity_tier(cP: float) -> int:
    if cP is None or (isinstance(cP, float) and math.isnan(cP)):
        return 0
    for i in range(len(C.TIER_EDGES) - 1):
        if C.TIER_EDGES[i] <= cP < C.TIER_EDGES[i + 1]:
            return i
    return len(C.TIER_EDGES) - 2


# ===========================================================================
#  Viscosity read from analyze-N.zip  (verbatim logic from v6)
# ===========================================================================

_ANALYZE_ZIP_RE = re.compile(r"^analyze-(\d+)\.zip$", re.IGNORECASE)


def _find_analyze_zip(run_dir: Path) -> Optional[Path]:
    best_idx, best_path = -1, None
    try:
        for cand in run_dir.iterdir():
            m = _ANALYZE_ZIP_RE.match(cand.name)
            if m and int(m.group(1)) > best_idx:
                best_idx, best_path = int(m.group(1)), cand
    except (FileNotFoundError, NotADirectoryError):
        return None
    return best_path


def read_run_viscosity(run_dir: Path) -> Optional[float]:
    zip_path = _find_analyze_zip(run_dir)
    if zip_path is None:
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            inner = next(
                (n for n in zf.namelist() if n.lower().endswith("analyze_out.csv")),
                None,
            )
            if inner is None:
                return None
            with zf.open(inner) as fh:
                df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
        if "viscosity_raw" not in df.columns:
            return None
        vals = pd.to_numeric(df["viscosity_raw"], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else None
    except Exception:
        return None


# ===========================================================================
#  Run discovery  (adapted from dataset_builder.discover_runs)
# ===========================================================================


def discover_runs(runs_root: Path, n_workers: int = 8) -> List[RunSpec]:
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise NotADirectoryError(f"runs_root not a directory: {runs_root}")

    candidates = sorted([d for d in runs_root.iterdir() if d.is_dir()])
    LOG.info("Scanning %d candidate dirs under %s", len(candidates), runs_root)
    if not candidates:
        return []

    visc_map: Dict[str, Optional[float]] = {}
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
        futs = {ex.submit(read_run_viscosity, d): d for d in candidates}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                visc_map[d.name] = fut.result()
            except Exception:
                visc_map[d.name] = None

    runs: List[RunSpec] = []
    skip = {"no_files": 0, "no_pois": 0, "no_time": 0, "short": 0}

    for d in candidates:
        try:
            poi_file = next(d.glob("*_poi.csv"))
        except StopIteration:
            skip["no_files"] += 1
            continue
        try:
            data_file = next(p for p in d.glob("*.csv") if p != poi_file)
        except StopIteration:
            skip["no_files"] += 1
            continue

        try:
            poi_df = pd.read_csv(poi_file, header=None, names=["sample_index"])
        except Exception:
            skip["no_pois"] += 1
            continue

        # Need a time column to validate length/duration.
        try:
            raw_time = pd.read_csv(data_file, usecols=[C.COL_TIME])
        except (KeyError, ValueError):
            try:
                full = pd.read_csv(data_file)
                if C.COL_TIME not in full.columns:
                    skip["no_time"] += 1
                    continue
                raw_time = full[[C.COL_TIME]]
            except Exception:
                skip["no_time"] += 1
                continue
        except Exception:
            skip["no_time"] += 1
            continue

        n_rows = len(raw_time)
        if n_rows < C.MIN_ROWS:
            skip["short"] += 1
            continue
        try:
            dur = float(raw_time[C.COL_TIME].iloc[-1] - raw_time[C.COL_TIME].iloc[0])
        except Exception:
            dur = 0.0
        if dur < C.MIN_DURATION_SEC:
            skip["short"] += 1
            continue

        # Map POI rows -> raw sample indices (skip row 2, as in v6).
        poi_sample_idx: Dict[str, int] = {}
        for row_idx, poi_name in C.POI_ROW_MAP.items():
            if row_idx >= len(poi_df):
                continue
            si = poi_df.iloc[row_idx]["sample_index"]
            if pd.isna(si):
                continue
            try:
                si_int = int(si)
            except (TypeError, ValueError):
                continue
            if 0 <= si_int < n_rows:
                poi_sample_idx[poi_name] = si_int

        if not poi_sample_idx:
            skip["no_pois"] += 1
            continue

        v = visc_map.get(d.name)
        runs.append(
            RunSpec(
                csv_path=data_file,
                viscosity_cP=float(v) if (v is not None and v > 0) else float("nan"),
                poi_sample_idx=poi_sample_idx,
                run_id=d.name,
            )
        )

    LOG.info("Discovered %d valid runs (skipped: %s)", len(runs), skip)
    return runs


# ===========================================================================
#  Preprocessing  (uniform dt + Difference + median smooth — from v6)
# ===========================================================================


def _compute_difference_curve(df: pd.DataFrame) -> Optional[pd.Series]:
    if not all(c in df.columns for c in (C.COL_FREQ, C.COL_DISS, C.COL_TIME)):
        return None
    xs = df[C.COL_TIME].to_numpy(dtype=float)
    if len(xs) == 0:
        return None
    i = int(np.searchsorted(xs, C.BASELINE_START_SEC))
    j = int(np.searchsorted(xs, C.BASELINE_END_SEC))
    if i == j and j < len(xs):
        j = int(np.searchsorted(xs, xs[j] + C.BASELINE_OFFSET_SEC))
    if i >= len(df) or j > len(df) or i == j:
        i, j = 0, min(100, len(df))
    avg_f = float(df[C.COL_FREQ].iloc[i:j].mean())
    avg_d = float(df[C.COL_DISS].iloc[i:j].mean())
    ys_diss = (df[C.COL_DISS].to_numpy(dtype=float) - avg_d) * avg_f / 2.0
    ys_freq = avg_f - df[C.COL_FREQ].to_numpy(dtype=float)
    return pd.Series(ys_freq - C.DIFF_FACTOR * ys_diss, index=df.index)


def preprocess_dataframe(df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Uniform-dt resample + Difference + median smooth. Mirrors v6 exactly."""
    if df_raw is None or df_raw.empty:
        return None
    df = df_raw.copy()
    df.drop(columns=[c for c in C.DROP_COLS if c in df.columns], inplace=True)
    if C.COL_TIME not in df.columns:
        return None
    df.drop_duplicates(subset=[C.COL_TIME], keep="first", inplace=True)
    if len(df) < C.MIN_ROWS:
        return None

    t_min = float(df[C.COL_TIME].min())
    t_max = float(df[C.COL_TIME].max())
    if (t_max - t_min) < 1e-6:
        return None

    new_grid = np.arange(t_min, t_max, C.TARGET_DT_SEC)
    if len(new_grid) < C.MIN_ROWS:
        return None

    df = df.set_index(C.COL_TIME)
    combined = df.index.union(new_grid).sort_values()
    df = df.reindex(combined).interpolate(method="index").loc[new_grid]
    df = df.reset_index().rename(columns={"index": C.COL_TIME})

    diff = _compute_difference_curve(df)
    df[C.COL_DIFF] = diff if diff is not None else 0.0

    if C.MEDIAN_KERNEL and C.MEDIAN_KERNEL >= 3:
        for col in df.columns:
            if col == C.COL_TIME or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            df[col] = medfilt(df[col].to_numpy(dtype=float), kernel_size=C.MEDIAN_KERNEL)
    return df


# ===========================================================================
#  Build the model tensors for one run
# ===========================================================================


def _gaussian_bump(centre_frac: float, length: int, sigma_frac: float) -> np.ndarray:
    """Soft target: a Gaussian peaking (=1) at centre_frac along [0,1]."""
    x = np.linspace(0.0, 1.0, length, dtype=np.float64)
    g = np.exp(-0.5 * ((x - centre_frac) / sigma_frac) ** 2)
    return g.astype(np.float32)


@dataclass
class RunSample:
    sequence: np.ndarray  # (SEQ_LEN, N_INPUT_CHANNELS) float32, z-scored
    targets: np.ndarray  # (N_POI, SEQ_LEN) float32 soft heatmaps
    poi_pos: np.ndarray  # (N_POI,) float32 normalised [0,1] (-1 if absent)
    poi_mask: np.ndarray  # (N_POI,) bool
    viscosity_cP: float
    tier: int
    run_id: str


def build_run_sample(spec: RunSpec) -> Optional[RunSample]:
    """Full path: read CSV -> preprocess -> normalised SEQ_LEN tensors + targets."""
    try:
        raw = pd.read_csv(spec.csv_path)
    except Exception as exc:
        LOG.warning("read fail %s: %s", spec.run_id, exc)
        return None

    # Need the raw time array to translate POI *sample indices* (which index
    # the ORIGINAL csv rows) into physical times, before resampling.
    if C.COL_TIME not in raw.columns:
        return None
    raw_t = pd.to_numeric(raw[C.COL_TIME], errors="coerce").to_numpy(dtype=float)
    n_raw = len(raw_t)

    poi_time: Dict[str, float] = {}
    for name, si in spec.poi_sample_idx.items():
        if 0 <= si < n_raw and np.isfinite(raw_t[si]):
            poi_time[name] = float(raw_t[si])

    df = preprocess_dataframe(raw)
    if df is None:
        return None

    t = df[C.COL_TIME].to_numpy(dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    span = t1 - t0
    if span < 1e-9:
        return None

    # Fixed normalised time grid -> duration-invariant sequence.
    norm_grid = np.linspace(0.0, 1.0, C.SEQ_LEN)
    phys_grid = t0 + norm_grid * span

    seq = np.empty((C.SEQ_LEN, C.N_INPUT_CHANNELS), dtype=np.float32)
    for ci, col in enumerate(C.INPUT_CHANNELS):
        if col not in df.columns:
            return None
        vals = df[col].to_numpy(dtype=float)
        resampled = np.interp(phys_grid, t, vals)
        mu = float(np.nanmean(resampled))
        sd = float(np.nanstd(resampled))
        if sd < 1e-9:
            sd = 1.0
        seq[:, ci] = ((resampled - mu) / sd).astype(np.float32)

    # POI targets in normalised coords.
    targets = np.zeros((C.N_POI, C.SEQ_LEN), dtype=np.float32)
    poi_pos = np.full((C.N_POI,), -1.0, dtype=np.float32)
    poi_mask = np.zeros((C.N_POI,), dtype=bool)
    for pi, name in enumerate(C.POI_NAMES):
        if name not in poi_time:
            continue
        frac = (poi_time[name] - t0) / span
        if not (0.0 <= frac <= 1.0):
            continue
        poi_pos[pi] = np.float32(frac)
        poi_mask[pi] = True
        targets[pi] = _gaussian_bump(frac, C.SEQ_LEN, C.LABEL_SIGMA_FRAC)

    if not poi_mask.any():
        return None

    return RunSample(
        sequence=seq,
        targets=targets,
        poi_pos=poi_pos,
        poi_mask=poi_mask,
        viscosity_cP=spec.viscosity_cP,
        tier=viscosity_tier(spec.viscosity_cP),
        run_id=spec.run_id,
    )


# ===========================================================================
#  Stratified split on PHYSICAL runs (same key as v6)
# ===========================================================================


def stratified_split(
    specs: List[RunSpec],
    val_ratio: float = C.VAL_SPLIT,
    rng_seed: int = C.RNG_SEED,
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(rng_seed)
    by_tier: Dict[int, List[int]] = {}
    for i, r in enumerate(specs):
        by_tier.setdefault(viscosity_tier(r.viscosity_cP), []).append(i)
    train_idx, val_idx = [], []
    for _, idxs in by_tier.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio))) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    LOG.info("Split: train=%d val=%d (val_ratio=%.2f)", len(train_idx), len(val_idx), val_ratio)
    return train_idx, val_idx

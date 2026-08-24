"""Shared corpus-discovery library for evaluating and training models.

Walks `data/raw`-style run directories, parses ground-truth point-of-interest
(POI) times, and reads per-run viscosity. Every other stage (dataset building,
tier discovery, benchmarking, audit/triage tools) relies on this module as a
starting point.

Consolidates logic for determining valid POI marks via a strict acceptance rule
(strictly-ascending, non-tail POI). This is the single source of truth used both
to fit the spacing prior and to build the evaluation corpus.

Attributes:
    POI_ROW (Dict[str, int]): Mapping of chain-space POI names to their
        corresponding zero-indexed row in the POI CSV.
    TIER_EDGES (List[float]): Fixed viscosity-tier boundary edges (in cP)
        used for benchmark and report tables.
    TIER_LABELS (List[str]): Human-readable labels corresponding to the
        defined viscosity tiers.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

from .decode.spacing_prior import POI_ORDER

LOG = get_logger("qmodel_7_onyx.corpus")

# poi-csv row -> chain-space POI name (row 2 is the legacy shim, skipped).
POI_ROW = {"POI1": 0, "POI2": 1, "POI3": 3, "POI4": 4, "POI5": 5}

# Fixed viscosity-tier edges used for benchmark/report tables (distinct from
# the data-driven TierScheme in tiers.py, which is used for stratified
# dataset splitting/upsampling).
TIER_EDGES = [2.66, 6.16, 18.14, 73.4]
TIER_LABELS = ["<2.66 cP", "2.66-6.16 cP", "6.16-18.14 cP", "18.14-73.4 cP", "73.4+ cP", "unknown"]


def viscosity_tier(cp: Optional[float]) -> int:
    """Determines the viscosity tier index for a given viscosity value.

    Args:
        cp (Optional[float]): Viscosity in centipoise (cP), or None if unknown.

    Returns:
        int: The zero-based index of the tier corresponding to `TIER_LABELS`.
        Returns the "unknown" index if the input is None or not finite.
    """
    if cp is None or not np.isfinite(cp):
        return len(TIER_EDGES) + 1  # "unknown"
    for i, edge in enumerate(TIER_EDGES):
        if cp < edge:
            return i
    return len(TIER_EDGES)


@dataclass
class RunRecord:
    """A parsed representation of a single physical run directory.

    Attributes:
        run_id (str): Unique identifier for the run (typically the directory name).
        csv_path (Path): Path to the primary run data CSV file.
        poi_times (Dict[str, float]): Dictionary mapping POI names to their
            ground-truth chain-space time in seconds. Acts as a prefix on
            partial fills.
        viscosity_cP (Optional[float]): The mean viscosity of the run in
            centipoise, if available.
    """

    run_id: str
    csv_path: Path
    poi_times: Dict[str, float]  # chain-space truth (prefix on partial fills)
    viscosity_cP: Optional[float]


def truth_times(poi_path: Path, time_axis: np.ndarray) -> Dict[str, float]:
    """Parses and validates ground-truth POI times from a run's POI CSV.

    Extracts chain-space ground-truth times with a strict acceptance rule:
    A POI is accepted only if its row index is present, non-tail (at least
    `tail_tol` samples before the end of the run), in range, and strictly
    later than the previous accepted POI's time.

    This enforces a strictly-ascending, non-tail prefix constraint for both
    training (where only complete fills are used) and evaluation (where
    prefixes are accepted as partial fills).

    Args:
        poi_path (Path): Path to the `*_poi.csv` file containing raw POI indices.
        time_axis (np.ndarray): The 1D numeric array of time values for the run,
            used to map row indices to physical timestamps.

    Returns:
        Dict[str, float]: A dictionary mapping POI names (e.g., "POI1") to
        their validated timestamps.
    """
    try:
        raw_idx = pd.to_numeric(
            pd.read_csv(poi_path, header=None).iloc[:, 0], errors="coerce"
        ).to_numpy()
    except Exception:
        return {}
    n_rows = len(time_axis)
    last_idx = n_rows - 1
    tail_tol = max(2, int(0.001 * n_rows))
    out: Dict[str, float] = {}
    prev_t = -np.inf
    for name in POI_ORDER:
        row = POI_ROW[name]
        if row >= len(raw_idx) or np.isnan(raw_idx[row]):
            break
        idx = int(raw_idx[row])
        if idx >= last_idx - tail_tol or idx < 0 or idx > last_idx:
            break
        t = float(time_axis[idx])
        if t <= prev_t:
            break
        out[name] = t
        prev_t = t
    return out


def _viscosity_from_frame(df: pd.DataFrame) -> Optional[float]:
    """Extracts the mean viscosity from an analyze-output DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame parsed from an `analyze_out` CSV.

    Returns:
        Optional[float]: The mean viscosity in cP if a valid `viscosity_avg`
        column exists and contains finite numeric data; otherwise, None.
    """
    df.columns = [str(c).lstrip("# ").strip() for c in df.columns]
    if "viscosity_avg" in df.columns:
        v = pd.to_numeric(df["viscosity_avg"], errors="coerce").dropna()
        if len(v):
            return float(v.mean())
    return None


def run_viscosity(run_dir: Path) -> Optional[float]:
    """Locates and extracts the mean viscosity for a single run directory.

    Searches for `*analyze_out*.csv` files directly within the directory,
    falling back to probing inside `analyze-*.zip` archives if the loose
    CSV is missing.

    Args:
        run_dir (Path): The root directory path of the specific run.

    Returns:
        Optional[float]: The parsed mean viscosity in cP, or None if no
        valid analyze output was found.
    """
    for p in run_dir.glob("*analyze_out*.csv"):
        try:
            cp = _viscosity_from_frame(pd.read_csv(p))
            if cp is not None:
                return cp
        except Exception:
            continue

    for z in sorted(run_dir.glob("analyze-*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                for name in zf.namelist():
                    if "analyze_out" in name.lower() and name.lower().endswith(".csv"):
                        with zf.open(name) as fh:
                            cp = _viscosity_from_frame(pd.read_csv(io.BytesIO(fh.read())))
                        if cp is not None:
                            return cp
        except Exception:
            continue
    return None


def discover_runs(raw_root: Path, time_col: str = "Relative_time") -> List[RunRecord]:
    """Scans a root directory to discover and parse valid run records.

    Iterates over subdirectories looking for pairs of data CSVs and POI
    CSVs. Extracts the time axis, filters out malformed or empty data,
    extracts validated truth times, and attempts to resolve viscosity.

    Args:
        raw_root (Path): Path to the root directory containing run subdirectories
            (e.g., `data/raw`).
        time_col (str, optional): The name of the time column to use. Defaults to
            "Relative_time". Falls back to the first column if missing.

    Returns:
        List[RunRecord]: A list of successfully validated and parsed `RunRecord`
        instances representing the discoverable corpus.
    """
    runs: List[RunRecord] = []
    for d in sorted(Path(raw_root).iterdir()):
        if not d.is_dir():
            continue
        cands = [
            p
            for p in d.glob("*.csv")
            if not p.name.lower().endswith("_poi.csv") and "analyze_out" not in p.name.lower()
        ]
        poi_files = list(d.glob("*_poi.csv"))
        if not cands or not poi_files:
            continue
        try:
            data = pd.read_csv(cands[0])
        except Exception:
            continue
        tcol = time_col if time_col in data.columns else data.columns[0]
        ta = pd.to_numeric(data[tcol], errors="coerce").to_numpy()
        if len(ta) < 2 or np.isnan(ta).all():
            continue
        truth = truth_times(poi_files[0], ta)
        if not truth:
            continue
        runs.append(
            RunRecord(
                run_id=d.name,
                csv_path=cands[0],
                poi_times=truth,
                viscosity_cP=run_viscosity(d),
            )
        )
    return runs


def _run_fingerprint(rec: RunRecord) -> str:
    """Generates a unique content fingerprint for duplicate-run detection.

    Calculates a Blake2s hash based on the exact sequence of POI truth times.
    Two directories containing the same physical run (same POI times) are
    considered the same run regardless of their directory names. This prevents
    double-counting benchmark failures and avoids train/val data leakage.

    Args:
        rec (RunRecord): The run record to fingerprint.

    Returns:
        str: An 8-byte hexadecimal string representing the run's content fingerprint.
    """
    key = "|".join(f"{k}:{v:.4f}" for k, v in sorted(rec.poi_times.items()))
    return hashlib.blake2s(key.encode(), digest_size=8).hexdigest()


def dedupe_runs(runs: List[RunRecord]) -> List[RunRecord]:
    """Removes duplicate runs from a corpus using content fingerprints.

    Args:
        runs (List[RunRecord]): The initial list of parsed run records.

    Returns:
        List[RunRecord]: A deduplicated list of run records. Warnings are logged
        for any duplicates dropped during this process.
    """
    seen: Dict[str, str] = {}
    out: List[RunRecord] = []
    n_dup = 0
    for r in runs:
        fp = _run_fingerprint(r)
        if fp in seen:
            n_dup += 1
            LOG.warning("Duplicate run content: {} == {} (skipping)", r.run_id, seen[fp])
            continue
        seen[fp] = r.run_id
        out.append(r)
    if n_dup:
        LOG.warning("Deduplicated corpus: {} duplicate run(s) removed, {} remain", n_dup, len(out))
    return out


def load_run_filter(path: Path) -> set:
    """Loads a set of allowed or isolated run IDs for stratified evaluation.

    Accepts either a JSON manifest from `build_dataset.py` (extracting its
    `val_ids` key to ensure evaluation strictly on unseen runs) or a plain
    text/JSON list of run IDs. This prevents benchmarking on runs the model
    may have already memorized during training.

    Args:
        path (Path): Path to the JSON manifest or text list of run IDs.

    Returns:
        set: A set of unique run ID strings parsed from the file.

    Raises:
        SystemExit: If the file format is unrecognized or lacks the expected schema.
    """
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {ln.strip() for ln in text.splitlines() if ln.strip()}
    if isinstance(data, dict) and "val_ids" in data:
        return set(data["val_ids"])
    if isinstance(data, list):
        return set(data)
    raise SystemExit(f"unrecognized run-filter format: {path}")

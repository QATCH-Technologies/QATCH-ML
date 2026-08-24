"""
corpus.py
=========

Shared corpus-discovery library: walks ``data/raw``-style run directories,
parses ground-truth POI times, and reads per-run viscosity. Every other
stage (dataset building, tier discovery, benchmarking, audit/triage tools)
starts from :func:`discover_runs`.

This module consolidates logic that used to be duplicated across
``benchmark_decode.py`` (``_truth_times``) and ``fit_prior.py``
(``parse_present``) - both implemented the same "strictly-ascending,
non-tail POI" acceptance rule independently, which is exactly the kind of
duplication that drifts out of sync silently. :func:`truth_times` is now the
single source of truth for what counts as a valid POI mark, used both to fit
the spacing prior (:mod:`.decode.fit_prior`) and to build the evaluation
corpus (:func:`discover_runs`).
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
    if cp is None or not np.isfinite(cp):
        return len(TIER_EDGES) + 1  # "unknown"
    for i, edge in enumerate(TIER_EDGES):
        if cp < edge:
            return i
    return len(TIER_EDGES)


@dataclass
class RunRecord:
    run_id: str
    csv_path: Path
    poi_times: Dict[str, float]  # chain-space truth (prefix on partial fills)
    viscosity_cP: Optional[float]


def truth_times(poi_path: Path, time_axis: np.ndarray) -> Dict[str, float]:
    """Chain-space ground-truth POI times: strictly-ascending, non-tail
    prefix. A POI is accepted only if its row index is present, non-tail (at
    least ``tail_tol`` samples before the end of the run), in range, and
    strictly later than the previous accepted POI's time. This is the single
    acceptance rule used both to fit the spacing prior (only complete-fill
    configurations, i.e. ``len(truth_times(...)) == len(POI_ORDER)``, are
    used) and to build the evaluation corpus (prefixes are accepted as
    partial fills)."""
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
    df.columns = [str(c).lstrip("# ").strip() for c in df.columns]
    if "viscosity_avg" in df.columns:
        v = pd.to_numeric(df["viscosity_avg"], errors="coerce").dropna()
        if len(v):
            return float(v.mean())
    return None


def run_viscosity(run_dir: Path) -> Optional[float]:
    """Mean viscosity_avg from a run's analyze output. Looks at loose
    ``*analyze_out*.csv`` files first, then inside ``analyze-N.zip``
    archives."""
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
    """Content fingerprint for duplicate-run detection. Two directories
    containing the same physical run (same POI truth times) are the same run
    regardless of directory name. Duplicates double-count benchmark failures
    and - far worse - defeat group-by-run_id train/val splitting: the same
    run on both sides of the split is leakage."""
    key = "|".join(f"{k}:{v:.4f}" for k, v in sorted(rec.poi_times.items()))
    return hashlib.blake2s(key.encode(), digest_size=8).hexdigest()


def dedupe_runs(runs: List[RunRecord]) -> List[RunRecord]:
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
    """Accepts either a build_dataset manifest.json (uses its val_ids - the
    runs the CURRENT model never trained on) or a plain text/JSON list of run
    ids. Benchmarking on the full corpus rewards whichever system memorized
    the corpus harder; a system trained (even partially) on the evaluation
    runs reports memorization, not generalization."""
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

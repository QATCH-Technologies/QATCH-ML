"""Fit and persist a spacing prior from complete-fill run configurations.

Collects accepted POI timestamps from raw runs using the same truth-selection
logic used by the evaluation corpus, retaining only configurations in which
all expected POIs are present. The resulting complete-fill configurations are
used to fit a :class:`SpacingPrior`, which is then written to the configured
output path for use during decoding.

Usage:
    python -m src.systems.qmodel_7_onyx.decode.fit_prior \
        --raw-root path/to/data/raw --out configs/spacing_prior.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

from .. import paths
from ..corpus import truth_times
from .spacing_prior import POI_ORDER, SpacingPrior

LOG = get_logger("qmodel_7_onyx.decode.fit_prior")


def _find_run_csv(run_dir: Path) -> Optional[Path]:
    """Locate the primary signal CSV within a run directory.

    Searches for CSV files while excluding POI annotation files and returns
    the first matching signal file.

    Args:
        run_dir (pathlib.Path): Directory containing the files associated with
            a single run.

    Returns:
        pathlib.Path | None: Path to the run's signal CSV, or `None` when no
        suitable CSV is found.
    """
    cands = [p for p in run_dir.glob("*.csv") if not p.name.lower().endswith("_poi.csv")]
    return cands[0] if cands else None


def collect_complete_configs(
    raw_root: Path, time_col: str, limit: Optional[int] = None
) -> np.ndarray:
    """Collect POI timing configurations from complete runs.

    Iterates through run directories, loads the signal timestamps, derives
    accepted POI times using the shared truth-selection rule, and retains only
    runs containing the full expected POI sequence. The resulting configurations
    provide the training data for the spacing prior.

    Args:
        raw_root (pathlib.Path): Root directory containing per-run data.
        time_col (str): Preferred signal column containing timestamps. The
            first data column is used when this column is unavailable.
        limit (int | None, optional): Maximum number of runs to inspect.
            Defaults to `None`.

    Returns:
        numpy.ndarray: Two-dimensional array containing one complete POI timing
        configuration per row, ordered according to :data:`POI_ORDER`.

    Raises:
        SystemExit: If no complete-fill configurations are found.
    """
    rows = []
    n = 0
    for d in sorted(Path(raw_root).iterdir()):
        if not d.is_dir():
            continue
        if limit and n >= limit:
            break
        run_csv = _find_run_csv(d)
        poi_files = list(d.glob("*_poi.csv"))
        if run_csv is None or not poi_files:
            continue
        try:
            data = pd.read_csv(run_csv)
        except Exception:
            continue
        tcol = time_col if time_col in data.columns else data.columns[0]
        ta = pd.to_numeric(data[tcol], errors="coerce").to_numpy()
        if len(ta) < 2 or np.isnan(ta).all():
            continue
        times = list(truth_times(poi_files[0], ta).values())
        n += 1
        if len(times) == len(POI_ORDER):  # complete fills only
            rows.append(times)
    if not rows:
        raise SystemExit("No complete-fill configurations found.")
    return np.array(rows, dtype=float)


def main() -> None:
    """Fit and save the spacing prior from the configured raw dataset.

    Parses command-line options, collects complete-fill POI configurations,
    fits a :class:`SpacingPrior`, reports fitted gap statistics, and saves the
    resulting prior to the requested output path.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=paths.DATA_ROOT, type=Path)
    ap.add_argument("--time-col", default="Relative_time")
    ap.add_argument("--out", type=Path, default=paths.SPACING_PRIOR_JSON)
    ap.add_argument("--frac-blend", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    configs = collect_complete_configs(args.raw_root, args.time_col, args.limit)
    LOG.info("Fitting spacing prior on {} complete-fill configs.", len(configs))
    prior = SpacingPrior.fit(configs, frac_blend=args.frac_blend)
    for pair, gs in prior.gap.items():
        print(
            f"  {pair}: median gap {np.exp(gs.log_mu_sec):7.2f}s "
            f"(log-sd {gs.log_sd_sec:.2f}) bounds [{gs.min_gap_sec:.2f}, "
            f"{gs.max_gap_sec:.2f}]s  n={gs.n}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prior.save(args.out)
    LOG.info("Wrote {}", args.out)


if __name__ == "__main__":
    main()

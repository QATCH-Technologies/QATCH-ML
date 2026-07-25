"""
fit_prior.py
============

Fit the SpacingPrior from data/raw complete-fill configurations, using the
same POI acceptance rule the evaluation corpus uses (:func:`corpus.truth_times`)
so partial fills are correctly excluded from the prior (we only learn spacing
from runs where all POIs are genuinely present).

Usage
-----
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
    cands = [p for p in run_dir.glob("*.csv") if not p.name.lower().endswith("_poi.csv")]
    return cands[0] if cands else None


def collect_complete_configs(
    raw_root: Path, time_col: str, limit: Optional[int] = None
) -> np.ndarray:
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


def main():
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

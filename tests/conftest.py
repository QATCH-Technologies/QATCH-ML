"""Shared fixtures for the qmodel_7_onyx test suite.

``make_run`` builds a synthetic run directory in the same layout
``corpus.discover_runs`` expects (a signal CSV + a headerless ``*_poi.csv``
of raw row indices, with row 2 being the legacy POI2-shim row that
``corpus.truth_times`` skips), without needing any real QATCH data on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

# Every qmodel_7_onyx path default is resolved once at import time via
# paths.py's REPO_ROOT-based constants; no env-var indirection is needed for
# tests since they always pass explicit paths to the functions under test.

POI_ROW = {"POI1": 0, "POI2": 1, "POI3": 3, "POI4": 4, "POI5": 5}


def _write_run(
    run_dir: Path,
    run_id: str,
    poi_times: Dict[str, float],
    dt: float = 0.02,
    end_pad: float = 2.0,
    viscosity_cP: Optional[float] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    last_t = max(poi_times.values()) + end_pad if poi_times else 10.0
    t = np.arange(0.0, last_t, dt)
    rng = np.random.default_rng(abs(hash(run_id)) % (2**32))
    diss = np.cumsum(rng.normal(0, 1e-7, len(t))) + 3e-5
    freq = 1.5e7 - np.linspace(0, 500, len(t))
    pd.DataFrame({"Relative_time": t, "Dissipation": diss, "Resonance_Frequency": freq}).to_csv(
        run_dir / f"{run_id}.csv", index=False
    )

    # Map each present POI time to the nearest row index, then lay indices
    # out at POI_ROW positions (row 2 duplicates POI2 as the legacy shim).
    idx_by_name = {name: int(np.searchsorted(t, tt)) for name, tt in poi_times.items()}
    max_row = max(POI_ROW.values())
    rows: List[object] = [""] * (max_row + 1)
    for name, idx in idx_by_name.items():
        rows[POI_ROW[name]] = idx
    if "POI2" in idx_by_name:
        rows[2] = idx_by_name["POI2"]  # legacy shim row, skipped by truth_times
    pd.Series(rows).to_csv(run_dir / f"{run_id}_poi.csv", index=False, header=False)

    if viscosity_cP is not None:
        pd.DataFrame({"shear_rate": [100.0], "viscosity_avg": [viscosity_cP]}).to_csv(
            run_dir / f"{run_id}_analyze_out.csv", index=False
        )


@pytest.fixture
def make_run(tmp_path):
    """Factory: make_run(raw_root, run_id, poi_times, viscosity_cP=None) creates
    one synthetic run directory under raw_root and returns its Path."""

    def _make(raw_root: Path, run_id: str, poi_times: Dict[str, float], **kwargs) -> Path:
        run_dir = Path(raw_root) / run_id
        _write_run(run_dir, run_id, poi_times, **kwargs)
        return run_dir

    return _make


@pytest.fixture
def complete_poi_times() -> Dict[str, float]:
    """A well-separated, strictly-ascending complete-fill configuration."""
    return {"POI1": 5.0, "POI2": 6.5, "POI3": 25.0, "POI4": 60.0, "POI5": 120.0}

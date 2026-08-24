"""
tiers.py
========

Data-driven viscosity tier binning, replacing the arbitrary 5-bin scheme.

Tiers are fitted on log10(viscosity_cP) - fill dynamics scale roughly
multiplicatively with viscosity, so log-space is where cluster structure
lives. Three fitters:

  * `log_uniform` (default): bins of equal width in log10(cP), spanning
    the observed min..max. This is the only method that guarantees the
    top edge tracks the actual tail of the corpus - the corpus here has
    a 95th percentile around 46 cP but individual runs up to ~1800 cP, and
    both of the count-balanced methods below place their top edge around
    11-17 cP because that's where the *count* quantile falls, silently
    lumping every run from there to 1800 cP into one "N+" bucket.
  * `gmm`: Gaussian mixture with BIC model selection over k (uses
    scikit-learn when available). Bin edges are placed at the posterior
    decision boundaries between adjacent components. Cluster-seeking, not
    range-seeking: BIC penalizes extra components that only serve a
    sparsely-populated tail, so it tends to underrepresent high-viscosity
    runs the same way quantile does (just less severely).
  * `quantile`: equal-*count* bins (no dependency). Guarantees balanced
    support per tier but, for a right-skewed corpus, that balance is
    exactly what collapses the tail.

Whichever method is used, bins with fewer than `min_support` runs are merged into their
neighbour, because a stratification bin you cannot populate in BOTH train
and val splits is worse than no bin: it silently degrades to noise in the
sampler and in the benchmark's per-tier tables (the 150+ cP tier with n=15
in the current corpus is exactly this).

The result is persisted to `configs/tiers.json` and consumed by:
  * dataset/build_detectors.py, dataset/build_fill_classifier.py - stratified
    group split + per-tier upsampling
  * qa/benchmark.py - per-tier reporting

Usage
-----
    python -m src.systems.qmodel_7_onyx.tiers --raw-root data/raw --out configs/tiers.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.utils.logger import get_logger

from . import paths

LOG = get_logger("qmodel_7_onyx.tiers")


@dataclass
class TierScheme:
    """Edges are in cP, ascending, defining len(edges)+1 bins plus an
    implicit "unknown" bin for runs without a viscosity estimate."""

    edges_cp: List[float]
    labels: List[str] = field(default_factory=list)
    n_per_tier: List[int] = field(default_factory=list)
    method: str = "quantile"

    def __post_init__(self):
        if not self.labels:
            self.labels = self._make_labels()

    def _make_labels(self) -> List[str]:
        labels = [f"<{self.edges_cp[0]:g} cP"]
        for a, b in zip(self.edges_cp[:-1], self.edges_cp[1:], strict=True):
            labels.append(f"{a:g}-{b:g} cP")
        labels.append(f"{self.edges_cp[-1]:g}+ cP")
        labels.append("unknown")
        return labels

    @property
    def n_tiers(self) -> int:
        return len(self.edges_cp) + 2  # bins + unknown

    def tier_of(self, cp: Optional[float]) -> int:
        if cp is None or not np.isfinite(cp):
            return len(self.edges_cp) + 1  # unknown
        for i, e in enumerate(self.edges_cp):
            if cp < e:
                return i
        return len(self.edges_cp)

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(
                dict(
                    edges_cp=self.edges_cp,
                    labels=self.labels,
                    n_per_tier=self.n_per_tier,
                    method=self.method,
                ),
                indent=2,
            )
        )

    @staticmethod
    def load(path: Path) -> "TierScheme":
        d = json.loads(Path(path).read_text())
        return TierScheme(
            edges_cp=d["edges_cp"],
            labels=d.get("labels", []),
            n_per_tier=d.get("n_per_tier", []),
            method=d.get("method", "loaded"),
        )


def _merge_small_bins(edges: List[float], log_v: np.ndarray, min_support: int) -> List[float]:
    """Drop interior edges until every bin has >= min_support members."""
    edges = sorted(edges)
    while edges:
        counts = np.histogram(log_v, bins=[-np.inf] + edges + [np.inf])[0]
        if counts.min() >= min_support or len(edges) == 0:
            break
        # remove the edge adjacent to the smallest bin (merge into neighbour)
        k = int(np.argmin(counts))
        drop = (
            min(max(k - 1, 0), len(edges) - 1) if k == len(counts) - 1 else min(k, len(edges) - 1)
        )
        edges.pop(drop)
    return edges


def _log_uniform_edges(log_v: np.ndarray, n_bins: int) -> List[float]:
    """Interior edges of `n_bins` bins of equal width spanning
    `log_v`'s observed range - the only scheme whose top edge tracks the
    actual max of a right-skewed corpus rather than a count quantile."""
    lo, hi = float(log_v.min()), float(log_v.max())
    return list(np.linspace(lo, hi, n_bins + 1)[1:-1])


def _quantile_edges(log_v: np.ndarray, n_bins: int) -> List[float]:
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return list(np.quantile(log_v, qs))


def _gmm_edges(log_v: np.ndarray, max_tiers: int) -> Optional[tuple]:
    """Returns (edges_log, n_components) via BIC-selected GaussianMixture,
    or None if scikit-learn isn't installed."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None

    best_bic, best_gm = np.inf, None
    x = log_v.reshape(-1, 1)
    for k in range(2, max_tiers + 1):
        gm = GaussianMixture(n_components=k, n_init=3, random_state=0).fit(x)
        bic = gm.bic(x)
        if bic < best_bic:
            best_bic, best_gm = bic, gm
    # decision boundaries between adjacent (sorted) components via a
    # dense posterior scan - robust to unequal variances/weights.
    grid = np.linspace(log_v.min(), log_v.max(), 4000).reshape(-1, 1)
    lab = best_gm.predict(grid)
    change = np.where(np.diff(lab) != 0)[0]
    edges_log = sorted(float(grid[i + 1, 0]) for i in change)
    return edges_log, best_gm.n_components


def fit_tiers(
    viscosities_cp: np.ndarray,
    max_tiers: int = 8,
    min_support: int = 40,
    method: str = "auto",
) -> TierScheme:
    """Fit a TierScheme on the corpus viscosities.

    min_support: minimum runs per tier AFTER merging. Should comfortably
    exceed 2x the validation fraction's reciprocal sampling needs (a tier
    needs presence in both splits to be a meaningful stratum).

    method: "auto" (== "log_uniform"), "log_uniform", "gmm", or "quantile".
    "gmm" falls back to "log_uniform" (not "quantile") when scikit-learn is
    unavailable or its decision-boundary scan comes up empty, since
    log_uniform is the better range-preserving default either way.
    """
    v = np.asarray(viscosities_cp, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 3 * min_support:
        raise SystemExit(f"Too few viscosity-labelled runs ({len(v)}) for tier discovery.")
    log_v = np.log10(v)

    edges_log: Optional[List[float]] = None
    used: Optional[str] = None
    if method == "gmm":
        result = _gmm_edges(log_v, max_tiers)
        if result is not None:
            edges_log, k = result
            used = f"gmm_bic(k={k})"
    elif method == "quantile":
        edges_log = _quantile_edges(log_v, max_tiers)
        used = "quantile"

    if not edges_log:
        edges_log = _log_uniform_edges(log_v, max_tiers)
        used = "log_uniform"

    edges_log = _merge_small_bins(edges_log, log_v, min_support)
    edges_cp = [round(float(10**e), 2) for e in edges_log]
    scheme = TierScheme(edges_cp=edges_cp, method=used)
    counts = np.histogram(log_v, bins=[-np.inf] + edges_log + [np.inf])[0]
    scheme.n_per_tier = [int(c) for c in counts] + [0]  # unknown count filled by caller
    return scheme


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=paths.DATA_ROOT)
    ap.add_argument("--out", type=Path, default=paths.TIERS_JSON)
    ap.add_argument("--max-tiers", type=int, default=8)
    ap.add_argument("--min-support", type=int, default=40)
    ap.add_argument(
        "--method",
        choices=["auto", "log_uniform", "gmm", "quantile"],
        default="auto",
    )
    args = ap.parse_args()

    from .corpus import discover_runs  # reuses zip-aware viscosity

    runs = discover_runs(args.raw_root)
    v = np.array([r.viscosity_cP for r in runs if r.viscosity_cP is not None], dtype=float)
    n_unknown = sum(1 for r in runs if r.viscosity_cP is None)
    LOG.info("{} runs, {} with viscosity, {} unknown", len(runs), len(v), n_unknown)

    scheme = fit_tiers(
        v, max_tiers=args.max_tiers, min_support=args.min_support, method=args.method
    )
    scheme.n_per_tier[-1] = n_unknown
    args.out.parent.mkdir(parents=True, exist_ok=True)
    scheme.save(args.out)
    print(f"method: {scheme.method}")
    for lab, n in zip(scheme.labels, scheme.n_per_tier, strict=True):
        print(f"  {lab:<14} n={n}")
    LOG.info("Wrote {}", args.out)


if __name__ == "__main__":
    main()

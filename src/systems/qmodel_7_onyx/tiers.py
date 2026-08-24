"""Data-driven viscosity tier binning for stratified dataset splitting.

Replaces arbitrary binning schemes by fitting tiers on log10(viscosity_cP),
since fill dynamics scale roughly multiplicatively with viscosity, making
log-space ideal for cluster structure.

Supports multiple fitting methods:
  * `log_uniform` (default): Equal-width bins in log-space, ensuring the
    top tier captures the true high-viscosity tail of the corpus.
  * `gmm`: Gaussian mixture model with BIC model selection, placing bin edges
    at posterior decision boundaries. Tends to underrepresent the tail slightly.
  * `quantile`: Equal-count bins. Guarantees balanced support but collapses
    the right-skewed tail into a single bucket.

Regardless of the selected method, bins with fewer than `min_support` runs
are iteratively merged into their neighbors to ensure viable stratification
across train and validation splits.

Example:
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
    """Defines viscosity tiers and handles categorization of run data.

    Edges are stored in centipoise (cP) in ascending order, defining
    `len(edges) + 1` bins plus an implicit "unknown" bin for runs without
    a viscosity estimate.

    Attributes:
        edges_cp (List[float]): Ascending tier boundary edges in cP.
        labels (List[str]): Human-readable labels for each tier.
        n_per_tier (List[int]): Number of samples falling into each tier.
        method (str): The fitting method used to generate the tiers (e.g.,
            "log_uniform", "gmm", "quantile").
    """

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
        """int: Total number of defined tiers, including the 'unknown' tier."""
        return len(self.edges_cp) + 2  # bins + unknown

    def tier_of(self, cp: Optional[float]) -> int:
        """Determines the corresponding tier index for a given viscosity value.

        Args:
            cp (Optional[float]): Viscosity value in cP.

        Returns:
            int: The zero-indexed tier bin, or the 'unknown' tier index
            if `cp` is None or non-finite.
        """
        if cp is None or not np.isfinite(cp):
            return len(self.edges_cp) + 1  # unknown
        for i, e in enumerate(self.edges_cp):
            if cp < e:
                return i
        return len(self.edges_cp)

    def save(self, path: Path) -> None:
        """Serializes the tier scheme to a JSON configuration file.

        Args:
            path (Path): Destination file path for the JSON configuration.
        """
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
        """Deserializes a TierScheme from a JSON configuration file.

        Args:
            path (Path): Path to the saved JSON configuration.

        Returns:
            TierScheme: The instantiated tier scheme loaded from the file.
        """
        d = json.loads(Path(path).read_text())
        return TierScheme(
            edges_cp=d["edges_cp"],
            labels=d.get("labels", []),
            n_per_tier=d.get("n_per_tier", []),
            method=d.get("method", "loaded"),
        )


def _merge_small_bins(edges: List[float], log_v: np.ndarray, min_support: int) -> List[float]:
    """Merges interior bin edges until all bins meet a minimum support threshold.

    Iteratively drops the boundary adjacent to the smallest bin, effectively
    merging it into its neighbor, until all remaining bins contain at least
    `min_support` items or no interior edges remain.

    Args:
        edges (List[float]): Initial list of interior bin edges in log space.
        log_v (np.ndarray): Array of log10(viscosity) values.
        min_support (int): Minimum number of samples required per bin.

    Returns:
        List[float]: Filtered list of interior edges ensuring minimum support.
    """
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
    """Calculates equal-width bin edges in log space.

    Ensures the top edge tracks the actual maximum of a right-skewed corpus
    rather than a count quantile.

    Args:
        log_v (np.ndarray): Array of log10(viscosity) values.
        n_bins (int): Target number of bins.

    Returns:
        List[float]: Interior bin edges dividing the range evenly.
    """
    lo, hi = float(log_v.min()), float(log_v.max())
    return list(np.linspace(lo, hi, n_bins + 1)[1:-1])


def _quantile_edges(log_v: np.ndarray, n_bins: int) -> List[float]:
    """Calculates equal-count bin edges using quantiles.

    Args:
        log_v (np.ndarray): Array of log10(viscosity) values.
        n_bins (int): Target number of bins.

    Returns:
        List[float]: Interior bin edges mapping to equal percentiles.
    """
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return list(np.quantile(log_v, qs))


def _gmm_edges(log_v: np.ndarray, max_tiers: int) -> Optional[tuple]:
    """Calculates cluster-seeking bin boundaries using a Gaussian Mixture Model.

    Selects the optimal number of components up to `max_tiers` using the
    Bayesian Information Criterion (BIC), then places edges at the dense
    posterior decision boundaries between adjacent components.

    Args:
        log_v (np.ndarray): Array of log10(viscosity) values.
        max_tiers (int): Maximum number of mixture components to evaluate.

    Returns:
        Optional[tuple]: A tuple containing a list of interior log edges
        and the number of components selected, or None if `scikit-learn`
        is not installed.
    """
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
    """Fits a data-driven TierScheme to the provided corpus viscosities.

    Transforms viscosities to log10 space, applies the requested binning
    strategy, and merges underpopulated bins to ensure valid stratification
    during train/val splitting.

    Args:
        viscosities_cp (np.ndarray): Array of corpus viscosity values in cP.
        max_tiers (int, optional): Maximum number of tiers to generate initially.
            Defaults to 8.
        min_support (int, optional): Minimum required runs per tier after merging.
            Should exceed 2x the reciprocal validation fraction. Defaults to 40.
        method (str, optional): Fitting strategy. One of "auto" (defaults
            to "log_uniform"), "log_uniform", "gmm", or "quantile".
            "gmm" falls back to "log_uniform" if unsupported or if the scan fails.
            Defaults to "auto".

    Returns:
        TierScheme: The fitted scheme containing final tier boundaries and counts.

    Raises:
        SystemExit: If the number of valid viscosity samples is insufficient
            for the required minimum support.
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
    """Parses command-line arguments and fits a new viscosity TierScheme.

    Discovers runs in the configured raw data root, extracts viscosities, fits
    the requested tier structure, and saves the configuration to JSON for use
    in downstream dataset building and benchmarking.
    """
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

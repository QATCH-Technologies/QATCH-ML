"""Shared run-level split and per-tier upsampling utilities for dataset builders.

Provides the common train/validation partitioning and viscosity-tier
upsampling logic used by multiple dataset builders. Keeping these operations
centralized ensures that builders apply consistent leakage prevention,
stratification, and tier-balancing behavior.

The module exposes:

    :class:`SplitResult`
        Container for the run identifiers assigned to each dataset split.

    :func:`stratified_group_split`
        Creates a run-level train/validation partition stratified by viscosity
        tier and POI count.

    :func:`repeat_factor`
        Computes a bounded per-tier rendering multiplier based on the relative
        representation of each tier in the training set.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..corpus import RunRecord
from ..tiers import TierScheme


@dataclass
class SplitResult:
    """Represent the run-level partition produced by a dataset split.

    Attributes:
        train_ids (List[str]): Run identifiers assigned to the training split.
        val_ids (List[str]): Run identifiers assigned to the validation split.
    """

    train_ids: List[str]
    val_ids: List[str]


def stratified_group_split(
    runs: List[RunRecord], tiers: TierScheme, val_frac: float, seed: int
) -> SplitResult:
    """Create a leakage-safe, stratified train/validation run partition.

    Assigns each run wholly to either the training or validation split so that
    rendered variants and derived samples from the same run cannot cross the
    split boundary. Runs are grouped by viscosity tier and POI count before
    sampling the validation subset, preserving representation across relevant
    strata where sufficient runs are available.

    Singleton strata remain in the training split because they cannot provide
    an independent validation example without removing the stratum entirely
    from training.

    Args:
        runs (List[RunRecord]): Run records to partition.
        tiers (TierScheme): Viscosity tier scheme used to assign each run to a
            stratification group.
        val_frac (float): Target fraction of runs from each stratum assigned to
            validation.
        seed (int): Random seed used to shuffle runs within each stratum.

    Returns:
        SplitResult: Run identifiers partitioned into non-overlapping training
        and validation splits.
    """
    rng = np.random.default_rng(seed)
    strata: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for r in runs:
        strata[(tiers.tier_of(r.viscosity_cP), len(r.poi_times))].append(r.run_id)
    train_ids: List[str] = []
    val_ids: List[str] = []
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        n_val = int(round(val_frac * len(ids)))
        if len(ids) >= 2:
            n_val = max(1, min(n_val, len(ids) - 1))
        else:
            n_val = 0  # singleton stratum: keep it trainable
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])
    assert not set(train_ids) & set(val_ids), "split leakage: run in both splits"
    return SplitResult(train_ids, val_ids)


def repeat_factor(tier: int, tier_counts: Dict[int, int], cap: int) -> int:
    """Calculate a bounded rendering multiplier for a viscosity tier.

    Computes the repetition factor relative to the most represented training
    tier using a square-root scaling rule. Underrepresented tiers therefore
    receive additional rendered variants while avoiding the excessive sample
    growth that direct inverse-frequency weighting could produce.

    Args:
        tier (int): Viscosity tier index for which the repetition factor is
            calculated.
        tier_counts (Dict[int, int]): Number of training runs associated with
            each tier index.
        cap (int): Maximum permitted repetition factor.

    Returns:
        int: Bounded repetition factor calculated as
        `clip(round(sqrt(n_max / n_tier)), 1, cap)`.
    """
    n_max = max(tier_counts.values())
    n = max(1, tier_counts.get(tier, 1))
    return int(np.clip(round(np.sqrt(n_max / n)), 1, cap))

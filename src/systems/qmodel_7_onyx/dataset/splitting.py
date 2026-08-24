"""Shared run-level split and per-tier upsampling logic for dataset builders.

Used by both :mod:`.build_detectors` and :mod:`.build_fill_classifier` so the
two dataset builders agree on which runs land in train vs. val and on how
much each viscosity tier is upsampled, rather than each reimplementing (and
risking drifting apart on) the same leakage-sensitive logic independently.

  * :func:`stratified_group_split` - leakage-proof split. Train/val is split
    by run (group split): every rendered variant - augmented or not, every
    stage frame - of a run lands on one side only. The split is stratified
    by (viscosity tier x fill count) so rare tiers are present in both
    splits.

  * :func:`repeat_factor` - per-tier upsampling. `repeat = clip(sqrt(n_max /
    n_tier), 1, cap)`, so rare tiers are rendered proportionally more often.
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
    """Run-id partition produced by :func:`stratified_group_split`.

    Attributes:
        train_ids (List[str]): Run ids assigned to the training split.
        val_ids (List[str]): Run ids assigned to the validation split.
    """

    train_ids: List[str]
    val_ids: List[str]


def stratified_group_split(
    runs: List[RunRecord], tiers: TierScheme, val_frac: float, seed: int
) -> SplitResult:
    """Split runs into train/val, grouped by run id and stratified by stratum.

    Each run is assigned to exactly one split (a group split by run id), so
    no rendered variant of a run - augmented or not, any stage frame - can
    leak across the train/val boundary. Runs are stratified by
    `(tier, n_pois)`: every stratum with at least two runs contributes at
    least one run to val, so no tier is invisible to validation.

    Args:
        runs (List[RunRecord]): Runs to split.
        tiers (TierScheme): Viscosity tier scheme used to compute each run's
            stratum.
        val_frac (float): Target fraction of each stratum assigned to val.
        seed (int): Seed for the per-stratum shuffle.

    Returns:
        SplitResult: The train/val run-id partition.
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
    """Compute how many extra times a tier's runs should be rendered.

    Scales inversely with a tier's representation in the training split so
    rare tiers get proportionally more rendered variants, without letting a
    very rare tier explode past a sane multiple of the best-represented
    tier's render cost.

    Args:
        tier (int): Tier index to compute the repeat factor for.
        tier_counts (Dict[int, int]): Number of training runs per tier index.
        cap (int): Maximum allowed repeat factor.

    Returns:
        int: Repeat factor, `clip(sqrt(n_max / n_tier), 1, cap)`.
    """
    n_max = max(tier_counts.values())
    n = max(1, tier_counts.get(tier, 1))
    return int(np.clip(round(np.sqrt(n_max / n)), 1, cap))

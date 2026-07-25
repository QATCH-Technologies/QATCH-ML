"""
dataset/splitting.py
=====================

Shared run-level split and per-tier upsampling logic used by BOTH dataset
builders (``dataset/build_detectors.py`` and
``dataset/build_fill_classifier.py``). Extracted out of build_dataset.py,
which the fill-classifier builder used to import these two functions from
directly — a sibling CLI script reaching into another CLI script's library
functions. Both builders now import from here instead.

  * :func:`stratified_group_split` — LEAKAGE-PROOF SPLIT. Train/val is split
    by RUN (group split): every rendered variant — augmented or not, every
    stage frame — of a run lands on one side only. The split is stratified
    by (viscosity tier x fill count) so rare tiers are present in BOTH
    splits.

  * :func:`repeat_factor` — PER-TIER UPSAMPLING. repeat = clip(sqrt(n_max /
    n_tier), 1, cap), so rare tiers are rendered proportionally more often.
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
    train_ids: List[str]
    val_ids: List[str]


def stratified_group_split(
    runs: List[RunRecord], tiers: TierScheme, val_frac: float, seed: int
) -> SplitResult:
    """Group split by run id, stratified by (tier, n_pois). Every stratum
    with >= 2 runs contributes at least one run to val so no tier is
    invisible to validation."""
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
    n_max = max(tier_counts.values())
    n = max(1, tier_counts.get(tier, 1))
    return int(np.clip(round(np.sqrt(n_max / n)), 1, cap))

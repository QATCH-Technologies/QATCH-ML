"""Build the fill-type classification dataset for a live fill classifier.

Renders prefix-based classification samples for each fill state in a
classifier-compatible directory structure. Dataset generation preserves
run-level train/validation separation, matches the temporal distribution seen
by the live classifier, excludes ambiguous transition regions, oversamples
latency-critical post-transition frames, and applies signal-domain
augmentation with corresponding POI timing adjustments.

Training samples are balanced across achievable fill states rather than being
weighted by the amount of time each state occupies. Validation uses only
unaugmented variants so evaluation reflects classifier performance rather than
augmentation behavior.

Rendered samples use the same preprocessing and final input preparation as
the deployed classifier, keeping training and inference image distributions
aligned.

Usage:
    python -m src.systems.qmodel_7_onyx.dataset.build_fill_classifier \
        --raw-root data/raw --tiers tiers.json --out datasets/onyx_fill \
        [--base-variants 2] [--cuts-per-class 2] [--hard-cuts 1] \
        [--val-frac 0.15] [--repeat-cap 8] [--seed 7]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

from .. import paths
from ..augmentation import COL_TIME, augment_run, dynamic_box_width_sec
from ..corpus import dedupe_runs, discover_runs
from ..rendering.dataprocessor import QModelOnyxDataProcessor as DP
from ..rendering.fill_render import prepare_cls_input
from ..tiers import TierScheme
from .splitting import repeat_factor, stratified_group_split

LOG = get_logger("qmodel_7_onyx.dataset.build_fill_classifier")

FILL_RENDER_VERSION = 3  # must match the predictor's fill render version

# Ordinal class order - index == channels + 1
CLASS_NAMES = ["no_fill", "initial_fill", "1ch", "2ch", "3ch"]

# State boundaries in POI space
BOUNDARY_POI = {1: "POI1", 2: "POI3", 3: "POI4", 4: "POI5"}

MIN_PREFIX_S = 3.0  # matches MIN_SLICE_S: shortest usable prefix
MIN_PREFIX_PTS = 64
SETTLE_FRAC = 0.75  # cut must sit past this fraction of the transition extent
PRE_FRAC = 0.35  # exclusion before the NEXT transition begins
CUT_MARGIN_S = 0.2  # additive slop on both exclusions
HARD_SPAN_S = 4.0  # width of the oversampled just-confirmed band


def fill_state_at(t: float, poi: Dict[str, float]) -> int:
    """Return the fill-state class active at a given time.

    Determines the ordinal fill state by comparing the requested time against
    the configured POI state boundaries. Missing boundaries are ignored, so
    the latest confirmed state remains active until another available
    boundary is reached.

    Args:
        t (float): Signal time at which the fill state should be evaluated.
        poi (dict[str, float]): Mapping of POI names to their timestamps.

    Returns:
        int: Ordinal fill-state index corresponding to the state active at
        `t`.
    """
    state = 0
    for k in (1, 2, 3, 4):
        pt = poi.get(BOUNDARY_POI[k])
        if pt is not None and t >= pt:
            state = k
    return state


def class_intervals(
    poi: Dict[str, float], df_p: pd.DataFrame, t0: float, t1: float
) -> Dict[int, Tuple[float, float]]:
    """Determine valid temporal sampling intervals for each fill state.

    Builds one sampleable interval per state while excluding transition
    regions whose measured temporal extent makes the state visually ambiguous.
    The beginning of each state is delayed until the transition has sufficiently
    settled, while the end is shortened before the next transition begins.

    States with missing boundaries or insufficient usable duration are omitted
    from the returned mapping.

    Args:
        poi (dict[str, float]): Mapping of POI names to their timestamps.
        df_p (pandas.DataFrame): Preprocessed signal used to estimate transition
            extents.
        t0 (float): Start time of the usable signal.
        t1 (float): End time of the usable signal.

    Returns:
        dict[int, tuple[float, float]]: Mapping from ordinal fill-state indices
        to inclusive sampling intervals represented as `(start, end)`.
    """
    bounds: List[Tuple[int, Optional[float], float]] = []  # (state, t_boundary, width)
    for k in (1, 2, 3, 4):
        pt = poi.get(BOUNDARY_POI[k])
        w = dynamic_box_width_sec(df_p, pt) if pt is not None else 0.0
        bounds.append((k, pt, w))

    intervals: Dict[int, Tuple[float, float]] = {}

    # State 0 (no_fill): from the earliest usable prefix to just before the
    # first transition begins.
    first = next(((pt, w) for _, pt, w in bounds if pt is not None), None)
    hi0 = t1 if first is None else first[0] - PRE_FRAC * first[1] - CUT_MARGIN_S
    intervals[0] = (t0 + MIN_PREFIX_S, min(hi0, t1))

    for i, (k, pt, w) in enumerate(bounds):
        if pt is None:
            continue
        lo = pt + SETTLE_FRAC * w + CUT_MARGIN_S
        nxt = next(((p2, w2) for _, p2, w2 in bounds[i + 1 :] if p2 is not None), None)
        hi = t1 if nxt is None else nxt[0] - PRE_FRAC * nxt[1] - CUT_MARGIN_S
        intervals[k] = (max(lo, t0 + MIN_PREFIX_S), min(hi, t1))
    return {k: (lo, hi) for k, (lo, hi) in intervals.items() if hi > lo}


def sample_cuts(
    rng: np.random.Generator,
    intervals: Dict[int, Tuple[float, float]],
    cuts_per_class: int,
    hard_cuts: int,
) -> List[Tuple[float, int, bool]]:
    """Sample classifier cut times across the available fill states.

    Draws uniformly distributed cuts within each state's valid interval and
    optionally adds extra samples near the beginning of each nonzero state.
    These additional hard-band samples emphasize frames immediately after a
    transition has visually settled.

    Args:
        rng (numpy.random.Generator): Random-number generator used to sample
            cut positions.
        intervals (dict[int, tuple[float, float]]): Sampleable temporal
            intervals keyed by fill-state index.
        cuts_per_class (int): Number of regular cuts to sample for each
            available state.
        hard_cuts (int): Number of additional post-transition cuts to sample
            for each nonzero state.

    Returns:
        list[tuple[float, int, bool]]: Sample specifications containing the cut
        time, fill-state index, and whether the sample belongs to a
        transition-adjacent hard band.
    """
    out: List[Tuple[float, int, bool]] = []
    for state, (lo, hi) in intervals.items():
        for _ in range(cuts_per_class):
            out.append((float(rng.uniform(lo, hi)), state, False))
        if state > 0 and hard_cuts > 0:
            h_hi = min(lo + HARD_SPAN_S, hi)
            for _ in range(hard_cuts):
                out.append((float(rng.uniform(lo, h_hi)), state, True))
    return out


def build(
    raw_root: Path,
    tiers_path: Path,
    out_root: Path,
    base_variants: int = 2,
    cuts_per_class: int = 2,
    hard_cuts: int = 1,
    val_frac: float = 0.15,
    repeat_cap: int = 8,
    seed: int = 7,
    limit: Optional[int] = None,
) -> None:
    """Build the complete fill-state classification dataset.

    Discovers and deduplicates runs, creates a leakage-safe stratified
    train/validation split, generates clean and augmented signal variants,
    samples state-balanced prefix cuts, and renders classifier-ready images
    for each fill state.

    Training runs receive tier-dependent repetition and fresh augmentation,
    while validation runs use only the clean variant. Each run also contributes
    a completed-run sample representing the final analysis-time state.

    The output directory is recreated when it already exists. Samples are
    organized by dataset split and class name, and a `manifest.json` records
    generation settings, split membership, tier information, and sample
    counts.

    Args:
        raw_root (pathlib.Path): Root directory containing raw per-run data.
        tiers_path (pathlib.Path): Path to the serialized tier scheme used for
            stratification and per-tier repetition.
        out_root (pathlib.Path): Destination directory for the generated
            classification dataset.
        base_variants (int, optional): Base number of training variants
            generated for each run before tier-specific repetition. Defaults
            to 2.
        cuts_per_class (int, optional): Number of regular prefix cuts sampled
            for each achievable fill state per variant. Defaults to 2.
        hard_cuts (int, optional): Number of additional transition-adjacent
            cuts sampled for each nonzero fill state. Defaults to 1.
        val_frac (float, optional): Target validation fraction within each
            stratification group. Defaults to 0.15.
        repeat_cap (int, optional): Maximum tier-specific repetition factor
            applied to training runs. Defaults to 8.
        seed (int, optional): Random seed controlling splitting, sampling, and
            augmentation. Defaults to 7.
        limit (int, optional): Maximum number of discovered runs to process.
            If `None`, all discovered runs are used.

    Raises:
        SystemExit: If no usable runs are discovered beneath `raw_root`.
    """
    rng = np.random.default_rng(seed)
    tiers = TierScheme.load(tiers_path)
    runs = dedupe_runs(discover_runs(raw_root))
    if limit:
        runs = runs[:limit]
    if not runs:
        raise SystemExit(f"no runs under {raw_root}")
    LOG.info("{} runs discovered", len(runs))

    split = stratified_group_split(runs, tiers, val_frac, seed)
    by_id = {r.run_id: r for r in runs}
    tier_counts: Dict[int, int] = defaultdict(int)
    for rid in split.train_ids:
        tier_counts[tiers.tier_of(by_id[rid].viscosity_cP)] += 1

    out_root = Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    for split_name in ("train", "val"):
        for cname in CLASS_NAMES:
            (out_root / split_name / cname).mkdir(parents=True)

    counts = defaultdict(int)

    def emit(split_name: str, state: int, name: str, img: np.ndarray) -> None:
        """Write a rendered classifier sample to its state-specific directory.

        Args:
            split_name (str): Dataset split receiving the sample.
            state (int): Ordinal fill-state index used to select the class
                directory.
            name (str): Base identifier for the generated sample.
            img (numpy.ndarray): Prepared classifier input image.
        """
        cname = CLASS_NAMES[state]
        h = hashlib.blake2s(f"fill/{name}".encode(), digest_size=4).hexdigest()
        cv2.imwrite(str(out_root / split_name / cname / f"{h}_{name}.png"), img)
        counts[(split_name, cname)] += 1

    def process_run(rid: str, split_name: str) -> None:
        """Generate classification samples for a single run.

        Loads the run data, creates the required clean or augmented variants,
        determines valid state-specific sampling intervals, generates prefix cuts,
        and emits both sampled prefix frames and the completed-run analysis frame.

        Args:
            rid (str): Identifier of the run to process.
            split_name (str): Dataset split receiving the generated samples.
        """
        rec = by_id[rid]
        try:
            df_raw = pd.read_csv(rec.csv_path)
        except Exception as exc:
            LOG.warning("skip {}: {}", rid, exc)
            return
        tier = tiers.tier_of(rec.viscosity_cP)
        n_variants = (
            base_variants * repeat_factor(tier, tier_counts, repeat_cap)
            if split_name == "train"
            else 1
        )
        for v in range(n_variants):
            # Variant 0 is always the clean signal; val ONLY gets variant 0.
            if v == 0 or split_name == "val":
                df_a, poi_a = df_raw, dict(rec.poi_times)
            else:
                df_a, poi_a, _ = augment_run(df_raw, dict(rec.poi_times), rng)
            df_p = DP.preprocess_dataframe(df_a)
            if df_p is None or df_p.empty:
                continue
            t = df_p[COL_TIME].to_numpy(dtype=float)
            t0, t1 = float(t[0]), float(t[-1])
            if t1 - t0 < MIN_PREFIX_S:
                continue

            intervals = class_intervals(poi_a, df_p, t0, t1)
            cuts = sample_cuts(rng, intervals, cuts_per_class, hard_cuts)
            # Analysis-time frame: the completed run, labeled by final state.
            cuts.append((t1 + 1e-6, fill_state_at(t1, poi_a), False))

            for k, (cut_t, state, is_hard) in enumerate(cuts):
                sl = df_p[df_p[COL_TIME] < cut_t]
                if len(sl) < MIN_PREFIX_PTS:
                    continue
                if float(sl[COL_TIME].iloc[-1]) - t0 < MIN_PREFIX_S:
                    continue
                img = prepare_cls_input(sl, version=FILL_RENDER_VERSION)
                tag = "h" if is_hard else "u"
                emit(split_name, state, f"{rid}_v{v}_{tag}{k}", img)

    for i, rid in enumerate(split.train_ids):
        process_run(rid, "train")
        if (i + 1) % 100 == 0:
            LOG.info("train {}/{}", i + 1, len(split.train_ids))
    for rid in split.val_ids:
        process_run(rid, "val")

    manifest = dict(
        fill_render_version=FILL_RENDER_VERSION,
        seed=seed,
        val_frac=val_frac,
        base_variants=base_variants,
        cuts_per_class=cuts_per_class,
        hard_cuts=hard_cuts,
        repeat_cap=repeat_cap,
        class_names=CLASS_NAMES,
        tiers=tiers.labels,
        tier_counts_train={tiers.labels[k]: v for k, v in sorted(tier_counts.items())},
        n_train_runs=len(split.train_ids),
        n_val_runs=len(split.val_ids),
        train_ids=sorted(split.train_ids),
        val_ids=sorted(split.val_ids),
        sample_counts={f"{sp}/{c}": n for (sp, c), n in sorted(counts.items())},
    )
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["sample_counts"], indent=2))
    print(f"train runs {len(split.train_ids)} | val runs {len(split.val_ids)}")
    LOG.info("Dataset -> {}", out_root)


def main() -> None:
    """Run the fill-classification dataset builder from the command line.

    Parses dataset-generation options and delegates construction to
    :func:`build`.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=paths.DATA_ROOT)
    ap.add_argument("--tiers", type=Path, default=paths.TIERS_JSON)
    ap.add_argument("--out", type=Path, default=paths.DATASETS_ROOT / "onyx_fill")
    ap.add_argument("--base-variants", type=int, default=2)
    ap.add_argument("--cuts-per-class", type=int, default=2)
    ap.add_argument("--hard-cuts", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--repeat-cap", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(
        args.raw_root,
        args.tiers,
        args.out,
        base_variants=args.base_variants,
        cuts_per_class=args.cuts_per_class,
        hard_cuts=args.hard_cuts,
        val_frac=args.val_frac,
        repeat_cap=args.repeat_cap,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

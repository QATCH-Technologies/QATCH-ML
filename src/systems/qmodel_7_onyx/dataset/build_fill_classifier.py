"""
dataset/build_fill_classifier.py
==================================

Builds the v7 fill-TYPE classification dataset (no_fill / initial_fill /
1ch / 2ch / 3ch) in ultralytics classify folder format, porting every
dataset-side fix the detector rebuild demanded:

  1. LEAKAGE-PROOF SPLIT — reuses ``dataset.splitting.stratified_group_split``
     (shared with ``dataset/build_detectors.py``): train/val split by RUN,
     stratified by (viscosity tier x fill count). Every prefix cut and
     augmented variant of a run lands on one side only. (The classifier has
     the same exposure the old detector split had: dozens of prefix frames
     per run means run-level leakage inflates val accuracy dramatically — a
     classifier that memorizes a run's noise texture aces val on that run's
     other prefixes.)

  2. LIVE-MATCHED SLICING. The live classifier never sees a full run until
     the run is over — it sees a GROWING PREFIX, one frame per chunk. So
     the training distribution is prefix cuts: for each run, cut times are
     sampled per achievable class and the label is the fill state AT the
     cut (state(t): t<POI1 -> no_fill; POI1<=t<POI3 -> initial_fill;
     POI3<=t<POI4 -> 1ch; POI4<=t<POI5 -> 2ch; t>=POI5 -> 3ch). The full
     run is always emitted too — that is the analysis-time distribution,
     which is just the prefix at t=end. One model, both duties, because
     both duties are points on the same prefix continuum.

  3. TRANSITION DEAD ZONES + HARD BANDS (the dynamic-box insight, inverted).
     dynamic_box_width_sec measures each transition's actual temporal
     extent. DURING a transition the label is genuinely ambiguous — the
     ridge is half-formed — so cuts inside the measured extent are
     SKIPPED rather than taught with a hard label the pixels don't yet
     support (mid-transition frames are what the live debounce exists to
     ride out; training a confident label there just teaches confident
     flicker). Immediately AFTER the transition completes sits the hard
     band: the earliest moment the new state is visually present (one
     barely-grown ridge). These are the latency-critical live frames —
     every second of delayed confirmation is a second of delayed operator
     feedback — so the hard band is deliberately oversampled.

  4. SIGNAL-DOMAIN AUGMENTATION (v7_augment), labels exact. time_warp is
     again the high-viscosity synthesizer, and for the classifier it is
     doubly load-bearing: stretching manufactures the slow-fill geometry
     where late transitions flatten — the 2ch/3ch confusion zone. POI
     times warp with the signal, so every cut's state label stays exact
     by construction.

  5. PER-TIER UPSAMPLING with fresh augmentation per repeat, and
     PER-CLASS-BALANCED cuts per run (each achievable state contributes
     the same number of cuts), so class balance does not simply mirror
     state dwell times (long 3ch tails would otherwise dominate).

  6. TRAIN/DEPLOY RENDER MATCH, taken one step further than the detector:
     images are saved as the EXACT 224x224 prepare_cls_input output the
     predictor feeds the model — including the 640->224 INTER_AREA resize
     — so there is no resize-kernel mismatch between training and
     inference at all.

Val is variant-0 (un-augmented) only: it measures the model, not the
augmentation pipeline.

Usage
-----
    python -m src.systems.qmodel_7_onyx.dataset.build_fill_classifier \
        --raw-root data/raw --tiers tiers.json --out datasets/v7_fill \
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
from ..rendering.fill_render import prepare_cls_input
from ..rendering.legacy_dataprocessor import QModelV6YOLO_DataProcessor as DP
from ..tiers import TierScheme
from .splitting import repeat_factor, stratified_group_split

LOG = get_logger("qmodel_7_onyx.dataset.build_fill_classifier")

FILL_RENDER_VERSION = 3  # must match the predictor's fill render version

# Ordinal class order — index == channels + 1. Folder names match
# QModelV6Config.FILL_CLASS_MAP keys so the trained model's label names map
# straight through _map_label_to_channels.
CLASS_NAMES = ["no_fill", "initial_fill", "1ch", "2ch", "3ch"]

# State boundaries in POI space: state k begins at BOUNDARY_POI[k].
# (POI2 — end of initial fill — is not a state boundary: initial_fill spans
# POI1..POI3.)
BOUNDARY_POI = {1: "POI1", 2: "POI3", 3: "POI4", 4: "POI5"}

MIN_PREFIX_S = 3.0  # matches MIN_SLICE_S: shortest usable prefix
MIN_PREFIX_PTS = 64
SETTLE_FRAC = 0.75  # cut must sit past this fraction of the transition extent
PRE_FRAC = 0.35  # exclusion before the NEXT transition begins
CUT_MARGIN_S = 0.2  # additive slop on both exclusions
HARD_SPAN_S = 4.0  # width of the oversampled just-confirmed band


def fill_state_at(t: float, poi: Dict[str, float]) -> int:
    """Ordinal state index (0..4) of the run at time t."""
    state = 0
    for k in (1, 2, 3, 4):
        pt = poi.get(BOUNDARY_POI[k])
        if pt is not None and t >= pt:
            state = k
    return state


def class_intervals(
    poi: Dict[str, float], df_p: pd.DataFrame, t0: float, t1: float
) -> Dict[int, Tuple[float, float]]:
    """Per-state sampleable cut interval [lo, hi], with dead zones carved
    out around each boundary transition using its MEASURED extent:

        lo(state k) = boundary_k + SETTLE_FRAC * width_k + margin
        hi(state k) = boundary_{k+1} - PRE_FRAC * width_{k+1} - margin

    States whose interval is empty (e.g. two transitions nearly back to
    back) simply contribute no cuts — better absent than mislabeled."""
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
    """Returns (cut_time, state, is_hard). Uniform cuts across each state's
    interval plus hard-band cuts hugging the interval's left edge — the
    just-confirmed frames where live latency is decided. State 0 has no
    'just confirmed' moment, so it takes only uniform cuts."""
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
        cname = CLASS_NAMES[state]
        h = hashlib.blake2s(f"fill/{name}".encode(), digest_size=4).hexdigest()
        cv2.imwrite(str(out_root / split_name / cname / f"{h}_{name}.png"), img)
        counts[(split_name, cname)] += 1

    def process_run(rid: str, split_name: str) -> None:
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=paths.DATA_ROOT)
    ap.add_argument("--tiers", type=Path, default=paths.TIERS_JSON)
    ap.add_argument("--out", type=Path, default=paths.DATASETS_ROOT / "v7_fill")
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

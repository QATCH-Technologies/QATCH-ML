"""Builds the per-stage YOLO detection datasets for the cascade detectors.

Renders one detection dataset per cascade stage (init / ch1 / ch2 / ch3,
plus zoom-refinement variants) from discovered runs, with the following
properties:

  1. LEAKAGE-PROOF SPLIT. Train/val is split by run (group split, via
     :func:`.splitting.stratified_group_split`): every rendered variant -
     augmented or not, every stage frame - of a run lands on one side only.
     The split is stratified by (viscosity tier x fill count) so rare tiers
     are present in BOTH splits. Validation images are rendered from
     UN-augmented signal only: val measures the model, not the augmentation
     pipeline. A split that lets augmented or partial-fill variants of a
     training run leak into val makes every metric used for model selection
     optimistic.

  2. SIGNAL-DOMAIN AUGMENTATION (:mod:`..augmentation`): monotone time-warp
     (the high-viscosity synthesizer), noise injection, amplitude jitter. POI
     labels are warped exactly with the signal.

  3. PER-TIER UPSAMPLING (:func:`.splitting.repeat_factor`). Each training
     run is rendered ``base_variants`` x ``repeat(tier)`` times. Each repeat
     draws FRESH augmentation, so upsampling adds geometry diversity rather
     than duplicating pixels - duplication alone cannot fix a sparsely
     populated tier, but a handful of runs stretch-warped across a wide
     scale range covers the slow-fill manifold that raw duplication cannot.

  4. CASCADE-MATCHED SLICING + NEGATIVES. Each stage's frames are sliced the
     way inference slices them: canonical cuts (between the target POI and
     the next), WIDE cuts (anywhere after the target - matching the decode
     layer's conservative harvest slices), and NEGATIVE cuts (before the
     target, or from partial fills): the target POI is absent and the label
     file is empty. A detector that never sees its target absent during
     training hallucinates a box on every frame at inference and relies
     entirely on the fill classifier being right.

  5. DYNAMIC BOXES. Box width = measured temporal extent of the transition
     (:func:`..augmentation.dynamic_box_width_sec`), clamped in pixels - so
     stretched slow-fill events keep IoU-trainable boxes instead of a fixed
     pixel size that starves them.

Rendering uses the same :func:`..rendering.detector_render.generate_det_image`
as inference, at the same 2560x384 geometry, so train and deploy
distributions match by construction.

Split and per-tier upsampling logic (:func:`.splitting.stratified_group_split`,
:func:`.splitting.repeat_factor`) lives in :mod:`.splitting`, shared with
:mod:`.build_fill_classifier`.

Usage
-----
    python -m src.systems.qmodel_7_onyx.dataset.build_detectors \
        --raw-root data/raw --tiers tiers.json --out datasets/v7 \
        [--base-variants 2] [--val-frac 0.15] [--repeat-cap 8] [--seed 7]
"""

from __future__ import annotations

import argparse
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
from ..rendering.legacy_dataprocessor import QModelOnyx_DataProcessor as DP
from ..tiers import TierScheme
from .splitting import repeat_factor, stratified_group_split

LOG = get_logger("qmodel_7_onyx.dataset.build_detectors")

IMG_W, IMG_H = 2560, 384
RENDER_VERSION = 3  # must match QModelOnyxConfig.RENDER_VERSION at inference

# Per-stage box geometry. Boxes are FULL HEIGHT (h=1.0): on these renders
# the vertical dimension carries no localization information (this is a 1D
# time-interval detection task drawn as 2D), and the previous 0.85-height
# centered boxes clipped the dissipation trace near strip tops and the
# difference trace near strip bottoms - only the middle strip was reliably
# inside the box, making the visual evidence inconsistent across samples.
# init boxes are TIGHT: POI1/POI2 are millisecond-scale events that can sit
# a few pixels apart; a wide box covering both events teaches the model
# that either event satisfies both classes. Widths are additionally clamped
# at the inter-POI midpoint (see _render_and_label).
BOX_SPEC = {
    "init": dict(min_px=6, max_px=48, max_width_frac=0.008),
    "ch1": dict(min_px=12, max_px=220, max_width_frac=0.06),
    "ch2": dict(min_px=12, max_px=220, max_width_frac=0.06),
    "ch3": dict(min_px=12, max_px=220, max_width_frac=0.06),
    # zoom windows: the event may legitimately span a large fraction of the
    # frame on slow runs, so the width ceiling is far higher.
    "ch1_zoom": dict(min_px=16, max_px=1280, max_width_frac=0.5),
    "ch2_zoom": dict(min_px=16, max_px=1280, max_width_frac=0.5),
    "ch3_zoom": dict(min_px=16, max_px=1280, max_width_frac=0.5),
}
BOX_H_FRAC = 1.0

# Stage spec in chain space. "next" bounds the canonical cut window; the
# init stage detects two classes, channel stages one.
STAGES: Dict[str, Dict] = {
    "init": {
        "targets": {"POI1": 0, "POI2": 1},
        "anchor": "POI2",
        "next": "POI3",
        "nc": 2,
        "names": {0: "poi1", 1: "poi2"},
    },
    "ch1": {"targets": {"POI3": 0}, "anchor": "POI3", "next": "POI4", "nc": 1, "names": {0: "ch1"}},
    "ch2": {"targets": {"POI4": 0}, "anchor": "POI4", "next": "POI5", "nc": 1, "names": {0: "ch2"}},
    "ch3": {"targets": {"POI5": 0}, "anchor": "POI5", "next": None, "nc": 1, "names": {0: "ch3"}},
}

# Zoom refinement stages: small windows around the target POI, re-rendered
# at full image width. At full-run scale a slow (high-viscosity) transition
# is a faint ~200 px smear; in a 8-40 s window it fills a large fraction of
# the frame. These detectors serve the post-decode refinement pass in
# v6_yolo (REFINE settings), which targets the 1-5 s error band where
# oracle@5s exceeds oracle@1s. Windows deliberately do NOT start at the run
# head - that is their distribution, unlike the cascade stages.
ZOOM_STAGES: Dict[str, Dict] = {
    "ch1_zoom": {"targets": {"POI3": 0}, "nc": 1, "names": {0: "ch1z"}},
    "ch2_zoom": {"targets": {"POI4": 0}, "nc": 1, "names": {0: "ch2z"}},
    "ch3_zoom": {"targets": {"POI5": 0}, "nc": 1, "names": {0: "ch3z"}},
}
ALL_STAGES: Dict[str, Dict] = {**STAGES, **ZOOM_STAGES}
ZOOM_W_RANGE_S = (8.0, 40.0)  # sampled window span
ZOOM_POI_FRAC = (0.25, 0.75)  # where the POI lands inside the window

# Cut-mode sampling probabilities (canonical / wide / negative).
P_CANONICAL, P_WIDE, P_NEGATIVE = 0.60, 0.25, 0.15
CUT_MARGIN_S = 0.4  # keep the anchor event fully inside positive frames
MIN_SLICE_S = 3.0


# ===========================================================================
#  Per-stage sample generation
# ===========================================================================


def _sample_cut(
    rng: np.random.Generator,
    anchor_t: Optional[float],
    next_t: Optional[float],
    t0: float,
    t1: float,
) -> Tuple[Optional[float], bool]:
    """Pick a cut time for one stage sample, matching inference's slicing.

    Samples a canonical cut (between the anchor POI and the next one), a
    wide cut (anywhere after the anchor, matching the decode layer's
    conservative harvest slices), or a negative cut (before the anchor, so
    the target is absent from the frame) according to the module-level
    ``P_CANONICAL`` / ``P_WIDE`` / ``P_NEGATIVE`` probabilities.

    Args:
        rng (numpy.random.Generator): Source of randomness for the cut draw.
        anchor_t (float, optional): Time of the stage's target POI, or
            ``None`` for a partial fill where the target never happened.
        next_t (float, optional): Time of the following stage's POI, if any.
        t0 (float): Start time of the run's usable signal.
        t1 (float): End time of the run's usable signal.

    Returns:
        Tuple[float, bool]: ``(cut_time, is_negative)``. ``cut_time`` is
        ``None`` when no usable cut exists for this run/stage combination.
    """
    if anchor_t is None:
        # partial fill: the stage's target never happened -> negative frame,
        # cut anywhere that leaves a sane slice.
        if t1 - t0 < MIN_SLICE_S:
            return None, True
        return float(rng.uniform(t0 + MIN_SLICE_S, t1)), True
    u = rng.random()
    lo = anchor_t + CUT_MARGIN_S
    if u < P_NEGATIVE:
        # negative: cut before the anchor so the target is absent. Falls
        # through to a positive cut when the run's head is too short.
        hi = anchor_t - CUT_MARGIN_S
        if hi - t0 >= MIN_SLICE_S:
            return float(rng.uniform(t0 + MIN_SLICE_S, hi)), True
    if next_t is not None and next_t > lo and u < P_NEGATIVE + P_CANONICAL:
        return float(rng.uniform(lo, next_t)), False
    # wide: anywhere after the target, matching the decode layer's
    # conservative harvest slices (and the only positive mode when there is
    # no next POI, i.e. the ch3/EOF stage).
    if t1 > lo:
        return float(rng.uniform(lo, t1)), False
    return None, False


def _render_and_label(
    df_p: pd.DataFrame,
    cut_t: float,
    stage: str,
    poi_times: Dict[str, float],
    is_negative: bool,
    t_start: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """Slice a run, render it, and emit the stage's YOLO label lines.

    Args:
        df_p (pandas.DataFrame): Preprocessed run signal.
        cut_t (float): End of the slice (exclusive).
        stage (str): Stage key into :data:`ALL_STAGES` / :data:`BOX_SPEC`.
        poi_times (Dict[str, float]): POI name -> time, for this run/variant.
        is_negative (bool): Whether this sample should carry no label boxes
            (the target POI is outside the slice).
        t_start (float, optional): Start of the slice. ``None`` gives the
            cascade's prefix slice ``[..., cut_t)``; a value gives a zoom
            window ``[t_start, cut_t)``.

    Returns:
        Tuple[numpy.ndarray, List[str]]: The rendered image and its YOLO
        label lines (empty list for a negative sample), or ``None`` when the
        slice is too short to render.
    """
    if t_start is None:
        sl = df_p[df_p[COL_TIME] < cut_t]
    else:
        sl = df_p[(df_p[COL_TIME] >= t_start) & (df_p[COL_TIME] < cut_t)]
    if len(sl) < 64:
        return None
    t = sl[COL_TIME].to_numpy(dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    span = t1 - t0
    if span < MIN_SLICE_S:
        return None
    from ..rendering.detector_render import generate_det_image

    img = generate_det_image(sl, IMG_W, IMG_H, version=RENDER_VERSION)

    lines: List[str] = []
    if not is_negative:
        spec = BOX_SPEC[stage]
        boxes: List[Tuple[int, float, float]] = []  # (cls, xc_px, w_px)
        for poi, cls_id in ALL_STAGES[stage]["targets"].items():
            pt = poi_times.get(poi)
            if pt is None or not (t0 < pt < t1):
                continue
            w_sec = dynamic_box_width_sec(sl, pt, max_width_frac=spec["max_width_frac"])
            w_px = float(np.clip(w_sec / span * IMG_W, spec["min_px"], spec["max_px"]))
            xc_px = (pt - t0) / span * IMG_W
            boxes.append((cls_id, xc_px, w_px))
        # Overlap clamp for multi-target stages (init): when two events sit
        # closer than their box widths, shrink each toward 90% of the
        # center-to-center gap so each box covers ONLY its own event. Floor
        # of 4 px keeps the box trainable even at near-coincident events.
        if len(boxes) == 2:
            gap_px = abs(boxes[1][1] - boxes[0][1])
            boxes = [(c, x, float(max(4.0, min(w, 0.9 * gap_px)))) for (c, x, w) in boxes]
        for cls_id, xc_px, w_px in boxes:
            lines.append(f"{cls_id} {xc_px / IMG_W:.6f} 0.5 {w_px / IMG_W:.6f} {BOX_H_FRAC:.4f}")
    return img, lines


# ===========================================================================
#  Build
# ===========================================================================


def build(
    raw_root: Path,
    tiers_path: Path,
    out_root: Path,
    base_variants: int = 2,
    val_frac: float = 0.15,
    repeat_cap: int = 8,
    seed: int = 7,
    limit: Optional[int] = None,
) -> None:
    """Discover runs, split them, and render the per-stage YOLO datasets.

    Writes an ``images/`` + ``labels/`` tree and a ``data.yaml`` per stage
    under ``out_root``, plus a top-level ``manifest.json`` recording the
    split, tier counts, and per-stage/split sample counts. ``out_root`` is
    replaced if it already exists.

    Args:
        raw_root (Path): Root directory of raw per-run data.
        tiers_path (Path): Path to a saved :class:`.TierScheme` (see
            :mod:`..tiers`).
        out_root (Path): Destination directory for the rendered dataset.
        base_variants (int, optional): Base number of rendered variants per
            training run, before per-tier upsampling. Defaults to 2.
        val_frac (float, optional): Target validation fraction per stratum.
            Defaults to 0.15.
        repeat_cap (int, optional): Maximum per-tier upsampling repeat
            factor (see :func:`.splitting.repeat_factor`). Defaults to 8.
        seed (int, optional): Seed for the split and all sampling. Defaults
            to 7.
        limit (int, optional): If set, only the first ``limit`` discovered
            runs are used. Defaults to None.

    Raises:
        SystemExit: If no runs are discovered under ``raw_root``.
    """
    rng = np.random.default_rng(seed)
    tiers = TierScheme.load(tiers_path)
    # Dedupe BEFORE splitting: the same physical run under two directory
    # names on opposite sides of the split is train/val leakage.
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
    for stage in ALL_STAGES:
        for split_name in ("train", "val"):
            (out_root / stage / "images" / split_name).mkdir(parents=True)
            (out_root / stage / "labels" / split_name).mkdir(parents=True)

    counts = defaultdict(int)

    def emit(stage: str, split_name: str, name: str, img: np.ndarray, lines: List[str]) -> None:
        # Rect training disables per-epoch shuffling (ultralytics), so disk
        # order IS the sample order. Hash-prefixing the filename turns the
        # sorted directory listing into a fixed pseudo-random permutation,
        # decorrelating neighbouring samples (same run / same tier) without
        # shuffle support.
        import hashlib

        h = hashlib.blake2s(f"{stage}/{name}".encode(), digest_size=4).hexdigest()
        name = f"{h}_{name}"
        cv2.imwrite(str(out_root / stage / "images" / split_name / f"{name}.png"), img)
        (out_root / stage / "labels" / split_name / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        counts[(stage, split_name, "neg" if not lines else "pos")] += 1

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
            for stage, spec in STAGES.items():
                anchor_t = poi_a.get(spec["anchor"])
                next_t = poi_a.get(spec["next"]) if spec["next"] else None
                cut_t, neg = _sample_cut(rng, anchor_t, next_t, t0, t1)
                if cut_t is None:
                    continue
                res = _render_and_label(df_p, cut_t, stage, poi_a, neg)
                if res is None:
                    continue
                img, lines = res
                emit(stage, split_name, f"{rid}_v{v}", img, lines)

            # ---- zoom samples (window around the target POI)
            for zstage, zspec in ZOOM_STAGES.items():
                target = next(iter(zspec["targets"]))
                pt = poi_a.get(target)
                span = t1 - t0
                W = float(np.clip(rng.uniform(*ZOOM_W_RANGE_S), 4.0, 0.6 * span))
                neg = rng.random() < P_NEGATIVE
                if pt is not None and (t0 < pt < t1) and not neg:
                    f = rng.uniform(*ZOOM_POI_FRAC)
                    w0 = pt - f * W
                else:
                    # negative window: anywhere that does NOT contain the POI
                    placed = False
                    for _ in range(8):
                        w0 = rng.uniform(t0, max(t0 + 1e-6, t1 - W))
                        if pt is None or not (w0 - 2.0 < pt < w0 + W + 2.0):
                            placed = True
                            break
                    if not placed:
                        continue
                    neg = True
                w0 = max(t0, w0)
                w1 = min(t1, w0 + W)
                if w1 - w0 < 4.0:
                    continue
                res = _render_and_label(df_p, w1, zstage, poi_a, neg, t_start=w0)
                if res is None:
                    continue
                img, lines = res
                emit(zstage, split_name, f"{rid}_v{v}", img, lines)

    for i, rid in enumerate(split.train_ids):
        process_run(rid, "train")
        if (i + 1) % 100 == 0:
            LOG.info("train {}/{}", i + 1, len(split.train_ids))
    for rid in split.val_ids:
        process_run(rid, "val")

    # data.yaml per stage + manifest
    for stage, spec in ALL_STAGES.items():
        names = "\n".join(f"  {k}: {v}" for k, v in spec["names"].items())
        (out_root / stage / "data.yaml").write_text(
            f"path: {(out_root / stage).resolve()}\n"
            f"train: images/train\nval: images/val\nnc: {spec['nc']}\nnames:\n{names}\n"
        )
    manifest = dict(
        render_version=RENDER_VERSION,
        seed=seed,
        val_frac=val_frac,
        base_variants=base_variants,
        repeat_cap=repeat_cap,
        tiers=tiers.labels,
        tier_counts_train={tiers.labels[k]: v for k, v in sorted(tier_counts.items())},
        n_train_runs=len(split.train_ids),
        n_val_runs=len(split.val_ids),
        train_ids=sorted(split.train_ids),
        val_ids=sorted(split.val_ids),
        sample_counts={f"{s}/{sp}/{k}": v for (s, sp, k), v in sorted(counts.items())},
    )
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["sample_counts"], indent=2))
    print(f"train runs {len(split.train_ids)} | val runs {len(split.val_ids)}")
    LOG.info("Dataset -> {}", out_root)


def main() -> None:
    """CLI entry point: parse arguments and run :func:`build`."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=paths.DATA_ROOT)
    ap.add_argument("--tiers", type=Path, default=paths.TIERS_JSON)
    ap.add_argument("--out", type=Path, default=paths.DATASETS_ROOT / "v7")
    ap.add_argument("--base-variants", type=int, default=2)
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
        val_frac=args.val_frac,
        repeat_cap=args.repeat_cap,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

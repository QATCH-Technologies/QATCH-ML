"""
QModel V7 — Top-level pipeline driver
======================================

Run::

    python -m qmodel_v7.pipeline

Or as a script::

    python pipeline.py

The pipeline reads its toggles from :mod:`config` (RUN_BUILD_DATASETS,
RUN_TRAIN, RUN_BENCHMARK, USE_SUBPIXEL_REFINE) and executes the
selected steps in order. No CLI flags — edit ``config.py`` to change
behaviour, matching the existing ``train_v6_yolo_balanced`` workflow.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import (
    BENCHMARK_OUTPUT,
    CASCADE_CHANNELS,
    DATASET_ROOT,
    RNG_SEED,
    RUNS_ROOT,
    RUN_BENCHMARK,
    RUN_BUILD_DATASETS,
    RUN_TRAIN,
    TRAIN_PROJECT,
    USE_SUBPIXEL_REFINE,
    VAL_SPLIT,
)

LOG = logging.getLogger("v7")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _summarize_plan() -> None:
    steps = []
    if RUN_BUILD_DATASETS:
        steps.append("build-datasets")
    if RUN_TRAIN:
        steps.append("train")
    if RUN_BENCHMARK:
        steps.append("benchmark")
    LOG.info("=" * 70)
    LOG.info("QModel V7 pipeline — %s", " → ".join(steps) if steps else "(no steps enabled)")
    LOG.info("=" * 70)
    LOG.info("Runs root        : %s", RUNS_ROOT)
    LOG.info("Dataset root     : %s", DATASET_ROOT)
    LOG.info("Train project    : %s", TRAIN_PROJECT)
    LOG.info("Benchmark output : %s", BENCHMARK_OUTPUT)
    LOG.info("Sub-pixel refine : %s", "ON" if USE_SUBPIXEL_REFINE else "OFF")
    LOG.info("Channels (%d):", len(CASCADE_CHANNELS))
    for c in CASCADE_CHANNELS:
        if c.slice_mode == "backward":
            slice_str = f"start→{c.cutoff_poi}"
        elif c.slice_mode == "forward":
            slice_str = f"{c.anchor_poi}→end"
        elif c.slice_mode == "windowed":
            slice_str = f"{c.anchor_poi}→{c.cutoff_poi}"
        else:
            slice_str = "full"
        LOG.info(
            "  %-15s target=%-4s  slice=%-15s  res=%dx%d  stretch=%.2f  boost=%.2f",
            c.name,
            c.target,
            slice_str,
            c.resolution.img_w,
            c.resolution.img_h,
            c.stretch_prob,
            c.high_cp_boost,
        )


def main() -> None:
    _setup_logging(verbose=False)
    _summarize_plan()

    if RUN_BUILD_DATASETS:
        from dataset_builder import build_all_datasets

        LOG.info("STEP: build-datasets")
        build_all_datasets(
            runs_root=Path(RUNS_ROOT),
            output_root=Path(DATASET_ROOT),
            channels=CASCADE_CHANNELS,
            val_split=VAL_SPLIT,
            rng_seed=RNG_SEED,
        )

    if RUN_TRAIN:
        from train import train_all

        LOG.info("STEP: train")
        train_all(
            channels=CASCADE_CHANNELS,
            dataset_root=Path(DATASET_ROOT),
            output_dir=Path(TRAIN_PROJECT),
        )

    if RUN_BENCHMARK:
        from benchmark import run_benchmark

        LOG.info("STEP: benchmark")
        run_benchmark(
            runs_root=Path(RUNS_ROOT),
            project=Path(TRAIN_PROJECT),
            output_dir=Path(BENCHMARK_OUTPUT),
            use_subpixel_refine=USE_SUBPIXEL_REFINE,
        )

    LOG.info("=" * 70)
    LOG.info("Pipeline complete.")
    LOG.info("=" * 70)


if __name__ == "__main__":
    main()

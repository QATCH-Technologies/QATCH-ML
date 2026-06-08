"""
QModel-MOIRAI pilot — Pipeline driver
=====================================

Run::

    python pipeline.py

Edit ``config.py`` to change behaviour (mirrors the v6 no-CLI-flags
convention). Steps, in order:

  1. discover runs under RUNS_ROOT  (same contract as QModel v6)
  2. stratified-by-tier split on physical runs  (no leakage)
  3. build per-run .npz cache (normalised SEQ_LEN tensors + soft targets)
  4. fine-tune MOIRAI + POI head
  5. report val localisation error (normalised AND in seconds)

The eval error is reported in normalised-time units (directly) and, to be
comparable with the YOLO |Δx-center| localisation benchmark, also converted
to seconds using each run's duration at inference time. For the pilot we
report normalised error as the primary metric since it is duration-invariant
and therefore fair across the long viscous runs that motivated this.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

import config as C

LOG = logging.getLogger("moirai")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _setup_logging()
    from data import discover_runs, stratified_split
    from dataset import build_cache, cache_path_for

    LOG.info("=" * 70)
    LOG.info("QModel-MOIRAI pilot")
    LOG.info("Runs root : %s", C.RUNS_ROOT)
    LOG.info("Cache root: %s", C.CACHE_ROOT)
    LOG.info("Output    : %s", C.OUTPUT_ROOT)
    LOG.info(
        "SEQ_LEN=%d PATCH=%d tokens=%d  channels=%s",
        C.SEQ_LEN,
        C.PATCH_SIZE,
        C.SEQ_LEN // C.PATCH_SIZE,
        ",".join(C.INPUT_CHANNELS),
    )
    LOG.info("=" * 70)

    specs = discover_runs(C.RUNS_ROOT, n_workers=8)
    if not specs:
        raise FileNotFoundError(f"No valid runs under {C.RUNS_ROOT}")

    if C.LIMIT_RUNS is not None and len(specs) > C.LIMIT_RUNS:
        rng = np.random.default_rng(C.RNG_SEED)
        idx = rng.choice(len(specs), size=C.LIMIT_RUNS, replace=False)
        specs = [specs[int(i)] for i in idx]
        LOG.info("LIMIT_RUNS=%d → using %d runs", C.LIMIT_RUNS, len(specs))

    # Split FIRST on physical runs, then cache — keeps the comparison to YOLO
    # honest (same split key) and means no sample crosses the train/val line.
    train_idx, val_idx = stratified_split(specs)
    train_specs = [specs[i] for i in train_idx]
    val_specs = [specs[i] for i in val_idx]

    if C.RUN_BUILD_CACHE:
        LOG.info("STEP: build cache (train)")
        build_cache(train_specs)
        LOG.info("STEP: build cache (val)")
        build_cache(val_specs)

    train_paths = [cache_path_for(s.run_id) for s in train_specs]
    val_paths = [cache_path_for(s.run_id) for s in val_specs]
    train_paths = [p for p in train_paths if p.exists()]
    val_paths = [p for p in val_paths if p.exists()]
    LOG.info("Usable cache: %d train, %d val", len(train_paths), len(val_paths))

    if C.RUN_TRAIN:
        from train import train as run_train

        LOG.info("STEP: train")
        best = run_train(train_paths, val_paths)
        LOG.info("Best checkpoint: %s", best)

    LOG.info("=" * 70)
    LOG.info("Pilot complete.")
    LOG.info("=" * 70)


if __name__ == "__main__":
    main()

"""
QModel v6 — Parallel dataset renderer
======================================

Hot-loop replacement for the serial variant-rendering inside
:func:`dataset_builder.build_dataset_for_channel`. Each worker process
takes one ``(RunSpec, ChannelConfig, split)`` task, preprocesses the run,
generates all variants for that channel, and writes the images and YOLO
label files.

Why ProcessPoolExecutor (not threads):
    Per-run work is dominated by scipy interpolation, scipy
    Savitzky-Golay, numpy multiscale gradients, and cv2 PNG/JPG encoding.
    Most of that releases the GIL, but the surrounding glue is Python.
    Processes give predictable speedup without GIL surprises and avoid
    the shared-state hazards of writing thousands of files concurrently
    from one Python process.

Determinism guarantee
---------------------
Worker order is not deterministic, but the OUTPUT files are. Each
worker derives its RNG from ``rng_seed XOR hash(run_id)``, so a given
``(run_id, channel, split, rng_seed)`` tuple always produces the same
variant set regardless of how many processes are used.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.1.0
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from augmentation import (
    compute_box_width_norm,
    sample_stretch_factor,
    time_stretch_df,
    viscosity_tier,
)
from config import ChannelConfig
from signal_processing import COL_TIME, preprocess_dataframe, render_detection_image

LOG = logging.getLogger("v6.parallel")


# ===========================================================================
#  Task / result containers
# ===========================================================================


@dataclass
class RenderTask:
    """One unit of work for the worker pool."""

    run_id: str
    csv_path: str
    viscosity_cP: float
    poi_times: Dict[str, float]
    channel: ChannelConfig
    output_dir: str
    split: str
    n_variants: int
    allow_stretch: bool
    rng_seed_base: int


@dataclass
class RenderResult:
    run_id: str
    n_images: int
    n_stretched: int
    error: Optional[str] = None


# ===========================================================================
#  Slicing — duplicated from dataset_builder to avoid a circular import.
# ===========================================================================


def _apply_slice(
    df: pd.DataFrame,
    cfg: ChannelConfig,
    poi_times: Dict[str, float],
) -> Optional[pd.DataFrame]:
    if cfg.slice_mode == "full":
        return df
    if cfg.slice_mode == "backward":
        if cfg.cutoff_poi not in poi_times:
            return None
        return df[df[COL_TIME] <= poi_times[cfg.cutoff_poi]].reset_index(drop=True)
    if cfg.slice_mode == "forward":
        if cfg.anchor_poi not in poi_times:
            return None
        return df[df[COL_TIME] >= poi_times[cfg.anchor_poi]].reset_index(drop=True)
    if cfg.slice_mode == "windowed":
        if cfg.anchor_poi not in poi_times or cfg.cutoff_poi not in poi_times:
            return None
        return df[
            (df[COL_TIME] >= poi_times[cfg.anchor_poi])
            & (df[COL_TIME] <= poi_times[cfg.cutoff_poi])
        ].reset_index(drop=True)
    return None


# ===========================================================================
#  Worker function — runs in a fresh process
# ===========================================================================


def _worker_render(task: RenderTask) -> RenderResult:
    """Preprocess one run, render its variants, write images + labels."""
    cfg = task.channel
    try:
        # ── Validity gates ──────────────────────────────────────────
        if cfg.target not in task.poi_times:
            return RenderResult(task.run_id, 0, 0)
        if cfg.slice_mode == "backward" and cfg.cutoff_poi not in task.poi_times:
            return RenderResult(task.run_id, 0, 0)
        if cfg.slice_mode == "forward" and cfg.anchor_poi not in task.poi_times:
            return RenderResult(task.run_id, 0, 0)
        if cfg.slice_mode == "windowed" and (
            cfg.anchor_poi not in task.poi_times or cfg.cutoff_poi not in task.poi_times
        ):
            return RenderResult(task.run_id, 0, 0)

        # ── Load + preprocess ───────────────────────────────────────
        raw = pd.read_csv(task.csv_path)
        df_full = preprocess_dataframe(raw)
        if df_full is None or df_full.empty:
            return RenderResult(task.run_id, 0, 0)

        df_base = _apply_slice(df_full, cfg, task.poi_times)
        if df_base is None or len(df_base) < 32:
            return RenderResult(task.run_id, 0, 0)

        poi_t = task.poi_times[cfg.target]
        t_min = float(df_base[COL_TIME].min())
        t_max = float(df_base[COL_TIME].max())
        if not (t_min < poi_t < t_max):
            return RenderResult(task.run_id, 0, 0)

        # ── Deterministic per-run RNG ───────────────────────────────
        rng = np.random.default_rng(
            (task.rng_seed_base ^ (hash(task.run_id) & 0x7FFFFFFF)) & 0x7FFFFFFF
        )

        output_dir = Path(task.output_dir)
        img_dir = output_dir / "images" / task.split
        lbl_dir = output_dir / "labels" / task.split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        tier_idx = viscosity_tier(task.viscosity_cP)

        n_imgs = 0
        n_stretch = 0
        seen_keys: set = set()

        for i in range(task.n_variants):
            if i == 0:
                df_view = df_base
                tag = "full"
                stretched = False
            else:
                tail_frac = float(rng.uniform(0.05, 1.0))
                cut_t = poi_t + (t_max - poi_t) * tail_frac
                df_view = df_base[df_base[COL_TIME] <= cut_t].reset_index(drop=True)
                if len(df_view) < 32:
                    continue
                tag = f"trunc{i:02d}_p{tail_frac:.2f}"

                stretched = False
                if task.allow_stretch and i >= 2 and rng.random() < cfg.stretch_prob:
                    factor = sample_stretch_factor(tier_idx, rng)
                    df_view = time_stretch_df(df_view, factor)
                    tag += f"_s{factor:.2f}"
                    stretched = True

            cut_key = round(float(df_view[COL_TIME].iloc[-1]), 2)
            if cut_key in seen_keys:
                continue
            seen_keys.add(cut_key)

            slice_t_min = float(df_view[COL_TIME].min())
            slice_t_max = float(df_view[COL_TIME].max())
            slice_dur = max(slice_t_max - slice_t_min, 1e-6)
            x_center = (poi_t - slice_t_min) / slice_dur
            if not (0.0 < x_center < 1.0):
                continue

            box_w = compute_box_width_norm(
                poi_name=cfg.target,
                slice_duration_s=slice_dur,
                viscosity_cP=task.viscosity_cP,
                img_w=cfg.resolution.img_w,
            )

            img = render_detection_image(df_view, cfg)
            if img is None:
                continue

            stem = f"{task.run_id}_{tag}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
            with open(lbl_dir / f"{stem}.txt", "w") as f:
                f.write(f"0 {x_center:.6f} 0.5 {box_w:.6f} 1.0\n")
            n_imgs += 1
            if stretched:
                n_stretch += 1

        return RenderResult(task.run_id, n_imgs, n_stretch)

    except Exception as exc:
        return RenderResult(task.run_id, 0, 0, error=str(exc))


# ===========================================================================
#  Public driver
# ===========================================================================


def render_parallel(
    tasks: List[RenderTask],
    n_workers: int,
    log_every: int = 100,
) -> Tuple[int, int, int]:
    """Render every task in parallel.

    Returns:
        (total_images, total_stretched, total_runs_used).
    """
    if not tasks:
        return 0, 0, 0

    total_imgs = 0
    total_stretched = 0
    total_used = 0
    n_done = 0
    errors: List[Tuple[str, str]] = []

    # On systems where fork is unavailable (Windows), ProcessPoolExecutor
    # uses spawn — each worker re-imports our modules. That's fine; the
    # worker function is at module top-level so it pickles cleanly.
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker_render, t): t.run_id for t in tasks}
        for fut in as_completed(futures):
            n_done += 1
            try:
                res = fut.result()
            except Exception as exc:
                errors.append((futures[fut], f"worker crash: {exc}"))
                continue

            if res.error:
                errors.append((res.run_id, res.error))
                continue
            if res.n_images > 0:
                total_used += 1
            total_imgs += res.n_images
            total_stretched += res.n_stretched

            if n_done % log_every == 0:
                LOG.info(
                    "  rendered %d / %d runs   images=%d  stretched=%d",
                    n_done,
                    len(tasks),
                    total_imgs,
                    total_stretched,
                )

    if errors:
        LOG.warning("Render errors: %d", len(errors))
        for run_id, err in errors[:5]:
            LOG.warning("  %s: %s", run_id, err)
        if len(errors) > 5:
            LOG.warning("  ... %d more suppressed", len(errors) - 5)

    return total_imgs, total_stretched, total_used

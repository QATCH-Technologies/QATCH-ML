"""
QModel v6 — Dataset builder  (parallel rendering)
==================================================

Walks a runs root, computes per-tier statistics, splits train/val
stratified by viscosity tier, and renders YOLO datasets in parallel
for every channel in :data:`config.CASCADE_CHANNELS`.

Critical leakage fix vs the 10k approach
----------------------------------------
The 10k approach upsampled high-cP runs THEN shuffled-and-split.
Variants of the same physical run leaked between train and val.

v6 inverts the order: split first by physical run, then render variants
inside each split via :mod:`parallel_renderer`. No variant can cross a
split boundary.

Hardware-optimised changes vs 7.0
---------------------------------
  * Variant rendering is now ProcessPool-parallel (RENDER_WORKERS).
  * Variant count is capped by :func:`augmentation.effective_variant_count`
    using the lower :class:`ChannelConfig.tier_cap` and ``high_cp_boost``
    defaults from v6.1.
  * ``LIMIT_RUNS`` / ``LIMIT_CHANNELS`` toggles support a fast
    "iterate on one channel" loop.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.1.0
"""

from __future__ import annotations

import io
import logging
import math
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from augmentation import (
    compute_tier_multipliers,
    effective_variant_count,
    viscosity_tier,
)
from config import (
    CASCADE_CHANNELS,
    LIMIT_CHANNELS,
    LIMIT_RUNS,
    POI_ROW_MAP,
    RENDER_WORKERS,
    RNG_SEED,
    TIER_LABELS,
    VAL_SPLIT,
    ChannelConfig,
)
from parallel_renderer import RenderTask, render_parallel

LOG = logging.getLogger("v6.dataset")


# ===========================================================================
#  RunSpec
# ===========================================================================


@dataclass
class RunSpec:
    csv_path: Path
    viscosity_cP: float
    poi_times: Dict[str, float]
    run_id: str = ""


# ===========================================================================
#  Run discovery
# ===========================================================================

_ANALYZE_ZIP_RE = re.compile(r"^analyze-(\d+)\.zip$", re.IGNORECASE)


def _find_analyze_zip(run_dir: Path) -> Optional[Path]:
    best_idx, best_path = -1, None
    try:
        for cand in run_dir.iterdir():
            m = _ANALYZE_ZIP_RE.match(cand.name)
            if m and int(m.group(1)) > best_idx:
                best_idx = int(m.group(1))
                best_path = cand
    except (FileNotFoundError, NotADirectoryError):
        return None
    return best_path


def read_run_viscosity(run_dir: Path) -> Optional[float]:
    """Mean viscosity_raw from analyze-N.zip/analyze_out.csv."""
    zip_path = _find_analyze_zip(run_dir)
    if zip_path is None:
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            inner = next(
                (n for n in zf.namelist() if n.lower().endswith("analyze_out.csv")),
                None,
            )
            if inner is None:
                return None
            with zf.open(inner) as fh:
                df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
        if "viscosity_raw" not in df.columns:
            return None
        vals = pd.to_numeric(df["viscosity_raw"], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else None
    except Exception:
        return None


def discover_runs(
    runs_root: Path,
    min_duration_sec: float = 2.0,
    n_workers: int = 8,
) -> List[RunSpec]:
    """Walk runs_root and build a RunSpec for every valid run."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise NotADirectoryError(f"runs_root not a directory: {runs_root}")

    candidates = sorted([d for d in runs_root.iterdir() if d.is_dir()])
    LOG.info("Scanning %d candidate directories under %s", len(candidates), runs_root)
    if not candidates:
        return []

    visc_map: Dict[str, Optional[float]] = {}
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
        futures = {ex.submit(read_run_viscosity, d): d for d in candidates}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                visc_map[d.name] = fut.result()
            except Exception:
                visc_map[d.name] = None

    runs: List[RunSpec] = []
    skip = {"no_files": 0, "no_pois": 0, "no_time": 0, "short": 0}

    for d in candidates:
        try:
            poi_file = next(d.glob("*_poi.csv"))
        except StopIteration:
            skip["no_files"] += 1
            continue
        try:
            data_file = next(p for p in d.glob("*.csv") if p != poi_file)
        except StopIteration:
            skip["no_files"] += 1
            continue

        try:
            poi_df = pd.read_csv(poi_file, header=None, names=["sample_index"])
        except Exception:
            skip["no_pois"] += 1
            continue

        try:
            raw_time = pd.read_csv(data_file, usecols=["Relative_time"])
        except (KeyError, ValueError):
            try:
                full = pd.read_csv(data_file)
                if "Relative_time" not in full.columns:
                    skip["no_time"] += 1
                    continue
                raw_time = full[["Relative_time"]]
            except Exception:
                skip["no_time"] += 1
                continue
        except Exception:
            skip["no_time"] += 1
            continue

        n_rows = len(raw_time)
        if n_rows < 50:
            skip["short"] += 1
            continue
        try:
            dur = float(raw_time["Relative_time"].iloc[-1] - raw_time["Relative_time"].iloc[0])
        except Exception:
            dur = 0.0
        if dur < min_duration_sec:
            skip["short"] += 1
            continue

        poi_times: Dict[str, float] = {}
        for row_idx, poi_name in POI_ROW_MAP.items():
            if row_idx >= len(poi_df):
                continue
            si = poi_df.iloc[row_idx]["sample_index"]
            if pd.isna(si):
                continue
            try:
                si_int = int(si)
            except (TypeError, ValueError):
                continue
            if 0 <= si_int < n_rows:
                poi_times[poi_name] = float(raw_time.iloc[si_int]["Relative_time"])

        if not poi_times:
            skip["no_pois"] += 1
            continue

        v = visc_map.get(d.name)
        runs.append(
            RunSpec(
                csv_path=data_file,
                viscosity_cP=float(v) if (v is not None and v > 0) else float("nan"),
                poi_times=poi_times,
                run_id=d.name,
            )
        )

    LOG.info("Discovered %d valid runs (skipped: %s)", len(runs), skip)
    n_visc = sum(1 for r in runs if not math.isnan(r.viscosity_cP))
    LOG.info(
        "Viscosity available for %d / %d runs (%.1f%%)",
        n_visc,
        len(runs),
        100.0 * n_visc / max(len(runs), 1),
    )
    return runs


# ===========================================================================
#  Stratified train/val split
# ===========================================================================


def stratified_split(
    runs: List[RunSpec],
    val_ratio: float = VAL_SPLIT,
    rng_seed: int = RNG_SEED,
) -> Tuple[List[int], List[int]]:
    """Per-tier split on PHYSICAL RUNS. Variants stay inside their split."""
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    by_tier: Dict[int, List[int]] = {}
    for i, r in enumerate(runs):
        by_tier.setdefault(viscosity_tier(r.viscosity_cP), []).append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for tier, idxs in by_tier.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_ratio))) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    LOG.info(
        "Stratified split: train=%d  val=%d  (val_ratio=%.2f)",
        len(train_idx),
        len(val_idx),
        val_ratio,
    )
    return train_idx, val_idx


# ===========================================================================
#  Per-channel dataset assembly
# ===========================================================================


def build_dataset_for_channel(
    cfg: ChannelConfig,
    runs: List[RunSpec],
    output_root: Path,
    val_split: float = VAL_SPLIT,
    rng_seed: int = RNG_SEED,
    n_workers: int = RENDER_WORKERS,
) -> Optional[Path]:
    """Render a YOLO-format detection dataset for one cascade channel."""
    output_dir = Path(output_root) / cfg.name
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Split FIRST (no variant leakage) ───────────────────────────────
    train_idx, val_idx = stratified_split(runs, val_ratio=val_split, rng_seed=rng_seed)
    tier_mult = compute_tier_multipliers([r.viscosity_cP for r in runs], cap=cfg.tier_cap)
    LOG.info(
        "Tier multipliers for %s (cap=%d): %s",
        cfg.name,
        cfg.tier_cap,
        {TIER_LABELS[k]: v for k, v in sorted(tier_mult.items())},
    )

    # ── Build task lists ───────────────────────────────────────────────
    train_tasks = _build_tasks(
        cfg,
        [runs[i] for i in train_idx],
        output_dir,
        "train",
        tier_mult,
        allow_stretch=True,
        rng_seed_base=rng_seed,
    )
    val_tasks = _build_tasks(
        cfg,
        [runs[i] for i in val_idx],
        output_dir,
        "val",
        tier_mult,
        allow_stretch=False,
        rng_seed_base=rng_seed,
    )

    # Estimated dataset size — handy for spotting variant-explosion regressions.
    est_train = sum(t.n_variants for t in train_tasks)
    est_val = sum(t.n_variants for t in val_tasks)
    LOG.info(
        "%s: estimated %d train + %d val variants (%d eligible runs)",
        cfg.name,
        est_train,
        est_val,
        len(train_tasks) + len(val_tasks),
    )

    # ── Render in parallel ─────────────────────────────────────────────
    LOG.info("%s: rendering with %d worker(s)", cfg.name, n_workers)
    n_train_img, n_train_str, _ = render_parallel(train_tasks, n_workers)
    n_val_img, _, _ = render_parallel(val_tasks, n_workers, log_every=50)

    LOG.info(
        "%s built: %d train (%d stretched), %d val images",
        cfg.name,
        n_train_img,
        n_train_str,
        n_val_img,
    )

    if n_train_img == 0:
        LOG.warning("%s: no train images produced — skipping yaml write", cfg.name)
        return None

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("# Auto-generated by qmodel_v6.dataset_builder\n")
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write(f"  0: {cfg.target}\n")
    return yaml_path


def _build_tasks(
    cfg: ChannelConfig,
    runs: List[RunSpec],
    output_dir: Path,
    split: str,
    tier_mult: Dict[int, int],
    allow_stretch: bool,
    rng_seed_base: int,
) -> List[RenderTask]:
    """Materialise one RenderTask per (run, channel, split)."""
    tasks: List[RenderTask] = []
    for r in runs:
        # Quick pre-filter — skip runs that can't satisfy this channel.
        if cfg.target not in r.poi_times:
            continue
        if cfg.slice_mode == "backward" and cfg.cutoff_poi not in r.poi_times:
            continue
        if cfg.slice_mode == "forward" and cfg.anchor_poi not in r.poi_times:
            continue
        if cfg.slice_mode == "windowed" and (
            cfg.anchor_poi not in r.poi_times or cfg.cutoff_poi not in r.poi_times
        ):
            continue

        tier_idx = viscosity_tier(r.viscosity_cP)
        n_variants = effective_variant_count(
            base=cfg.base_truncations,
            tier_multiplier=tier_mult.get(tier_idx, 1),
            viscosity_tier_idx=tier_idx,
            high_cp_boost=cfg.high_cp_boost,
        )
        tasks.append(
            RenderTask(
                run_id=r.run_id,
                csv_path=str(r.csv_path),
                viscosity_cP=r.viscosity_cP,
                poi_times=dict(r.poi_times),
                channel=cfg,
                output_dir=str(output_dir),
                split=split,
                n_variants=n_variants,
                allow_stretch=allow_stretch,
                rng_seed_base=rng_seed_base,
            )
        )
    return tasks


# ===========================================================================
#  Top-level driver
# ===========================================================================


def build_all_datasets(
    runs_root: Path,
    output_root: Path,
    channels: List[ChannelConfig] = CASCADE_CHANNELS,
    val_split: float = VAL_SPLIT,
    rng_seed: int = RNG_SEED,
    n_workers: int = RENDER_WORKERS,
    continue_on_error: bool = True,
    limit_runs: Optional[int] = LIMIT_RUNS,
    limit_channels: Optional[List[str]] = LIMIT_CHANNELS,
) -> Dict[str, Optional[Path]]:
    """Discover runs once, then build a dataset for every selected channel.

    Args:
        runs_root, output_root, val_split, rng_seed: standard.
        channels: full cascade list (the default).
        n_workers: passed through to :func:`render_parallel`.
        limit_runs: if set, randomly sample this many discovered runs.
            Useful for a "build a tiny dataset to validate the pipeline"
            iteration loop.
        limit_channels: if set, only build the listed channel names.
            Pair with ``LIMIT_RUNS`` to validate one channel end-to-end
            before committing to the full 6-channel build.
    """
    import numpy as np

    runs = discover_runs(runs_root, n_workers=8)
    if not runs:
        raise FileNotFoundError(f"No valid runs found under {runs_root}")

    if limit_runs is not None and len(runs) > limit_runs:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(runs), size=limit_runs, replace=False)
        runs = [runs[int(i)] for i in idx]
        LOG.info("LIMIT_RUNS=%d → restricted to %d runs", limit_runs, len(runs))

    if limit_channels:
        channels = [c for c in channels if c.name in limit_channels]
        LOG.info(
            "LIMIT_CHANNELS=%s → building %d channel(s)",
            limit_channels,
            len(channels),
        )

    out: Dict[str, Optional[Path]] = {}
    for i, cfg in enumerate(channels, start=1):
        slice_str = _slice_desc(cfg)
        LOG.info("=" * 70)
        LOG.info(
            "[%d/%d] %s  target=%s  slice=%s  res=%dx%d  batch=%d  stretch=%.2f  boost=%.2f",
            i,
            len(channels),
            cfg.name,
            cfg.target,
            slice_str,
            cfg.resolution.img_w,
            cfg.resolution.img_h,
            cfg.effective_batch,
            cfg.stretch_prob,
            cfg.high_cp_boost,
        )
        LOG.info("=" * 70)
        try:
            yaml_path = build_dataset_for_channel(
                cfg,
                runs,
                Path(output_root),
                val_split=val_split,
                rng_seed=rng_seed,
                n_workers=n_workers,
            )
            out[cfg.name] = yaml_path
        except Exception as exc:
            LOG.error("%s build failed: %s", cfg.name, exc, exc_info=True)
            out[cfg.name] = None
            if not continue_on_error:
                raise
    return out


def _slice_desc(cfg: ChannelConfig) -> str:
    if cfg.slice_mode == "full":
        return "full"
    if cfg.slice_mode == "backward":
        return f"start→{cfg.cutoff_poi}"
    if cfg.slice_mode == "forward":
        return f"{cfg.anchor_poi}→end"
    if cfg.slice_mode == "windowed":
        return f"{cfg.anchor_poi}→{cfg.cutoff_poi}"
    return cfg.slice_mode

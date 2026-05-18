"""
QModel V7 — Dataset builder
============================

Walks a runs root, computes per-tier statistics, splits train/val
stratified by viscosity tier, and renders YOLO datasets for every
channel in :data:`config.CASCADE_CHANNELS`.

Critical leakage fix vs the 10k approach
----------------------------------------
The 10k approach upsampled high-cP runs (made many copies) THEN shuffled
and split. Variants of the same physical run ended up in both train and
val, inflating val metrics.

v7 inverts the order: split first by physical run, then generate
variants. Every variant of a run lives in the same split as its parent.
The high-cP boost (more variants for rare-tier runs) operates within a
split, not across splits, so no leakage is possible.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.0.0
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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from augmentation import (
    compute_box_width_norm,
    compute_tier_multipliers,
    effective_variant_count,
    sample_stretch_factor,
    time_stretch_df,
    viscosity_tier,
)
from config import (
    POI_ROW_MAP,
    RNG_SEED,
    TIER_LABELS,
    VAL_SPLIT,
    CASCADE_CHANNELS,
    ChannelConfig,
)
from signal_processing import COL_TIME, preprocess_dataframe, render_detection_image

LOG = logging.getLogger("v7.dataset")


# ===========================================================================
#  RunSpec
# ===========================================================================


@dataclass
class RunSpec:
    """Metadata for one training run."""

    csv_path: Path
    viscosity_cP: float
    poi_times: Dict[str, float]
    run_id: str = ""


# ===========================================================================
#  Run discovery (mirrors train_v6_yolo_balanced.discover_runs)
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
    """Mean ``viscosity_raw`` from analyze-N.zip/analyze_out.csv."""
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
    """Walk runs_root and build a RunSpec for every valid run directory."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise NotADirectoryError(f"runs_root not a directory: {runs_root}")

    candidates = sorted([d for d in runs_root.iterdir() if d.is_dir()])
    LOG.info("Scanning %d candidate directories under %s", len(candidates), runs_root)
    if not candidates:
        return []

    # Parallel viscosity read (zip extraction is the slow part).
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
#  Stratified train / val split
# ===========================================================================


def stratified_split(
    runs: List[RunSpec],
    val_ratio: float = VAL_SPLIT,
    rng_seed: int = RNG_SEED,
) -> Tuple[List[int], List[int]]:
    """Split runs by viscosity tier. Returns (train_indices, val_indices).

    The split is on PHYSICAL RUNS. Variants are generated per-split
    downstream, so no variant of a run can leak across splits.
    """
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
#  Slicing
# ===========================================================================


def _apply_slice(
    df: pd.DataFrame,
    cfg: ChannelConfig,
    poi_times: Dict[str, float],
) -> Optional[pd.DataFrame]:
    """Slice df according to cfg.slice_mode. Returns None when invalid."""
    if cfg.slice_mode == "full":
        return df

    if cfg.slice_mode == "backward":
        if cfg.cutoff_poi not in poi_times:
            return None
        cutoff_t = poi_times[cfg.cutoff_poi]
        return df[df[COL_TIME] <= cutoff_t].reset_index(drop=True)

    if cfg.slice_mode == "forward":
        if cfg.anchor_poi not in poi_times:
            return None
        anchor_t = poi_times[cfg.anchor_poi]
        return df[df[COL_TIME] >= anchor_t].reset_index(drop=True)

    if cfg.slice_mode == "windowed":
        if cfg.anchor_poi not in poi_times or cfg.cutoff_poi not in poi_times:
            return None
        anchor_t = poi_times[cfg.anchor_poi]
        cutoff_t = poi_times[cfg.cutoff_poi]
        return df[(df[COL_TIME] >= anchor_t) & (df[COL_TIME] <= cutoff_t)].reset_index(drop=True)

    return None


# ===========================================================================
#  Per-channel dataset assembly
# ===========================================================================


def build_dataset_for_channel(
    cfg: ChannelConfig,
    runs: List[RunSpec],
    output_root: Path,
    val_split: float = VAL_SPLIT,
    rng_seed: int = RNG_SEED,
) -> Optional[Path]:
    """
    Render a YOLO-format detection dataset for one cascade channel.

    Args:
        cfg: Channel configuration.
        runs: All discovered runs.
        output_root: Output directory root (e.g. ``datasets_v7/``).
        val_split: Validation fraction (per-tier).
        rng_seed: RNG seed for reproducibility.

    Returns:
        Path to ``data.yaml`` for the rendered dataset, or ``None``
        if no images were produced.
    """
    rng = np.random.default_rng(rng_seed)

    output_dir = Path(output_root) / cfg.name
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Split FIRST (no variant leakage) ───────────────────────────────
    train_idx, val_idx = stratified_split(runs, val_ratio=val_split, rng_seed=rng_seed)
    tier_mult = compute_tier_multipliers([r.viscosity_cP for r in runs], cap=cfg.tier_cap)
    LOG.info(
        "Tier multipliers for %s: %s",
        cfg.name,
        {TIER_LABELS[k]: v for k, v in sorted(tier_mult.items())},
    )

    n_train_img = 0
    n_val_img = 0
    n_stretched = 0
    n_runs_used = 0

    for j, i in enumerate(train_idx):
        rendered = _render_run_variants(
            cfg,
            runs[i],
            output_dir,
            "train",
            rng,
            tier_mult,
            allow_stretch=True,
        )
        if rendered:
            n_runs_used += 1
            n_train_img += rendered.n_images
            n_stretched += rendered.n_stretched
        if (j + 1) % 100 == 0:
            LOG.info(
                "%s train: %d / %d runs processed (%d images, %d stretched)",
                cfg.name,
                j + 1,
                len(train_idx),
                n_train_img,
                n_stretched,
            )

    # Val: never stretch (val should reflect natural inference distribution).
    for i in val_idx:
        rendered = _render_run_variants(
            cfg,
            runs[i],
            output_dir,
            "val",
            rng,
            tier_mult,
            allow_stretch=False,
        )
        if rendered:
            n_val_img += rendered.n_images

    LOG.info(
        "%s built: %d train (%d stretched), %d val images",
        cfg.name,
        n_train_img,
        n_stretched,
        n_val_img,
    )

    if n_train_img == 0:
        LOG.warning("%s: no train images produced — skipping yaml write", cfg.name)
        return None

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("# Auto-generated by qmodel_v7.dataset_builder\n")
        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write(f"  0: {cfg.target}\n")
    return yaml_path


# ---------------------------------------------------------------------------


@dataclass
class _RenderTally:
    n_images: int
    n_stretched: int


def _render_run_variants(
    cfg: ChannelConfig,
    run: RunSpec,
    output_dir: Path,
    split: str,
    rng: np.random.Generator,
    tier_mult: Dict[int, int],
    allow_stretch: bool,
) -> Optional[_RenderTally]:
    """Render the configured number of variants for one (channel, run, split)."""
    # ── Validity gates ────────────────────────────────────────────────
    if cfg.target not in run.poi_times:
        return None
    if cfg.slice_mode == "backward" and cfg.cutoff_poi not in run.poi_times:
        return None
    if cfg.slice_mode == "forward" and cfg.anchor_poi not in run.poi_times:
        return None
    if cfg.slice_mode == "windowed" and (
        cfg.anchor_poi not in run.poi_times or cfg.cutoff_poi not in run.poi_times
    ):
        return None

    # ── Load + preprocess once per run ────────────────────────────────
    try:
        raw = pd.read_csv(run.csv_path)
    except Exception as exc:
        LOG.debug("%s read failed (%s)", run.csv_path, exc)
        return None

    df_full = preprocess_dataframe(raw)
    if df_full is None or df_full.empty:
        return None

    df_base = _apply_slice(df_full, cfg, run.poi_times)
    if df_base is None or len(df_base) < 32:
        return None

    poi_t = run.poi_times[cfg.target]
    t_min = float(df_base[COL_TIME].min())
    t_max = float(df_base[COL_TIME].max())
    if not (t_min < poi_t < t_max):
        return None

    # ── Variant count for this run ────────────────────────────────────
    tier_idx = viscosity_tier(run.viscosity_cP)
    n_variants = effective_variant_count(
        base=cfg.base_truncations,
        tier_multiplier=tier_mult.get(tier_idx, 1),
        viscosity_tier_idx=tier_idx,
        high_cp_boost=cfg.high_cp_boost,
    )

    n_imgs = 0
    n_stretch = 0
    seen_keys: set = set()

    for i in range(n_variants):
        if i == 0:
            df_view = df_base
            tag = "full"
            stretched = False
        else:
            # Truncate the post-POI tail to a random fraction in (0.05, 1.0).
            tail_frac = float(rng.uniform(0.05, 1.0))
            cut_t = poi_t + (t_max - poi_t) * tail_frac
            df_view = df_base[df_base[COL_TIME] <= cut_t].reset_index(drop=True)
            if len(df_view) < 32:
                continue
            tag = f"trunc{i:02d}_p{tail_frac:.2f}"

            # Optional stretch: train only, never on the first or second
            # variant (so every run reliably contributes at least one
            # un-augmented truncated image).
            stretched = False
            if allow_stretch and i >= 2 and rng.random() < cfg.stretch_prob:
                factor = sample_stretch_factor(tier_idx, rng)
                df_view = time_stretch_df(df_view, factor)
                tag += f"_s{factor:.2f}"
                stretched = True

        # Deduplicate near-identical truncation cuts.
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

        # Dynamic box width: physical seconds, then normalised against
        # this variant's slice duration.
        box_w = compute_box_width_norm(
            poi_name=cfg.target,
            slice_duration_s=slice_dur,
            viscosity_cP=run.viscosity_cP,
        )

        img = render_detection_image(df_view, cfg)
        if img is None:
            continue

        stem = f"{run.run_id or run.csv_path.stem}_{tag}"
        img_path = output_dir / "images" / split / f"{stem}.jpg"
        lbl_path = output_dir / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_path), img)
        with open(lbl_path, "w") as f:
            f.write(f"0 {x_center:.6f} 0.5 {box_w:.6f} 1.0\n")
        n_imgs += 1
        if stretched:
            n_stretch += 1

    return _RenderTally(n_imgs, n_stretch) if n_imgs else None


# ===========================================================================
#  Top-level driver
# ===========================================================================


def build_all_datasets(
    runs_root: Path,
    output_root: Path,
    channels: List[ChannelConfig] = CASCADE_CHANNELS,
    val_split: float = VAL_SPLIT,
    rng_seed: int = RNG_SEED,
    n_workers: int = 8,
    continue_on_error: bool = True,
) -> Dict[str, Optional[Path]]:
    """Discover runs once, then build a dataset for every channel.

    Returns a dict mapping channel name → yaml path (or None on failure).
    """
    runs = discover_runs(runs_root, n_workers=n_workers)
    if not runs:
        raise FileNotFoundError(f"No valid runs found under {runs_root}")

    out: Dict[str, Optional[Path]] = {}
    for i, cfg in enumerate(channels, start=1):
        slice_str = _slice_desc(cfg)
        LOG.info("=" * 70)
        LOG.info(
            "[%d/%d] Channel %s  target=%s  slice=%s  res=%dx%d  stretch=%.2f  boost=%.2f",
            i,
            len(channels),
            cfg.name,
            cfg.target,
            slice_str,
            cfg.resolution.img_w,
            cfg.resolution.img_h,
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

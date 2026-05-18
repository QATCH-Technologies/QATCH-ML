"""
QModel V7 — Training driver
============================

Trains every cascade channel sequentially using Ultralytics YOLO. Per-
channel hyperparameters merge in this order (later overrides earlier):

    YOLO_DEFAULTS          (config.py — sensible defaults)
    ↓
    channel.yolo_extra     (channel-specific overrides — box, dfl, …)

After each training run the absolute path to ``best.pt`` is written to
``<run_dir>/best_weights_path.txt`` so :mod:`cascade_inference` can
locate weights even if the working directory is different at inference
time.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.0.0
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
    BASE_WEIGHTS,
    CASCADE_CHANNELS,
    DATASET_ROOT,
    TRAIN_DEVICE,
    TRAIN_PROJECT,
    YOLO_DEFAULTS,
    NUM_WORKERS,
    ChannelConfig,
)

LOG = logging.getLogger("v7.train")

try:
    from ultralytics import YOLO

    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


# ===========================================================================
#  Per-channel training config
# ===========================================================================


@dataclass
class TrainConfig:
    """Resolved training configuration for one channel."""

    channel: ChannelConfig
    dataset_yaml: Path
    output_dir: Path
    base_weights: str = BASE_WEIGHTS
    workers: int = NUM_WORKERS
    device: str = TRAIN_DEVICE
    resume: bool = False

    def merged_hyperparams(self) -> Dict[str, Any]:
        params = dict(YOLO_DEFAULTS)
        params.update(self.channel.yolo_extra)
        return params


# ===========================================================================
#  Training a single channel
# ===========================================================================


def train_channel(tc: TrainConfig) -> Path:
    """Train one channel. Returns the path to its best.pt."""
    if not _HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics is not installed. Install with `pip install ultralytics`.")

    cfg = tc.channel
    tc.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = tc.output_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    hyper = tc.merged_hyperparams()
    imgsz = max(cfg.resolution.img_w, cfg.resolution.img_h)

    # Snapshot the full resolved config for traceability.
    snapshot = {
        "channel": {
            "name": cfg.name,
            "target": cfg.target,
            "slice_mode": cfg.slice_mode,
            "cutoff_poi": cfg.cutoff_poi,
            "anchor_poi": cfg.anchor_poi,
            "resolution": {
                "img_w": cfg.resolution.img_w,
                "img_h": cfg.resolution.img_h,
                "strip_h": cfg.resolution.strip_h,
                "time_strip_h": cfg.resolution.time_strip_h,
            },
            "smooth_signal_window": cfg.smooth_signal_window,
            "include_engineered_features": cfg.include_engineered_features,
            "base_truncations": cfg.base_truncations,
            "stretch_prob": cfg.stretch_prob,
            "high_cp_boost": cfg.high_cp_boost,
            "epochs": cfg.epochs,
            "batch": cfg.batch,
            "conf_threshold": cfg.conf_threshold,
        },
        "merged_hyperparams": hyper,
        "dataset_yaml": str(tc.dataset_yaml),
        "base_weights": tc.base_weights,
        "imgsz": imgsz,
    }
    with open(run_dir / "training_config.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    LOG.info("─" * 70)
    LOG.info(
        "Training %s  (target=%s, %dx%d, epochs=%d, batch=%d, device=%s)",
        cfg.name,
        cfg.target,
        cfg.resolution.img_w,
        cfg.resolution.img_h,
        cfg.epochs,
        cfg.batch,
        tc.device,
    )
    LOG.info("─" * 70)
    LOG.debug("Merged hyperparams:\n%s", json.dumps(hyper, indent=2, default=str))

    model = YOLO(tc.base_weights)
    model.train(
        data=str(tc.dataset_yaml),
        epochs=cfg.epochs,
        imgsz=imgsz,
        rect=True,
        batch=cfg.batch,
        workers=tc.workers,
        device=tc.device,
        project=str(tc.output_dir),
        name=cfg.name,
        resume=tc.resume,
        exist_ok=True,
        **hyper,
    )

    try:
        best: Path = Path(model.trainer.best)
    except AttributeError:
        best = run_dir / "weights" / "best.pt"

    sidecar = run_dir / "best_weights_path.txt"
    sidecar.write_text(str(best))
    LOG.info("Best checkpoint: %s  (sidecar: %s)", best, sidecar)
    return best


# ===========================================================================
#  Top-level driver
# ===========================================================================


def train_all(
    channels: List[ChannelConfig] = CASCADE_CHANNELS,
    dataset_root: Path = Path(DATASET_ROOT),
    output_dir: Path = Path(TRAIN_PROJECT),
    base_weights: str = BASE_WEIGHTS,
    workers: int = NUM_WORKERS,
    device: str = TRAIN_DEVICE,
    resume: bool = False,
    continue_on_error: bool = True,
) -> Dict[str, Optional[Path]]:
    """Train every cascade channel sequentially.

    Returns a dict mapping channel name → best.pt path (or None on failure).
    """
    out: Dict[str, Optional[Path]] = {}
    failures: List[str] = []

    for i, cfg in enumerate(channels, start=1):
        LOG.info("=" * 70)
        LOG.info("[%d/%d] Training channel %s", i, len(channels), cfg.name)
        LOG.info("=" * 70)

        dataset_yaml = Path(dataset_root) / cfg.name / "data.yaml"
        if not dataset_yaml.is_file():
            LOG.warning("%s: dataset yaml missing at %s — skipping", cfg.name, dataset_yaml)
            out[cfg.name] = None
            failures.append(cfg.name)
            continue

        tc = TrainConfig(
            channel=cfg,
            dataset_yaml=dataset_yaml,
            output_dir=Path(output_dir),
            base_weights=base_weights,
            workers=workers,
            device=device,
            resume=resume,
        )
        try:
            out[cfg.name] = train_channel(tc)
        except Exception as exc:
            LOG.error("%s training failed: %s", cfg.name, exc, exc_info=True)
            out[cfg.name] = None
            failures.append(cfg.name)
            if not continue_on_error:
                raise

    if failures:
        LOG.warning("train_all: %d channel(s) failed: %s", len(failures), ", ".join(failures))
    return out

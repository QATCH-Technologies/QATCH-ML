"""
QModel v6 — Single-channel training subprocess entry point
===========================================================

This module is invoked by :mod:`train` as a separate Python process for
each cascade channel. Running each channel in its own subprocess gives
us two things on Windows:

  1. CUDA context isolation. If a channel OOMs, the bad context is torn
     down when the process exits. The next channel gets a fresh CUDA
     context instead of failing in ``torch.manual_seed`` on the corrupted
     residual state.

  2. Memory release. AdamW state, gradient buffers, and any cached
     CUDA workspaces are reclaimed in full by the OS when the process
     exits — not subject to in-process ``torch.cuda.empty_cache``
     incompleteness.

Invocation::

    python -m qmodel_v6._train_subprocess \\
        --channel ch_poi5 \\
        --dataset-yaml /path/to/data.yaml \\
        --output-dir /path/to/runs/v6

Exit codes:
    0   success — best.pt written and sidecar updated.
    2   OOM — caller should consider lowering the batch size.
    3   other training error — see stderr.
    4   missing channel or dataset — see stderr.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from config import (
    AUTO_BATCH,
    AUTO_BATCH_FRACTION,
    BASE_WEIGHTS,
    CACHE_MODE,
    CACHE_RAM_LIMIT_GB,
    CASCADE_CHANNELS,
    ENABLE_AMP,
    ENABLE_COMPILE,
    NUM_WORKERS,
    PIN_MEMORY,
    TRAIN_DEVICE,
    YOLO_DEFAULTS,
    ChannelConfig,
)

LOG = logging.getLogger("v6.train_sub")


# ===========================================================================
#  Cache mode resolution
# ===========================================================================


def _resolve_cache_mode(
    dataset_yaml: Path,
    cfg: ChannelConfig,
    requested: Optional[str],
) -> Any:
    if requested is None:
        return False
    if requested == "disk":
        return "disk"
    if requested != "ram":
        LOG.warning("Unknown CACHE_MODE=%r — falling back to None", requested)
        return False

    images_dir = dataset_yaml.parent / "images" / "train"
    if not images_dir.is_dir():
        return "disk"

    n_imgs = sum(1 for _ in images_dir.glob("*.jpg"))
    bytes_per_img = cfg.resolution.img_w * cfg.resolution.img_h * 3
    estimated_gb = (n_imgs * bytes_per_img) / (1024**3)

    if estimated_gb > CACHE_RAM_LIMIT_GB:
        LOG.warning(
            "RAM cache would need ~%.1f GB > limit %.1f GB — downgrading to 'disk'",
            estimated_gb,
            CACHE_RAM_LIMIT_GB,
        )
        return "disk"
    return "ram"


# ===========================================================================
#  The actual training call
# ===========================================================================


def _train_one(
    cfg: ChannelConfig,
    dataset_yaml: Path,
    output_dir: Path,
    base_weights: str,
    workers: int,
    device: str,
    resume: bool,
) -> Path:
    from ultralytics import YOLO

    run_dir = output_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    hyper = dict(YOLO_DEFAULTS)
    hyper.update(cfg.yolo_extra)

    # imgsz must be the longer side so YOLO's internal resize logic works.
    # We keep it as a plain int — Ultralytics rect=True handles the aspect
    # ratio during actual training. But AutoBatch ALWAYS probes with square
    # (B, 3, imgsz, imgsz) tensors regardless of rect, so it overestimates
    # memory by (imgsz² / actual_area) relative to what training will use.
    # We compensate by scaling AUTO_BATCH_FRACTION DOWN by that same ratio
    # so the probe lands at the intended VRAM utilisation once rect kicks in.
    #
    # Example — MIDRES 1600×704:
    #   area_ratio  = (1600×704) / (1600×1600) = 0.44
    #   probe frac  = 0.65 × 0.44              = 0.286
    #   → AutoBatch finds a batch that uses 28.6 % of VRAM with square imgs,
    #     which corresponds to 65 % with the real 1600×704 rectangular imgs.
    imgsz = max(cfg.resolution.img_w, cfg.resolution.img_h)
    _area_ratio = (cfg.resolution.img_w * cfg.resolution.img_h) / (imgsz * imgsz)

    cache_arg = _resolve_cache_mode(dataset_yaml, cfg, CACHE_MODE)

    # Batch selection. AUTO_BATCH passes a float fraction to YOLO which
    # invokes its built-in autobatch probe; otherwise use the manual
    # fallback from effective_batch.
    if AUTO_BATCH:
        _probe_fraction = AUTO_BATCH_FRACTION * _area_ratio
        batch_arg: Any = _probe_fraction
        batch_str = (
            f"auto (target={AUTO_BATCH_FRACTION:.0%} VRAM, "
            f"probe_fraction={_probe_fraction:.1%}, "
            f"area_ratio={_area_ratio:.3f})"
        )
    else:
        batch_arg = cfg.effective_batch
        batch_str = str(cfg.effective_batch)

    LOG.info("─" * 70)
    LOG.info("Training %s  target=%s", cfg.name, cfg.target)
    LOG.info("  resolution  : %dx%d", cfg.resolution.img_w, cfg.resolution.img_h)
    LOG.info("  batch       : %s", batch_str)
    LOG.info("  base wts    : %s", base_weights)
    LOG.info("  amp         : %s", ENABLE_AMP)
    LOG.info("  cache       : %s", cache_arg)
    LOG.info("  workers     : %d  pin_memory=%s", workers, PIN_MEMORY)
    LOG.info("  device      : %s", device)
    LOG.info("─" * 70)

    # Snapshot for reproducibility
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
            "epochs": cfg.epochs,
            "batch_arg": batch_arg,
        },
        "merged_hyperparams": hyper,
        "dataset_yaml": str(dataset_yaml),
        "base_weights": base_weights,
        "imgsz": imgsz,
        "autobatch_area_ratio": _area_ratio,
        "autobatch_probe_fraction": batch_arg if AUTO_BATCH else None,
        "cache": cache_arg,
        "amp": ENABLE_AMP,
        "compile": ENABLE_COMPILE,
        "auto_batch": AUTO_BATCH,
    }
    with open(run_dir / "training_config.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    model = YOLO(base_weights)

    train_kwargs: Dict[str, Any] = dict(
        data=str(dataset_yaml),
        epochs=cfg.epochs,
        imgsz=imgsz,
        rect=True,
        batch=batch_arg,
        workers=workers,
        device=device,
        project=str(output_dir),
        name=cfg.name,
        resume=resume,
        exist_ok=True,
        cache=cache_arg,
    )
    train_kwargs.update(hyper)
    if ENABLE_COMPILE:
        train_kwargs["compile"] = True

    model.train(**train_kwargs)

    try:
        best: Path = Path(model.trainer.best)
    except AttributeError:
        best = run_dir / "weights" / "best.pt"

    (run_dir / "best_weights_path.txt").write_text(str(best))
    LOG.info("Best checkpoint: %s", best)
    return best


# ===========================================================================
#  CLI entry point
# ===========================================================================


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train one cascade channel.")
    parser.add_argument(
        "--channel", required=True, help="Channel name from CASCADE_CHANNELS (e.g. ch_poi5)"
    )
    parser.add_argument("--dataset-yaml", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=TRAIN_DEVICE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Resolve the channel config by name.
    cfg = next((c for c in CASCADE_CHANNELS if c.name == args.channel), None)
    if cfg is None:
        print(f"ERROR: channel {args.channel!r} not in CASCADE_CHANNELS", file=sys.stderr)
        return 4
    if not args.dataset_yaml.is_file():
        print(f"ERROR: dataset yaml missing: {args.dataset_yaml}", file=sys.stderr)
        return 4

    base_weights = cfg.base_weights_override or BASE_WEIGHTS

    try:
        _train_one(
            cfg=cfg,
            dataset_yaml=args.dataset_yaml,
            output_dir=args.output_dir,
            base_weights=base_weights,
            workers=args.workers,
            device=args.device,
            resume=args.resume,
        )
        return 0
    except Exception as exc:
        msg = str(exc).lower()
        # Distinguish OOM from other failures so the parent can react.
        is_oom = (
            "out of memory" in msg
            or "cuda error" in msg
            or "cudnn" in msg
            or "host_allocation" in msg
        )
        traceback.print_exc()
        return 2 if is_oom else 3


if __name__ == "__main__":
    sys.exit(main())

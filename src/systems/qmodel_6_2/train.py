"""
QModel v6 — Training orchestrator  (OOM-resilient)
===================================================

Orchestrates per-channel training. Each channel is, by default, trained
in its own subprocess (see :mod:`_train_subprocess`) so a CUDA OOM in
one channel cannot poison the CUDA context for the next.

Two modes:

  USE_SUBPROCESS_TRAINING=True   (default, recommended on Windows)
      Spawns ``python -m _train_subprocess`` per channel. Exit code
      tells us OOM vs other failure vs success.

  USE_SUBPROCESS_TRAINING=False  (in-process, useful for debugging)
      Imports and calls :func:`_train_subprocess._train_one` directly.
      The 7.1 behaviour; do not use on Windows for multi-channel runs.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.2.0
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    BASE_WEIGHTS,
    CASCADE_CHANNELS,
    DATASET_ROOT,
    LIMIT_CHANNELS,
    NUM_WORKERS,
    TRAIN_DEVICE,
    TRAIN_PROJECT,
    USE_SUBPROCESS_TRAINING,
    ChannelConfig,
)

LOG = logging.getLogger("v6.train")


# ===========================================================================
#  Subprocess driver
# ===========================================================================


def _train_channel_subprocess(
    cfg: ChannelConfig,
    dataset_yaml: Path,
    output_dir: Path,
    workers: int,
    device: str,
    resume: bool,
) -> Optional[Path]:
    """Spawn train_subprocess.py for one channel. Returns best.pt path or None.

    Exit code mapping (see :mod:`train_subprocess`):
        0  success
        2  OOM
        3  other error
        4  bad inputs (channel not found, etc.)

    Note: Python exits with code 2 when it cannot open the script file
    ("can't open file ... No such file or directory"). A pre-flight
    existence check catches this before launch so it is never
    misclassified as OOM.
    """
    # File is train_subprocess.py — no leading underscore.
    script_path = Path(__file__).parent / "train_subprocess.py"
    if not script_path.is_file():
        LOG.error(
            "%s: training script not found at %s — verify file name/location",
            cfg.name,
            script_path,
        )
        return None

    cmd = [
        sys.executable,
        str(script_path),
        "--channel",
        cfg.name,
        "--dataset-yaml",
        str(dataset_yaml),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--workers",
        str(workers),
    ]
    if resume:
        cmd.append("--resume")

    LOG.info("Subprocess: %s", " ".join(cmd))

    # Inherit env so the child sees CUDA_VISIBLE_DEVICES, PATH, conda env, etc.
    # PYTHONUNBUFFERED=1 forces line-buffered stdout/stderr so progress
    # lines stream into our log instead of buffering for minutes.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Reduce CUDA allocator fragmentation. Without this the allocator can
    # fail to find a contiguous block even when total free VRAM is sufficient,
    # turning a borderline-fit batch into a hard OOM. Safe to set globally.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            check=False,
            # Stream child output to our stdout/stderr in real time.
            stdout=None,
            stderr=None,
        )
    except FileNotFoundError as exc:
        LOG.error("%s: subprocess launch failed (%s)", cfg.name, exc)
        return None

    rc = proc.returncode
    if rc == 0:
        # Recover best.pt path from the sidecar.
        sidecar = output_dir / cfg.name / "best_weights_path.txt"
        if sidecar.is_file():
            best = Path(sidecar.read_text().strip())
            LOG.info("%s: success — best.pt at %s", cfg.name, best)
            return best
        LOG.warning(
            "%s: subprocess exited 0 but sidecar missing at %s",
            cfg.name,
            sidecar,
        )
        return None
    if rc == 2:
        # Exit code 2 from train_subprocess means CUDA OOM.
        # Python itself also exits 2 for "can't open file" errors, but
        # the pre-flight is_file() check above makes that unreachable.
        LOG.error(
            "%s: subprocess OOM. Consider:\n"
            "  - Lowering AUTO_BATCH_FRACTION from current value\n"
            "  - Setting AUTO_BATCH=False and reducing BATCH_BY_RESOLUTION\n"
            "  - Reducing NUM_WORKERS further (4 instead of 8)",
            cfg.name,
        )
        return None
    if rc == 3:
        LOG.error("%s: subprocess training error (see traceback above)", cfg.name)
        return None
    if rc == 4:
        LOG.error("%s: subprocess invocation error (bad channel name or dataset)", cfg.name)
        return None
    LOG.error("%s: subprocess exited with unexpected code %d", cfg.name, rc)
    return None


# ===========================================================================
#  In-process fallback (debugging only)
# ===========================================================================


def _train_channel_inprocess(
    cfg: ChannelConfig,
    dataset_yaml: Path,
    output_dir: Path,
    workers: int,
    device: str,
    resume: bool,
) -> Optional[Path]:
    """Run training inside this process. NOT recommended for multi-channel
    runs on Windows — a CUDA OOM corrupts the context for all subsequent
    channels."""
    from _train_subprocess import _train_one

    base_weights = cfg.base_weights_override or BASE_WEIGHTS
    try:
        return _train_one(
            cfg=cfg,
            dataset_yaml=dataset_yaml,
            output_dir=output_dir,
            base_weights=base_weights,
            workers=workers,
            device=device,
            resume=resume,
        )
    except Exception as exc:
        LOG.error("%s: in-process training failed: %s", cfg.name, exc, exc_info=True)
        return None


# ===========================================================================
#  Top-level driver
# ===========================================================================


def train_all(
    channels: List[ChannelConfig] = CASCADE_CHANNELS,
    dataset_root: Path = Path(DATASET_ROOT),
    output_dir: Path = Path(TRAIN_PROJECT),
    workers: int = NUM_WORKERS,
    device: str = TRAIN_DEVICE,
    resume: bool = False,
    continue_on_error: bool = True,
    limit_channels: Optional[List[str]] = LIMIT_CHANNELS,
) -> Dict[str, Optional[Path]]:
    """Train every cascade channel.

    Honours :data:`LIMIT_CHANNELS` so you can validate one channel
    end-to-end before committing to the full 6-channel run.
    """
    if limit_channels:
        channels = [c for c in channels if c.name in limit_channels]
        LOG.info("LIMIT_CHANNELS=%s → training %d channel(s)", limit_channels, len(channels))

    mode_label = "subprocess" if USE_SUBPROCESS_TRAINING else "in-process"
    LOG.info("Training mode: %s", mode_label)

    out: Dict[str, Optional[Path]] = {}
    failures: List[str] = []

    for i, cfg in enumerate(channels, start=1):
        LOG.info("=" * 70)
        LOG.info(
            "[%d/%d] Channel %s  (target=%s, %dx%d, epochs=%d)",
            i,
            len(channels),
            cfg.name,
            cfg.target,
            cfg.resolution.img_w,
            cfg.resolution.img_h,
            cfg.epochs,
        )
        LOG.info("=" * 70)

        dataset_yaml = Path(dataset_root) / cfg.name / "data.yaml"
        if not dataset_yaml.is_file():
            LOG.warning("%s: dataset yaml missing at %s — skipping", cfg.name, dataset_yaml)
            out[cfg.name] = None
            failures.append(cfg.name)
            continue

        if USE_SUBPROCESS_TRAINING:
            result = _train_channel_subprocess(
                cfg,
                dataset_yaml,
                Path(output_dir),
                workers=workers,
                device=device,
                resume=resume,
            )
        else:
            result = _train_channel_inprocess(
                cfg,
                dataset_yaml,
                Path(output_dir),
                workers=workers,
                device=device,
                resume=resume,
            )

        out[cfg.name] = result
        if result is None:
            failures.append(cfg.name)
            if not continue_on_error:
                LOG.error("Aborting: %s failed and continue_on_error=False", cfg.name)
                break

    if failures:
        LOG.warning("train_all: %d channel(s) failed: %s", len(failures), ", ".join(failures))
    return out

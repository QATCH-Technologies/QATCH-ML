"""
train_fill_classifier.py
========================

Trains the v7 fill-type classifier on datasets built by
build_fill_dataset.py, using YOLO26-cls at a selectable size.

Ultralytics' classify defaults are LABEL-CORRUPTING on this task
----------------------------------------------------------------
The detector work established that pixel-space geometric augmentation is
wrong when x IS time. For the classifier the stock classify augmentations
are worse than wrong - several of them silently CHANGE THE TRUE CLASS
while keeping the label:

  * RandomResizedCrop (driven by ``scale``) - a crop that clips the
    right edge of the frame removes the most recent transition ridge:
    a 3ch image becomes a 2ch image wearing a 3ch label. It also crops
    time, which the prefix-cut dataset already covers correctly (with
    labels that follow the cut).
  * ``erasing`` - random erasing can delete a ridge outright: again a
    2ch image labeled 3ch.
  * ``fliplr`` - time reversal, exactly as on the detector side.
  * HSV / auto_augment color ops - channel identity IS signal identity
    in these renders (R=dissipation, G=resonance, B=derivative energy);
    hue rotation teaches that the strips are interchangeable. They are
    not.

All of it is off. Augmentation happened in the signal domain
(v7_augment, inside build_fill_dataset.py) where every transform warps
the POI times - and therefore the state labels - exactly.

Schedule rationale (carried over from train_detectors.py)
---------------------------------------------------------
The dataset has heavy per-tier upsampling duplication, the condition under
which the auto-optimizer's lr0=0.01 made every detector stage peak early
and degrade. Same medicine: explicit SGD, gentle lr0, cosine schedule.
Images are 224x224 as saved (the exact prepare_cls_input output), so
imgsz=224 means ultralytics performs NO resize at all - the train/deploy
pixel contract survives the loader.

Usage
-----
    python train_fill_classifier.py --data-root datasets/v7_fill \
        --size s [--epochs 120] [--batch 128] [--project runs/v7_fill]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logger import get_logger

from .. import paths
from .env import StageResult, extract_metrics, setup_cuda_env

setup_cuda_env()

LOG = get_logger("qmodel_7_onyx.training.train_fill_classifier")

DEFAULT_EPOCHS = 120
DEFAULT_LR0 = 0.005
DEFAULT_PATIENCE = 30


def train(
    data_root: Path,
    size: str,
    epochs: int,
    project: Path,
    batch: int,
    seed: int,
    resume: bool,
    device: str,
) -> StageResult:
    import shutil

    from ultralytics import YOLO

    if not (data_root / "train").exists():
        raise SystemExit(f"missing {data_root}/train - run build_fill_dataset.py first")

    # Ultralytics reuses this folder in place (exist_ok=True below) rather
    # than starting clean, so a fresh (non-resume) run can otherwise inherit
    # stale plots/logs/checkpoints from a previous training session on a
    # since-regenerated dataset. Purge it first unless we're resuming, which
    # needs last.pt still in place.
    run_dir = Path(project) / f"fill_yolo26{size}"
    if not resume and run_dir.exists():
        LOG.info("[fill] purging stale run dir {}", run_dir)
        shutil.rmtree(run_dir)

    model = YOLO(f"yolo26{size}-cls.pt")
    train_return = model.train(
        data=str(data_root),
        epochs=epochs,
        imgsz=224,  # == saved image size: loader performs no resize
        batch=batch,
        device=device,
        amp=True,
        cache=False,
        workers=2,  # established Windows constraints
        seed=seed,
        deterministic=True,
        cos_lr=True,
        patience=DEFAULT_PATIENCE,
        optimizer="SGD",  # "auto" ignores lr0; name it so lr0 sticks
        lr0=DEFAULT_LR0,
        momentum=0.937,
        warmup_epochs=5.0,
        # ---- label-corrupting pixel augmentation OFF (module docstring) ----
        scale=0.0,  # RandomResizedCrop scale -> (1.0, 1.0): no crop
        erasing=0.0,
        auto_augment=None,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        crop_fraction=1.0,  # val: no center crop - evaluate the full frame
        # --------------------------------------------------------------------
        project=str(project),
        name=f"fill_yolo26{size}",
        exist_ok=True,
        resume=resume,
        plots=True,
        val=True,
    )
    best = Path(project) / f"fill_yolo26{size}" / "weights" / "best.pt"
    LOG.info("[fill] best weights -> {}", best)
    LOG.info("Ship as classifiers/fill_classifier/type_cls.pt WITH FILL_RENDER_VERSION=2:")
    LOG.info("weights and render version travel together, exactly as on the detector side.")
    return StageResult(
        stage="fill_classifier", weights_path=best, metrics=extract_metrics(train_return, model)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=paths.DATASETS_ROOT / "v7_fill")
    ap.add_argument("--size", choices=["n", "s", "m", "l", "xl"], default="s")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", type=Path, default=paths.RUNS_ROOT / "v7_fill")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    LOG.info(
        "=== training fill classifier (yolo26{}-cls, {} epochs, batch {}) ===",
        args.size,
        args.epochs,
        args.batch,
    )
    train(
        args.data_root,
        args.size,
        args.epochs,
        args.project,
        args.batch,
        args.seed,
        args.resume,
        args.device,
    )


if __name__ == "__main__":
    main()

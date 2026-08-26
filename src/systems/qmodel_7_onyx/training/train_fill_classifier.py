"""Trains the onyx fill-type classifier on datasets built by build_fill_dataset.py.

Uses YOLO26-cls at a selectable size. Ultralytics' classify defaults are
LABEL-CORRUPTING on this task. The detector work established that pixel-space
geometric augmentation is wrong when x IS time. For the classifier, the stock
classify augmentations are worse than wrong - several of them silently CHANGE
THE TRUE CLASS while keeping the label (e.g., RandomResizedCrop, erasing,
fliplr, and HSV augmentations).

Because of this, all pixel-space augmentation is turned off. Augmentation
happens in the signal domain (`augment_run`, inside `build_fill_dataset.py`) where
every transform warps the POI times and state labels exactly.

Schedule rationale: The dataset has heavy per-tier upsampling duplication,
so explicit SGD, gentle lr0, and a cosine schedule are used. Images are
224x224 as saved (no resize).

Example:
    python train_fill_classifier.py --data-root datasets/onyx_fill \\
        --size s --epochs 120 --batch 128 --project runs/onyx_fill

Attributes:
    DEFAULT_EPOCHS (int): Default maximum number of training epochs.
    DEFAULT_LR0 (float): Default initial learning rate for the SGD optimizer.
    DEFAULT_PATIENCE (int): Default early stopping patience.
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
    """Trains the fill classifier model and returns its best checkpoint.

    A fresh run directory is removed before training unless `resume` is
    enabled, preventing stale checkpoints, plots, and logs from a previous
    dataset generation from being reused accidentally. Training uses
    deterministic seeding, cosine learning-rate scheduling, and completely
    disables pixel-space augmentation.

    Args:
        data_root (Path): Root directory containing the classification dataset
            (must contain a "train" subdirectory).
        size (str): YOLO26-cls model-size suffix, such as "n", "s", "m",
            "l", or "xl".
        epochs (int): Maximum number of training epochs.
        project (Path): Directory under which the Ultralytics run directory
            is created.
        batch (int): Training batch size.
        seed (int): Random seed used for deterministic training.
        resume (bool): Whether to resume an existing run instead of removing
            its run directory and starting fresh.
        device (str): CUDA or device specification passed to Ultralytics.

    Returns:
        StageResult: Result containing the stage name ("fill_classifier"),
        path to the best checkpoint, and best-effort validation metrics.

    Raises:
        SystemExit: If the `data_root / "train"` directory does not exist,
            indicating `build_fill_dataset.py` needs to be run first.
    """
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
        # label-corrupting pixel augmentation OFF
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
        project=str(project),
        name=f"fill_yolo26{size}",
        exist_ok=True,
        resume=resume,
        plots=True,
        val=True,
    )
    best = Path(project) / f"fill_yolo26{size}" / "weights" / "best.pt"
    LOG.info("[fill] best weights -> {}", best)
    LOG.info("Ship as classifiers/fill_classifier/type_cls.pt.")
    return StageResult(
        stage="fill_classifier", weights_path=best, metrics=extract_metrics(train_return, model)
    )


def main() -> None:
    """Parses command-line arguments and executes the classifier training process.

    Command-line options control the dataset root, YOLO26 model size, epoch count,
    batch size, device, output project directory, random seed, and resume behavior.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=paths.DATASETS_ROOT / "onyx_fill")
    ap.add_argument("--size", choices=["n", "s", "m", "l", "xl"], default="s")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", type=Path, default=paths.RUNS_ROOT / "onyx_fill")
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

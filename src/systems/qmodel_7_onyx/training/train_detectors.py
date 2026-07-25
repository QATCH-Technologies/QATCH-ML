"""
train_detectors.py
==================

Trains the v7 cascade detectors (init / ch1 / ch2 / ch3) on datasets built
by build_dataset.py, using YOLO26 at a selectable size (n/s/m/l/xl), tuned
to train on a SINGLE 24 GB GPU (RTX 4090) in hours, not days.

The single-GPU pathology this fixes
-----------------------------------
Ultralytics' DetectionTrainer hard-codes ``rect=mode == "val"`` in
``build_dataset`` — the ``rect=True`` train argument is silently ignored
for the TRAINING loader. Every 2560x384 render therefore gets letterboxed
into a 2560x2560 square that is ~85% black padding: ~6.5 MP/image, ~63 GB
at batch 8, which spills past 24 GB into the Windows CUDA system-memory
fallback. The 200+ s/it and multi-day ETAs are PCIe thrashing, not compute.
(The huge initial cls_loss is the same symptom: the loss is dominated by an
ocean of padded background.)

``RectDetectionTrainer`` below overrides that one method so train batches
are genuinely rectangular (384x2560 at full res — ~1 MP, ~6x less). All
images share one aspect ratio, so rect costs nothing: a single batch shape,
no aspect bucketing. Ultralytics disables per-epoch shuffling for rect
datasets; build_dataset.py compensates by hashing sample filenames so disk
order IS a fixed random permutation.

Defaults are sized for a 4090:
  * imgsz=1536 (content ~1536x232 after rect): 0.36 MP/img. At a ~300 s run
    this is ~0.2 s/pixel of native time resolution — localization headroom
    is preserved because boxes are 10-220 px wide, the head regresses
    sub-cell centers, and final precision belongs to the fine stage +
    decode, not the coarse detector grid. Use --imgsz 2560 to train at full
    render resolution: WITH the rect fix that is ~1 MP and also fits 24 GB
    comfortably; it is simply ~2.5x slower.
  * batch=16 at imgsz=1536 (drop to 8 if you raise imgsz to 2560).
  * cache=False, workers=2 — the established Windows constraints
    (RAM-cache duplication across workers, pagefile exhaustion).

Also set the NVIDIA driver's "CUDA - Sysmem Fallback Policy" to
"Prefer No Sysmem Fallback" (NVIDIA Control Panel > Manage 3D Settings) so
any future overflow fails fast with an OOM instead of silently running 50x
slower.

Augmentation rationale (unchanged): ALL pixel-space geometric/photometric
augmentation is OFF (mosaic, mixup, copy_paste, flips, degrees, translate,
scale, shear, perspective, hsv, erasing). On these renders x IS time:
fliplr teaches time-reversal invariance, mosaic destroys global fill
context, translate/scale break the time<->pixel map. Augmentation already
happened in the signal domain (v7_augment) where labels warp exactly with
the data.

Usage
-----
    python train_detectors.py --data-root datasets/v7 --size s \
        [--stages init ch1 ch2 ch3] [--imgsz 1536] [--batch 16] \
        [--epochs 150] [--project runs/v7]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logger import get_logger

from .. import paths
from .env import StageResult, extract_metrics, setup_cuda_env

# Reduce fragmentation-driven OOMs before torch is imported (harmless if
# unsupported by the installed torch).
setup_cuda_env()

LOG = get_logger("qmodel_7_onyx.training.train_detectors")

STAGE_CHOICES = [
    "init",
    "ch1",
    "ch2",
    "ch3",
    "ch1_zoom",
    "ch2_zoom",
    "ch3_zoom",
]

# Per-stage epoch overrides: the init stage converges fast (huge support,
# easy events); channel stages need the long tail of warp-augmented epochs.
STAGE_EPOCHS = {
    "init": 100,
    "ch1": 150,
    "ch2": 175,
    "ch3": 150,
    "ch1_zoom": 120,
    "ch2_zoom": 120,
    "ch3_zoom": 120,
}

# Zoom stages need a gentler schedule than the cascade stages: locally
# normalized windows with no run-head anchor and wide variable boxes make
# the loss landscape noisier — at the cascade lr they peaked at epochs 3-4
# and then collapsed. Lower lr + shorter patience (their peaks come early,
# so a long patience just burns epochs walking downhill).
STAGE_LR0 = {"ch1_zoom": 0.0015, "ch2_zoom": 0.0015, "ch3_zoom": 0.0015}
STAGE_PATIENCE = {"ch1_zoom": 15, "ch2_zoom": 15, "ch3_zoom": 15}
DEFAULT_LR0 = 0.003
DEFAULT_PATIENCE = 30


def _make_rect_trainer():
    """Trainer subclass forcing genuinely rectangular TRAIN batches.

    Built lazily so this module imports without ultralytics installed.
    Mirrors the upstream build_dataset exactly except rect=True for both
    modes. With a single shared aspect ratio across the dataset this yields
    one fixed batch shape (e.g. 384x2560) instead of square letterboxing.
    """
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer

    class RectDetectionTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode: str = "train", batch=None):
            try:  # stride accessor name varies across ultralytics versions
                from ultralytics.utils.torch_utils import unwrap_model

                gs = max(int(unwrap_model(self.model).stride.max()), 32)
            except (ImportError, AttributeError):
                try:
                    from ultralytics.utils.torch_utils import de_parallel

                    gs = max(int(de_parallel(self.model).stride.max()), 32)
                except Exception:
                    gs = 32
            ds = build_yolo_dataset(
                self.args, img_path, batch, self.data, mode=mode, rect=True, stride=gs
            )
            if mode == "train" and getattr(ds, "augment", False):
                # The augment branch of build_transforms (v8_transforms)
                # letterboxes to a SQUARE (imgsz, imgsz) unconditionally,
                # ignoring rect batch shapes — that square padding is the
                # whole memory blow-up. All pixel-space augmentations are
                # zeroed for this task anyway (augmentation lives in the
                # signal domain), so route the train dataset through the
                # non-augment transform path, which honours rect_shape.
                ds.augment = False
                ds.transforms = ds.build_transforms(hyp=self.args)
            return ds

    return RectDetectionTrainer


def train_stage(
    data_root: Path,
    stage: str,
    size: str,
    epochs: int,
    project: Path,
    batch: int,
    imgsz: int,
    seed: int,
    resume: bool,
    device: str,
) -> StageResult:
    from ultralytics import YOLO

    data_yaml = data_root / stage / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"missing {data_yaml} — run build_dataset.py first")

    model = YOLO(f"yolo26{size}.pt")
    train_return = model.train(
        trainer=_make_rect_trainer(),  # <- the 24 GB fix; see module docstring
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        rect=True,  # honoured by val; train handled by the trainer subclass
        batch=batch,
        device=device,
        amp=True,
        cache=False,
        workers=2,
        seed=seed,
        deterministic=True,
        cos_lr=True,
        patience=STAGE_PATIENCE.get(stage, DEFAULT_PATIENCE),
        # Stability: every v7 stage peaked very early (ch1 @ epoch 8, ch3 @
        # 17) then degraded until patience fired — the signature of the
        # auto-optimizer's lr0=0.01 being too hot for a dataset with heavy
        # upsampling duplication. Pin a gentler explicit schedule; "auto"
        # ignores lr0/momentum, so name the optimizer to make lr0 stick.
        optimizer="SGD",
        lr0=STAGE_LR0.get(stage, DEFAULT_LR0),
        momentum=0.937,
        warmup_epochs=5.0,
        # ---- pixel-space augmentation OFF (see module docstring) ----
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        erasing=0.0,
        multi_scale=0.0,
        # -------------------------------------------------------------
        project=str(project),
        name=f"{stage}_yolo26{size}",
        exist_ok=True,
        resume=resume,
        plots=True,
        val=True,
    )
    best = Path(project) / f"{stage}_yolo26{size}" / "weights" / "best.pt"
    LOG.info("[{}] best weights -> {}", stage, best)
    return StageResult(stage=stage, weights_path=best, metrics=extract_metrics(train_return, model))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=paths.DATASETS_ROOT / "v7")
    ap.add_argument("--size", choices=["n", "s", "m", "l", "xl"], default="s")
    ap.add_argument("--stages", nargs="+", choices=STAGE_CHOICES, default=STAGE_CHOICES)
    ap.add_argument("--epochs", type=int, default=None, help="override per-stage defaults")
    ap.add_argument("--batch", type=int, default=16, help="use 8 at --imgsz 2560")
    ap.add_argument("--imgsz", type=int, default=1536, help="2560 = full render resolution")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", type=Path, default=paths.RUNS_ROOT / "v7")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    for stage in args.stages:
        epochs = args.epochs or STAGE_EPOCHS[stage]
        LOG.info(
            "=== training {} (yolo26{}, {} epochs, imgsz {}, batch {}) ===",
            stage,
            args.size,
            epochs,
            args.imgsz,
            args.batch,
        )
        train_stage(
            args.data_root,
            stage,
            args.size,
            epochs,
            args.project,
            args.batch,
            args.imgsz,
            args.seed,
            args.resume,
            args.device,
        )


if __name__ == "__main__":
    main()

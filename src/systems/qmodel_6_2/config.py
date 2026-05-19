"""
QModel v6 — Unified Configuration  (hardware-optimised for RTX 4090)
=====================================================================

Single source of truth for all training, dataset, and inference constants.

Hardware target
---------------
This config is tuned for a single-GPU workstation with:
  * RTX 4090 (24 GB VRAM)
  * 32-core CPU class
  * 128 GB system RAM
  * NVMe scratch disk

Knobs that ship from here:
  * Auto-batch discovery       (AUTO_BATCH, AUTO_BATCH_FRACTION)
  * Per-resolution batch caps  (manual fallback only)
  * Capped variant explosion   (tier_cap, high_cp_boost)
  * Mixed precision + caching  (ENABLE_AMP, CACHE_MODE)
  * Parallel dataset rendering (RENDER_WORKERS)
  * Subprocess channel isolation (USE_SUBPROCESS_TRAINING)

Important lessons from v6.1 → v6.2
----------------------------------
* On Windows, a CUDA OOM corrupts the CUDA context for the rest of the
  Python process. Every subsequent ``torch.manual_seed`` (~8 bytes)
  fails. Subprocess-per-channel is the only reliable fix.
* WDDM happily spills GPU allocations to system RAM ("shared GPU memory"
  in Task Manager). Training appears to work but runs at 50–150× slower
  per iteration. Conservative batch sizing avoids this trap.
* 32 dataloader workers × ``pin_memory=True`` exhausts the page-locked
  host pool and triggers ``CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION``.
  Keep ``NUM_WORKERS`` ≤ 8 on Windows.

See HARDWARE.md for the full troubleshooting walkthrough.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.2.0  (OOM-resilient iteration of 7.1.0)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ===========================================================================
#  POI taxonomy and run discovery
# ===========================================================================

POI_NAMES: Tuple[str, ...] = ("POI1", "POI2", "POI3", "POI4", "POI5")
POI_ROW_MAP: Dict[int, str] = {0: "POI1", 1: "POI2", 3: "POI3", 4: "POI4", 5: "POI5"}


# ===========================================================================
#  Viscosity tiers
# ===========================================================================

TIER_EDGES: Tuple[float, ...] = (0.0, 2.5, 5.0, 10.0, 22.0, 150.0, float("inf"))
TIER_LABELS: Tuple[str, ...] = (
    "<2.5 cP",
    "2.5-5 cP",
    "5-10 cP",
    "10-22 cP",
    "22-150 cP",
    "150+ cP",
)


# ===========================================================================
#  Standardised dt resampling
# ===========================================================================

TARGET_DT_SEC: float = 0.005


# ===========================================================================
#  Difference curve
# ===========================================================================

DIFF_FACTOR: float = 2.50
BASELINE_START_SEC: float = 0.5
BASELINE_END_SEC: float = 2.5
BASELINE_OFFSET_SEC: float = 2.0


# ===========================================================================
#  Engineered feature channels
# ===========================================================================

FINE_SCALES_PX: Tuple[int, ...] = (7, 25)
COARSE_SCALES_PX: Tuple[int, ...] = (50, 101)
ALL_SCALES_PX: Tuple[int, ...] = (7, 25, 75)
LOCAL_NORM_WIN_PX: int = 150


# ===========================================================================
#  Image layout
# ===========================================================================


@dataclass(frozen=True)
class ResolutionPreset:
    img_w: int
    strip_h: int
    time_strip_h: int

    @property
    def img_h(self) -> int:
        """Computed from the global strip-visibility toggles in this module."""
        # Import lazily to avoid a circular-reference at class-definition time.
        # At runtime these are always already defined.
        import sys

        cfg = sys.modules[__name__]
        n_signal_strips = 3  # Diss / Freq / Diff — always on
        if getattr(cfg, "RENDER_ENGINEERED_STRIP", True):
            n_signal_strips += 1
        h = n_signal_strips * self.strip_h
        if getattr(cfg, "RENDER_TIME_STRIP", True):
            h += self.time_strip_h
        return h


# ===========================================================================
#  Strip visibility toggles
# ===========================================================================
# Turning a strip off reclaims its vertical pixels for the signal strips.
# ResolutionPreset.img_h reads these at call time so every downstream
# path (rendering, dataset building, YOLO imgsz) automatically adjusts.
#
#   RENDER_ENGINEERED_STRIP  — Strip 3: multiscale gradient heatmap
#                              (diff_pos / diff_neg_fine / diff_neg_coarse).
#                              Turn off to give more vertical resolution to
#                              the three raw-signal strips.
#
#   RENDER_TIME_STRIP        — Strip 4: left→right luminance ramp that
#                              encodes absolute time position within the
#                              slice. Turn off if positional context is
#                              already implicit from slice_mode.

RENDER_ENGINEERED_STRIP: bool = False
RENDER_TIME_STRIP: bool = False


# ===========================================================================
#  Tick-mark overlay (burned into the rendered images)
# ===========================================================================
# When enabled, thin tick lines are drawn on the image in a neutral grey
# so the model can use temporal / amplitude landmarks without them
# dominating the signal. Both axes are independently toggleable.
#
#   RENDER_X_TICKS           — Vertical tick lines at fixed time intervals.
#   X_TICK_INTERVAL_SEC      — Physical seconds between X ticks.
#
#   RENDER_Y_TICKS           — Horizontal tick lines at fixed fractional
#                              positions within each signal strip.
#   Y_TICK_FRACTIONS         — Normalised [0, 1] positions within each
#                              strip height at which Y ticks are drawn
#                              (0 = top edge, 1 = bottom edge).
#
#   TICK_COLOR_BGR           — BGR colour tuple for all tick marks.
#   TICK_THICKNESS           — Line thickness in pixels.

RENDER_X_TICKS: bool = True
X_TICK_INTERVAL_SEC: float = 10.0  # one vertical tick every 10 s

RENDER_Y_TICKS: bool = True
Y_TICK_FRACTIONS: Tuple[float, ...] = (0.25, 0.50, 0.75)  # quartile lines

TICK_COLOR_BGR: Tuple[int, int, int] = (80, 80, 80)
TICK_THICKNESS: int = 1


HIRES_PRESET = ResolutionPreset(img_w=2560, strip_h=160, time_strip_h=64)
MIDRES_PRESET = ResolutionPreset(img_w=1600, strip_h=160, time_strip_h=64)
ZOOM_PRESET = ResolutionPreset(img_w=1280, strip_h=160, time_strip_h=64)


# ===========================================================================
#  Dynamic bounding-box sizing
# ===========================================================================


@dataclass(frozen=True)
class BoxSizeProfile:
    """Pixel-space bounding-box target for one POI class.

    Design rationale
    ----------------
    The previous ``base_seconds``-based approach divided a fixed physical
    time by the slice duration to get the normalised width.  This made
    pixel box size *inversely* proportional to run length — a 5 s feature
    produced 267 px in a 30 s run but only 40 px in a 200 s run.  YOLO
    learns in pixel space, so inconsistent pixel widths hurt training.

    The new approach targets a *pixel width* directly (``base_px``,
    calibrated at ``reference_img_w=1600`` px) and applies two scalings:

    1. **Resolution scaling** — ``base_px × (img_w / 1600)`` keeps the
       normalised fraction constant across HIRES / MIDRES / ZOOM presets.

    2. **Duration boost** — a sub-linear power-law ``(duration / ref)^exp``
       captures the physical reality that longer runs genuinely have broader
       features (higher-viscosity runs are slower to fill *and* run longer).
       Exponent < 1 keeps the growth gentle; the result is clamped to
       ``[0.5, 2.5]`` so extreme outliers don't dominate.

    Result: within a viscosity tier, pixel box width is roughly constant
    across run lengths (the floor), and grows modestly for very long runs
    (the boost).  Shorter runs get proportionally smaller boxes because the
    duration ratio < 1 shrinks the base.
    """

    poi_name: str
    base_px: int  # target pixel width @ ref_img_w=1600, ref_duration, tier 0
    tier_multipliers: Tuple[float, ...] = (1.0, 1.0, 1.2, 1.5, 2.5, 4.0)
    min_px: int = 20  # absolute pixel floor (detectability guarantee)
    max_norm_width: float = 0.20  # normalised ceiling (prevents box > 20% of image)
    reference_duration_s: float = 30.0  # slice duration at which base_px was calibrated
    duration_exponent: float = 0.30  # power-law exponent; 0 = no scaling, 1 = linear

    def target_pixel_width(
        self,
        viscosity_tier_idx: int,
        img_w: int,
        slice_duration_s: float,
    ) -> float:
        """Return the raw target pixel width before clamping.

        Args:
            viscosity_tier_idx: 0-indexed tier from :func:`viscosity_tier`.
            img_w: Pixel width of the rendered image (resolution-dependent).
            slice_duration_s: Duration of the rendered time slice in seconds.

        Returns:
            Float pixel width.  Caller normalises by ``img_w`` and clips.
        """
        idx = max(0, min(viscosity_tier_idx, len(self.tier_multipliers) - 1))

        # 1. Resolution scaling: keep the fraction constant across presets
        res_scaled = self.base_px * (img_w / 1600.0)

        # 2. Viscosity-tier multiplier
        tier_scaled = res_scaled * self.tier_multipliers[idx]

        # 3. Duration boost: sub-linear, bounded so outliers don't explode
        duration_ratio = max(0.1, slice_duration_s / self.reference_duration_s)
        duration_boost = min(2.5, max(0.5, duration_ratio**self.duration_exponent))

        return float(tier_scaled * duration_boost)


# ---------------------------------------------------------------------------
# base_px calibration notes (at MIDRES 1600px, reference_duration_s run):
#
#   POI1 / POI2  Sharp baseline events.  At low-cP these are sub-second
#                transitions; 40 px ≈ 2.5 % of the image gives the model
#                a clear signal without an over-wide box that dilutes the
#                centroid signal. reference_duration_s=25 s (typical short
#                low-cP run); duration_exponent=0.25 (gentle scaling —
#                baseline events don't lengthen much with viscosity).
#
#   POI3 / POI4  Loading events.  Broader in seconds than POI1/2.
#                80 px ≈ 5 % at reference 40 s run.
#
#   POI5         Filling event — the widest feature.  100 px ≈ 6.25 %
#                at reference 50 s run; tier5 multiplier ×5 → 500 px raw
#                but max_norm_width=0.20 clips to 320 px (20 % of 1600).
#                duration_exponent=0.35 gives slightly stronger scaling
#                because filling duration is tightly coupled to viscosity.
# ---------------------------------------------------------------------------
BOX_SIZE_PROFILES: Dict[str, BoxSizeProfile] = {
    "POI1": BoxSizeProfile(
        "POI1",
        base_px=40,
        tier_multipliers=(1.0, 1.0, 1.2, 1.5, 2.0, 3.0),
        min_px=16,
        max_norm_width=0.08,
        reference_duration_s=25.0,
        duration_exponent=0.25,
    ),
    "POI2": BoxSizeProfile(
        "POI2",
        base_px=40,
        tier_multipliers=(1.0, 1.0, 1.2, 1.5, 2.0, 3.0),
        min_px=16,
        max_norm_width=0.08,
        reference_duration_s=25.0,
        duration_exponent=0.25,
    ),
    "POI3": BoxSizeProfile(
        "POI3",
        base_px=80,
        tier_multipliers=(1.0, 1.0, 1.2, 1.5, 2.5, 4.0),
        min_px=28,
        max_norm_width=0.20,
        reference_duration_s=40.0,
        duration_exponent=0.30,
    ),
    "POI4": BoxSizeProfile(
        "POI4",
        base_px=80,
        tier_multipliers=(1.0, 1.0, 1.2, 1.5, 2.5, 4.0),
        min_px=28,
        max_norm_width=0.20,
        reference_duration_s=40.0,
        duration_exponent=0.30,
    ),
    "POI5": BoxSizeProfile(
        "POI5",
        base_px=100,
        tier_multipliers=(1.0, 1.0, 1.2, 1.5, 3.0, 5.0),
        min_px=36,
        max_norm_width=0.20,
        reference_duration_s=50.0,
        duration_exponent=0.35,
    ),
}


# ===========================================================================
#  HARDWARE OPTIMISATION
# ===========================================================================
# v6.2 changes vs 7.1 (OOM crash fixes):
#
#   * AUTO_BATCH=True            Ultralytics probes for the right batch
#                                per channel instead of using fixed sizes
#                                that proved too large on a 4090.
#   * USE_SUBPROCESS_TRAINING    OOM in one channel won't poison the rest.
#   * NUM_WORKERS=8              Was 32 — too many for Windows pinned pool.
#   * PIN_MEMORY=False           Avoid the cuDNN host-alloc failure.
#   * Manual batch fallbacks     Halved across the board. Used only if
#                                AUTO_BATCH is turned off.
# ---------------------------------------------------------------------------

ENABLE_AMP: bool = True
ENABLE_COMPILE: bool = False

# Auto-batch via Ultralytics' built-in probe. When AUTO_BATCH is True,
# YOLO passes ``batch=AUTO_BATCH_FRACTION`` to ``model.train``, which
# tells Ultralytics to use that fraction of GPU memory and probe upward.
# This is dramatically more reliable than picking numbers by hand.
#
# Set AUTO_BATCH=False to use the manual values in BATCH_BY_RESOLUTION
# (e.g. for reproducible CI runs where you want a fixed batch).
AUTO_BATCH: bool = False
AUTO_BATCH_FRACTION: float = 0.65  # 65% of 24 GB ≈ 15.5 GB ceiling

# Subprocess-per-channel training. Strongly recommended on Windows.
# Each channel trains in a fresh Python process, so a CUDA OOM tears
# the process down cleanly and the next channel gets a fresh CUDA
# context. Costs ~5 s of startup overhead per channel.
USE_SUBPROCESS_TRAINING: bool = True

# YOLO dataloader cache mode.
#   None    : decode every epoch (slow, no extra disk)
#   'disk'  : decode once → .npy alongside the JPG (default)
#   'ram'   : decompress to RAM
CACHE_MODE: Optional[str] = "disk"
CACHE_RAM_LIMIT_GB: float = 80.0

# Dataset-build parallelism. 32-core box → 16 workers leaves room for I/O.
RENDER_WORKERS: int = min(16, max(1, (os.cpu_count() or 4) - 2))

# Train-time dataloader workers. Was 32 in v6.1 — too many for Windows;
# pinned host memory pool exhausted and cuDNN crashed. 8 is the sweet
# spot for a single-GPU workstation on Windows.
NUM_WORKERS: int = 8

# Pinned memory — disabled by default. With 32 workers and pin_memory=True
# the host pool was exhausting before the dataloader could feed the GPU.
# At 8 workers you may be able to re-enable this for a 5–10 % speedup,
# but only after baseline metrics are stable.
PIN_MEMORY: bool = False


# ---------------------------------------------------------------------------
# Manual batch fallbacks. ONLY used when AUTO_BATCH=False.
#
# These are conservative — they leave ~6 GB of headroom on a 4090.
# v6.1's values were too aggressive for rect=True training at non-square
# aspect ratios; the actual safe ceiling is roughly half of what I
# estimated. The auto-batch path (recommended) ignores these.
# ---------------------------------------------------------------------------
BATCH_BY_RESOLUTION: Dict[str, int] = {
    "HIRES": 16,  # was 6  — ~13.5 GB on 2560×704; 3 is possible but probe first
    "MIDRES": 16,  # was 10 — ~16.8 GB on 1600×704; batch=5 hits ~21 GB (OOM risk)
    "ZOOM": 16,  # was 16 — ~19.1 GB on 1280×704; 8 would exceed 24 GB
}


def batch_for_resolution(preset: ResolutionPreset) -> int:
    if preset.img_w >= 2400:
        return BATCH_BY_RESOLUTION["HIRES"]
    if preset.img_w >= 1500:
        return BATCH_BY_RESOLUTION["MIDRES"]
    return BATCH_BY_RESOLUTION["ZOOM"]


# ===========================================================================
#  Per-channel cascade configuration
# ===========================================================================


@dataclass
class ChannelConfig:
    name: str
    target: str

    slice_mode: str = "full"
    cutoff_poi: Optional[str] = None
    anchor_poi: Optional[str] = None

    resolution: ResolutionPreset = field(default_factory=lambda: MIDRES_PRESET)
    smooth_signal_window: int = 0
    include_engineered_features: bool = True

    # Augmentation
    base_truncations: int = 4
    stretch_prob: float = 0.4
    tier_cap: int = 8
    high_cp_boost: float = 2.0

    # Training
    epochs: int = 60
    batch: int = 0
    conf_threshold: float = 0.20
    base_weights_override: Optional[str] = None

    yolo_extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_batch(self) -> int:
        """Manual batch fallback (only used when AUTO_BATCH=False).

        When ``batch`` is explicitly set on the channel, that wins.
        Otherwise we fall back to the per-resolution table.
        """
        if self.batch and self.batch > 0:
            return self.batch
        return batch_for_resolution(self.resolution)

    @property
    def dataset_dir(self) -> str:
        return f"datasets_v6/{self.name}"

    @property
    def dataset_yaml(self) -> str:
        return f"{self.dataset_dir}/data.yaml"


CASCADE_CHANNELS: List[ChannelConfig] = [
    ChannelConfig(
        name="ch_poi5",
        target="POI5",
        slice_mode="full",
        resolution=MIDRES_PRESET,
        base_truncations=5,
        stretch_prob=0.5,
        high_cp_boost=2.0,
        smooth_signal_window=15,
        epochs=60,
        conf_threshold=0.15,
        yolo_extra={"box": 14.0, "cls": 0.4, "dfl": 2.0, "patience": 25},
    ),
    ChannelConfig(
        name="ch_poi4",
        target="POI4",
        slice_mode="backward",
        cutoff_poi="POI5",
        resolution=MIDRES_PRESET,
        base_truncations=5,
        stretch_prob=0.5,
        high_cp_boost=2.0,
        smooth_signal_window=11,
        epochs=60,
        conf_threshold=0.15,
        yolo_extra={
            "box": 16.0,
            "cls": 0.5,
            "dfl": 2.5,
            "patience": 30,
            "weight_decay": 0.003,
            "dropout": 0.20,
        },
    ),
    ChannelConfig(
        name="ch_poi3",
        target="POI3",
        slice_mode="backward",
        cutoff_poi="POI4",
        resolution=MIDRES_PRESET,
        base_truncations=5,
        stretch_prob=0.5,
        high_cp_boost=2.0,
        smooth_signal_window=11,
        epochs=60,
        conf_threshold=0.15,
        yolo_extra={"box": 14.0, "cls": 0.4, "dfl": 2.0, "patience": 25},
    ),
    ChannelConfig(
        name="ch_poi2",
        target="POI2",
        slice_mode="backward",
        cutoff_poi="POI3",
        resolution=HIRES_PRESET,
        base_truncations=4,
        stretch_prob=0.3,
        high_cp_boost=2.5,
        smooth_signal_window=0,
        epochs=60,
        conf_threshold=0.20,
        yolo_extra={"box": 20.0, "dfl": 2.5, "patience": 25},
    ),
    ChannelConfig(
        name="ch_poi1",
        target="POI1",
        slice_mode="backward",
        cutoff_poi="POI3",
        resolution=HIRES_PRESET,
        base_truncations=4,
        stretch_prob=0.3,
        high_cp_boost=2.5,
        smooth_signal_window=0,
        epochs=60,
        conf_threshold=0.20,
        yolo_extra={"box": 20.0, "dfl": 2.5, "patience": 25},
    ),
    ChannelConfig(
        name="ch_poi5_fine",
        target="POI5",
        slice_mode="forward",
        anchor_poi="POI4",
        resolution=ZOOM_PRESET,
        base_truncations=4,
        stretch_prob=0.4,
        high_cp_boost=1.5,
        smooth_signal_window=11,
        conf_threshold=0.40,
        epochs=40,
        base_weights_override="yolo26s.pt",
        yolo_extra={"box": 14.0, "dfl": 2.0, "patience": 20},
    ),
]


# ===========================================================================
#  YOLO training defaults
# ===========================================================================
YOLO_DEFAULTS: Dict[str, Any] = {
    "box": 12.0,
    "cls": 0.5,
    "dfl": 2.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "degrees": 0.0,
    "perspective": 0.0,
    "shear": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "translate": 0.02,
    "scale": 0.0,
    "erasing": 0.0,
    "auto_augment": None,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 3.0,
    "weight_decay": 0.002,
    "dropout": 0.15,
    "patience": 25,
    "cos_lr": True,
    "save_period": 10,
    "amp": ENABLE_AMP,
    "pretrained": True,
}


# ===========================================================================
#  Cascade inference gates
# ===========================================================================

FINE_MIN_CONF: float = 0.50
FINE_MAX_DISP_FRAC: float = 0.15
FINE_LOG_DECISIONS: bool = False


# ===========================================================================
#  Paths and pipeline toggles
# ===========================================================================

RUNS_ROOT: str = "data/raw"
DATASET_ROOT: str = "data/datasets"
TRAIN_PROJECT: str = "runs/v6"
BENCHMARK_OUTPUT: str = "test/benchmark"

BASE_WEIGHTS: str = "yolo26s.pt"
NUM_WORKERS: int = 32
TRAIN_DEVICE: str = "0"

VAL_SPLIT: float = 0.15
RNG_SEED: int = 42

RUN_BUILD_DATASETS: bool = True
RUN_TRAIN: bool = True
RUN_BENCHMARK: bool = True
USE_SUBPIXEL_REFINE: bool = True

BENCHMARK_N_RUNS: Optional[int] = None
BENCHMARK_GROSS_THRESHOLD: float = 5.0
# Iteration aids — cap runs and/or channels for fast validation loops.
LIMIT_RUNS: Optional[int] = None  # None = use all discovered runs
LIMIT_CHANNELS: Optional[List[str]] = None  # e.g. ["ch_poi5"] for single-channel

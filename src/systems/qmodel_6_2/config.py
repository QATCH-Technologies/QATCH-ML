"""
QModel V7 — Unified Configuration
==================================

Single source of truth for all training, dataset, and inference constants.

Combines best ideas from three prior iterations:

  10k approach
      Reverse cascade architecture, EOF detector on POI5. POI3/POI4 fine
      detectors found marginal vs coarse; POI5-fine is the cascade-
      stabilising win and stays.

  Balanced approach
      Viscosity-stratified train/val split (no leakage), time-axis
      stretching to fill inter-tier gaps in event morphology, per-tier
      run multipliers.

  Zoom approach
      Engineered composite feature channels: multiscale signed gradients
      of the Difference curve, split into fine-scale (sharp events for
      POI1/2 low-cP) and coarse-scale (slow events for POI1/2 high-cP).

Plus the four bucket-list items:

  1. Dynamic bounding boxes sized in PHYSICAL TIME (seconds), not
     normalised width. A 60 s high-cP fill gets a 60-second box; a 2 s
     low-cP fill gets a 2-second box. Both still produce the same
     normalised x_center; only the width adapts.
  2. All gradients / derivatives are computed w.r.t. ``Relative_time``,
     not sample index. Preprocessing resamples every run to a uniform
     dt grid so cross-rate runs are commensurable.
  3. Per-POI image_width: 2560 px for POI1/POI2 (sharp events need
     lateral resolution), 1600 px for POI3-5 (sub-second resolution
     sufficient). Strip height bumped to 160 px from 128 to avoid the
     feature flattening Paul observed on POI3/POI4 channel curves.
  4. High-viscosity oversampling via variant multiplication: the rare
     high-cP runs (~11 in the dataset at 150+ cP) get more truncations
     and more stretches so the model sees more high-cP morphology than
     a 1:1 run count would imply, without resorting to the
     post-shuffle leakage of the 10k approach.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ===========================================================================
#  POI taxonomy and run discovery
# ===========================================================================

POI_NAMES: Tuple[str, ...] = ("POI1", "POI2", "POI3", "POI4", "POI5")

# Row index in <run>_poi.csv → POI name.
POI_ROW_MAP: Dict[int, str] = {
    0: "POI1",
    1: "POI2",
    3: "POI3",
    4: "POI4",
    5: "POI5",
}


# ===========================================================================
#  Viscosity tiers — for stratification and per-tier augmentation
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
# Every run is resampled to this uniform sample interval before feature
# computation and rendering. Different QATCH devices ship with different
# sampling rates (typically 50-200 Hz); without resampling, the same
# multi-scale gradient window covers a different number of physical
# seconds across runs. 0.005 s (200 Hz) is the v6 default and is fine
# enough to preserve sharp POI1/POI2 events while keeping memory
# manageable on 600 s+ high-cP runs.
#
# Setting this here also makes "all gradients/derivatives are w.r.t.
# Relative_time" automatic: once the dt is uniform, sample-index
# gradients and Relative_time gradients are exactly proportional.

TARGET_DT_SEC: float = 0.005  # 200 Hz uniform — matches v6 preprocess


# ===========================================================================
#  Difference curve
# ===========================================================================
# Locked by the config_c_analysis sweep at 2.50. Kept overridable per
# channel for ablation work.

DIFF_FACTOR: float = 2.50
BASELINE_START_SEC: float = 0.5
BASELINE_END_SEC: float = 2.5
BASELINE_OFFSET_SEC: float = 2.0


# ===========================================================================
#  Engineered feature channels
# ===========================================================================
# These are the core "zoom approach" win: instead of feeding the raw
# Difference signal and asking YOLO to learn its derivative implicitly,
# precompute the multiscale signed gradient up front and present three
# pre-decomposed bands as a dedicated strip in the rendered image.
#
#   diff_pos          : positive gradient of Difference, all scales.
#                       Loading-direction events (POI3 / POI4 / POI5).
#   diff_neg_fine     : negative gradient, fine smoothing scales.
#                       Sharp baseline-event boundaries (POI1/POI2 on
#                       low-cP runs).
#   diff_neg_coarse   : negative gradient, coarse smoothing scales.
#                       Slow baseline-event boundaries (POI1/POI2 on
#                       high-cP runs — the failure mode of v6).
#
# Smoothing window sizes are pixels of the RENDERED image. After dt
# resampling and per-channel interp to img_w, pixel position is exactly
# proportional to physical time, so a 7-pixel window is the same
# physical span regardless of run length within a single channel config.

FINE_SCALES_PX: Tuple[int, ...] = (7, 25)
COARSE_SCALES_PX: Tuple[int, ...] = (50, 101)
ALL_SCALES_PX: Tuple[int, ...] = (7, 25, 75)  # for diff_pos

# Adaptive normalisation: rolling p90 over this window keeps each
# channel calibrated against local activity rather than global maxima.
LOCAL_NORM_WIN_PX: int = 150


# ===========================================================================
#  Image layout (5-strip canvas)
# ===========================================================================
# Each detection image is composed of horizontal strips:
#
#     Strip 0 : Dissipation                  (R channel filled)
#     Strip 1 : Resonance Frequency          (G channel filled)
#     Strip 2 : Difference                   (B channel filled)
#     Strip 3 : Engineered feature channels  (R = diff_pos,
#                                             G = diff_neg_fine,
#                                             B = diff_neg_coarse)
#     Strip 4 : Time-position gradient       (black → white horizontal)
#
# Strip 3 is the new ingredient relative to the balanced approach.
# Strip 4 (time gradient) is the explicit positional encoding that
# survives stride-32 pooling.


@dataclass(frozen=True)
class ResolutionPreset:
    """Pixel geometry for one cascade channel."""

    img_w: int
    strip_h: int
    time_strip_h: int

    @property
    def img_h(self) -> int:
        # 4 signal/feature strips + 1 time-gradient strip
        return 4 * self.strip_h + self.time_strip_h


# Sharp-event channels (POI1, POI2): high lateral resolution.
HIRES_PRESET = ResolutionPreset(
    img_w=2560,
    strip_h=160,
    time_strip_h=64,
)
# Slow-event channels (POI3, POI4, POI5): balanced.
MIDRES_PRESET = ResolutionPreset(
    img_w=1600,
    strip_h=160,
    time_strip_h=64,
)
# Fine refinement channels (run on small slices).
ZOOM_PRESET = ResolutionPreset(
    img_w=1280,
    strip_h=160,
    time_strip_h=64,
)


# ===========================================================================
#  Dynamic bounding-box sizing
# ===========================================================================
# Box width is specified in PHYSICAL SECONDS, not normalised units. At
# render time, the box is converted to normalised width by dividing by
# the slice duration. Consequences:
#
#   * The same physical box covers the same physical event regardless
#     of how the run is sliced.
#   * Long high-cP runs naturally get wider normalised boxes (because
#     the slice is longer) without overshooting on short low-cP runs.
#
# Per-POI base value reflects event timescale:
#   POI1/POI2 : ~0.5 s — very sharp boundary events
#   POI3/POI4 : ~3.0 s base, scales with viscosity tier
#   POI5      : ~5.0 s base, can stretch to ~25 s for 150+ cP runs


@dataclass(frozen=True)
class BoxSizeProfile:
    """Physical bounding-box width (seconds) per viscosity tier.

    Indexing matches :data:`TIER_LABELS`:
        0 = <2.5 cP, 1 = 2.5-5 cP, 2 = 5-10 cP,
        3 = 10-22 cP, 4 = 22-150 cP, 5 = 150+ cP
    """

    poi_name: str
    base_seconds: float
    tier_multipliers: Tuple[float, ...] = (1.0, 1.0, 1.2, 1.5, 2.5, 4.0)
    # Hard floor / ceiling in NORMALISED units to keep YOLO sane even
    # when slice duration explodes (e.g. a single-POI run of 1200 s).
    min_norm_width: float = 0.004
    max_norm_width: float = 0.20

    def width_seconds(self, viscosity_tier: int) -> float:
        idx = max(0, min(viscosity_tier, len(self.tier_multipliers) - 1))
        return self.base_seconds * self.tier_multipliers[idx]


BOX_SIZE_PROFILES: Dict[str, BoxSizeProfile] = {
    "POI1": BoxSizeProfile(
        "POI1",
        base_seconds=0.5,
        tier_multipliers=(1.0, 1.0, 1.2, 1.5, 2.0, 3.0),
        min_norm_width=0.003,
        max_norm_width=0.08,
    ),
    "POI2": BoxSizeProfile(
        "POI2",
        base_seconds=0.5,
        tier_multipliers=(1.0, 1.0, 1.2, 1.5, 2.0, 3.0),
        min_norm_width=0.003,
        max_norm_width=0.08,
    ),
    "POI3": BoxSizeProfile("POI3", base_seconds=3.0),
    "POI4": BoxSizeProfile("POI4", base_seconds=3.0),
    "POI5": BoxSizeProfile(
        "POI5",
        base_seconds=5.0,
        tier_multipliers=(1.0, 1.0, 1.2, 1.5, 3.0, 5.0),
    ),
}


# ===========================================================================
#  Per-channel cascade configuration
# ===========================================================================


@dataclass
class ChannelConfig:
    """Full configuration for one cascade channel (detector)."""

    # Identity
    name: str
    target: str  # POI name (must be one of POI_NAMES)

    # Slicing
    slice_mode: str = "full"  # full | backward | forward | windowed
    cutoff_poi: Optional[str] = None  # required for backward / windowed
    anchor_poi: Optional[str] = None  # required for forward / windowed

    # Resolution & rendering
    resolution: ResolutionPreset = field(default_factory=lambda: MIDRES_PRESET)
    smooth_signal_window: int = 0  # 0 disables — keep POI1/POI2 sharpness
    include_engineered_features: bool = True

    # Augmentation
    base_truncations: int = 4
    stretch_prob: float = 0.4
    tier_cap: int = 12
    # Extra variant multiplier for high-cP tiers (4 and 5), applied on
    # top of the stratified tier-balance multiplier. Driving force for
    # rare-tier coverage without cross-split leakage.
    high_cp_boost: float = 2.0

    # Training
    epochs: int = 120
    batch: int = 16
    conf_threshold: float = 0.20

    # Per-channel YOLO hyperparameter overrides (box / dfl / patience / …)
    yolo_extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def dataset_dir(self) -> str:
        return f"datasets_v7/{self.name}"

    @property
    def dataset_yaml(self) -> str:
        return f"{self.dataset_dir}/data.yaml"


# Reverse cascade. Training and inference both walk this order.
CASCADE_CHANNELS: List[ChannelConfig] = [
    # --- Coarse detectors (reverse cascade) ----------------------------
    ChannelConfig(
        name="ch_poi5",
        target="POI5",
        slice_mode="full",
        resolution=MIDRES_PRESET,
        base_truncations=6,
        stretch_prob=0.5,
        high_cp_boost=2.5,
        smooth_signal_window=15,
        epochs=120,
        conf_threshold=0.15,
        yolo_extra={"box": 14.0, "cls": 0.4, "dfl": 2.0, "patience": 35},
    ),
    ChannelConfig(
        name="ch_poi4",
        target="POI4",
        slice_mode="backward",
        cutoff_poi="POI5",
        resolution=MIDRES_PRESET,
        base_truncations=6,
        stretch_prob=0.5,
        high_cp_boost=2.5,
        smooth_signal_window=11,
        epochs=120,
        conf_threshold=0.15,
        yolo_extra={
            "box": 16.0,
            "cls": 0.5,
            "dfl": 2.5,
            "patience": 40,
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
        base_truncations=6,
        stretch_prob=0.5,
        high_cp_boost=2.0,
        smooth_signal_window=11,
        epochs=120,
        conf_threshold=0.15,
        yolo_extra={"box": 14.0, "cls": 0.4, "dfl": 2.0, "patience": 35},
    ),
    ChannelConfig(
        name="ch_poi2",
        target="POI2",
        slice_mode="backward",
        cutoff_poi="POI3",
        resolution=HIRES_PRESET,  # sharp event → high lateral res
        base_truncations=4,
        stretch_prob=0.3,
        high_cp_boost=2.5,  # high-cP POI2 was the v6 miss
        smooth_signal_window=0,  # preserve event sharpness
        epochs=120,
        conf_threshold=0.20,
        yolo_extra={"box": 20.0, "dfl": 2.5, "patience": 35},
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
        epochs=120,
        conf_threshold=0.20,
        yolo_extra={"box": 20.0, "dfl": 2.5, "patience": 35},
    ),
    # --- Fine refinement detectors ------------------------------------
    # Per the 10k approach observation, POI3 and POI4 fine detectors
    # add marginal value over their coarse counterparts. Keep only the
    # POI5 fine (EOF) detector, which IS a stable cascade win.
    ChannelConfig(
        name="ch_poi5_fine",
        target="POI5",
        slice_mode="forward",
        anchor_poi="POI4",
        resolution=ZOOM_PRESET,
        base_truncations=5,
        stretch_prob=0.4,
        high_cp_boost=2.0,
        smooth_signal_window=11,
        conf_threshold=0.40,
        epochs=80,
        yolo_extra={"box": 14.0, "dfl": 2.0, "patience": 25},
    ),
]


# ===========================================================================
#  YOLO training defaults (channel.yolo_extra overrides these)
# ===========================================================================
YOLO_DEFAULTS: Dict[str, Any] = {
    "box": 12.0,
    "cls": 0.5,
    "dfl": 2.0,
    # Augmentations: all zeroed except mild translate. Time direction is
    # meaningful, so no flips; channel colour encodes signal identity,
    # so no HSV; mosaic / mixup destroy temporal layout.
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
    # Optimiser
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 3.0,
    "weight_decay": 0.002,
    "dropout": 0.15,
    "patience": 30,
    "cos_lr": True,
    "save_period": 10,
}


# ===========================================================================
#  Cascade inference gates (from balanced approach)
# ===========================================================================

FINE_MIN_CONF: float = 0.50
FINE_MAX_DISP_FRAC: float = 0.15
FINE_LOG_DECISIONS: bool = False


# ===========================================================================
#  Paths and pipeline toggles
# ===========================================================================

RUNS_ROOT: str = "C:/Users/QATCH/dev/QATCH-ML/QModel/data/raw"
DATASET_ROOT: str = "datasets_v7"
TRAIN_PROJECT: str = "runs/v7"
BENCHMARK_OUTPUT: str = "runs/v7/benchmark"

BASE_WEIGHTS: str = "yolo26m.pt"
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

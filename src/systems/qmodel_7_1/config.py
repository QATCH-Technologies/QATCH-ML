"""
QModel-MOIRAI pilot — Configuration
===================================

A foundation-model alternative to the YOLO cascade for POI localisation.

Why this exists
---------------
The YOLO approach treats POI detection as object detection on a *rendered
image* of one signal. It is strong inside the bulk of the distribution but
degrades sharply when the POI feature is faint, ambiguous, or absent — the
exact regime of very long viscous runs. A bounding-box detector with no
prior over signal dynamics has nothing to fall back on when the local
visual cue is missing.

This pilot reframes the task:

  * Input  : the three raw channels (Dissipation, Resonance_Frequency,
             Relative_time-derived Difference) as a MULTIVARIATE TIME SERIES,
             not an image. No rasterisation.
  * Backbone: MOIRAI (Salesforce ``uni2ts``) — a multivariate time-series
             foundation model. We use its patch-embedding encoder to get a
             contextual representation that carries a learned prior over how
             these signals evolve.
  * Head    : DENSE per-timestep POI probability. Each POI is a soft
             Gaussian-bump target over the time axis. Localisation = the
             argmax / soft-argmax of that channel's probability curve.

The dense formulation is the key change vs YOLO: every timestep gets a
prediction, so the model reasons about where a POI *should* be from
sequence context even when the feature is degenerate, rather than failing
to emit a box.

This file mirrors the constants and run-discovery contract of the existing
QModel v6 pipeline so the loader is a drop-in against the same ``data/raw``.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    0.1.0  (pilot)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ===========================================================================
#  Paths  (edit these — they are the only machine-specific bits)
# ===========================================================================

RUNS_ROOT: Path = Path("data/raw")
OUTPUT_ROOT: Path = Path("runs_moirai")
CACHE_ROOT: Path = Path("cache_moirai")  # cached resampled tensors (.npz)

# ===========================================================================
#  POI definition — IDENTICAL to QModel v6 so labels line up 1:1.
#  Row 2 of the POI csv is intentionally skipped (matches POI_ROW_MAP).
# ===========================================================================

POI_ROW_MAP: Dict[int, str] = {0: "POI1", 1: "POI2", 3: "POI3", 4: "POI4", 5: "POI5"}
POI_NAMES: Tuple[str, ...] = ("POI1", "POI2", "POI3", "POI4", "POI5")
N_POI: int = len(POI_NAMES)

# ===========================================================================
#  Viscosity tiers — IDENTICAL to v6 (used for stratified, leak-free split).
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
#  Preprocessing — matches signal_processing.preprocess_dataframe.
# ===========================================================================

COL_TIME = "Relative_time"
COL_DISS = "Dissipation"
COL_FREQ = "Resonance_Frequency"
COL_DIFF = "Difference"
DROP_COLS = ("Date", "Time", "Ambient", "Peak Magnitude (RAW)", "Temperature")

TARGET_DT_SEC: float = 0.005
MEDIAN_KERNEL: int = 5
DIFF_FACTOR: float = 2.50
BASELINE_START_SEC: float = 0.5
BASELINE_END_SEC: float = 2.5
BASELINE_OFFSET_SEC: float = 2.0

# The three model input channels, in fixed order. MOIRAI is multivariate-
# native so all three go in together and the model can use cross-channel
# structure (freq/diss coupling) to reason about POI position.
INPUT_CHANNELS: Tuple[str, ...] = (COL_DISS, COL_FREQ, COL_DIFF)
N_INPUT_CHANNELS: int = len(INPUT_CHANNELS)

# ===========================================================================
#  Sequence resampling for the model
# ===========================================================================
#  TARGET_DT_SEC=0.005 gives ~10k+ samples on a 50s run and far more on long
#  viscous runs. Feeding raw length to a transformer is wasteful and blows up
#  memory on the long runs we care about. We resample every run to a FIXED
#  sequence length on a normalised [0,1] time axis. POI targets are placed in
#  the same normalised coordinate, so the model is scale-invariant to run
#  duration — directly helpful for "very long viscous runs".

SEQ_LEN: int = 2048  # fixed model sequence length (normalised time grid)
PATCH_SIZE: int = 32  # MOIRAI patch length; SEQ_LEN must be divisible
assert SEQ_LEN % PATCH_SIZE == 0

# Soft-label Gaussian bump: sigma in units of normalised-time *fraction*.
# 0.004 ≈ 8 samples at SEQ_LEN=2048. Wide enough to give gradient, tight
# enough to localise.
LABEL_SIGMA_FRAC: float = 0.004

# ===========================================================================
#  Split
# ===========================================================================

VAL_SPLIT: float = 0.15
RNG_SEED: int = 42

# ===========================================================================
#  MOIRAI backbone
# ===========================================================================
#  Pretrained weights pulled from HuggingFace by uni2ts. small is plenty for
#  a pilot and fits comfortably in VRAM alongside SEQ_LEN=2048.

MOIRAI_SIZE: str = "small"  # small | base | large
MOIRAI_HF_REPO: str = "Salesforce/moirai-1.0-R-small"
FREEZE_BACKBONE_EPOCHS: int = 3  # warm up the head first, then unfreeze

# ===========================================================================
#  Training
# ===========================================================================


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 8
    lr_head: float = 3e-4
    lr_backbone: float = 2e-5  # smaller LR once backbone unfreezes
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.05
    num_workers: int = 4
    device: str = "cuda"  # falls back to cpu automatically
    amp: bool = True  # mixed precision
    # Loss: weighted BCE over the dense heatmap + a soft-argmax localisation
    # term. pos_weight counters the heavy background/positive imbalance
    # (only a handful of timesteps near each POI are "hot").
    bce_pos_weight: float = 50.0
    loc_loss_weight: float = 0.5  # weight on |soft-argmax - true| term
    early_stop_patience: int = 8


TRAIN = TrainConfig()

# ===========================================================================
#  Pilot toggles  (mirror the v6 pipeline ergonomics)
# ===========================================================================

LIMIT_RUNS: int | None = 100  # pilot: cap discovered runs. None = all.
RUN_BUILD_CACHE: bool = True
RUN_TRAIN: bool = True
RUN_EVAL: bool = True

MIN_DURATION_SEC: float = 2.0
MIN_ROWS: int = 50

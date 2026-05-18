# QModel V6.2 — Unified POI Detection Pipeline

Combines the strongest ideas from your three prior approaches into a single
coherent system, with the four bucket-list items wired in throughout.

## What's inherited from where

| Approach | What v6.2 keeps | What v6.2 fixes |
| --- | --- | --- |
| **10k (oversample)** | Reverse cascade architecture; POI5 fine (EOF) detector kept, POI3/4 fine detectors dropped (marginal vs coarse) | Variant generation now happens **after** the train/val split, eliminating cross-split leakage |
| **Balanced** | Viscosity-stratified per-tier split; time-axis stretching with tier-biased factors; gated fine refinement with confidence + displacement | High-cP runs now get a **per-channel boost** to variant count so their morphology is densely covered without crowding the population stats |
| **Zoom (feature engineering)** | Multiscale signed-gradient channels (`diff_pos` / `diff_neg_fine` / `diff_neg_coarse`) become a dedicated rendered strip every channel sees | Bound to the engineered features alone — v6.2 keeps the raw signal strips too, so the model gets both representations |

## What's new (the four bucket-list items)

1. **Dynamic bounding boxes.** Box width is specified in **physical seconds** via `BOX_SIZE_PROFILES`, then converted to normalised width per-variant. A 60 s high-cP POI5 fill gets a 60-second box; a 2 s low-cP fill gets a 2-second box. Min/max norm-width clamps prevent degenerate boxes on extreme slice durations. See `augmentation.compute_box_width_norm`.

2. **Relative-time x-axis everywhere.** `preprocess_dataframe` resamples every run to `TARGET_DT_SEC = 0.005 s` (200 Hz uniform). After that, sample index and `Relative_time` are proportional, so multiscale gradient windows expressed in pixels cover the same physical span across runs regardless of native sample rate. See `signal_processing.preprocess_dataframe`.

3. **Per-POI resolution.** Three `ResolutionPreset` profiles in `config.py`:
   - **HIRES** (POI1 / POI2): `img_w=2560, strip_h=160` — sharp events need lateral resolution.
   - **MIDRES** (POI3 / POI4 / POI5 coarse): `img_w=1600, strip_h=160` — sub-second resolution is enough.
   - **ZOOM** (POI5 fine): `img_w=1280, strip_h=160` — refinement on a small slice.

   Strip height is bumped from 128 → 160 to avoid the channel-curve flattening you noticed.

4. **High-cP oversampling via variants.** Per-channel `high_cp_boost` (default 2.0–2.5) multiplies the variant count specifically for tier 4 (22–150 cP) and tier 5 (150+ cP) runs. This is applied **inside each split** (no leakage), unlike the 10k approach.

## Module layout

```
qmodel_v6.2/
├── config.py              # All constants, channel list, per-POI profiles
├── signal_processing.py   # dt-uniform preprocess, feature channels, rendering
├── augmentation.py        # Stretch sampler, dynamic boxes, variant counts
├── dataset_builder.py     # Run discovery, stratified split, dataset assembly
├── train.py               # Ultralytics YOLO training driver
├── cascade_inference.py   # Reverse cascade with gates + sub-pixel refine
├── benchmark.py           # Per-POI + per-tier time-error evaluation
└── pipeline.py            # Top-level driver — reads toggles from config.py
```

## Image layout (rendered)

Every channel renders into a 5-strip canvas:

```
┌────────────────────────────────────┐  ┐
│  Strip 0 : Dissipation     (R)     │  │
├────────────────────────────────────┤  │
│  Strip 1 : Frequency       (G)     │  │  4 × strip_h
├────────────────────────────────────┤  │
│  Strip 2 : Difference      (B)     │  │
├────────────────────────────────────┤  │
│  Strip 3 : R = diff_pos            │  │
│            G = diff_neg_fine       │  │
│            B = diff_neg_coarse     │  │
├────────────────────────────────────┤  ┘
│  Strip 4 : time-position ramp      │  ← time_strip_h
└────────────────────────────────────┘
```

Strip 3 is the dedicated engineered-feature heatmap (the zoom-approach win) and Strip 4 is the explicit absolute-x positional encoding from the balanced approach.

## Running the pipeline

Edit toggles at the top of `config.py`:

```python
RUN_BUILD_DATASETS: bool = True
RUN_TRAIN: bool = True
RUN_BENCHMARK: bool = True
USE_SUBPIXEL_REFINE: bool = True

RUNS_ROOT: str = "C:/Users/QATCH/dev/QATCH-ML/QModel/data/raw"
DATASET_ROOT: str = "datasets_v6.2"
TRAIN_PROJECT: str = "runs/v6.2"
```

Then:

```
python pipeline.py
```

Each step can be toggled independently — you can `RUN_BUILD_DATASETS=True` once, then iterate on `RUN_TRAIN=True; RUN_BENCHMARK=True` without re-rendering the dataset.

## Per-channel tuning

The cascade is defined as a list of `ChannelConfig` objects in `config.CASCADE_CHANNELS`. Each entry carries everything that varies per-channel:

```python
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
    yolo_extra={"box": 16.0, "dfl": 2.5, "patience": 40, ...},
)
```

To add a new channel (e.g. a POI4 fine detector for an ablation), append a `ChannelConfig` to `CASCADE_CHANNELS` — the pipeline will pick it up.

## Integration notes

- The renderer is **drop-in compatible** with your `QModelV6YOLO_DataProcessor` column conventions (`Relative_time`, `Resonance_Frequency`, `Dissipation`, derived `Difference`).
- The run-discovery code (`discover_runs`, `read_run_viscosity`, `RunSpec`) mirrors the equivalent code in `train_v6_yolo_balanced.py` so you can swap layers if needed.
- Sub-pixel refinement uses your existing `refine_poi.refine_poi_x_default`. If `refine_poi.py` is not on the Python path, `cascade_inference` falls back silently to coarse YOLO output.

## What to look at first in the benchmark output

`benchmark_metrics_by_tier.csv` is the headline diagnostic. The columns to watch:
- **POI1 / POI2 at tier 4 + 5**: this was the v6 failure mode. The `diff_neg_coarse` channel + `high_cp_boost=2.5` on `ch_poi1` / `ch_poi2` is the intervention. If MAE in those cells is still > 5 s, increase `high_cp_boost` further (3.0–4.0) or widen `BoxSizeProfile("POI2").tier_multipliers`.
- **POI5 bias**: a non-zero `bias_s` in the high-cP tiers indicates the box-anchor displacement gate may be over-rejecting fine refinements. Loosen `FINE_MAX_DISP_FRAC` from 0.15 to 0.20.
- **gross_failures.csv**: filter by `tier` to find which specific 150+ cP runs are still hard. Cross-reference with `run_id` to inspect.
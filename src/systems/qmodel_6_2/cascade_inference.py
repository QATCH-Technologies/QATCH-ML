"""
QModel v6 — Cascade inference
==============================

Reverse-cascade POI detection. The cascade walks coarse-to-fine in
reverse run order (POI5 first, POI1/POI2 last) so each detector
operates on a slice constrained by the previous detector's prediction:

    1.  POI5 coarse                 (full run)
    2.  POI4 coarse                 (start → POI5)
    3.  POI5 fine, gated            (POI4 → end)
    4.  POI3 coarse                 (start → POI4)
    5.  ordering sanity check       (POI3 < POI4 < POI5)
    6.  POI1, POI2 coarse           (start → POI3)

Fine-refinement gates
---------------------
Fine predictions only replace coarse predictions when BOTH:
  * confidence ≥ FINE_MIN_CONF
  * |t_fine − t_coarse| ≤ FINE_MAX_DISP_FRAC × slice_duration

This is the v10/v11 lesson from the balanced approach: unguarded fine
detectors silently degrade the cascade by latching onto the wrong
feature in hard runs and propagating bias downstream.

Optional sub-pixel refinement
-----------------------------
If ``USE_SUBPIXEL_REFINE`` is True and ``refine_poi`` is importable, the
YOLO box center is snapped to a derivative-peak / inflection inside the
predicted box. Falls back silently to coarse YOLO output if the module
isn't on the path.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.0.0
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BASE_WEIGHTS,
    CASCADE_CHANNELS,
    FINE_LOG_DECISIONS,
    FINE_MAX_DISP_FRAC,
    FINE_MIN_CONF,
    GROUP_TO_CHANNELS,
    SIGNAL_FUSION_METHOD,
    SIGNAL_MIN_AGREE,
    SIGNAL_MIN_CONF,
    SIGNAL_OUTLIER_FRAC,
    TRAIN_PROJECT,
    USE_SUBPIXEL_REFINE,
    ChannelConfig,
)
from signal_processing import COL_TIME, preprocess_dataframe, render_detection_image

LOG = logging.getLogger("v6.inference")

# Optional sub-pixel refinement.
try:
    from refine_poi import refine_poi_x_default

    _HAS_REFINE = True
except ImportError:
    _HAS_REFINE = False

try:
    from ultralytics import YOLO

    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False


# ===========================================================================
#  Quiet helper — Ultralytics is loud
# ===========================================================================


@contextlib.contextmanager
def _quiet_stdout():
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


# ===========================================================================
#  Render + detect on one slice
# ===========================================================================


def _render_and_detect(
    df_view: pd.DataFrame,
    model: Any,
    cfg: ChannelConfig,
    use_subpixel_refine: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """Render df_view, run YOLO, return (t_pred, conf).

    Returning conf alongside the prediction is what lets the cascade
    gate fine refinements. Returns ``(None, None)`` on any short-slice
    or no-detection path.
    """
    if df_view is None or len(df_view) < 32:
        return None, None

    img = render_detection_image(df_view, cfg)
    if img is None:
        return None, None

    # v8 renderer returns a 2-D (H, W) grayscale array. During training the
    # images are written to disk as grayscale PNG and read back via
    # cv2.imread, which promotes them to 3-channel BGR. The weights are
    # therefore 3-channel. model.predict() on a raw 2-D ndarray skips that
    # promotion and feeds [1, 1, H, W] to a 3-channel conv stem. Broadcast
    # to 3 identical channels here so inference matches the training path.
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)  # (H, W) -> (H, W, 3)

    with _quiet_stdout():
        results = model.predict(img, conf=cfg.conf_threshold, iou=0.3, verbose=False)

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None

    boxes = results[0].boxes
    best = int(boxes.conf.argmax())
    conf = float(boxes.conf[best])
    x_norm = float(boxes.xywhn[best][0])
    w_norm = float(boxes.xywhn[best][2])

    t_min = float(df_view[COL_TIME].iloc[0])
    t_max = float(df_view[COL_TIME].iloc[-1])
    t_span = t_max - t_min
    t_coarse = t_min + x_norm * t_span

    if use_subpixel_refine and _HAS_REFINE:
        box_w_time = max(1e-6, w_norm * t_span)
        try:
            t_refined = refine_poi_x_default(
                df=df_view,
                poi_name=cfg.target,
                t_coarse=t_coarse,
                box_w_time=box_w_time,
            )
            return float(t_refined), conf
        except Exception as exc:
            LOG.debug(
                "Sub-pixel refine failed for %s (%s) — falling back to coarse",
                cfg.target,
                exc,
            )

    return t_coarse, conf


# ===========================================================================
#  Per-signal fusion
# ===========================================================================


def _fuse_signal_predictions(
    preds: List[Tuple[float, float, str]],
    slice_dur: float,
    method: str = SIGNAL_FUSION_METHOD,
    min_conf: float = SIGNAL_MIN_CONF,
    min_agree: int = SIGNAL_MIN_AGREE,
    outlier_frac: float = SIGNAL_OUTLIER_FRAC,
    poi_label: str = "?",
) -> Tuple[Optional[float], Optional[float]]:
    """Fuse the three single-signal detections for one POI into one time.

    Args:
        preds: list of ``(t_pred, conf, signal_key)`` from each signal
            detector that produced a detection.
        slice_dur: duration of the slice (for outlier rejection scaling).
        method: "conf_weighted" | "median" | "best".
        min_conf: drop any signal below this confidence first.
        min_agree: require at least this many surviving detections.
        outlier_frac: drop a signal whose prediction is further than
            ``outlier_frac × slice_dur`` from the median of the rest.

    Returns:
        ``(t_fused, conf_fused)`` or ``(None, None)`` if agreement fails.
        ``conf_fused`` is the mean confidence of the contributing signals,
        used downstream by the fine-refinement gate.
    """
    # 1. Confidence floor.
    kept = [(t, c, s) for (t, c, s) in preds if c >= min_conf]
    if len(kept) < max(1, min_agree):
        if FINE_LOG_DECISIONS:
            LOG.info(
                "[%s fuse] reject: only %d signal(s) ≥ conf %.2f", poi_label, len(kept), min_conf
            )
        return None, None

    # 2. Outlier rejection against the median (needs ≥ 3 to be meaningful).
    if len(kept) >= 3 and outlier_frac > 0 and slice_dur > 0:
        times = np.array([t for (t, _, _) in kept])
        med = float(np.median(times))
        tol = outlier_frac * slice_dur
        survivors = [(t, c, s) for (t, c, s) in kept if abs(t - med) <= tol]
        if len(survivors) >= max(1, min_agree):
            kept = survivors

    times = np.array([t for (t, _, _) in kept], dtype=float)
    confs = np.array([c for (_, c, _) in kept], dtype=float)

    # 3. Fuse.
    if method == "best":
        bi = int(np.argmax(confs))
        t_fused = float(times[bi])
    elif method == "median":
        t_fused = float(np.median(times))
    else:  # conf_weighted (default)
        w = confs / (confs.sum() + 1e-9)
        t_fused = float((times * w).sum())

    conf_fused = float(confs.mean())
    if FINE_LOG_DECISIONS:
        sigs = ",".join(s for (_, _, s) in kept)
        LOG.info(
            "[%s fuse] %s → t=%.3f conf=%.2f (signals: %s)",
            poi_label,
            method,
            t_fused,
            conf_fused,
            sigs,
        )
    return t_fused, conf_fused


def _detect_group(
    df_view: pd.DataFrame,
    group: str,
    models: Dict[str, Any],
    channel_map: Dict[str, ChannelConfig],
    use_subpixel_refine: bool = False,
) -> Tuple[Optional[float], Optional[float]]:
    """Run every available signal detector for ``group`` on ``df_view`` and fuse.

    Returns ``(t_fused, conf_fused)``. ``(None, None)`` if the group has no
    loaded models or no signal produced a detection.
    """
    if df_view is None or len(df_view) < 32:
        return None, None

    chan_names = GROUP_TO_CHANNELS.get(group, [])
    preds: List[Tuple[float, float, str]] = []
    for cname in chan_names:
        if cname not in models:
            continue
        cfg = channel_map[cname]
        t_pred, conf = _render_and_detect(df_view, models[cname], cfg, use_subpixel_refine)
        if t_pred is not None and conf is not None:
            preds.append((t_pred, conf, cfg.signal))

    if not preds:
        return None, None

    t_min = float(df_view[COL_TIME].iloc[0])
    t_max = float(df_view[COL_TIME].iloc[-1])
    return _fuse_signal_predictions(preds, slice_dur=t_max - t_min, poi_label=group)


# ===========================================================================
#  Fine-refinement gate
# ===========================================================================


def _apply_fine_refinement(
    t_coarse: Optional[float],
    t_fine: Optional[float],
    conf_fine: Optional[float],
    slice_dur: float,
    poi_label: str = "?",
    min_conf: Optional[float] = None,
    max_disp_frac: Optional[float] = None,
) -> Optional[float]:
    """Decide whether to accept a fine-detector refinement.

    Two gates in sequence:
      1. Confidence: the fine detector must clear ``min_conf``.
      2. Displacement: |t_fine − t_coarse| must be within
         ``max_disp_frac × slice_dur``. Blocks the right-edge-feature
         latching mode where a confused fine detector predicts the
         location of the NEXT POI.

    Special case: if coarse failed but fine cleared the confidence gate,
    the fine prediction is accepted unconditionally (no anchor to gate
    against).
    """
    if min_conf is None:
        min_conf = FINE_MIN_CONF
    if max_disp_frac is None:
        max_disp_frac = FINE_MAX_DISP_FRAC

    if t_fine is None or conf_fine is None:
        if FINE_LOG_DECISIONS:
            LOG.info("[%s fine] reject: no detection", poi_label)
        return t_coarse

    if conf_fine < min_conf:
        if FINE_LOG_DECISIONS:
            LOG.info("[%s fine] reject: low conf %.2f < %.2f", poi_label, conf_fine, min_conf)
        return t_coarse

    if t_coarse is None:
        if FINE_LOG_DECISIONS:
            LOG.info("[%s fine] accept: no coarse, fine conf=%.2f", poi_label, conf_fine)
        return t_fine

    disp = abs(t_fine - t_coarse)
    max_disp = max_disp_frac * max(slice_dur, 1e-6)
    if disp > max_disp:
        if FINE_LOG_DECISIONS:
            LOG.info(
                "[%s fine] reject: disp %.2fs > %.2fs (frac=%.2f of %.1fs slice)",
                poi_label,
                disp,
                max_disp,
                max_disp_frac,
                slice_dur,
            )
        return t_coarse

    if FINE_LOG_DECISIONS:
        LOG.info(
            "[%s fine] accept: conf=%.2f, disp=%.2fs (within %.2fs)",
            poi_label,
            conf_fine,
            disp,
            max_disp,
        )
    return t_fine


# ===========================================================================
#  Sanity checks
# ===========================================================================


def _sanity_check_poi5(
    t_poi5: Optional[float],
    t_min: float,
    t_max: float,
) -> Optional[float]:
    """POI5 (end-of-fill) must be in the back half of the run."""
    if t_poi5 is None:
        return None
    run_dur = max(t_max - t_min, 1e-6)
    rel = (t_poi5 - t_min) / run_dur
    if rel < 0.50 or rel > 0.99:
        return None
    return t_poi5


def _sanity_check_ordering(
    t_poi3: Optional[float],
    t_poi4: Optional[float],
    t_poi5: Optional[float],
    t_min: float,
    t_max: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Enforce POI3 < POI4 < POI5. Falls back heuristically when violated."""
    if t_poi4 is not None and t_poi5 is not None and t_poi4 >= t_poi5:
        # POI4 detected after POI5 — back POI4 off by 15% of the run.
        t_poi4 = t_poi5 - 0.15 * (t_max - t_min)
    if t_poi3 is not None and t_poi4 is not None and t_poi3 >= t_poi4:
        t_poi3 = None
    return t_poi3, t_poi4, t_poi5


# ===========================================================================
#  Main cascade
# ===========================================================================


def _group_loaded(group: str, models: Dict[str, Any]) -> bool:
    """True if at least one signal detector for ``group`` is loaded."""
    return any(c in models for c in GROUP_TO_CHANNELS.get(group, []))


def cascade_predict(
    df_raw: pd.DataFrame,
    models: Dict[str, Any],
    channel_map: Dict[str, ChannelConfig],
    use_subpixel_refine: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Run the reverse cascade end-to-end and return ``{POI1..5: time | None}``.

    Each POI is detected by fusing its three single-signal detectors
    (dissipation / resonance / difference) on the same slice — see
    :func:`_detect_group`.

    Args:
        df_raw: Raw run dataframe (will be preprocessed here).
        models: Dict of channel_name → loaded YOLO model (one per signal).
        channel_map: Dict of channel_name → ChannelConfig.
        use_subpixel_refine: When True and ``refine_poi.py`` is on the
            path, snap each YOLO box center to a derivative peak inside
            the box.
    """
    df = preprocess_dataframe(df_raw)
    if df is None or df.empty:
        return {}

    t = df[COL_TIME]
    t_min = float(t.iloc[0])
    t_max = float(t.iloc[-1])

    def _safe_cut(val: Optional[float], fallback: float) -> float:
        if val is not None and t_min < val < t_max:
            return val
        return fallback

    preds: Dict[str, Optional[float]] = {p: None for p in ("POI1", "POI2", "POI3", "POI4", "POI5")}

    # --- 1. POI5 coarse (signal-fused) -----------------------------------
    if _group_loaded("ch_poi5", models):
        t_poi5, _ = _detect_group(df, "ch_poi5", models, channel_map, use_subpixel_refine)
        t_poi5 = _sanity_check_poi5(t_poi5, t_min, t_max)
    else:
        t_poi5 = None

    # --- 2. POI4 coarse (signal-fused) -----------------------------------
    t5_cut = _safe_cut(t_poi5, t_max)
    df_4 = df[df[COL_TIME] <= t5_cut].reset_index(drop=True)
    if _group_loaded("ch_poi4", models):
        t_poi4, _ = _detect_group(df_4, "ch_poi4", models, channel_map, use_subpixel_refine)
    else:
        t_poi4 = None

    # --- 3. POI5 fine (gated, signal-fused) ------------------------------
    if _group_loaded("ch_poi5_fine", models) and t_poi4 is not None and t_min < t_poi4 < t_max:
        df_eof = df[df[COL_TIME] >= t_poi4].reset_index(drop=True)
        t5_fine, c5_fine = _detect_group(
            df_eof, "ch_poi5_fine", models, channel_map, use_subpixel_refine
        )
        t5_fine = _sanity_check_poi5(t5_fine, t_min, t_max)
        t_poi5 = _apply_fine_refinement(
            t_coarse=t_poi5,
            t_fine=t5_fine,
            conf_fine=c5_fine,
            slice_dur=t_max - t_poi4,
            poi_label="POI5",
        )

    # --- 4. POI3 coarse (signal-fused) -----------------------------------
    t4_cut = _safe_cut(t_poi4, t5_cut)
    df_3 = df[df[COL_TIME] <= t4_cut].reset_index(drop=True)
    if _group_loaded("ch_poi3", models):
        t_poi3, _ = _detect_group(df_3, "ch_poi3", models, channel_map, use_subpixel_refine)
    else:
        t_poi3 = None

    # --- 5. Ordering sanity check ----------------------------------------
    t_poi3, t_poi4, t_poi5 = _sanity_check_ordering(
        t_poi3,
        t_poi4,
        t_poi5,
        t_min,
        t_max,
    )

    # --- 6. POI1 & POI2 coarse (signal-fused) ----------------------------
    t3_cut = _safe_cut(t_poi3, _safe_cut(t_poi4, t5_cut))
    df_12 = df[df[COL_TIME] <= t3_cut].reset_index(drop=True)
    if _group_loaded("ch_poi2", models):
        t_poi2, _ = _detect_group(df_12, "ch_poi2", models, channel_map, use_subpixel_refine)
    else:
        t_poi2 = None
    if _group_loaded("ch_poi1", models):
        t_poi1, _ = _detect_group(df_12, "ch_poi1", models, channel_map, use_subpixel_refine)
    else:
        t_poi1 = None

    preds["POI1"] = t_poi1
    preds["POI2"] = t_poi2
    preds["POI3"] = t_poi3
    preds["POI4"] = t_poi4
    preds["POI5"] = t_poi5
    return preds


# ===========================================================================
#  Weight loading
# ===========================================================================


def load_trained_models(
    channels: List[ChannelConfig] = CASCADE_CHANNELS,
    project: Path = Path(TRAIN_PROJECT),
) -> Tuple[Dict[str, Any], Dict[str, ChannelConfig]]:
    """Load best.pt for every trained channel under ``project``."""
    if not _HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics must be installed to run inference.")

    project = Path(project)
    script_dir = Path(__file__).parent

    models: Dict[str, Any] = {}
    channel_map: Dict[str, ChannelConfig] = {}

    for cfg in channels:
        run_dir_abs = script_dir / project / cfg.name
        run_dir_cwd = project / cfg.name

        sidecar = run_dir_abs / "best_weights_path.txt"
        if not sidecar.is_file():
            sidecar = run_dir_cwd / "best_weights_path.txt"

        candidates: List[Path] = []
        if sidecar.is_file():
            recorded = Path(sidecar.read_text().strip())
            candidates.append(recorded)
            if not recorded.is_absolute():
                candidates.append(script_dir / recorded)

        candidates.append(run_dir_abs / "weights" / "best.pt")
        candidates.append(run_dir_cwd / "weights" / "best.pt")

        weights: Optional[Path] = next((p for p in candidates if p.is_file()), None)
        if weights is None:
            LOG.warning(
                "Cascade: best.pt not found for channel %s — tried:\n%s",
                cfg.name,
                "\n".join(f"  {p}" for p in candidates),
            )
            continue

        LOG.info("Cascade: loading %-15s ← %s", cfg.name, weights)
        with _quiet_stdout():
            models[cfg.name] = YOLO(str(weights))
        channel_map[cfg.name] = cfg

    loaded = list(models.keys())
    LOG.info("Cascade: loaded %d / %d channels: %s", len(loaded), len(channels), loaded)
    return models, channel_map

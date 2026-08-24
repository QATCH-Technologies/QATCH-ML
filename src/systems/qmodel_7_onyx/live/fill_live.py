"""
fill_live.py
============

Live-side upgrades for the fill-type classifier, porting the v7 detector
lessons to the streaming path. Three independent fixes, one drop-in class:

1. CONSTANT-COST PREPROCESSING (the RectDetectionTrainer lesson: find the
   silent resource pathology and cap it). The current live path reindexes
   the ENTIRE buffer to a 5 ms grid on every chunk: a 20-minute run is
   ~240k interpolated rows, re-built, re-differenced, and median-filtered
   several times per second — cost grows linearly with run duration for an
   image that is only 640 px wide. `preprocess_for_cls` widens the grid
   step so total points never exceed `max_points` (default 4096, ~6 per
   rendered column): identical to the production 5 ms grid for short
   buffers, graceful decimation for long ones. Inference cost becomes flat
   for the life of the run. (The median filter's 5-sample kernel then
   smooths a window proportional to run span — for a GLOBAL-shape
   classifier that scale-following smoothing is a feature, not a loss.)

2. PROBABILITIES OUT OF THE PREDICTOR. `predict` currently collapses the
   softmax to top-1 and discards confidence, so the debounce layer sees a
   hard integer and must treat a 51/49 flicker identically to a 99/1 call.
   `QModelV7FillClassifier.predict_probs` returns the full distribution
   over the ORDINAL state axis [-1, 0, 1, 2, 3], rendered with the v2
   derivative-energy fill render via the shared prepare_cls_input contract.

3. ORDINAL MONOTONE EVIDENCE (the decode-layer insight: the model's raw
   argmax is not the answer — a physical prior over the answer space is
   cheap and it is where the remaining errors live). Fill state is
   physically monotone non-decreasing: channels do not un-fill. The
   symmetric count-of-3 debounce ignores this — three flickers backward
   and the UI reports a channel emptying. `OrdinalEvidence` replaces it:

     * EMA over the probability vector (alpha tuned so confirmation
       latency matches the old 3-frame debounce on clean streams).
     * FORWARD moves confirm on the CUMULATIVE tail P(state >= k): if the
       model splits mass between 2ch and 3ch, both agree the state is at
       least 2ch — cumulative confirmation banks that agreement instead
       of stalling on a split argmax. Multi-step jumps are allowed, which
       is also what makes the same code correct at analysis time, where
       the first frame of a finished 3ch run should go straight to 3ch.
     * BACKWARD moves are treated as what they physically are: evidence
       that an earlier CONFIRMATION was wrong, not that the fill reversed.
       They require sustained strong contrary evidence (P(state >=
       current) below a low floor for many consecutive cycles) before the
       state is re-solved from scratch.

The public surface of QModelV6YOLO_Live (attempt_classification -> int,
STATUS_MAP, display-message machinery, drop-epoch gating, duration
thresholds) is preserved; QModelV7YOLO_Live is a drop-in replacement for
the class the LiveProcess constructs.

Migrated from the flat `v7_fill_live.py` module as part of the `live/`
subpackage split. The dead `buffer_window_size` fallback this docstring
used to describe (computed in `QModelV6YOLO_LiveProcess.__init__` but
always discarded in favour of `None` in `run()`) has been removed at
its source in `live/base_live.py`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import medfilt

try:
    from QATCH.common.logger import Logger as Log
except (ImportError, ModuleNotFoundError):
    from src.utils.logger import get_logger

    _log = get_logger("qmodel_7_onyx.live.fill_live")

    class Log:  # headless fallback, matching the rest of qmodel_7_onyx
        @staticmethod
        def d(tag, msg):
            _log.debug(f"{tag} {msg}")

        @staticmethod
        def i(tag, msg):
            _log.info(f"{tag} {msg}")

        @staticmethod
        def w(tag, msg):
            _log.warning(f"{tag} {msg}")

        @staticmethod
        def e(tag, msg):
            _log.error(f"{tag} {msg}")


try:
    from QATCH.QModel.src.models.v6_yolo.v6_yolo import (
        QModelV6Config,
        QModelV6YOLO_FillClassifier,
    )
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_dataprocessor import (
        QModelV6YOLO_DataProcessor as DP,
    )
    from QATCH.QModel.src.models.v6_yolo.v7_fill_render import prepare_cls_input
except (ImportError, ModuleNotFoundError):
    from ..inference.config import QModelV6Config
    from ..inference.controller import QModelV6YOLO_FillClassifier
    from ..rendering.fill_render import prepare_cls_input
    from ..rendering.legacy_dataprocessor import QModelV6YOLO_DataProcessor as DP

# The live base class lives app-side (base_live imports QATCH.common
# unconditionally when the app IS present), so it is imported SEPARATELY and
# optionally: headless consumers — replay.py, audits, notebooks — need
# preprocess_for_cls / QModelV7FillClassifier / OrdinalEvidence without
# dragging in the application. When the base is absent, QModelV7YOLO_Live
# is still defined (over the headless placeholder base_live already
# provides) but is not constructible.
try:
    from QATCH.QModel.src.models.v6_yolo.v6_yolo_live import QModelV6YOLO_Live

    _LIVE_BASE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    from .base_live import _LIVE_APP_AVAILABLE as _LIVE_BASE_AVAILABLE
    from .base_live import QModelV6YOLO_Live

TAG = "[QModelV7FillLive]"

# Ordinal state axis. Index i corresponds to channels i - 1.
N_STATES = 5
STATE_NO_FILL = 0  # channels == -1

# Fill render version the deployed type_cls.pt was trained on. Ships with
# the weights, exactly like QModelV6Config.RENDER_VERSION on the detector
# side: v2-trained weights <-> version 2.
FILL_RENDER_VERSION: int = 3

# Live preprocessing point budget: ~6 samples per rendered column at
# FILL_GEN_W=640. Below the budget the grid is the production 5 ms step, so
# short-buffer behaviour is bit-identical to the current path.
MAX_CLS_POINTS: int = 4096

# Probability temperature applied in predict_probs: p_i ∝ p_i^(1/T).
# The trained classifier is SATURATED (train loss ~0.005; val predictions
# are wall-to-wall 1.0), so raw outputs give OrdinalEvidence no distinction
# between a marginal call and a certain one — a saturated WRONG frame in a
# transition zone carries full weight into the EMA. T>1 softens the
# distribution back toward informative confidences. Fit T on the val split
# (fit_temperature.py / audit tooling) rather than hand-picking; 1.0 = off.
PROB_TEMPERATURE: float = 1.45  # val-NLL fit (audit_fill_val, first v7 weights)


def preprocess_for_cls(
    df: pd.DataFrame,
    baseline_freq: Optional[float] = None,
    baseline_diss: Optional[float] = None,
    max_points: int = MAX_CLS_POINTS,
) -> Optional[pd.DataFrame]:
    """DP.preprocess_dataframe with an adaptive grid step.

    step = max(TIME_STEP, span / max_points): identical to production for
    spans under TIME_STEP * max_points (~20 s at defaults), then the step
    widens so the interpolated frame never exceeds `max_points` rows —
    flat per-chunk cost regardless of run length. The classifier's render
    is 640 px wide; carrying 240k rows into it was pure waste.
    """
    cols_to_drop = [c for c in DP.DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    if DP.COL_TIME not in df.columns:
        return None
    df = df.drop_duplicates(subset=[DP.COL_TIME], keep="first")
    t_min = df[DP.COL_TIME].min()
    t_max = df[DP.COL_TIME].max()
    span = float(t_max - t_min)
    if not np.isfinite(span) or span <= 0:
        return None
    step = max(DP.TIME_STEP, span / float(max_points))
    new_time_grid = np.arange(t_min, t_max, step)
    if len(new_time_grid) < 2:
        return None
    df = df.set_index(DP.COL_TIME)
    combined_index = df.index.union(new_time_grid).sort_values()
    df = df.reindex(combined_index).interpolate(method="index").loc[new_time_grid]
    df = df.reset_index().rename(columns={"index": DP.COL_TIME})
    diff_series = DP._compute_difference_curve(df, baseline_freq, baseline_diss)
    df[DP.COL_DIFF] = diff_series if diff_series is not None else 0.0
    for col in df.columns:
        if col != DP.COL_TIME and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = medfilt(df[col], kernel_size=DP.MEDIAN_KERNEL)
    return df


class QModelV7FillClassifier(QModelV6YOLO_FillClassifier):
    """Fill classifier speaking probabilities over the ordinal state axis,
    rendering through the shared v2 prepare_cls_input contract."""

    TAG = "[QModelV7FillClassifier]"

    def predict_probs(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Returns p, shape (5,), p[i] = P(channels == i - 1); None on
        failure. Renders with FILL_RENDER_VERSION — the render the weights
        were trained on, not necessarily the legacy one."""
        if df is None or df.empty:
            return None
        try:
            img_input = prepare_cls_input(df, version=FILL_RENDER_VERSION)
        except Exception as e:
            Log.e(self.TAG, f"Error generating fill render: {e}")
            return None
        self._last_image = img_input
        try:
            results = self.model(img_input, verbose=False)
            if not results:
                return None
            probs = results[0].probs.data
            probs = probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)
            names = results[0].names
            vec = np.zeros(N_STATES, dtype=float)
            for idx, label in names.items():
                ch = self._map_label_to_channels(label)
                if -1 <= ch <= 3:
                    vec[ch + 1] += float(probs[idx])
            s = vec.sum()
            if s <= 0:
                return None
            vec /= s
            if PROB_TEMPERATURE != 1.0:
                vec = np.power(np.clip(vec, 1e-12, 1.0), 1.0 / PROB_TEMPERATURE)
                vec /= vec.sum()
            return vec
        except Exception as e:
            Log.e(self.TAG, f"Inference error: {e}")
            return None

    def predict(self, df: pd.DataFrame) -> int:
        """Back-compatible int path (analysis-time callers): plain argmax.

        The original v7 design used the ordinal maximum-tail rule here for
        live/analysis symmetry; the val decision-rule check killed that
        idea with data (T=1.45: tail 97.65% vs argmax 97.84% — it fixed 7
        frames and broke 21 by promoting borderline frames upward). The
        tail rule earns its keep only where it integrates evidence ACROSS
        frames (OrdinalEvidence, live); on a single frame it is just a
        lower decision threshold. Argmax here, evidence machinery live."""
        p = self.predict_probs(df)
        if p is None:
            return 0
        return int(np.argmax(p)) - 1


class OrdinalEvidence:
    """EMA evidence accumulator with monotone-state confirmation.

    Forward:  move to the HIGHEST k > current whose cumulative tail
              P(state >= k) clears the forward threshold in the smoothed
              vector. The threshold is uniform (CONF_FORWARD); the
              per-state override mechanism (CONF_FORWARD_PER_STATE)
              remains for sweeps but ships EMPTY, after the 547-run
              replay sweep delivered a clean post-mortem on the stricter
              terminal (3ch) bar it originally carried:

                term bar   final✓   missed   falseF
                0.75       89.9%      48        9
                0.70       91.8%      37        9
                0.65       95.8%      15        9
                0.60       96.0%      14        9

              False forwards were 9 at EVERY bar: the phantom 3ch regions
              are saturated (conf 0.93-1.0), so their EMA clears any
              plausible threshold — a threshold discriminates marginal
              evidence, and these errors are not marginal. The 0.75 bar
              therefore suppressed zero false stops while blocking 34
              statically-clean runs from ever confirming. False terminal
              confirmations are a MODEL problem (the persistent
              over-count runs), not a machinery problem, and belong to
              the label-review/data track.
    Backward: only after P(state >= current) stays below CONF_BACKWARD for
              BACKWARD_CYCLES consecutive updates — sustained proof that a
              prior confirmation was wrong — at which point the state is
              re-solved from the smoothed vector. (At the uniform bar the
              sweep showed backward moves rise 2 -> 5 across 547 runs:
              that is this guard correctly cleaning up after false
              forwards, i.e. the self-correction working.)

    ALPHA=0.45 puts a fresh unanimous signal past CONF_FORWARD in 2
    updates from a cold contrary EMA (measured) — at or under the latency
    of the old count-of-3 debounce on clean streams while being far
    harder to move with split or flickering evidence.
    """

    ALPHA = 0.45
    CONF_FORWARD = 0.60
    # Ships empty (see post-mortem above); kept as a sweep mechanism.
    CONF_FORWARD_PER_STATE: Dict[int, float] = {}
    CONF_BACKWARD = 0.20
    BACKWARD_CYCLES = 8

    def __init__(
        self,
        alpha: Optional[float] = None,
        conf_forward: Optional[float] = None,
        conf_forward_per_state: Optional[Dict[int, float]] = None,
        conf_backward: Optional[float] = None,
        backward_cycles: Optional[int] = None,
    ) -> None:
        # Instance-level overrides (class attributes remain the shipped
        # defaults) so replay.py can sweep configurations over ONE shared
        # probability stream instead of hand-picking thresholds.
        self.alpha = self.ALPHA if alpha is None else float(alpha)
        self.conf_forward = self.CONF_FORWARD if conf_forward is None else float(conf_forward)
        self.conf_forward_per_state = (
            dict(self.CONF_FORWARD_PER_STATE)
            if conf_forward_per_state is None
            else dict(conf_forward_per_state)
        )
        self.conf_backward = self.CONF_BACKWARD if conf_backward is None else float(conf_backward)
        self.backward_cycles = (
            self.BACKWARD_CYCLES if backward_cycles is None else int(backward_cycles)
        )
        self.ema: Optional[np.ndarray] = None
        self._contrary_count = 0

    def reset(self) -> None:
        self.ema = None
        self._contrary_count = 0

    @staticmethod
    def decide(p: np.ndarray, conf: float) -> int:
        """Highest state whose cumulative tail clears conf; falls back to
        argmax when nothing does (early, low-evidence frames)."""
        tail = np.cumsum(p[::-1])[::-1]
        ks = np.nonzero(tail >= conf)[0]
        return int(ks.max()) if len(ks) else int(np.argmax(p))

    def update(self, p: np.ndarray, current_state: int) -> int:
        """Feed one probability vector; returns the (possibly unchanged)
        confirmed state index."""
        self.ema = p.copy() if self.ema is None else self.alpha * p + (1 - self.alpha) * self.ema
        tail = np.cumsum(self.ema[::-1])[::-1]

        # Forward: highest state above current whose tail clears ITS bar.
        proposal = current_state
        for k in range(current_state + 1, len(tail)):
            if tail[k] >= self.conf_forward_per_state.get(k, self.conf_forward):
                proposal = k
        if proposal > current_state:
            self._contrary_count = 0
            return proposal

        # Backward guard: sustained collapse of support for the current state.
        if current_state > 0 and tail[current_state] < self.conf_backward:
            self._contrary_count += 1
            if self._contrary_count >= self.backward_cycles:
                self._contrary_count = 0
                return self.decide(self.ema, self.conf_forward)
        else:
            self._contrary_count = 0
        return current_state


class QModelV7YOLO_Live(QModelV6YOLO_Live):
    """Drop-in live classifier: v2 render, flat-cost preprocessing, ordinal
    monotone evidence in place of the symmetric count debounce.

    Buffer management, baseline caching, drop-epoch gating, duration
    thresholds, display messages, and the LiveProcess output contract are
    all inherited unchanged; only the inner classification step of
    attempt_classification is replaced.
    """

    TAG = "[QModelV7YOLO_Live]"

    def __init__(self, model_path: str, buffer_window_size: Optional[int] = None):
        if not _LIVE_BASE_AVAILABLE:
            raise RuntimeError(
                "QModelV6YOLO_Live base unavailable (QATCH app modules not "
                "importable) — headless consumers should use "
                "QModelV7FillClassifier / OrdinalEvidence / preprocess_for_cls "
                "directly, as live/replay.py does."
            )
        super().__init__(model_path, buffer_window_size)
        self._evidence = OrdinalEvidence()

    # ------------------------------------------------------------------
    # Classifier plumbing: reuse QModelV7FillClassifier's methods against
    # our own loaded model (super().__init__ chain already loaded it).
    # ------------------------------------------------------------------
    def predict_probs(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        return QModelV7FillClassifier.predict_probs(self, df)

    def attempt_classification(self) -> int:
        """Same contract and side effects as the v6 implementation; the
        raw-int debounce is replaced by ordinal evidence over probability
        vectors, and preprocessing runs at a fixed point budget."""
        if self._data is None or len(self._data) < QModelV6Config.MIN_SLICE_LENGTH:
            return self.current_prediction

        try:
            filtered = self._data[self._data["Relative_time"] > 0.05]
            if filtered.empty:
                Log.d(self.TAG, "Waiting for > 0.05s of buffer data.")
                return self.current_prediction

            self._try_cache_baseline()
            processed_df = preprocess_for_cls(
                filtered,
                baseline_freq=self._cached_baseline_freq,
                baseline_diss=self._cached_baseline_diss,
            )
            if processed_df is None or processed_df.empty:
                Log.w(self.TAG, "Preprocessing returned empty/None DataFrame.")
                return self.current_prediction

            p = self.predict_probs(processed_df)
            if p is None:
                return self.current_prediction

            # Drop gating, expressed in evidence space: until the UI
            # reports the drop applied, the state is no_fill by
            # definition. The accumulator is held in reset so pre-drop
            # frames (which can look fill-like during handling) cannot
            # bank evidence toward an instant post-drop confirmation.
            if not self._drop_applied_received:
                self._evidence.reset()
                self.current_prediction = -1
                return self.current_prediction

            current_state = self.current_prediction + 1  # -> ordinal index
            new_state = self._evidence.update(p, current_state)

            if new_state != current_state:
                previous_prediction = self.current_prediction
                self.current_prediction = new_state - 1
                if new_state < current_state:
                    Log.w(
                        self.TAG,
                        f"Backward state revision {previous_prediction} -> "
                        f"{self.current_prediction} after sustained contrary "
                        f"evidence — an earlier confirmation was wrong.",
                    )
                # A multi-step forward jump (e.g. 0 -> 2ch on a fast run or
                # at analysis time) must fire bookkeeping for every state it
                # passes through, so duration latches release in order.
                step = 1 if new_state > current_state else -1
                for s in range(current_state + step, new_state + step, step):
                    if step > 0:
                        self._on_channel_confirmed(s - 1)

            self._evaluate_duration_threshold(self.current_prediction)
            self._check_initial_fill_timeout()

        except Exception as e:
            n = len(self._data) if self._data is not None else 0
            Log.e(self.TAG, f"Inference failed at buffer size {n}: {str(e)}")

        return self.current_prediction

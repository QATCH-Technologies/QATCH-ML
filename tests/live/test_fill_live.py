import numpy as np
import pandas as pd

from src.systems.qmodel_7_onyx.live.fill_live import (
    MAX_CLS_POINTS,
    OrdinalEvidence,
    preprocess_for_cls,
)


def _one_hot(state: int, n_states: int = 5) -> np.ndarray:
    p = np.full(n_states, 1e-6)
    p[state] = 1.0 - 1e-6 * (n_states - 1)
    return p


def test_ordinal_evidence_forward_confirms_on_unanimous_evidence():
    ev = OrdinalEvidence()
    state = 0
    for _ in range(5):
        state = ev.update(_one_hot(2), state)
    assert state == 2


def test_ordinal_evidence_does_not_confirm_on_single_ambiguous_frame():
    ev = OrdinalEvidence()
    # Cumulative tail P(state>=1) = 0.55, below CONF_FORWARD (0.60): forward
    # confirmation is cumulative-tail-based, so a near-even split with most
    # mass still in state 0 should not confirm state 1 on one frame.
    p = np.array([0.45, 0.45, 0.05, 0.03, 0.02])
    state = ev.update(p, 0)
    assert state == 0


def test_ordinal_evidence_multi_step_forward_jump_allowed():
    """A run whose first observed frame is already fully 3ch should jump
    straight there, not step through every intermediate state."""
    ev = OrdinalEvidence()
    state = 0
    for _ in range(5):
        state = ev.update(_one_hot(4), state)
    assert state == 4


def test_ordinal_evidence_backward_requires_sustained_contrary_evidence():
    ev = OrdinalEvidence()
    state = 0
    for _ in range(5):
        state = ev.update(_one_hot(3), state)
    assert state == 3

    # A single contrary frame should not immediately revert the state.
    state_after_one = ev.update(_one_hot(0), state)
    assert state_after_one == 3

    # Sustained contrary evidence (many cycles) should eventually re-solve.
    for _ in range(ev.backward_cycles + 2):
        state = ev.update(_one_hot(0), state)
    assert state == 0


def test_ordinal_evidence_reset_clears_ema_and_counter():
    ev = OrdinalEvidence()
    ev.update(_one_hot(2), 0)
    ev.reset()
    assert ev.ema is None
    assert ev._contrary_count == 0


def test_ordinal_evidence_decide_falls_back_to_argmax_when_nothing_clears_bar():
    p = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
    assert OrdinalEvidence.decide(p, conf=0.99) == int(np.argmax(p))


def test_ordinal_evidence_instance_overrides_do_not_affect_class_defaults():
    custom = OrdinalEvidence(alpha=0.9, conf_forward=0.99)
    assert custom.alpha == 0.9
    assert OrdinalEvidence.ALPHA == 0.45  # class default untouched


def test_preprocess_for_cls_caps_output_rows_for_long_runs():
    n = 50_000
    t = np.linspace(0, 1000, n)  # long, slow run
    df = pd.DataFrame(
        {
            "Relative_time": t,
            "Dissipation": np.sin(t / 50.0),
            "Resonance_Frequency": np.cos(t / 50.0),
        }
    )
    out = preprocess_for_cls(df)
    assert out is not None
    assert len(out) <= MAX_CLS_POINTS + 2  # +slack for arange edge inclusion


def test_preprocess_for_cls_returns_none_for_degenerate_span():
    df = pd.DataFrame({"Relative_time": [1.0], "Dissipation": [0.0], "Resonance_Frequency": [0.0]})
    assert preprocess_for_cls(df) is None

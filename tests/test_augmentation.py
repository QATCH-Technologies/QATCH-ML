import numpy as np
import pandas as pd

from src.systems.qmodel_7_onyx.augmentation import (
    COL_DISS,
    COL_TIME,
    amplitude_jitter,
    dynamic_box_width_sec,
    inject_noise,
    make_monotone_warp,
    time_warp,
)


def test_make_monotone_warp_is_strictly_increasing():
    rng = np.random.default_rng(0)
    w, S = make_monotone_warp(0.0, 100.0, rng)
    t = np.linspace(0.0, 100.0, 500)
    wt = w(t)
    assert np.all(np.diff(wt) > 0)
    assert S > 0


def test_time_warp_preserves_poi_order_and_returns_positive_stretch():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 50, 200)
    df = pd.DataFrame({COL_TIME: t, COL_DISS: np.sin(t)})
    poi = {"POI1": 5.0, "POI2": 10.0, "POI3": 30.0}
    out_df, out_poi, stretch = time_warp(df, poi, rng)
    ordered = [out_poi["POI1"], out_poi["POI2"], out_poi["POI3"]]
    assert ordered == sorted(ordered)
    assert stretch > 0
    assert np.all(np.diff(out_df[COL_TIME].to_numpy()) > 0)


def test_inject_noise_changes_values_but_preserves_shape():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 10, 500)
    df = pd.DataFrame({COL_TIME: t, COL_DISS: np.sin(t) * 1e-6 + 3e-5})
    out = inject_noise(df, rng)
    assert out.shape == df.shape
    assert not np.allclose(out[COL_DISS].to_numpy(), df[COL_DISS].to_numpy())


def test_amplitude_jitter_preserves_early_baseline_scale():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 10, 1000)
    x = np.full_like(t, 5.0)  # constant signal: baseline == every value
    df = pd.DataFrame({COL_TIME: t, COL_DISS: x})
    out = amplitude_jitter(df, rng, gain_sigma=0.0)  # zero jitter -> identity
    np.testing.assert_allclose(out[COL_DISS].to_numpy(), x)


def test_dynamic_box_width_sec_widens_for_slower_transition():
    """A gradual (low-slope) transition should measure a wider active
    window than a sharp step of the same total amplitude."""
    t = np.linspace(0, 20, 2000)
    poi_t = 10.0

    def make_step(width):
        x = np.zeros_like(t)
        x[t < poi_t - width / 2] = 0.0
        x[t > poi_t + width / 2] = 1.0
        ramp = (t >= poi_t - width / 2) & (t <= poi_t + width / 2)
        x[ramp] = (t[ramp] - (poi_t - width / 2)) / max(width, 1e-9)
        return x

    df_sharp = pd.DataFrame({COL_TIME: t, COL_DISS: make_step(0.2)})
    df_slow = pd.DataFrame({COL_TIME: t, COL_DISS: make_step(4.0)})
    w_sharp = dynamic_box_width_sec(df_sharp, poi_t)
    w_slow = dynamic_box_width_sec(df_slow, poi_t)
    assert w_slow > w_sharp


def test_dynamic_box_width_sec_falls_back_on_flat_signal():
    t = np.linspace(0, 20, 500)
    df = pd.DataFrame({COL_TIME: t, COL_DISS: np.zeros_like(t)})
    w = dynamic_box_width_sec(df, 10.0, min_width_s=0.05)
    assert w == 0.05

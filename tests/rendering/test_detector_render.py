import numpy as np
import pandas as pd

from src.systems.qmodel_7_onyx.rendering.detector_render import (
    derivative_energy,
    generate_channel_det,
)


def _step_signal(n=2000, step_at=0.5, dt=0.005):
    t = np.arange(n) * dt
    x = np.zeros(n)
    step_idx = int(step_at * n)
    x[step_idx:] = 1.0
    return t, x


def test_derivative_energy_peaks_near_the_step():
    t, x = _step_signal()
    df = pd.DataFrame(
        {"Relative_time": t, "Dissipation": x, "Resonance_Frequency": np.zeros_like(x)}
    )
    e = derivative_energy(df)
    assert e.shape == (len(df),)
    peak_idx = int(np.argmax(e))
    step_idx = int(0.5 * len(df))
    # peak should land within a small window of the true step location
    assert abs(peak_idx - step_idx) < len(df) * 0.05


def test_derivative_energy_short_series_returns_zeros():
    df = pd.DataFrame({"Relative_time": [0, 1, 2], "Dissipation": [1, 2, 3]})
    e = derivative_energy(df)
    assert np.all(e == 0)


def test_generate_channel_det_shape_and_dtype():
    t, x = _step_signal(n=500)
    df = pd.DataFrame({"Relative_time": t, "Dissipation": x, "Resonance_Frequency": 1.0 - x})
    img = generate_channel_det(df, img_w=320, img_h=240)
    assert img.shape == (240, 320, 3)
    assert img.dtype == np.uint8

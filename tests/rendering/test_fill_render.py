import warnings

import numpy as np
import pandas as pd

from src.systems.qmodel_7_onyx.rendering.fill_render import (
    FILL_GEN_H,
    FILL_GEN_W,
    FILL_INFERENCE_H,
    FILL_INFERENCE_W,
    generate_fill_cls,
    prepare_cls_input,
    step_coincidence_energy,
)


def _step_df(n=2000, step_at=0.5, dt=0.005, both_signals=True):
    t = np.arange(n) * dt
    step_idx = int(step_at * n)
    diss = np.zeros(n)
    diss[step_idx:] = 1.0
    data = {"Relative_time": t, "Dissipation": diss}
    if both_signals:
        freq = np.zeros(n)
        freq[step_idx:] = 1.0
        data["Resonance_Frequency"] = freq
    return pd.DataFrame(data)


class TestStepCoincidenceEnergy:
    def test_peaks_near_the_step(self):
        df = _step_df()
        e = step_coincidence_energy(df)
        assert e.shape == (len(df),)
        peak_idx = int(np.argmax(e))
        step_idx = int(0.5 * len(df))
        assert abs(peak_idx - step_idx) < len(df) * 0.05

    def test_short_series_returns_zeros(self):
        df = pd.DataFrame({"Relative_time": [0, 1, 2], "Dissipation": [1, 2, 3]})
        e = step_coincidence_energy(df)
        assert e.shape == (3,)
        assert np.all(e == 0)

    def test_missing_signal_columns_returns_zeros(self):
        df = pd.DataFrame({"Relative_time": np.arange(100) * 0.005})
        e = step_coincidence_energy(df)
        assert e.shape == (100,)
        assert np.all(e == 0)

    def test_single_signal_still_produces_a_response(self):
        """Only Dissipation present (Resonance_Frequency missing) should fall
        back to the single-signal response rather than crash on the missing
        column or the geometric-mean coincidence term."""
        df = _step_df(both_signals=False)
        e = step_coincidence_energy(df)
        assert e.shape == (len(df),)
        assert e.max() > 0

    def test_coincident_steps_score_higher_than_a_single_channel_excursion(self):
        """A step that fires in BOTH dissipation and resonance should score at
        least as high as noise firing in only one channel - the whole point
        of the geometric-mean coincidence combination."""
        both = _step_df(both_signals=True)
        single = _step_df(both_signals=False)
        e_both = step_coincidence_energy(both)
        e_single = step_coincidence_energy(single)
        assert e_both.max() >= e_single.max()

    def test_short_series_skips_the_smoothing_pass(self):
        """n just above the n<16 floor is still shorter than the smoothing
        kernel k (~31 samples at the default 0.005s step), so the function
        must return the raw per-sample trace instead of convolving it."""
        df = _step_df(n=20, step_at=0.5)
        e = step_coincidence_energy(df)
        assert e.shape == (20,)
        assert np.all(np.isfinite(e))


class TestGenerateFillCls:
    def test_blank_image_for_none_dataframe(self):
        img = generate_fill_cls(None, img_w=100, img_h=90)
        assert img.shape == (90, 100, 3)
        assert img.dtype == np.uint8
        assert np.all(img == 0)

    def test_blank_image_for_empty_dataframe(self):
        img = generate_fill_cls(pd.DataFrame(), img_w=100, img_h=90)
        assert img.shape == (90, 100, 3)
        assert np.all(img == 0)

    def test_blank_image_for_single_row_dataframe(self):
        df = pd.DataFrame({"Relative_time": [0.0], "Dissipation": [1.0]})
        img = generate_fill_cls(df, img_w=100, img_h=90)
        assert np.all(img == 0)

    def test_renders_nonblank_image_for_valid_input(self):
        df = _step_df(n=500)
        img = generate_fill_cls(df, img_w=320, img_h=240)
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8
        assert (img > 0).any()

    def test_default_dimensions_match_module_constants(self):
        df = _step_df(n=500)
        img = generate_fill_cls(df)
        assert img.shape == (FILL_GEN_H, FILL_GEN_W, 3)

    def test_missing_resonance_column_still_renders_without_error(self):
        """Dissipation + the energy strip should still render even when
        Resonance_Frequency is absent from the dataframe."""
        df = _step_df(n=500, both_signals=False)
        img = generate_fill_cls(df, img_w=320, img_h=240)
        assert img.shape == (240, 320, 3)
        assert (img > 0).any()

    def test_all_nan_column_is_skipped_without_error(self):
        """A present-but-entirely-NaN column has fewer than 2 finite values,
        so _strip_points returns None for that strip and it must be skipped
        rather than raising. Dissipation also feeds step_coincidence_energy
        (it's one of the two coincidence-check signals), so an all-NaN
        Dissipation column propagates NaN into the energy strip too - only
        the untouched Resonance_Frequency strip is expected to render.
        """
        df = _step_df(n=500)
        df["Dissipation"] = np.nan
        with warnings.catch_warnings():
            # Expected: nanmedian/nanstd on an all-NaN column warn by design.
            warnings.simplefilter("ignore", RuntimeWarning)
            img = generate_fill_cls(df, img_w=320, img_h=240)
        assert img.shape == (240, 320, 3)
        assert (img > 0).any()  # resonance strip still drew


class TestPrepareClsInput:
    def test_output_shape_and_dtype_match_inference_geometry(self):
        df = _step_df(n=2000)
        img = prepare_cls_input(df)
        assert img.shape == (FILL_INFERENCE_H, FILL_INFERENCE_W, 3)
        assert img.dtype == np.uint8

    def test_empty_input_still_yields_inference_sized_blank_image(self):
        img = prepare_cls_input(pd.DataFrame())
        assert img.shape == (FILL_INFERENCE_H, FILL_INFERENCE_W, 3)
        assert np.all(img == 0)

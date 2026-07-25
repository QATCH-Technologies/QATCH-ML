import numpy as np

from src.systems.qmodel_7_onyx.rendering._common import _robust_mad, _strip_points


def test_robust_mad_zero_for_constant_signal():
    x = np.full(100, 5.0)
    assert _robust_mad(x) == 0.0


def test_robust_mad_positive_for_noisy_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1.0, 500)
    assert _robust_mad(x) > 0.0


def test_robust_mad_zero_below_min_sample_count():
    assert _robust_mad(np.array([1.0, 2.0, 3.0])) == 0.0


def test_robust_mad_ignores_nan():
    x = np.concatenate([np.full(50, 1.0), [np.nan] * 10])
    assert _robust_mad(x) == 0.0  # the finite part is still constant


def test_strip_points_returns_none_for_degenerate_input():
    assert _strip_points(np.array([1.0]), 100, 100, 0) is None
    assert _strip_points(np.array([]), 100, 100, 0) is None


def test_strip_points_maps_into_expected_pixel_band():
    values = np.linspace(0, 1, 50)
    img_w, strip_h, strip_idx = 200, 100, 1
    pts = _strip_points(values, img_w, strip_h, strip_idx)
    assert pts.shape == (50, 2)
    x, y = pts[:, 0], pts[:, 1]
    assert x.min() >= 0 and x.max() <= img_w - 1
    # y must fall within this strip's pixel band [strip_idx*strip_h, (strip_idx+1)*strip_h)
    assert y.min() >= strip_idx * strip_h
    assert y.max() < (strip_idx + 1) * strip_h


def test_strip_points_constant_values_do_not_divide_by_zero():
    values = np.full(20, 3.0)
    pts = _strip_points(values, 100, 100, 0)
    assert pts is not None
    assert np.all(np.isfinite(pts))

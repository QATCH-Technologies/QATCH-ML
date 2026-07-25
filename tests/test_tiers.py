import builtins

import numpy as np
import pytest

from src.systems.qmodel_7_onyx.tiers import TierScheme, _merge_small_bins, fit_tiers


def test_tierscheme_labels_and_tier_of():
    ts = TierScheme(edges_cp=[2.66, 6.16, 18.14, 73.4])
    assert ts.n_tiers == 6  # 4 edges -> 5 bins + unknown
    assert ts.tier_of(1.0) == 0
    assert ts.tier_of(2.66) == 1
    assert ts.tier_of(100.0) == 4
    assert ts.tier_of(None) == 5
    assert ts.tier_of(float("nan")) == 5


def test_tierscheme_save_load_roundtrip(tmp_path):
    ts = TierScheme(edges_cp=[1.0, 10.0], n_per_tier=[5, 10, 2], method="quantile")
    p = tmp_path / "tiers.json"
    ts.save(p)
    loaded = TierScheme.load(p)
    assert loaded.edges_cp == ts.edges_cp
    assert loaded.labels == ts.labels
    assert loaded.n_per_tier == ts.n_per_tier


def test_merge_small_bins_drops_underpopulated_edges():
    # Three bins via two edges; middle bin is tiny and should get merged away.
    log_v = np.concatenate([np.full(50, 0.0), np.full(2, 1.0), np.full(50, 2.0)])
    edges = _merge_small_bins([0.5, 1.5], log_v, min_support=10)
    assert len(edges) <= 1


def test_fit_tiers_quantile_fallback_without_sklearn(monkeypatch):
    """Force the ImportError path so the quantile fallback is exercised even
    though scikit-learn is installed in this environment."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn.mixture" or name.startswith("sklearn"):
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rng = np.random.default_rng(0)
    v = rng.lognormal(mean=1.0, sigma=1.0, size=500)
    scheme = fit_tiers(v, max_tiers=5, min_support=20, method="auto")
    assert scheme.method == "quantile"
    assert len(scheme.edges_cp) >= 1
    assert all(a < b for a, b in zip(scheme.edges_cp[:-1], scheme.edges_cp[1:], strict=True))


def test_fit_tiers_raises_on_too_few_runs():
    with pytest.raises(SystemExit):
        fit_tiers(np.array([1.0, 2.0, 3.0]), min_support=40)


def test_fit_tiers_gmm_path_produces_ascending_edges():
    rng = np.random.default_rng(1)
    # Two well-separated clusters in log-space -> GMM should find them.
    v = np.concatenate([rng.lognormal(0.0, 0.2, 200), rng.lognormal(4.0, 0.2, 200)])
    scheme = fit_tiers(v, max_tiers=6, min_support=30, method="gmm")
    assert "gmm" in scheme.method
    assert all(a < b for a, b in zip(scheme.edges_cp[:-1], scheme.edges_cp[1:], strict=True))

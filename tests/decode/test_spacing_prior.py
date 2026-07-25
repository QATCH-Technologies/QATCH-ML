import numpy as np

from src.systems.qmodel_7_onyx.decode.spacing_prior import POI_ORDER, SpacingPrior


def _synthetic_configs(n=500, seed=0):
    rng = np.random.default_rng(seed)
    t0 = rng.uniform(3, 10, n)
    gaps = rng.lognormal(mean=[0.0, 1.0, 2.0, 2.5], sigma=0.3, size=(n, 4))
    cum = np.cumsum(gaps, axis=1)
    return np.column_stack([t0, t0 + cum[:, 0], t0 + cum[:, 1], t0 + cum[:, 2], t0 + cum[:, 3]])


def test_fit_produces_one_gapstat_per_pair():
    prior = SpacingPrior.fit(_synthetic_configs())
    assert prior.pairs == [f"{a}->{b}" for a, b in zip(POI_ORDER[:-1], POI_ORDER[1:], strict=True)]
    assert set(prior.gap) == set(prior.pairs)
    for gs in prior.gap.values():
        assert gs.n > 0
        assert gs.min_gap_sec <= gs.max_gap_sec


def test_save_load_roundtrip(tmp_path):
    prior = SpacingPrior.fit(_synthetic_configs())
    p = tmp_path / "prior.json"
    prior.save(p)
    loaded = SpacingPrior.load(p)
    assert loaded.pairs == prior.pairs
    assert loaded.frac_blend == prior.frac_blend
    for pair in prior.pairs:
        assert loaded.gap[pair].log_mu_sec == prior.gap[pair].log_mu_sec


def test_gap_loglik_peaks_near_fitted_median():
    prior = SpacingPrior.fit(_synthetic_configs())
    gs = prior.gap["POI1->POI2"]
    median_gap = float(np.exp(gs.log_mu_sec))
    ll_at_median = prior.gap_loglik(0, median_gap, span_sec=0.0)
    ll_far = prior.gap_loglik(0, median_gap * 50, span_sec=0.0)
    assert ll_at_median > ll_far


def test_gap_loglik_nonpositive_gap_is_heavily_penalized():
    prior = SpacingPrior.fit(_synthetic_configs())
    assert prior.gap_loglik(0, 0.0, span_sec=0.0) <= -1e8
    assert prior.gap_loglik(0, -1.0, span_sec=0.0) <= -1e8


def test_composed_stat_matches_fitted_for_adjacent_pair():
    prior = SpacingPrior.fit(_synthetic_configs())
    gs_direct = prior.gap["POI1->POI2"]
    gs_composed = prior.composed_stat(0, 1)
    assert gs_composed is gs_direct


def test_composed_stat_spans_absent_poi():
    prior = SpacingPrior.fit(_synthetic_configs())
    # POI2 (index 1) absent: compose POI1->POI3 (indices 0->2) from the two
    # fitted consecutive gaps.
    gs = prior.composed_stat(0, 2)
    med_direct = sum(np.exp(prior.gap[p].log_mu_sec) for p in ("POI1->POI2", "POI2->POI3"))
    assert np.isclose(np.exp(gs.log_mu_sec), med_direct, rtol=1e-6)


def test_gap_feasible_respects_learned_bounds():
    prior = SpacingPrior.fit(_synthetic_configs())
    gs = prior.gap["POI3->POI4"]
    assert prior.gap_feasible(2, gs.min_gap_sec, slack=1.0)
    assert not prior.gap_feasible(2, gs.min_gap_sec / 100, slack=1.0)
    assert not prior.gap_feasible(2, gs.max_gap_sec * 100, slack=1.0)


def test_gap_loglik_scoped_full_span_matches_unscoped():
    prior = SpacingPrior.fit(_synthetic_configs())
    ll_scoped = prior.gap_loglik_scoped(0, 1, 5.0, 100.0, span_lo=0, span_hi=len(POI_ORDER) - 1)
    ll_unscoped = prior.gap_loglik_between(0, 1, 5.0, 100.0)
    assert np.isclose(ll_scoped, ll_unscoped)


def test_config_loglik_is_sum_of_consecutive_gaps():
    prior = SpacingPrior.fit(_synthetic_configs())
    times = [5.0, 6.5, 25.0, 60.0, 120.0]
    span = times[-1] - times[0]
    manual = sum(prior.gap_loglik(i, times[i + 1] - times[i], span) for i in range(len(times) - 1))
    assert np.isclose(prior.config_loglik(times), manual)

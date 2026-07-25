import numpy as np

from src.systems.qmodel_7_onyx.decode.dp_decode import (
    Candidate,
    dp_decode,
    greedy_baseline,
    score_configuration,
)
from src.systems.qmodel_7_onyx.decode.spacing_prior import POI_ORDER, SpacingPrior


def _fitted_prior(seed=0, n=500):
    rng = np.random.default_rng(seed)
    t0 = rng.uniform(3, 10, n)
    gaps = rng.lognormal(mean=[0.0, 1.0, 2.0, 2.5], sigma=0.3, size=(n, 4))
    cum = np.cumsum(gaps, axis=1)
    configs = np.column_stack([t0, t0 + cum[:, 0], t0 + cum[:, 1], t0 + cum[:, 2], t0 + cum[:, 3]])
    return SpacingPrior.fit(configs)


def _plausible_times(prior: SpacingPrior, t0: float = 5.0) -> dict:
    """A configuration built from the prior's own fitted median gaps, so it
    is guaranteed to land within the learned feasibility bounds."""
    times = {POI_ORDER[0]: t0}
    t = t0
    for i, pair in enumerate(prior.pairs):
        t += float(np.exp(prior.gap[pair].log_mu_sec))
        times[POI_ORDER[i + 1]] = t
    return times


def test_dp_decode_picks_the_only_feasible_ordered_path():
    prior = _fitted_prior()
    true_times = _plausible_times(prior)
    candidates = {p: [Candidate(time=t, conf=0.9)] for p, t in true_times.items()}
    result = dp_decode(candidates, POI_ORDER, prior, lam=1.0)
    assert result.feasible
    assert not result.fallback_used
    for p, t in true_times.items():
        assert result.chosen[p].time == t


def test_dp_decode_prefers_high_confidence_plausible_config_over_decoy():
    prior = _fitted_prior()
    true_times = _plausible_times(prior)
    candidates = {}
    for p, t in true_times.items():
        cands = [Candidate(time=t, conf=0.6)]
        candidates[p] = cands
    # Give POI3 a confident decoy that is implausibly close to POI2 in time
    # (violates the learned spacing prior even though it's high-confidence).
    candidates["POI3"] = [
        Candidate(time=true_times["POI3"], conf=0.6),
        Candidate(time=true_times["POI2"] + 0.01, conf=0.95),
    ]
    result = dp_decode(candidates, POI_ORDER, prior, lam=5.0)  # heavy prior weight
    # The prior should reject the near-zero-gap decoy in favor of the
    # plausibly-spaced true candidate.
    assert result.chosen["POI3"].time == true_times["POI3"]


def test_dp_decode_enforces_strict_time_ordering():
    prior = _fitted_prior()
    candidates = {
        "POI1": [Candidate(time=10.0, conf=0.9)],
        "POI2": [Candidate(time=10.0, conf=0.9)],  # identical time -> not > POI1
    }
    result = dp_decode(candidates, ["POI1", "POI2"], prior, lam=1.0, require_feasible=False)
    # With only one candidate each and a tied timestamp, no valid strictly-
    # ordered path exists; the decoder must fall back rather than emit an
    # illegal (non-increasing) configuration.
    if result.chosen:
        times = [result.chosen[p].time for p in ["POI1", "POI2"] if p in result.chosen]
        assert times == sorted(times, key=lambda x: x) if len(times) < 2 else times[0] < times[1]


def test_dp_decode_empty_present_pois_returns_infeasible():
    prior = _fitted_prior()
    result = dp_decode({}, [], prior)
    assert result.chosen == {}
    assert not result.feasible


def test_dp_decode_never_worse_than_greedy_floor_when_no_path_exists():
    """When candidates are pathological (e.g. every POI's only candidate is
    at the same instant), dp_decode must still return SOME chosen set (the
    greedy floor), never crash or return nothing for POIs that have data."""
    prior = _fitted_prior()
    candidates = {p: [Candidate(time=1.0, conf=0.5)] for p in POI_ORDER}
    result = dp_decode(candidates, POI_ORDER, prior, lam=1.0)
    assert set(result.chosen) == set(POI_ORDER)


def test_greedy_baseline_picks_highest_confidence_per_poi():
    candidates = {
        "POI1": [Candidate(time=1.0, conf=0.2), Candidate(time=2.0, conf=0.8)],
        "POI2": [Candidate(time=5.0, conf=0.9)],
    }
    out = greedy_baseline(candidates, ["POI1", "POI2"])
    assert out["POI1"].time == 2.0
    assert out["POI2"].time == 5.0


def test_score_configuration_is_consistent_with_dp_decode_scoring():
    prior = _fitted_prior()
    true_times = _plausible_times(prior)
    candidates = {p: [Candidate(time=t, conf=0.9)] for p, t in true_times.items()}
    result = dp_decode(candidates, POI_ORDER, prior, lam=1.0)
    rescored = score_configuration(result.chosen, prior, lam=1.0)
    assert np.isclose(rescored, result.total_score, rtol=1e-6)


def test_score_configuration_empty_chosen_returns_very_low_score():
    prior = _fitted_prior()
    assert score_configuration({}, prior) <= -1e17

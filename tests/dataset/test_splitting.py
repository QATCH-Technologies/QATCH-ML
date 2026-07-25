from src.systems.qmodel_7_onyx.corpus import RunRecord
from src.systems.qmodel_7_onyx.dataset.splitting import repeat_factor, stratified_group_split
from src.systems.qmodel_7_onyx.tiers import TierScheme


def _runs(n, tier_edges=(10.0,), poi_count=5):
    """n runs split evenly between two viscosity tiers."""
    out = []
    for i in range(n):
        visc = 1.0 if i % 2 == 0 else 100.0
        poi_times = {f"POI{k}": float(k) for k in range(1, poi_count + 1)}
        out.append(
            RunRecord(run_id=f"{i:05d}", csv_path=None, poi_times=poi_times, viscosity_cP=visc)
        )
    return out


def test_stratified_group_split_has_no_leakage():
    runs = _runs(40)
    tiers = TierScheme(edges_cp=[10.0])
    result = stratified_group_split(runs, tiers, val_frac=0.2, seed=0)
    assert set(result.train_ids).isdisjoint(set(result.val_ids))
    assert set(result.train_ids) | set(result.val_ids) == {r.run_id for r in runs}


def test_stratified_group_split_every_multi_run_stratum_present_in_val():
    runs = _runs(40)
    tiers = TierScheme(edges_cp=[10.0])
    result = stratified_group_split(runs, tiers, val_frac=0.2, seed=1)
    assert len(result.val_ids) >= 2  # both tiers (>=2 runs each) contribute >=1 to val


def test_stratified_group_split_singleton_stratum_stays_in_train():
    runs = _runs(2)  # one run per tier -> both singleton strata
    tiers = TierScheme(edges_cp=[10.0])
    result = stratified_group_split(runs, tiers, val_frac=0.5, seed=0)
    assert result.val_ids == []
    assert set(result.train_ids) == {r.run_id for r in runs}


def test_stratified_group_split_is_deterministic_given_seed():
    runs = _runs(40)
    tiers = TierScheme(edges_cp=[10.0])
    r1 = stratified_group_split(runs, tiers, val_frac=0.2, seed=42)
    r2 = stratified_group_split(runs, tiers, val_frac=0.2, seed=42)
    assert r1.train_ids == r2.train_ids
    assert r1.val_ids == r2.val_ids


def test_repeat_factor_upsamples_rare_tier_more():
    counts = {0: 1000, 1: 10}
    rare = repeat_factor(1, counts, cap=10)
    common = repeat_factor(0, counts, cap=10)
    assert rare > common
    assert common == 1


def test_repeat_factor_respects_cap():
    counts = {0: 100000, 1: 1}
    assert repeat_factor(1, counts, cap=5) == 5

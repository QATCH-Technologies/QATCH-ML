import numpy as np
import pandas as pd

from src.systems.qmodel_7_onyx.corpus import (
    RunRecord,
    dedupe_runs,
    discover_runs,
    load_run_filter,
    truth_times,
    viscosity_tier,
)


def test_discover_runs_finds_complete_fill(tmp_path, make_run, complete_poi_times):
    make_run(tmp_path, "00000", complete_poi_times, viscosity_cP=15.0)
    runs = discover_runs(tmp_path)
    assert len(runs) == 1
    r = runs[0]
    assert r.run_id == "00000"
    assert set(r.poi_times) == set(complete_poi_times)
    assert r.viscosity_cP == 15.0


def test_discover_runs_accepts_partial_prefix_fill(tmp_path, make_run):
    make_run(tmp_path, "00000", {"POI1": 5.0, "POI2": 6.5, "POI3": 25.0})
    runs = discover_runs(tmp_path)
    assert len(runs) == 1
    assert set(runs[0].poi_times) == {"POI1", "POI2", "POI3"}


def test_discover_runs_skips_dir_without_poi_file(tmp_path):
    d = tmp_path / "00000"
    d.mkdir()
    t = np.arange(0, 10, 0.02)
    pd.DataFrame({"Relative_time": t, "Dissipation": t, "Resonance_Frequency": t}).to_csv(
        d / "00000.csv", index=False
    )
    assert discover_runs(tmp_path) == []


def test_truth_times_rejects_non_ascending_index(tmp_path):
    """A POI whose row index is not strictly after the previous one is
    dropped, along with everything after it (the acceptance rule truncates,
    it does not skip-and-continue)."""
    poi_path = tmp_path / "bad_poi.csv"
    # POI1 idx=100, POI2 idx=50 (goes backward) -> POI2 and beyond rejected.
    rows = [100, 50, 50, 200, 300, 400]
    pd.Series(rows).to_csv(poi_path, index=False, header=False)
    time_axis = np.arange(0, 1000) * 0.1
    out = truth_times(poi_path, time_axis)
    assert list(out.keys()) == ["POI1"]


def test_truth_times_rejects_tail_poi(tmp_path):
    """A POI whose index lands in the last ~0.1% of the run (or beyond) is
    treated as unset — it's noise/tail artifact, not a real mark."""
    poi_path = tmp_path / "tail_poi.csv"
    n = 10_000
    # row2 is the legacy POI2-shim row (skipped by truth_times); POI3/POI4/POI5
    # read from rows 3/4/5 respectively.
    rows = [100, 200, 200, 300, 400, n - 1]  # POI5 (row 5) at the very last sample
    pd.Series(rows).to_csv(poi_path, index=False, header=False)
    time_axis = np.arange(n) * 0.01
    out = truth_times(poi_path, time_axis)
    assert "POI5" not in out
    assert set(out) == {"POI1", "POI2", "POI3", "POI4"}


def test_truth_times_missing_row_stops_prefix(tmp_path):
    poi_path = tmp_path / "short_poi.csv"
    rows = [100, 200]  # only POI1, POI2 rows present at all
    pd.Series(rows).to_csv(poi_path, index=False, header=False)
    time_axis = np.arange(0, 1000) * 0.1
    out = truth_times(poi_path, time_axis)
    assert set(out) == {"POI1", "POI2"}


def test_viscosity_tier_boundaries():
    assert viscosity_tier(1.0) == 0
    assert viscosity_tier(2.66) == 1  # boundary is exclusive on the low side of the next bin
    assert viscosity_tier(1000.0) == 4
    assert viscosity_tier(None) == 5
    assert viscosity_tier(float("nan")) == 5


def test_dedupe_runs_removes_identical_poi_content():
    a = RunRecord("run_a", None, {"POI1": 1.0, "POI2": 2.0}, None)
    b = RunRecord("run_b", None, {"POI1": 1.0, "POI2": 2.0}, None)  # same content, diff id
    c = RunRecord("run_c", None, {"POI1": 1.0, "POI2": 3.0}, None)  # genuinely different
    out = dedupe_runs([a, b, c])
    assert len(out) == 2
    assert {r.run_id for r in out} == {"run_a", "run_c"}


def test_load_run_filter_from_manifest_val_ids(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text('{"val_ids": ["00001", "00002"], "train_ids": ["00003"]}')
    assert load_run_filter(p) == {"00001", "00002"}


def test_load_run_filter_from_plain_list(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text('["00001", "00002"]')
    assert load_run_filter(p) == {"00001", "00002"}


def test_load_run_filter_from_text_lines(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("00001\n00002\n\n00003\n")
    assert load_run_filter(p) == {"00001", "00002", "00003"}

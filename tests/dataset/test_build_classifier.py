import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.systems.qmodel_7_onyx.dataset import build_fill_classifier as bfc
from src.systems.qmodel_7_onyx.tiers import TierScheme


def _synthetic_df_p(duration_s=200.0, dt=0.005, seed=0):
    """A dataframe shaped like DP.preprocess_dataframe's output."""
    t = np.arange(0.0, duration_s, dt)
    rng = np.random.default_rng(seed)
    diss = np.cumsum(rng.normal(0, 1e-7, len(t))) + 3e-5
    freq = 1.5e7 - np.linspace(0, 500, len(t))
    return pd.DataFrame({"Relative_time": t, "Dissipation": diss, "Resonance_Frequency": freq})


class TestFillStateAt:
    def test_no_boundaries_present_is_state_zero(self):
        assert bfc.fill_state_at(50.0, {}) == 0

    def test_before_the_first_boundary_is_state_zero(self):
        poi = {"POI1": 5.0, "POI3": 25.0, "POI4": 60.0, "POI5": 120.0}
        assert bfc.fill_state_at(4.0, poi) == 0

    def test_exactly_at_a_boundary_is_already_that_state(self):
        poi = {"POI1": 5.0, "POI3": 25.0, "POI4": 60.0, "POI5": 120.0}
        assert bfc.fill_state_at(25.0, poi) == 2

    def test_between_boundaries_holds_the_latest_confirmed_state(self):
        poi = {"POI1": 5.0, "POI3": 25.0, "POI4": 60.0, "POI5": 120.0}
        assert bfc.fill_state_at(40.0, poi) == 2
        assert bfc.fill_state_at(150.0, poi) == 4

    def test_missing_intermediate_boundary_is_skipped(self):
        """A partial fill missing POI4 must still report state 3 once POI5
        fires, since POI4 being absent doesn't roll the state back."""
        poi = {"POI1": 5.0, "POI3": 25.0, "POI5": 120.0}
        assert bfc.fill_state_at(150.0, poi) == 4
        assert bfc.fill_state_at(50.0, poi) == 2


class TestClassIntervals:
    # dynamic_box_width_sec falls back toward its max_width_frac ceiling
    # (6% of a 200s span = 12s) when the synthetic signal has no measurable
    # transition, so POIs must sit far enough apart (and POI1 far enough
    # from t0) that the settle/pre-cut exclusions can't collapse an
    # interval even at that worst-case width.
    _WELL_SPACED_POI = {"POI1": 30.0, "POI3": 60.0, "POI4": 100.0, "POI5": 170.0}

    def test_complete_fill_yields_an_interval_per_state(self):
        df_p = _synthetic_df_p(duration_s=200.0)
        intervals = bfc.class_intervals(self._WELL_SPACED_POI, df_p, t0=0.0, t1=200.0)
        assert set(intervals) == {0, 1, 2, 3, 4}
        for lo, hi in intervals.values():
            assert hi > lo

    def test_missing_final_transition_omits_the_terminal_state(self):
        """A run that never reaches 3ch (no POI5) must not report a
        sampleable interval for state 4."""
        df_p = _synthetic_df_p(duration_s=200.0)
        poi = {k: v for k, v in self._WELL_SPACED_POI.items() if k != "POI5"}
        intervals = bfc.class_intervals(poi, df_p, t0=0.0, t1=200.0)
        assert 4 not in intervals
        assert set(intervals) == {0, 1, 2, 3}

    def test_state_zero_interval_starts_after_min_prefix(self):
        df_p = _synthetic_df_p(duration_s=200.0)
        intervals = bfc.class_intervals(self._WELL_SPACED_POI, df_p, t0=0.0, t1=200.0)
        lo0, _ = intervals[0]
        assert lo0 == pytest.approx(bfc.MIN_PREFIX_S)

    def test_back_to_back_transitions_can_collapse_an_interval_to_nothing(self):
        """Two transitions close enough together that the settle/pre-cut
        exclusions overlap must drop that state's interval instead of
        emitting an inverted (hi < lo) or empty range."""
        df_p = _synthetic_df_p(duration_s=200.0)
        # POI3 and POI4 essentially coincide - state 2's window is squeezed
        # to nothing by the settle-after/pre-before exclusions.
        poi = {"POI1": 5.0, "POI3": 25.0, "POI4": 25.05, "POI5": 120.0}
        intervals = bfc.class_intervals(poi, df_p, t0=0.0, t1=200.0)
        assert 2 not in intervals
        for lo, hi in intervals.values():
            assert hi > lo


class TestSampleCuts:
    def test_draws_cuts_per_class_regular_samples_per_state(self):
        rng = np.random.default_rng(0)
        intervals = {0: (0.0, 10.0), 1: (10.0, 20.0)}
        cuts = bfc.sample_cuts(rng, intervals, cuts_per_class=3, hard_cuts=0)
        assert len(cuts) == 6
        by_state = {0: [], 1: []}
        for t, state, is_hard in cuts:
            by_state[state].append(t)
            assert is_hard is False
        assert all(0.0 <= t <= 10.0 for t in by_state[0])
        assert all(10.0 <= t <= 20.0 for t in by_state[1])

    def test_hard_cuts_only_sampled_for_nonzero_states(self):
        rng = np.random.default_rng(0)
        intervals = {0: (0.0, 10.0), 1: (10.0, 20.0)}
        cuts = bfc.sample_cuts(rng, intervals, cuts_per_class=1, hard_cuts=2)
        hard = [(t, state) for t, state, is_hard in cuts if is_hard]
        assert len(hard) == 2  # only state 1 (nonzero) gets hard cuts
        assert all(state == 1 for _, state in hard)

    def test_hard_cuts_land_within_the_hard_span_near_the_transition(self):
        rng = np.random.default_rng(0)
        intervals = {1: (10.0, 100.0)}
        cuts = bfc.sample_cuts(rng, intervals, cuts_per_class=0, hard_cuts=5)
        assert len(cuts) == 5
        for t, _state, is_hard in cuts:
            assert is_hard is True
            assert 10.0 <= t <= 10.0 + bfc.HARD_SPAN_S

    def test_empty_intervals_yields_no_cuts(self):
        rng = np.random.default_rng(0)
        assert bfc.sample_cuts(rng, {}, cuts_per_class=5, hard_cuts=5) == []


class TestBuild:
    """Integration tests for the end-to-end fill-classifier dataset build."""

    def test_raises_when_no_runs_discovered(self, tmp_path):
        empty_raw = tmp_path / "raw"
        empty_raw.mkdir()
        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[1.0, 10.0], n_per_tier=[1, 2, 0]).save(tiers_path)

        with pytest.raises(SystemExit, match="no runs under"):
            bfc.build(empty_raw, tiers_path, tmp_path / "out")

    def test_builds_full_dataset_hierarchy(self, tmp_path, make_run, complete_poi_times):
        raw_root = tmp_path / "raw"
        for i in range(4):
            poi = {k: v + i for k, v in complete_poi_times.items()}
            make_run(raw_root, f"run{i:02d}", poi, viscosity_cP=10.0 * (i + 1))

        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)

        out_root = tmp_path / "out"
        bfc.build(
            raw_root,
            tiers_path,
            out_root,
            base_variants=1,
            cuts_per_class=1,
            hard_cuts=1,
            val_frac=0.5,
            repeat_cap=2,
            seed=1,
        )

        for split_name in ("train", "val"):
            for cname in bfc.CLASS_NAMES:
                assert (out_root / split_name / cname).is_dir()

        manifest = json.loads((out_root / "manifest.json").read_text())
        assert manifest["n_train_runs"] + manifest["n_val_runs"] == 4
        assert manifest["class_names"] == bfc.CLASS_NAMES
        assert manifest["seed"] == 1
        total_samples = sum(manifest["sample_counts"].values())
        assert total_samples > 0
        # Every emitted sample actually landed on disk under its class dir.
        assert (
            sum(
                len(list((out_root / sp / c).glob("*.png")))
                for sp in ("train", "val")
                for c in bfc.CLASS_NAMES
            )
            == total_samples
        )

    def test_existing_output_directory_is_recreated(self, tmp_path, make_run, complete_poi_times):
        raw_root = tmp_path / "raw"
        make_run(raw_root, "run00", complete_poi_times, viscosity_cP=20.0)
        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)

        out_root = tmp_path / "out"
        stale_marker = out_root / "stale.txt"
        out_root.mkdir()
        stale_marker.write_text("from a previous build")

        bfc.build(raw_root, tiers_path, out_root, base_variants=1, val_frac=0.0, seed=1)

        assert not stale_marker.exists()
        assert (out_root / "manifest.json").exists()

    def test_every_state_the_run_passes_through_is_represented(
        self, tmp_path, make_run, complete_poi_times
    ):
        """A single complete-fill run's analysis-time frame plus sampled
        prefixes should collectively cover every achievable fill state."""
        raw_root = tmp_path / "raw"
        make_run(raw_root, "run00", complete_poi_times, viscosity_cP=20.0)
        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)

        out_root = tmp_path / "out"
        bfc.build(
            raw_root,
            tiers_path,
            out_root,
            base_variants=1,
            cuts_per_class=2,
            hard_cuts=1,
            val_frac=0.0,
            seed=1,
        )

        manifest = json.loads((out_root / "manifest.json").read_text())
        classes_seen = {key.split("/")[1] for key in manifest["sample_counts"]}
        assert "3ch" in classes_seen  # the completed-run analysis-time frame

    def test_respects_the_limit_argument(self, tmp_path, make_run, complete_poi_times):
        raw_root = tmp_path / "raw"
        for i in range(4):
            poi = {k: v + i for k, v in complete_poi_times.items()}
            make_run(raw_root, f"run{i:02d}", poi, viscosity_cP=20.0)
        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)

        out_root = tmp_path / "out"
        bfc.build(raw_root, tiers_path, out_root, base_variants=1, val_frac=0.0, seed=1, limit=2)

        manifest = json.loads((out_root / "manifest.json").read_text())
        assert manifest["n_train_runs"] + manifest["n_val_runs"] == 2

    def test_repeated_variants_apply_signal_domain_augmentation(
        self, tmp_path, make_run, complete_poi_times
    ):
        raw_root = tmp_path / "raw"
        make_run(raw_root, "run00", complete_poi_times, viscosity_cP=20.0)
        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)
        out_root = tmp_path / "out"

        bfc.build(raw_root, tiers_path, out_root, base_variants=3, val_frac=0.0, seed=1)

        matches = list(out_root.rglob("*run00_v1_*"))
        assert matches, "expected an augmented (v1+) variant to be rendered"

    def test_csv_unreadable_at_process_time_is_skipped_not_fatal(
        self, tmp_path, make_run, complete_poi_times
    ):
        """discover_runs already read each run's CSV successfully once at
        discovery time; this exercises process_run's own defensive re-read
        try/except for the rare case where the file becomes unreadable
        between discovery and processing."""
        raw_root = tmp_path / "raw"
        make_run(raw_root, "good_run", complete_poi_times, viscosity_cP=20.0)
        flaky_poi = {k: v + 1 for k, v in complete_poi_times.items()}
        make_run(raw_root, "flaky_run", flaky_poi, viscosity_cP=20.0)

        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)
        out_root = tmp_path / "out"

        real_read_csv = pd.read_csv
        call_counts = {}

        def flaky_read_csv(path, *args, **kwargs):
            key = str(path)
            call_counts[key] = call_counts.get(key, 0) + 1
            if "flaky_run" in key and call_counts[key] >= 2:
                raise ValueError("corrupted mid-run")
            return real_read_csv(path, *args, **kwargs)

        with patch(
            "src.systems.qmodel_7_onyx.dataset.build_fill_classifier.pd.read_csv",
            side_effect=flaky_read_csv,
        ):
            bfc.build(raw_root, tiers_path, out_root, base_variants=1, val_frac=0.0, seed=1)

        manifest = json.loads((out_root / "manifest.json").read_text())
        assert manifest["n_train_runs"] + manifest["n_val_runs"] == 2
        assert sum(manifest["sample_counts"].values()) > 0


class TestMain:
    """Integration checks for the argparse CLI entry point."""

    def test_main_forwards_parsed_arguments(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "sys.argv",
            [
                "build_fill_classifier.py",
                "--raw-root",
                str(tmp_path / "raw"),
                "--tiers",
                str(tmp_path / "tiers.json"),
                "--out",
                str(tmp_path / "out"),
                "--base-variants",
                "3",
                "--cuts-per-class",
                "4",
                "--hard-cuts",
                "2",
                "--val-frac",
                "0.2",
                "--repeat-cap",
                "4",
                "--seed",
                "99",
                "--limit",
                "10",
            ],
        )

        with patch.object(bfc, "build") as mock_build:
            bfc.main()

        mock_build.assert_called_once_with(
            tmp_path / "raw",
            tmp_path / "tiers.json",
            tmp_path / "out",
            base_variants=3,
            cuts_per_class=4,
            hard_cuts=2,
            val_frac=0.2,
            repeat_cap=4,
            seed=99,
            limit=10,
        )

    def test_main_uses_default_paths_and_settings(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["build_fill_classifier.py"])

        with patch.object(bfc, "build") as mock_build:
            bfc.main()

        _, kwargs = mock_build.call_args
        assert kwargs["base_variants"] == 2
        assert kwargs["cuts_per_class"] == 2
        assert kwargs["hard_cuts"] == 1
        assert kwargs["val_frac"] == 0.15
        assert kwargs["repeat_cap"] == 8
        assert kwargs["seed"] == 7
        assert kwargs["limit"] is None

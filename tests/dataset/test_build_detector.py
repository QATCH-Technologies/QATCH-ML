import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.systems.qmodel_7_onyx.dataset import build_detectors as bd
from src.systems.qmodel_7_onyx.tiers import TierScheme


class _FixedRNG:
    """Deterministic stand-in for numpy.random.Generator's random()/uniform().

    numpy.random.Generator is a C-extension type whose bound methods are
    read-only, so `rng.random = lambda: ...` raises AttributeError - this
    stub gives full, precise control over which branch `_sample_cut` takes
    (via `random_value`) and where in an interval it lands (via
    `uniform_value`, defaulting to the interval midpoint).
    """

    def __init__(self, random_value=0.5, uniform_value=None):
        self._random_value = random_value
        self._uniform_value = uniform_value

    def random(self):
        return self._random_value

    def uniform(self, lo, hi):
        if self._uniform_value is not None:
            return self._uniform_value
        return (lo + hi) / 2.0


def _synthetic_df_p(duration_s=200.0, dt=0.005, seed=0):
    """A dataframe shaped like DP.preprocess_dataframe's output: a fixed
    Relative_time grid plus Dissipation/Resonance_Frequency columns."""
    t = np.arange(0.0, duration_s, dt)
    rng = np.random.default_rng(seed)
    diss = np.cumsum(rng.normal(0, 1e-7, len(t))) + 3e-5
    freq = 1.5e7 - np.linspace(0, 500, len(t))
    return pd.DataFrame({"Relative_time": t, "Dissipation": diss, "Resonance_Frequency": freq})


class TestSampleCut:
    """Tests for _sample_cut's canonical/wide/negative slicing modes."""

    def test_missing_anchor_treats_run_as_negative(self):
        """When the stage target never happened (partial fill), the run
        should always be treated as a negative sample."""
        rng = np.random.default_rng(0)
        cut_t, is_negative = bd._sample_cut(rng, anchor_t=None, next_t=None, t0=0.0, t1=100.0)
        assert is_negative is True
        assert cut_t is not None
        assert 0.0 < cut_t <= 100.0

    def test_missing_anchor_and_too_short_signal_yields_no_cut(self):
        rng = np.random.default_rng(0)
        cut_t, is_negative = bd._sample_cut(
            rng, anchor_t=None, next_t=None, t0=0.0, t1=bd.MIN_SLICE_S - 0.1
        )
        assert cut_t is None
        assert is_negative is True

    def test_negative_draw_cuts_before_the_anchor(self):
        """u < P_NEGATIVE must produce a cut strictly before the anchor time."""
        rng = _FixedRNG(random_value=0.0)  # 0.0 < P_NEGATIVE
        cut_t, is_negative = bd._sample_cut(rng, anchor_t=50.0, next_t=90.0, t0=0.0, t1=100.0)
        assert is_negative is True
        assert cut_t < 50.0 - bd.CUT_MARGIN_S + 1e-9

    def test_negative_draw_falls_through_to_positive_when_head_too_short(self):
        """If the run's head (before the anchor) is too short for a negative
        slice, the negative branch must fall through to a positive cut
        rather than returning None."""
        rng = _FixedRNG(random_value=0.0)  # 0.0 < P_NEGATIVE
        cut_t, is_negative = bd._sample_cut(rng, anchor_t=1.0, next_t=90.0, t0=0.0, t1=100.0)
        assert cut_t is not None
        assert is_negative is False

    def test_canonical_draw_cuts_between_anchor_and_next_poi(self):
        rng = _FixedRNG(random_value=0.5)  # P_NEGATIVE <= 0.5 < P_NEGATIVE + P_CANONICAL
        cut_t, is_negative = bd._sample_cut(rng, anchor_t=50.0, next_t=90.0, t0=0.0, t1=100.0)
        assert is_negative is False
        assert 50.0 + bd.CUT_MARGIN_S <= cut_t <= 90.0

    def test_wide_draw_cuts_anywhere_after_the_anchor_when_no_next_poi(self):
        """With no next POI (e.g. the terminal ch3 stage), positive cuts must
        use the wide mode even on a canonical-probability draw."""
        rng = _FixedRNG(random_value=0.5)
        cut_t, is_negative = bd._sample_cut(rng, anchor_t=50.0, next_t=None, t0=0.0, t1=100.0)
        assert is_negative is False
        assert cut_t is not None
        assert cut_t > 50.0

    def test_no_positive_slice_available_returns_none(self):
        """When the anchor sits right at the end of the signal, neither a
        canonical nor a wide positive cut is possible."""
        rng = _FixedRNG(random_value=0.5)  # avoid the negative branch
        cut_t, is_negative = bd._sample_cut(
            rng, anchor_t=100.0 - bd.CUT_MARGIN_S, next_t=None, t0=0.0, t1=100.0
        )
        assert cut_t is None


class TestRenderAndLabel:
    """Tests for _render_and_label's slicing, rendering, and box geometry."""

    def test_too_short_time_span_returns_none(self):
        df_p = _synthetic_df_p(duration_s=1.0)  # well under MIN_SLICE_S
        result = bd._render_and_label(df_p, cut_t=1.0, stage="ch1", poi_times={}, is_negative=True)
        assert result is None

    def test_too_few_samples_returns_none_even_with_a_long_span(self):
        """A sparse slice (< 64 rows) must be rejected even when its time
        span alone would clear MIN_SLICE_S - the row-count and span checks
        are independent guards."""
        df_p = pd.DataFrame(
            {"Relative_time": np.linspace(0.0, 10.0, 50), "Dissipation": np.zeros(50)}
        )
        result = bd._render_and_label(df_p, cut_t=10.0, stage="ch1", poi_times={}, is_negative=True)
        assert result is None

    def test_negative_sample_has_no_label_lines(self):
        df_p = _synthetic_df_p()
        result = bd._render_and_label(
            df_p, cut_t=100.0, stage="ch1", poi_times={"POI3": 200.0}, is_negative=True
        )
        assert result is not None
        img, lines = result
        assert img.shape == (bd.IMG_H, bd.IMG_W, 3)
        assert lines == []

    def test_positive_sample_emits_one_label_line_per_target_in_frame(self):
        df_p = _synthetic_df_p()
        result = bd._render_and_label(
            df_p, cut_t=100.0, stage="ch1", poi_times={"POI3": 50.0}, is_negative=False
        )
        assert result is not None
        img, lines = result
        assert img.shape == (bd.IMG_H, bd.IMG_W, 3)
        assert len(lines) == 1
        cls_id, xc, yc, w, h = lines[0].split()
        assert cls_id == "0"
        assert 0.0 <= float(xc) <= 1.0
        assert float(yc) == 0.5
        assert float(h) == bd.BOX_H_FRAC

    def test_target_outside_the_slice_window_emits_no_label(self):
        """A POI at or beyond cut_t must not produce a label for a frame
        that does not actually contain it."""
        df_p = _synthetic_df_p()
        result = bd._render_and_label(
            df_p, cut_t=100.0, stage="ch1", poi_times={"POI3": 150.0}, is_negative=False
        )
        assert result is not None
        _, lines = result
        assert lines == []

    def test_init_stage_clamps_overlapping_boxes(self):
        """Two POI1/POI2 targets sitting closer together than their natural
        box widths must be clamped to 90% of their center-to-center gap so
        neither box swallows the other's event."""
        df_p = _synthetic_df_p()
        # POI1/POI2 only 0.05s apart - far closer than the init stage's
        # natural box width at this span.
        result = bd._render_and_label(
            df_p,
            cut_t=100.0,
            stage="init",
            poi_times={"POI1": 50.0, "POI2": 50.05},
            is_negative=False,
        )
        assert result is not None
        _, lines = result
        assert len(lines) == 2
        gap_px = abs(50.05 - 50.0) / 100.0 * bd.IMG_W
        # The clamp is min(natural_w, 0.9*gap_px), but a 4px floor still
        # applies on top of that (see _render_and_label) - here gap_px is
        # tiny, so the floor is what actually binds.
        expected_ceiling = max(4.0, 0.9 * gap_px)
        for line in lines:
            _, _, _, w_frac, _ = line.split()
            w_px = float(w_frac) * bd.IMG_W
            # Labels are written as a fraction with 6 decimal places, so up
            # to ~0.5e-6 * IMG_W of pixel-space rounding is expected here.
            assert w_px <= expected_ceiling + 1e-2

    def test_zoom_window_uses_explicit_start_time(self):
        """t_start slices an explicit [t_start, cut_t) window instead of a
        cascade prefix, as used for the zoom-refinement stages."""
        df_p = _synthetic_df_p()
        result = bd._render_and_label(
            df_p,
            cut_t=60.0,
            stage="ch1_zoom",
            poi_times={"POI3": 50.0},
            is_negative=False,
            t_start=40.0,
        )
        assert result is not None
        img, lines = result
        assert img.shape == (bd.IMG_H, bd.IMG_W, 3)
        assert len(lines) == 1


class TestBuild:
    """Integration tests for the end-to-end dataset build."""

    def test_raises_when_no_runs_discovered(self, tmp_path):
        empty_raw = tmp_path / "raw"
        empty_raw.mkdir()
        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[1.0, 10.0], n_per_tier=[1, 2, 0]).save(tiers_path)

        with pytest.raises(SystemExit, match="no runs under"):
            bd.build(empty_raw, tiers_path, tmp_path / "out")

    def test_builds_full_dataset_hierarchy(self, tmp_path, make_run, complete_poi_times):
        raw_root = tmp_path / "raw"
        for i in range(4):
            poi = {k: v + i for k, v in complete_poi_times.items()}
            make_run(raw_root, f"run{i:02d}", poi, viscosity_cP=10.0 * (i + 1))

        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)

        out_root = tmp_path / "out"
        bd.build(
            raw_root,
            tiers_path,
            out_root,
            base_variants=1,
            val_frac=0.5,
            repeat_cap=2,
            seed=1,
        )

        # Every cascade + zoom stage gets its own YOLO hierarchy.
        for stage in bd.ALL_STAGES:
            assert (out_root / stage / "data.yaml").exists()
            for split_name in ("train", "val"):
                images = list((out_root / stage / "images" / split_name).glob("*.png"))
                labels = list((out_root / stage / "labels" / split_name).glob("*.txt"))
                assert len(images) == len(labels)

        manifest = json.loads((out_root / "manifest.json").read_text())
        assert manifest["n_train_runs"] + manifest["n_val_runs"] == 4
        assert manifest["seed"] == 1
        assert set(manifest["train_ids"]) | set(manifest["val_ids"]) == {
            f"run{i:02d}" for i in range(4)
        }
        # At least some positive AND negative samples were generated overall.
        counts = manifest["sample_counts"]
        assert any(k.endswith("/pos") for k in counts)

    def test_existing_output_directory_is_recreated(self, tmp_path, make_run, complete_poi_times):
        raw_root = tmp_path / "raw"
        make_run(raw_root, "run00", complete_poi_times, viscosity_cP=20.0)
        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)

        out_root = tmp_path / "out"
        stale_marker = out_root / "stale.txt"
        out_root.mkdir()
        stale_marker.write_text("from a previous build")

        bd.build(raw_root, tiers_path, out_root, base_variants=1, val_frac=0.0, seed=1)

        assert not stale_marker.exists()
        assert (out_root / "manifest.json").exists()

    def test_repeated_variants_apply_signal_domain_augmentation(
        self, tmp_path, make_run, complete_poi_times
    ):
        """base_variants > 1 must render augmented (not just clean) variants
        for train runs, exercising the augment_run path."""
        raw_root = tmp_path / "raw"
        make_run(raw_root, "run00", complete_poi_times, viscosity_cP=20.0)
        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)
        out_root = tmp_path / "out"

        bd.build(raw_root, tiers_path, out_root, base_variants=3, val_frac=0.0, seed=1)

        images = list((out_root / "init" / "images" / "train").glob("*run00_v1*"))
        assert images, "expected an augmented (v1+) variant to be rendered"

    def test_csv_unreadable_at_process_time_is_skipped_not_fatal(
        self, tmp_path, make_run, complete_poi_times
    ):
        """discover_runs already read each run's CSV successfully once at
        discovery time; this exercises process_run's own defensive re-read
        try/except for the rare case where the file becomes unreadable
        between discovery and processing (deleted/corrupted mid-run) -
        a real, if narrow, race that can't be reached by simply feeding a
        bad CSV (discover_runs would have filtered that run out already)."""
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
            # First read is discover_runs' own validation pass (must
            # succeed, or the run never becomes discoverable at all);
            # only the SECOND read - process_run's - fails.
            if "flaky_run" in key and call_counts[key] >= 2:
                raise ValueError("corrupted mid-run")
            return real_read_csv(path, *args, **kwargs)

        with patch(
            "src.systems.qmodel_7_onyx.dataset.build_detectors.pd.read_csv",
            side_effect=flaky_read_csv,
        ):
            bd.build(raw_root, tiers_path, out_root, base_variants=1, val_frac=0.0, seed=1)

        manifest = json.loads((out_root / "manifest.json").read_text())
        assert manifest["n_train_runs"] + manifest["n_val_runs"] == 2  # both discovered
        assert sum(manifest["sample_counts"].values()) > 0  # good_run still produced output

    def test_respects_the_limit_argument(self, tmp_path, make_run, complete_poi_times):
        raw_root = tmp_path / "raw"
        for i in range(4):
            # Distinct POI times per run - identical content across runs
            # would otherwise be deduped down to a single run.
            poi = {k: v + i for k, v in complete_poi_times.items()}
            make_run(raw_root, f"run{i:02d}", poi, viscosity_cP=20.0)
        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)

        out_root = tmp_path / "out"
        bd.build(raw_root, tiers_path, out_root, base_variants=1, val_frac=0.0, seed=1, limit=2)

        manifest = json.loads((out_root / "manifest.json").read_text())
        assert manifest["n_train_runs"] + manifest["n_val_runs"] == 2


class TestMain:
    """Integration checks for the argparse CLI entry point."""

    def test_main_forwards_parsed_arguments(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "sys.argv",
            [
                "build_detectors.py",
                "--raw-root",
                str(tmp_path / "raw"),
                "--tiers",
                str(tmp_path / "tiers.json"),
                "--out",
                str(tmp_path / "out"),
                "--base-variants",
                "3",
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

        with patch.object(bd, "build") as mock_build:
            bd.main()

        mock_build.assert_called_once_with(
            tmp_path / "raw",
            tmp_path / "tiers.json",
            tmp_path / "out",
            base_variants=3,
            val_frac=0.2,
            repeat_cap=4,
            seed=99,
            limit=10,
        )

    def test_main_uses_default_paths_and_settings(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["build_detectors.py"])

        with patch.object(bd, "build") as mock_build:
            bd.main()

        _, kwargs = mock_build.call_args
        assert kwargs["base_variants"] == 2
        assert kwargs["val_frac"] == 0.15
        assert kwargs["repeat_cap"] == 8
        assert kwargs["seed"] == 7
        assert kwargs["limit"] is None

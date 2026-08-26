import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.systems.qmodel_7_onyx.decode.spacing_prior import SpacingPrior
from src.systems.qmodel_7_onyx.inference import controller as ctl
from src.systems.qmodel_7_onyx.inference.config import QModelOnyxConfig
from src.systems.qmodel_7_onyx.inference.crosscheck import CrosscheckResult, ZoomEvidence

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _raw_df(duration_s=400.0, dt=0.02, seed=0):
    """A raw (unpreprocessed) sensor dataframe, long/dense enough to survive
    preprocess_dataframe and every MIN_SLICE_LENGTH/MIN_PREFIX_S guard."""
    t = np.arange(0.0, duration_s, dt)
    rng = np.random.default_rng(seed)
    diss = np.cumsum(rng.normal(0, 1e-7, len(t))) + 3e-5
    freq = 1.5e7 - np.linspace(0, 500, len(t))
    return pd.DataFrame({"Relative_time": t, "Dissipation": diss, "Resonance_Frequency": freq})


def _synthetic_configs(n=200, seed=0):
    rng = np.random.default_rng(seed)
    t0 = rng.uniform(3, 10, n)
    gaps = rng.lognormal(mean=[0.0, 1.0, 2.0, 2.5], sigma=0.2, size=(n, 4))
    cum = np.cumsum(gaps, axis=1)
    return np.column_stack([t0, t0 + cum[:, 0], t0 + cum[:, 1], t0 + cum[:, 2], t0 + cum[:, 3]])


class _FakeDetector:
    """Stand-in for QModelOnyxDetector: no YOLO, fully scripted results."""

    def __init__(self, single=None, candidates=None, fail=False):
        self._single = single or {}
        self._candidates = candidates or {}
        self.fail = fail
        self.single_calls = []
        self.candidates_calls = []

    def predict_single(self, df, target_class_map=None):
        self.single_calls.append((len(df) if df is not None else 0, target_class_map))
        if self.fail:
            raise RuntimeError("detector boom")
        return dict(self._single)

    def predict_candidates(self, df, target_class_map=None):
        self.candidates_calls.append((len(df) if df is not None else 0, target_class_map))
        if self.fail:
            raise RuntimeError("detector boom")
        return {k: list(v) for k, v in self._candidates.items()}


class _FakeFillClassifier:
    def __init__(self, channels):
        self.channels = channels

    def predict(self, df):
        return self.channels


def _controller(model_assets=None):
    return ctl.QModelOnyx(model_assets=model_assets or {})


# ---------------------------------------------------------------------------
# QModelOnyxFillClassifier
# ---------------------------------------------------------------------------


class TestQModelOnyxFillClassifierInit:
    def test_missing_model_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ctl.QModelOnyxFillClassifier(str(tmp_path / "missing.pt"))

    def test_yolo_load_failure_raises_runtime_error(self, tmp_path):
        weights = tmp_path / "type_cls.pt"
        weights.write_bytes(b"")
        with patch.dict(
            "sys.modules",
            {"ultralytics": MagicMock(YOLO=MagicMock(side_effect=Exception("bad weights")))},
        ):
            with pytest.raises(RuntimeError):
                ctl.QModelOnyxFillClassifier(str(weights))


class TestQModelOnyxFillClassifierPredict:
    def _clf(self, tmp_path, model=None):
        weights = tmp_path / "type_cls.pt"
        weights.write_bytes(b"")
        fake_yolo_class = MagicMock(return_value=model or MagicMock())
        with patch.object(ctl, "YOLO", fake_yolo_class):
            clf = ctl.QModelOnyxFillClassifier(str(weights))
        return clf

    def test_empty_dataframe_returns_zero(self, tmp_path):
        clf = self._clf(tmp_path)
        assert clf.predict(pd.DataFrame()) == 0
        assert clf.predict(None) == 0

    def test_render_failure_returns_zero(self, tmp_path):
        clf = self._clf(tmp_path)
        df = _raw_df()
        with patch.object(ctl, "prepare_cls_input", side_effect=Exception("render boom")):
            assert clf.predict(df) == 0

    def test_no_model_results_returns_zero(self, tmp_path):
        model = MagicMock(return_value=[])
        clf = self._clf(tmp_path, model=model)
        assert clf.predict(_raw_df()) == 0

    def test_successful_classification_maps_label_to_channels(self, tmp_path):
        result = MagicMock()
        result.probs.top1 = 0
        result.names = {0: "2ch"}
        model = MagicMock(return_value=[result])
        clf = self._clf(tmp_path, model=model)
        assert clf.predict(_raw_df()) == 2
        assert clf._last_image is not None

    def test_inference_exception_returns_zero(self, tmp_path):
        model = MagicMock(side_effect=Exception("inference boom"))
        clf = self._clf(tmp_path, model=model)
        assert clf.predict(_raw_df()) == 0

    def test_none_render_output_returns_zero(self, tmp_path):
        clf = self._clf(tmp_path)
        with patch.object(ctl, "prepare_cls_input", return_value=None):
            assert clf.predict(_raw_df()) == 0


class TestMapLabelToChannels:
    def _clf(self, tmp_path):
        weights = tmp_path / "type_cls.pt"
        weights.write_bytes(b"")
        with patch.object(ctl, "YOLO", MagicMock(return_value=MagicMock())):
            return ctl.QModelOnyxFillClassifier(str(weights))

    def test_known_label(self, tmp_path):
        clf = self._clf(tmp_path)
        assert clf._map_label_to_channels("3ch") == 3
        assert clf._map_label_to_channels(" No_Fill ") == -1

    def test_numeric_label(self, tmp_path):
        clf = self._clf(tmp_path)
        assert clf._map_label_to_channels("2") == 2

    def test_unknown_label_defaults_to_zero(self, tmp_path):
        clf = self._clf(tmp_path)
        assert clf._map_label_to_channels("mystery") == 0


# ---------------------------------------------------------------------------
# QModelOnyxDetector
# ---------------------------------------------------------------------------


def _box(cls_id, conf, x_norm):
    box = MagicMock()
    box.cls = [MagicMock(item=MagicMock(return_value=cls_id))]
    box.conf = MagicMock(item=MagicMock(return_value=conf))
    box.xywhn = [[MagicMock(item=MagicMock(return_value=x_norm))]]
    return box


class TestQModelOnyxDetectorInit:
    def test_missing_model_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ctl.QModelOnyxDetector(str(tmp_path / "missing.pt"))


class TestQModelOnyxDetectorPredictSingle:
    def _detector(self, tmp_path, model):
        weights = tmp_path / "ch1.pt"
        weights.write_bytes(b"")
        with patch.object(ctl, "YOLO", MagicMock(return_value=model)):
            return ctl.QModelOnyxDetector(str(weights))

    def test_short_slice_returns_empty(self, tmp_path):
        det = self._detector(tmp_path, MagicMock())
        short_df = _raw_df(duration_s=0.1)
        assert det.predict_single(short_df) == {}
        assert det.predict_single(None) == {}

    def test_maps_best_box_per_class_through_target_class_map(self, tmp_path):
        res = MagicMock()
        res.boxes = [_box(cls_id=0, conf=0.9, x_norm=0.5)]
        model = MagicMock(return_value=[res])
        det = self._detector(tmp_path, model)
        df = _raw_df()

        out = det.predict_single(df, target_class_map={0: 42})

        assert set(out) == {42}
        assert out[42]["conf"] == 0.9
        t_min, t_max = df["Relative_time"].min(), df["Relative_time"].max()
        assert t_min <= out[42]["time"] <= t_max

    def test_keeps_highest_confidence_detection_per_class(self, tmp_path):
        res = MagicMock()
        res.boxes = [_box(0, 0.4, 0.2), _box(0, 0.9, 0.8)]
        model = MagicMock(return_value=[res])
        det = self._detector(tmp_path, model)

        out = det.predict_single(_raw_df())

        assert out[0]["conf"] == 0.9

    def test_without_target_class_map_uses_raw_class_ids(self, tmp_path):
        res = MagicMock()
        res.boxes = [_box(7, 0.5, 0.5)]
        model = MagicMock(return_value=[res])
        det = self._detector(tmp_path, model)

        out = det.predict_single(_raw_df())

        assert set(out) == {7}


class TestQModelOnyxDetectorPredictCandidates:
    def _detector(self, tmp_path, model):
        weights = tmp_path / "ch1.pt"
        weights.write_bytes(b"")
        with patch.object(ctl, "YOLO", MagicMock(return_value=model)):
            return ctl.QModelOnyxDetector(str(weights))

    def test_short_slice_returns_empty(self, tmp_path):
        det = self._detector(tmp_path, MagicMock())
        assert det.predict_candidates(_raw_df(duration_s=0.1)) == {}

    def test_returns_all_candidates_sorted_by_confidence_desc(self, tmp_path):
        res = MagicMock()
        res.boxes = [_box(0, 0.3, 0.1), _box(0, 0.9, 0.5), _box(0, 0.6, 0.8)]
        model = MagicMock(return_value=[res])
        det = self._detector(tmp_path, model)

        out = det.predict_candidates(_raw_df())

        confs = [c["conf"] for c in out[0]]
        assert confs == sorted(confs, reverse=True)
        assert len(out[0]) == 3

    def test_target_class_map_filters_and_renames_keys(self, tmp_path):
        res = MagicMock()
        res.boxes = [_box(0, 0.9, 0.5), _box(1, 0.5, 0.5)]
        model = MagicMock(return_value=[res])
        det = self._detector(tmp_path, model)

        out = det.predict_candidates(_raw_df(), target_class_map={0: 99})

        assert set(out) == {99}


# ---------------------------------------------------------------------------
# QModelOnyx: loaders
# ---------------------------------------------------------------------------


class TestLoaders:
    def test_load_fill_cls_returns_none_without_a_configured_path(self):
        c = _controller()
        assert c._load_fill_cls() is None

    def test_load_fill_cls_constructs_and_caches(self, tmp_path):
        weights = tmp_path / "type_cls.pt"
        weights.write_bytes(b"")
        c = _controller({"fill_classifier": str(weights)})
        with patch.object(ctl, "YOLO", MagicMock(return_value=MagicMock())):
            first = c._load_fill_cls()
            second = c._load_fill_cls()
        assert first is second

    def test_load_detector_by_name_returns_none_when_unconfigured(self):
        c = _controller()
        assert c._load_detector_by_name("ch1") is None

    def test_load_detector_by_name_returns_none_and_logs_on_failure(self, tmp_path):
        weights = tmp_path / "ch1.pt"
        weights.write_bytes(b"")
        c = _controller({"detectors": {"ch1": str(weights)}})
        with patch.object(ctl, "YOLO", MagicMock(side_effect=Exception("boom"))):
            assert c._load_detector_by_name("ch1") is None

    def test_load_detector_by_name_caches_across_calls(self, tmp_path):
        weights = tmp_path / "ch1.pt"
        weights.write_bytes(b"")
        c = _controller({"detectors": {"ch1": str(weights)}})
        with patch.object(ctl, "YOLO", MagicMock(return_value=MagicMock())) as mock_yolo:
            c._load_detector_by_name("ch1")
            c._load_detector_by_name("ch1")
        mock_yolo.assert_called_once()

    def test_load_spacing_prior_returns_none_when_missing_path(self):
        c = _controller()
        assert c._load_spacing_prior() is None

    def test_load_spacing_prior_returns_none_when_file_absent(self, tmp_path):
        c = _controller({"spacing_prior": str(tmp_path / "missing.json")})
        assert c._load_spacing_prior() is None

    def test_load_spacing_prior_loads_and_caches(self, tmp_path):
        prior_path = tmp_path / "prior.json"
        SpacingPrior.fit(_synthetic_configs()).save(prior_path)
        c = _controller({"spacing_prior": str(prior_path)})
        first = c._load_spacing_prior()
        second = c._load_spacing_prior()
        assert first is second
        assert first is not None

    def test_load_spacing_prior_returns_none_when_decode_unavailable(self, tmp_path):
        prior_path = tmp_path / "prior.json"
        SpacingPrior.fit(_synthetic_configs()).save(prior_path)
        c = _controller({"spacing_prior": str(prior_path)})
        with patch.object(ctl, "_DECODE_AVAILABLE", False):
            assert c._load_spacing_prior() is None

    def test_load_spacing_prior_returns_none_and_logs_on_corrupt_file(self, tmp_path):
        prior_path = tmp_path / "prior.json"
        prior_path.write_text("not valid json")
        c = _controller({"spacing_prior": str(prior_path)})
        assert c._load_spacing_prior() is None


# ---------------------------------------------------------------------------
# QModelOnyx: small formatting/utility helpers
# ---------------------------------------------------------------------------


class TestFormatHelpers:
    def test_default_predictions_cover_every_poi_with_placeholders(self):
        c = _controller()
        preds = c._get_default_predictions()
        assert set(preds) == set(c.POI_MAP.values())
        for rec in preds.values():
            assert rec == {"indices": [-1], "confidences": [-1]}

    def test_format_output_fills_missing_pois_with_placeholders(self):
        c = _controller()
        out = c._format_output({1: {"index": 5, "conf": 0.8}})
        assert out["POI1"] == {"indices": [5], "confidences": [0.8]}
        assert out["POI2"] == {"indices": [-1], "confidences": [-1]}

    def test_validate_file_buffer_accepts_a_path_string(self, tmp_path):
        csv_path = tmp_path / "run.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(csv_path, index=False)
        c = _controller()
        df = c._validate_file_buffer(str(csv_path))
        assert list(df["a"]) == [1, 2]

    def test_validate_file_buffer_rewinds_a_seekable_buffer(self):
        buf = io.StringIO("a,b\n1,2\n")
        buf.read()  # advance past the data
        c = _controller()
        df = c._validate_file_buffer(buf)
        assert list(df["a"]) == [1]

    def test_validate_file_buffer_raises_on_unparseable_content(self):
        c = _controller()
        with pytest.raises(pd.errors.EmptyDataError):
            c._validate_file_buffer(io.StringIO(""))

    def test_get_raw_index_finds_nearest_row(self):
        c = _controller()
        raw_df = pd.DataFrame({"Relative_time": [0.0, 1.0, 2.0, 3.0]})
        assert c._get_raw_index(raw_df, 2.1) == 2

    def test_get_raw_index_falls_back_to_first_column(self):
        c = _controller()
        raw_df = pd.DataFrame({"idx": [0.0, 5.0, 10.0]})
        assert c._get_raw_index(raw_df, 4.0) == 1


class TestVisualize:
    def test_writes_a_debug_plot(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        c = _controller()
        df = _raw_df(duration_s=20.0)
        c._visualize(df, {1: {"time": 5.0}}, [("CH1_Cut", 10.0)], save_path="debug.png")
        matches = list(tmp_path.glob("debug_*.png"))
        assert len(matches) == 1

    def test_none_dataframe_is_a_noop(self):
        c = _controller()
        c._visualize(None, {}, [])  # must not raise

    def test_empty_dataframe_is_a_noop(self):
        c = _controller()
        c._visualize(pd.DataFrame(), {}, [])  # must not raise


# ---------------------------------------------------------------------------
# QModelOnyx: _decode_with_prior
# ---------------------------------------------------------------------------


def _prior(tmp_path):
    p = tmp_path / "prior.json"
    SpacingPrior.fit(_synthetic_configs()).save(p)
    return p


class TestDecodeWithPrior:
    def test_no_prior_asset_configured_is_a_noop(self):
        c = _controller()
        meta = c._decode_with_prior({}, {}, num_channels=3, raw_df=pd.DataFrame())
        assert meta["used"] is False
        assert meta["reason"] == "spacing prior unavailable"

    def test_decode_unavailable_short_circuits(self, tmp_path):
        c = _controller({"spacing_prior": str(_prior(tmp_path))})
        with patch.object(ctl, "_DECODE_AVAILABLE", False):
            meta = c._decode_with_prior({}, {}, num_channels=3, raw_df=pd.DataFrame())
        assert meta["used"] is False
        assert meta["reason"] == "decode modules unavailable"

    def test_no_candidates_harvested_is_a_noop(self, tmp_path):
        c = _controller({"spacing_prior": str(_prior(tmp_path))})
        meta = c._decode_with_prior({}, {}, num_channels=3, raw_df=pd.DataFrame())
        assert meta["used"] is False
        assert meta["reason"] == "no candidates harvested"

    def test_decode_replaces_greedy_placements_when_it_wins_by_the_margin(self, tmp_path):
        c = _controller({"spacing_prior": str(_prior(tmp_path))})
        raw_df = pd.DataFrame({"Relative_time": np.linspace(0, 200, 2000)})

        # A complete 3-channel chain time-aligned with the fitted prior's
        # typical spacing, harvested with high confidence at every stage -
        # the decode should confidently choose this exact configuration.
        t = {"POI1": 5.0, "POI2": 6.0, "POI3": 25.0, "POI4": 60.0, "POI5": 120.0}
        harvested = {
            1: [{"time": t["POI1"], "conf": 0.95, "index": 0}],
            2: [{"time": t["POI2"], "conf": 0.95, "index": 0}],
            4: [{"time": t["POI3"], "conf": 0.95, "index": 0}],
            5: [{"time": t["POI4"], "conf": 0.95, "index": 0}],
            6: [{"time": t["POI5"], "conf": 0.95, "index": 0}],
        }
        # Greedy cascade picks deliberately off from the true configuration,
        # so the decode has an actual improvement to make.
        final_results = {
            1: {"index": 0, "conf": 0.5, "time": t["POI1"] + 5.0},
            2: {"index": 0, "conf": 0.5, "time": t["POI2"] + 5.0},
            4: {"index": 0, "conf": 0.5, "time": t["POI3"] + 5.0},
            5: {"index": 0, "conf": 0.5, "time": t["POI4"] + 5.0},
            6: {"index": 0, "conf": 0.5, "time": t["POI5"] + 5.0},
        }

        meta = c._decode_with_prior(final_results, harvested, num_channels=3, raw_df=raw_df)

        assert meta["used"] is True
        # The decoded configuration should have moved off the deliberately
        # displaced greedy picks toward the harvested (true) times.
        assert final_results[4]["time"] == pytest.approx(t["POI3"], abs=1.0)

    def test_decode_keeps_cascade_when_margin_not_cleared(self, tmp_path):
        c = _controller({"spacing_prior": str(_prior(tmp_path))})
        raw_df = pd.DataFrame({"Relative_time": np.linspace(0, 200, 2000)})
        t = {"POI1": 5.0, "POI2": 6.0, "POI3": 25.0, "POI4": 60.0, "POI5": 120.0}
        # Only ONE candidate per POI, identical to the greedy pick: the
        # decode cannot possibly find anything better than the cascade.
        harvested = {
            1: [{"time": t["POI1"], "conf": 0.9, "index": 0}],
            2: [{"time": t["POI2"], "conf": 0.9, "index": 0}],
            4: [{"time": t["POI3"], "conf": 0.9, "index": 0}],
            5: [{"time": t["POI4"], "conf": 0.9, "index": 0}],
            6: [{"time": t["POI5"], "conf": 0.9, "index": 0}],
        }
        final_results = {
            poi_id: {"index": 0, "conf": 0.9, "time": tt}
            for poi_id, tt in zip((1, 2, 4, 5, 6), t.values(), strict=True)
        }

        meta = c._decode_with_prior(dict(final_results), harvested, num_channels=3, raw_df=raw_df)

        assert meta["used"] is True
        assert meta["kept_cascade"] is True
        assert meta["changed"] == []

    def test_decode_error_is_caught_and_reported(self, tmp_path):
        c = _controller({"spacing_prior": str(_prior(tmp_path))})
        raw_df = pd.DataFrame({"Relative_time": np.linspace(0, 200, 200)})
        harvested = {1: [{"time": 5.0, "conf": 0.9, "index": 0}]}
        final_results = {1: {"index": 0, "conf": 0.9, "time": 5.0}}

        with patch.object(ctl, "dp_decode", side_effect=RuntimeError("solver exploded")):
            meta = c._decode_with_prior(final_results, harvested, num_channels=3, raw_df=raw_df)

        assert meta["used"] is False
        assert "decode error" in meta["reason"]


# ---------------------------------------------------------------------------
# QModelOnyx: _crosscheck_fill
# ---------------------------------------------------------------------------


class TestCrosscheckFill:
    def test_no_zoom_detectors_is_a_noop(self):
        c = _controller()
        n, meta = c._crosscheck_fill({}, pd.DataFrame(), pd.DataFrame(), num_channels=2)
        assert n == 2
        assert meta["used"] is False
        assert meta["reason"] == "no zoom detector assets"

    def test_rescue_upgrades_channel_count_and_writes_the_new_poi(self, tmp_path):
        c = _controller()
        c._detectors["ch2_zoom"] = _FakeDetector()  # presence alone activates the stage

        rescue = CrosscheckResult(
            channels=2,
            upgraded=True,
            evidence=[ZoomEvidence(channel=2, time=42.0, conf=0.8, window=(30.0, 50.0))],
            windows_scanned=3,
        )
        raw_df = pd.DataFrame({"Relative_time": np.linspace(0, 100, 100)})
        final_results = {4: {"index": 0, "conf": 0.9, "time": 10.0}}

        with patch.object(ctl, "verify_fill_count", return_value=rescue):
            n, meta = c._crosscheck_fill(final_results, pd.DataFrame(), raw_df, num_channels=1)

        assert n == 2
        assert meta["rescue"]["upgraded"] is True
        assert final_results[5]["time"] == 42.0

    def test_rescue_failure_is_swallowed(self):
        c = _controller()
        c._detectors["ch1_zoom"] = _FakeDetector()

        with patch.object(ctl, "verify_fill_count", side_effect=RuntimeError("boom")):
            n, meta = c._crosscheck_fill({}, pd.DataFrame(), pd.DataFrame(), num_channels=1)

        assert n == 1
        assert meta["rescue"] == {"upgraded": False}

    def test_veto_attaches_advisory_metadata_without_mutating_state(self):
        c = _controller()
        c._detectors["ch1_zoom"] = _FakeDetector()
        veto = ZoomEvidence(channel=1, time=11.0, conf=0.1, window=(5.0, 20.0))
        final_results = {4: {"index": 0, "conf": 0.9, "time": 10.0}}

        with (
            patch.object(ctl, "verify_fill_count", return_value=CrosscheckResult(channels=1)),
            patch.object(ctl, "verify_claimed_poi", return_value=veto),
        ):
            n, meta = c._crosscheck_fill(dict(final_results), pd.DataFrame(), pd.DataFrame(), 1)

        assert n == 1
        assert meta["veto"]["advisory_only"] is True
        assert final_results[4]["time"] == 10.0  # untouched by the veto

    def test_veto_failure_is_swallowed(self):
        c = _controller()
        c._detectors["ch1_zoom"] = _FakeDetector()
        final_results = {4: {"index": 0, "conf": 0.9, "time": 10.0}}

        with (
            patch.object(ctl, "verify_fill_count", return_value=CrosscheckResult(channels=1)),
            patch.object(ctl, "verify_claimed_poi", side_effect=RuntimeError("boom")),
        ):
            n, meta = c._crosscheck_fill(final_results, pd.DataFrame(), pd.DataFrame(), 1)

        assert meta["veto"] is None

    def test_zero_channels_skips_both_rescue_and_veto(self):
        c = _controller()
        c._detectors["ch1_zoom"] = _FakeDetector()
        with (
            patch.object(ctl, "verify_fill_count") as mock_rescue,
            patch.object(ctl, "verify_claimed_poi") as mock_veto,
        ):
            n, meta = c._crosscheck_fill({}, pd.DataFrame(), pd.DataFrame(), num_channels=0)
        mock_rescue.assert_not_called()
        mock_veto.assert_not_called()
        assert n == 0


# ---------------------------------------------------------------------------
# QModelOnyx: _refine_with_zoom
# ---------------------------------------------------------------------------


class TestRefineWithZoom:
    def test_no_zoom_assets_configured_is_a_noop(self):
        c = _controller()
        meta = c._refine_with_zoom({}, _raw_df(), pd.DataFrame())
        assert meta["used"] is False
        assert meta["reason"] == "no zoom detector assets"

    def test_missing_poi_in_final_results_is_skipped(self, tmp_path):
        weights = tmp_path / "ch1_zoom.pt"
        weights.write_bytes(b"")
        c = _controller({"detectors": {"ch1_zoom": str(weights)}})
        meta = c._refine_with_zoom({}, _raw_df(), pd.DataFrame())
        assert meta["used"] is False
        assert meta["moved"] == {}

    def test_confident_refinement_within_trust_region_moves_the_poi(self, tmp_path):
        weights = tmp_path / "ch1_zoom.pt"
        weights.write_bytes(b"")
        c = _controller({"detectors": {"ch1_zoom": str(weights)}})
        master_df = _raw_df(duration_s=200.0)
        raw_df = master_df.copy()
        fake_det = _FakeDetector(single={4: {"time": 51.0, "conf": 0.9}})
        c._detectors["ch1_zoom"] = fake_det
        final_results = {4: {"index": 0, "conf": 0.5, "time": 50.0}}

        meta = c._refine_with_zoom(final_results, master_df, raw_df)

        assert meta["used"] is True
        assert final_results[4]["time"] == 51.0
        assert "POI4" in meta["moved"]  # POI_MAP[4], not DECODE_ID_TO_NAME[4]

    def test_low_confidence_refinement_is_rejected(self, tmp_path):
        weights = tmp_path / "ch1_zoom.pt"
        weights.write_bytes(b"")
        c = _controller({"detectors": {"ch1_zoom": str(weights)}})
        master_df = _raw_df(duration_s=200.0)
        fake_det = _FakeDetector(
            single={4: {"time": 51.0, "conf": QModelOnyxConfig.REFINE_MIN_CONF - 0.01}}
        )
        c._detectors["ch1_zoom"] = fake_det
        final_results = {4: {"index": 0, "conf": 0.5, "time": 50.0}}

        meta = c._refine_with_zoom(final_results, master_df, master_df)

        assert meta["used"] is False
        assert final_results[4]["time"] == 50.0

    def test_large_shift_is_rejected_as_a_different_event(self, tmp_path):
        weights = tmp_path / "ch1_zoom.pt"
        weights.write_bytes(b"")
        c = _controller({"detectors": {"ch1_zoom": str(weights)}})
        master_df = _raw_df(duration_s=200.0)
        # Window is REFINE_WINDOW_S wide; a huge jump should be rejected.
        fake_det = _FakeDetector(single={4: {"time": 90.0, "conf": 0.99}})
        c._detectors["ch1_zoom"] = fake_det
        final_results = {4: {"index": 0, "conf": 0.5, "time": 50.0}}

        meta = c._refine_with_zoom(final_results, master_df, master_df)

        assert meta["used"] is False
        assert final_results[4]["time"] == 50.0

    def test_detector_exception_is_swallowed_and_poi_unchanged(self, tmp_path):
        weights = tmp_path / "ch1_zoom.pt"
        weights.write_bytes(b"")
        c = _controller({"detectors": {"ch1_zoom": str(weights)}})
        master_df = _raw_df(duration_s=200.0)
        c._detectors["ch1_zoom"] = _FakeDetector(fail=True)
        final_results = {4: {"index": 0, "conf": 0.5, "time": 50.0}}

        meta = c._refine_with_zoom(final_results, master_df, master_df)

        assert meta["used"] is False
        assert final_results[4]["time"] == 50.0

    def test_index_minus_one_poi_is_skipped(self, tmp_path):
        weights = tmp_path / "ch1_zoom.pt"
        weights.write_bytes(b"")
        c = _controller({"detectors": {"ch1_zoom": str(weights)}})
        final_results = {4: {"index": -1, "conf": 0.5, "time": 50.0}}

        meta = c._refine_with_zoom(final_results, _raw_df(), pd.DataFrame())

        assert meta["used"] is False


# ---------------------------------------------------------------------------
# QModelOnyx: predict() - the full orchestration
# ---------------------------------------------------------------------------


def _cascade_controller(ch3_hit=True, ch2_hit=True, ch1_hit=True, init_hit=True):
    """A controller pre-populated with fake detectors that produce a
    deterministic, decreasing-in-time cascade: ch3 @150s, ch2 @100s,
    ch1 @50s, init POI1/POI2 @1s/2s. Bypasses YOLO/model-asset loading
    entirely by injecting straight into `_detectors`/`_fill_classifier`."""
    c = _controller()
    c._detectors["ch3"] = _FakeDetector(single={6: {"time": 150.0, "conf": 0.9}} if ch3_hit else {})
    c._detectors["ch2"] = _FakeDetector(single={5: {"time": 100.0, "conf": 0.9}} if ch2_hit else {})
    c._detectors["ch1"] = _FakeDetector(single={4: {"time": 50.0, "conf": 0.9}} if ch1_hit else {})
    c._detectors["init"] = _FakeDetector(
        single={1: {"time": 1.0, "conf": 0.9}, 2: {"time": 2.0, "conf": 0.9}} if init_hit else {}
    )
    return c


class TestPredictInputHandling:
    def test_no_data_provided_returns_default_predictions(self):
        c = _controller()
        output, num_channels = c.predict()
        assert num_channels == 0
        assert output == c._get_default_predictions()

    def test_preprocessing_failure_returns_default_predictions(self):
        c = _controller()
        # Missing the time column entirely -> preprocess_dataframe returns None.
        output, num_channels = c.predict(df=pd.DataFrame({"Dissipation": [1.0, 2.0, 3.0]}))
        assert num_channels == 0
        assert output == c._get_default_predictions()

    def test_reads_from_a_file_buffer_path(self, tmp_path):
        csv_path = tmp_path / "run.csv"
        _raw_df().to_csv(csv_path, index=False)
        c = _cascade_controller()
        output, num_channels = c.predict(file_buffer=str(csv_path), num_channels=0)
        assert num_channels == 0
        assert output["POI1"]["indices"] != [-1]

    def test_unparseable_file_buffer_returns_default_predictions(self):
        c = _controller()
        output, num_channels = c.predict(file_buffer=io.StringIO(""))
        assert num_channels == 0
        assert output == c._get_default_predictions()


class TestPredictFillClassifier:
    def test_uses_fill_classifier_when_num_channels_omitted(self):
        c = _cascade_controller()
        c._fill_classifier = _FakeFillClassifier(channels=2)
        _, num_channels = c.predict(df=_raw_df())
        assert num_channels == 2

    def test_defaults_to_three_channels_when_no_fill_classifier_configured(self):
        c = _cascade_controller()
        _, num_channels = c.predict(df=_raw_df(), refine_pois=False, crosscheck=False)
        assert num_channels == 3

    def test_explicit_num_channels_bypasses_the_fill_classifier(self):
        c = _cascade_controller()
        c._fill_classifier = _FakeFillClassifier(channels=3)
        with patch.object(_FakeFillClassifier, "predict") as mock_predict:
            c.predict(df=_raw_df(), num_channels=1)
        mock_predict.assert_not_called()

    def test_no_fill_verdict_short_circuits_with_a_progress_signal(self):
        c = _cascade_controller()
        c._fill_classifier = _FakeFillClassifier(channels=-1)
        progress = MagicMock()
        output, num_channels = c.predict(df=_raw_df(), progress_signal=progress)
        assert num_channels == -1
        assert output == c._get_default_predictions()
        progress.emit.assert_any_call(100, "No channels detected!")

    def test_no_fill_verdict_without_a_progress_signal_is_caught_by_the_outer_handler(self):
        """KNOWN BUG (documented, not silently patched around): the
        num_channels == -1 branch calls `progress_signal.emit(...)`
        unconditionally, unlike every other call site in this method which
        guards with `if progress_signal:`. With the default `progress_signal
        =None`, this raises AttributeError, which the outer try/except
        swallows - so num_channels comes back as 0, not the intended -1."""
        c = _cascade_controller()
        c._fill_classifier = _FakeFillClassifier(channels=-1)
        output, num_channels = c.predict(df=_raw_df())
        assert num_channels == 0  # NOT -1, because of the bug described above
        assert output == c._get_default_predictions()


class TestPredictCascade:
    def test_three_channels_runs_the_full_cascade_and_cuts_progressively(self):
        c = _cascade_controller()
        output, num_channels = c.predict(df=_raw_df(), num_channels=3, refine_pois=False)

        assert num_channels == 3
        assert output["POI1"]["indices"] != [-1]
        assert output["POI2"]["indices"] != [-1]
        assert output["POI3"]["indices"] == [-1]  # legacy id-3 shim is never populated
        assert output["POI4"]["indices"] != [-1]
        assert output["POI5"]["indices"] != [-1]
        assert output["POI6"]["indices"] != [-1]

    def test_zero_channels_only_runs_the_init_detector(self):
        c = _cascade_controller()
        output, num_channels = c.predict(df=_raw_df(), num_channels=0, refine_pois=False)

        assert num_channels == 0
        assert c._detectors["ch3"].single_calls == []
        assert c._detectors["ch2"].single_calls == []
        assert c._detectors["ch1"].single_calls == []
        assert output["POI1"]["indices"] != [-1]
        assert output["POI4"]["indices"] == [-1]

    def test_one_channel_only_runs_ch1_and_init(self):
        c = _cascade_controller()
        c.predict(df=_raw_df(), num_channels=1, refine_pois=False)

        assert c._detectors["ch3"].single_calls == []
        assert c._detectors["ch2"].single_calls == []
        assert len(c._detectors["ch1"].single_calls) == 1

    def test_a_stage_with_no_configured_detector_is_skipped(self):
        c = _controller()  # no detectors configured at all
        output, num_channels = c.predict(
            df=_raw_df(), num_channels=3, refine_pois=False, crosscheck=False
        )
        assert num_channels == 3
        assert output == c._get_default_predictions()

    def test_poi5_fine_refines_ch3s_placement_when_available(self):
        c = _cascade_controller()
        c._detectors["poi5_fine"] = _FakeDetector(single={6: {"time": 155.0, "conf": 0.95}})
        output, _ = c.predict(df=_raw_df(), num_channels=3, refine_pois=False)
        assert len(c._detectors["poi5_fine"].single_calls) == 1
        # POI6's final index should reflect the fine-refined time, not ch3's.
        raw_df = _raw_df()
        expected_idx = c._get_raw_index(raw_df, 155.0)
        assert output["POI6"]["indices"] == [expected_idx]

    def test_a_stage_that_detects_nothing_does_not_cut_the_signal(self):
        """When a stage's detector returns no hit, process_detection returns
        None and the cascade must NOT slice current_df at that (nonexistent)
        cut time - the next stage still sees the full remaining signal."""
        c = _cascade_controller(ch3_hit=False)
        output, num_channels = c.predict(df=_raw_df(), num_channels=3, refine_pois=False)
        assert num_channels == 3
        assert output["POI6"]["indices"] == [-1]
        # ch2 still ran and found its hit despite ch3 missing.
        assert output["POI5"]["indices"] != [-1]

    def test_init_detecting_nothing_leaves_poi1_and_poi2_unfilled(self):
        c = _cascade_controller(init_hit=False)
        output, _ = c.predict(df=_raw_df(), num_channels=0, refine_pois=False)
        assert output["POI1"]["indices"] == [-1]
        assert output["POI2"]["indices"] == [-1]

    def test_poi5_fine_detecting_nothing_keeps_ch3s_original_placement(self):
        c = _cascade_controller()
        c._detectors["poi5_fine"] = _FakeDetector(single={})
        output, _ = c.predict(df=_raw_df(), num_channels=3, refine_pois=False)
        raw_df = _raw_df()
        expected_idx = c._get_raw_index(raw_df, 150.0)  # ch3's original hit
        assert output["POI6"]["indices"] == [expected_idx]

    def test_harvest_candidate_failure_on_one_stage_does_not_abort_the_cascade(self):
        c = _cascade_controller()
        c._detectors["ch3"].predict_candidates = MagicMock(side_effect=RuntimeError("harvest boom"))
        output, num_channels = c.predict(
            df=_raw_df(), num_channels=3, harvest_candidates=True, refine_pois=False
        )
        assert num_channels == 3
        assert output["POI6"]["indices"] != [-1]  # predict_single result still used

    def test_progress_signal_is_invoked_at_each_stage(self):
        c = _cascade_controller()
        progress = MagicMock()
        c.predict(df=_raw_df(), num_channels=3, progress_signal=progress, refine_pois=False)
        stages = [call.args[1] for call in progress.emit.call_args_list]
        assert "Data Loaded" in stages
        assert "Preprocessing Data..." in stages
        assert "Detecting Channel 3..." in stages
        assert "Detecting Initialization Points..." in stages
        assert "Complete!" in stages


class TestPredictOptionalStages:
    def test_harvest_candidates_attaches_reserved_output_key(self):
        c = _cascade_controller()
        c._detectors["ch3"]._candidates = {6: [{"time": 150.0, "conf": 0.9}]}
        output, _ = c.predict(
            df=_raw_df(), num_channels=3, harvest_candidates=True, refine_pois=False
        )
        assert "_candidates" in output
        assert "POI6" in output["_candidates"]

    def test_decode_config_attaches_reserved_output_key(self):
        c = _cascade_controller()
        output, _ = c.predict(df=_raw_df(), num_channels=3, decode_config=True, refine_pois=False)
        assert "_decode" in output
        assert output["_decode"]["used"] is False  # no spacing_prior asset configured

    def test_crosscheck_attaches_reserved_output_key(self):
        c = _cascade_controller()
        output, _ = c.predict(df=_raw_df(), num_channels=3, crosscheck=True, refine_pois=False)
        assert "_crosscheck" in output
        assert output["_crosscheck"]["used"] is False  # no zoom detector assets

    def test_crosscheck_disabled_omits_the_key(self):
        c = _cascade_controller()
        output, _ = c.predict(df=_raw_df(), num_channels=3, crosscheck=False, refine_pois=False)
        assert "_crosscheck" not in output

    def test_refine_pois_omits_the_key_when_nothing_moved(self):
        c = _cascade_controller()
        output, _ = c.predict(df=_raw_df(), num_channels=3, refine_pois=True)
        assert "_refine" not in output  # no zoom detector assets configured

    def test_visualize_failure_does_not_fail_predict(self):
        c = _cascade_controller()
        with patch.object(ctl.QModelOnyx, "_visualize", side_effect=Exception("plot boom")):
            output, num_channels = c.predict(
                df=_raw_df(), num_channels=3, visualize=True, refine_pois=False
            )
        assert num_channels == 3  # the failure was caught and logged, not propagated

    def test_visualize_true_calls_the_visualizer(self):
        c = _cascade_controller()
        with patch.object(ctl.QModelOnyx, "_visualize") as mock_viz:
            c.predict(df=_raw_df(), num_channels=3, visualize=True, refine_pois=False)
        mock_viz.assert_called_once()


class TestPredictErrorHandling:
    def test_unexpected_exception_is_caught_and_returns_default_predictions(self):
        c = _cascade_controller()
        with patch.object(
            ctl.QModelOnyxDataProcessor, "preprocess_dataframe", side_effect=RuntimeError("boom")
        ):
            output, num_channels = c.predict(df=_raw_df())
        assert num_channels == 0
        assert output == c._get_default_predictions()

import numpy as np
import pandas as pd

from src.systems.qmodel_7_onyx.inference.crosscheck import verify_claimed_poi, verify_fill_count

COL_TIME = "Relative_time"


def _long_run_df(span_s=200.0, n=20000):
    t = np.linspace(0, span_s, n)
    return pd.DataFrame(
        {COL_TIME: t, "Dissipation": np.zeros(n), "Resonance_Frequency": np.zeros(n)}
    )


class FakeZoomDetector:
    """Duck-typed stand-in for QModelV6YOLO_Detector: always reports a hit
    at a fixed confidence somewhere inside whatever window it's given."""

    def __init__(self, conf: float, time_frac: float = 0.5):
        self.conf = conf
        self.time_frac = time_frac
        self.calls = 0

    def predict_single(self, df_slice: pd.DataFrame):
        self.calls += 1
        if df_slice.empty:
            return {}
        t = df_slice[COL_TIME].to_numpy()
        return {0: {"time": float(t[int(len(t) * self.time_frac)]), "conf": self.conf}}


class NullDetector:
    """Never finds anything."""

    def predict_single(self, df_slice: pd.DataFrame):
        return {}


def test_verify_fill_count_noop_when_channels_below_one():
    df = _long_run_df()
    result = verify_fill_count(df, fill_channels=0, poi_times={}, zoom_detectors={})
    assert result.channels == 0
    assert not result.upgraded


def test_verify_fill_count_climbs_with_confident_detector():
    df = _long_run_df()
    detectors = {2: FakeZoomDetector(conf=0.9), 3: FakeZoomDetector(conf=0.9)}
    result = verify_fill_count(
        df, fill_channels=1, poi_times={"POI2": 5.0, "POI3": 20.0}, zoom_detectors=detectors
    )
    assert result.upgraded
    assert result.channels == 3
    assert len(result.evidence) == 2


def test_verify_fill_count_stops_below_confidence_threshold():
    df = _long_run_df()
    detectors = {2: FakeZoomDetector(conf=0.1)}  # below DEFAULT_UPGRADE_CONF
    result = verify_fill_count(
        df, fill_channels=1, poi_times={"POI2": 5.0, "POI3": 20.0}, zoom_detectors=detectors
    )
    assert not result.upgraded
    assert result.channels == 1


def test_verify_fill_count_stops_when_detector_missing():
    df = _long_run_df()
    result = verify_fill_count(
        df, fill_channels=1, poi_times={"POI2": 5.0, "POI3": 20.0}, zoom_detectors={}
    )
    assert result.channels == 1
    assert not result.upgraded


def test_verify_fill_count_stops_when_anchor_missing():
    df = _long_run_df()
    detectors = {2: FakeZoomDetector(conf=0.9)}
    # No POI2/POI3 anchor known -> cannot search a tail.
    result = verify_fill_count(df, fill_channels=1, poi_times={}, zoom_detectors=detectors)
    assert result.channels == 1
    assert not result.upgraded


def test_verify_claimed_poi_zero_confidence_when_nothing_found():
    df = _long_run_df()
    ev = verify_claimed_poi(df, channel=3, poi_time=100.0, zoom_detector=NullDetector())
    assert ev.conf == 0.0
    assert ev.time == 100.0


def test_verify_claimed_poi_reports_confident_nearby_hit():
    df = _long_run_df()
    det = FakeZoomDetector(conf=0.8, time_frac=0.5)
    ev = verify_claimed_poi(df, channel=3, poi_time=100.0, zoom_detector=det, window_s=24.0)
    assert ev.conf > 0.0
    assert ev.channel == 3

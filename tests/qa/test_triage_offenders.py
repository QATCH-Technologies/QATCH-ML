import numpy as np
import pandas as pd

from src.systems.qmodel_7_onyx.qa import triage_offenders as triage
from src.systems.qmodel_7_onyx.tiers import TierScheme


def _synthetic_df_p(duration_s=200.0, dt=0.005, seed=0):
    """A dataframe shaped like DP.preprocess_dataframe's output, with a
    genuine step transition at t=100s so salience windows have something
    real to measure."""
    t = np.arange(0.0, duration_s, dt)
    rng = np.random.default_rng(seed)
    diss = np.full(len(t), 3e-5) + rng.normal(0, 1e-9, len(t))
    step_idx = int(len(t) * 0.5)
    diss[step_idx:] += 5e-5
    freq = 1.5e7 - np.linspace(0, 500, len(t))
    return pd.DataFrame({"Relative_time": t, "Dissipation": diss, "Resonance_Frequency": freq})


class TestSalienceReport:
    def test_reports_only_pois_present_in_the_mapping(self):
        df_p = _synthetic_df_p()
        poi = {"POI1": 5.0, "POI3": 100.0}
        rep = triage.salience_report(df_p, poi)
        assert set(rep) == {"POI1", "POI3"}

    def test_reports_expected_keys_for_a_poi_with_data_in_window(self):
        df_p = _synthetic_df_p()
        poi = {"POI3": 100.0}
        rep = triage.salience_report(df_p, poi)
        rec = rep["POI3"]
        assert rec["peak"] is not None
        for key in ("window_s", "v2_pctile", "v2_vs_median", "v3_pctile", "v3_vs_median"):
            assert key in rec

    def test_poi_outside_the_signal_range_has_no_data_in_window(self):
        df_p = _synthetic_df_p(duration_s=50.0)
        poi = {"POI5": 500.0}  # far past the end of the signal
        rep = triage.salience_report(df_p, poi)
        assert rep["POI5"] == {"peak": None}

    def test_peak_equals_the_v2_ratio(self):
        """`peak` is documented as an alias for the v2 (derivative-energy)
        ratio, kept for backward compatibility with earlier callers."""
        df_p = _synthetic_df_p()
        poi = {"POI3": 100.0}
        rec = triage.salience_report(df_p, poi)["POI3"]
        assert rec["peak"] == rec["v2_vs_median"]

    def test_a_real_transition_scores_above_the_trace_median(self):
        df_p = _synthetic_df_p()
        poi = {"POI3": 100.0}  # sits right on the injected step
        rec = triage.salience_report(df_p, poi)["POI3"]
        assert rec["v2_vs_median"] > 1.0
        assert rec["v3_vs_median"] > 1.0


class TestAnnotateRender:
    def test_writes_an_image_file(self, tmp_path):
        df_p = _synthetic_df_p()
        poi = {"POI1": 5.0, "POI3": 100.0}
        out_path = tmp_path / "annotated.png"

        triage.annotate_render(df_p, poi, out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_poi_outside_the_run_span_is_skipped_without_error(self, tmp_path):
        df_p = _synthetic_df_p(duration_s=50.0)
        poi = {"POI5": 500.0}  # outside [t0, t1]
        out_path = tmp_path / "annotated.png"

        triage.annotate_render(df_p, poi, out_path)  # must not raise

        assert out_path.exists()


class TestMain:
    """End-to-end CLI test: a misses.csv naming one offender run, a matching
    raw run on disk, and a tier scheme."""

    def test_writes_annotated_renders_and_salience_csv(
        self, tmp_path, monkeypatch, make_run, complete_poi_times
    ):
        raw_root = tmp_path / "raw"
        make_run(raw_root, "00000", complete_poi_times, viscosity_cP=20.0)

        # Two misses for run "00000": one tagged as the full-run (uniform,
        # k == FULLRUN_K["2ch"]) analysis-time miss, one an ordinary miss.
        misses = pd.DataFrame(
            [
                {
                    "path": "hash_00000_v0_u11.png",
                    "run": "00000",
                    "tag": "u",
                    "true": "2ch",
                    "pred": "1ch",
                },
                {
                    "path": "hash_00000_v0_h3.png",
                    "run": "00000",
                    "tag": "h",
                    "true": "1ch",
                    "pred": "2ch",
                },
            ]
        )
        misses_path = tmp_path / "misses.csv"
        misses.to_csv(misses_path, index=False)

        tiers_path = tmp_path / "tiers.json"
        TierScheme(edges_cp=[15.0, 35.0], n_per_tier=[1, 1, 1]).save(tiers_path)

        out_dir = tmp_path / "out"
        monkeypatch.setattr(
            "sys.argv",
            [
                "triage_offenders.py",
                "--raw-root",
                str(raw_root),
                "--misses",
                str(misses_path),
                "--tiers",
                str(tiers_path),
                "--min-misses",
                "1",
                "--out",
                str(out_dir),
            ],
        )

        triage.main()

        assert (out_dir / "00000_annotated.png").exists()
        salience = pd.read_csv(out_dir / "salience.csv")
        # pandas round-trips the zero-padded run id as an int; compare on
        # the zero-filled string form to match what main() actually wrote.
        assert (salience["run"].astype(str).str.zfill(5) == "00000").all()
        assert set(salience["poi"]) <= set(triage.POI_ORDER)

    def test_offender_run_missing_from_raw_root_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        raw_root = tmp_path / "raw"
        raw_root.mkdir()

        misses = pd.DataFrame(
            [
                {
                    "path": "hash_00000_v0_u11.png",
                    "run": "00000",
                    "tag": "u",
                    "true": "2ch",
                    "pred": "1ch",
                }
            ]
        )
        misses_path = tmp_path / "misses.csv"
        misses.to_csv(misses_path, index=False)

        out_dir = tmp_path / "out"
        monkeypatch.setattr(
            "sys.argv",
            [
                "triage_offenders.py",
                "--raw-root",
                str(raw_root),
                "--misses",
                str(misses_path),
                "--tiers",
                str(tmp_path / "nonexistent_tiers.json"),
                "--min-misses",
                "1",
                "--out",
                str(out_dir),
            ],
        )

        triage.main()  # must not raise despite the offender run not existing

        assert not (out_dir / "00000_annotated.png").exists()
        assert (out_dir / "salience.csv").exists()

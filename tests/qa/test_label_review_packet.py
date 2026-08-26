import numpy as np
import pandas as pd
import pytest

from src.systems.qmodel_7_onyx.corpus import RunRecord
from src.systems.qmodel_7_onyx.qa import label_review_packet as packet


def _write_csv(path, duration_s=200.0, dt=0.05):
    t = np.arange(0.0, duration_s, dt)
    diss = np.cumsum(np.random.default_rng(0).normal(0, 1e-7, len(t))) + 3e-5
    freq = 1.5e7 - np.linspace(0, 500, len(t))
    pd.DataFrame({"Relative_time": t, "Dissipation": diss, "Resonance_Frequency": freq}).to_csv(
        path, index=False
    )


class TestMakePage:
    def test_writes_a_review_png_for_a_complete_fill(self, tmp_path, complete_poi_times):
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path)
        rec = RunRecord(
            run_id="00042", csv_path=csv_path, poi_times=complete_poi_times, viscosity_cP=20.0
        )
        out = tmp_path / "out"
        out.mkdir()

        packet.make_page(rec, pois=["POI4", "POI5"], zoom_s=30.0, out=out)

        review = out / "00042_review.png"
        assert review.exists()
        assert review.stat().st_size > 0

    def test_missing_time_column_is_skipped_without_writing(self, tmp_path, complete_poi_times):
        csv_path = tmp_path / "run.csv"
        pd.DataFrame({"Dissipation": [1.0, 2.0, 3.0]}).to_csv(csv_path, index=False)
        rec = RunRecord(
            run_id="00099", csv_path=csv_path, poi_times=complete_poi_times, viscosity_cP=20.0
        )
        out = tmp_path / "out"
        out.mkdir()

        packet.make_page(rec, pois=["POI4", "POI5"], zoom_s=30.0, out=out)

        assert not (out / "00099_review.png").exists()

    def test_pois_absent_from_the_run_are_not_zoomed(self, tmp_path):
        """A partial fill missing POI5 should still produce a page - only
        zoomed on the POIs it actually has."""
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path)
        rec = RunRecord(
            run_id="00007",
            csv_path=csv_path,
            poi_times={"POI1": 5.0, "POI3": 25.0, "POI4": 60.0},
            viscosity_cP=15.0,
        )
        out = tmp_path / "out"
        out.mkdir()

        packet.make_page(rec, pois=["POI4", "POI5"], zoom_s=30.0, out=out)

        assert (out / "00007_review.png").exists()

    def test_no_requested_pois_present_still_writes_the_full_run_page(self, tmp_path):
        csv_path = tmp_path / "run.csv"
        _write_csv(csv_path)
        rec = RunRecord(
            run_id="00008",
            csv_path=csv_path,
            poi_times={"POI1": 5.0, "POI2": 6.0},
            viscosity_cP=15.0,
        )
        out = tmp_path / "out"
        out.mkdir()

        packet.make_page(rec, pois=["POI4", "POI5"], zoom_s=30.0, out=out)

        assert (out / "00008_review.png").exists()


class TestMain:
    def test_generates_pages_for_requested_runs_and_skips_missing_ones(
        self, tmp_path, monkeypatch, make_run, complete_poi_times
    ):
        raw_root = tmp_path / "raw"
        make_run(raw_root, "00001", complete_poi_times, viscosity_cP=20.0)
        out_dir = tmp_path / "out"

        monkeypatch.setattr(
            "sys.argv",
            [
                "label_review_packet.py",
                "--raw-root",
                str(raw_root),
                "--out",
                str(out_dir),
                "--runs",
                "00001",
                "99999",  # not on disk - must be skipped, not fatal
                "--pois",
                "POI4",
                "POI5",
                "--zoom-s",
                "30",
            ],
        )

        packet.main()

        assert (out_dir / "00001_review.png").exists()
        assert not (out_dir / "99999_review.png").exists()

    def test_run_ids_are_zero_padded_to_five_digits(
        self, tmp_path, monkeypatch, make_run, complete_poi_times
    ):
        raw_root = tmp_path / "raw"
        make_run(raw_root, "00007", complete_poi_times, viscosity_cP=20.0)
        out_dir = tmp_path / "out"

        monkeypatch.setattr(
            "sys.argv",
            [
                "label_review_packet.py",
                "--raw-root",
                str(raw_root),
                "--out",
                str(out_dir),
                "--runs",
                "7",  # unpadded - main() must zfill(5) before lookup
            ],
        )

        packet.main()

        assert (out_dir / "00007_review.png").exists()

    def test_missing_runs_argument_errors_out(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["label_review_packet.py", "--raw-root", str(tmp_path / "raw")],
        )
        with pytest.raises(SystemExit):
            packet.main()

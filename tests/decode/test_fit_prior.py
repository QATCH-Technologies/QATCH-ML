import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.systems.qmodel_7_onyx.decode import fit_prior


class TestFindRunCsv:
    """Tests for the _find_run_csv helper function."""

    def test_finds_valid_signal_csv(self, tmp_path):
        """It correctly identifies the signal CSV and ignores POI files."""
        (tmp_path / "run_data.csv").touch()
        (tmp_path / "annotations_poi.csv").touch()

        result = fit_prior._find_run_csv(tmp_path)

        assert result is not None
        assert result.name == "run_data.csv"

    def test_returns_none_when_no_signal_csv_exists(self, tmp_path):
        """It returns None if only POI files or no CSVs are present."""
        (tmp_path / "annotations_poi.csv").touch()
        (tmp_path / "other_file.txt").touch()

        result = fit_prior._find_run_csv(tmp_path)

        assert result is None


class TestCollectCompleteConfigs:
    """Tests for the collect_complete_configs pipeline."""

    @pytest.fixture
    def setup_run_dir(self, tmp_path):
        """Helper to create a fake run directory with necessary dummy files."""

        def _create(run_name: str):
            run_dir = tmp_path / run_name
            run_dir.mkdir()
            (run_dir / "signal.csv").touch()
            (run_dir / "truth_poi.csv").touch()
            return run_dir

        return _create

    # Use patch.object to avoid string-import resolution issues
    @patch.object(fit_prior, "truth_times")
    @patch.object(fit_prior.pd, "read_csv")
    def test_collects_complete_fills_only(
        self, mock_read_csv, mock_truth_times, setup_run_dir, tmp_path
    ):
        """It extracts complete configurations and explicitly ignores partial fills."""
        setup_run_dir("run_complete")
        setup_run_dir("run_partial")

        mock_read_csv.return_value = pd.DataFrame(
            {"Relative_time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        )

        mock_truth_times.side_effect = [
            {"POI1": 1.0, "POI2": 2.0, "POI3": 3.0, "POI4": 4.0, "POI5": 5.0},  # Complete
            {"POI1": 1.0, "POI2": 2.0},  # Partial (missing POI3, POI4, POI5)
        ]

        configs = fit_prior.collect_complete_configs(tmp_path, time_col="Relative_time")

        assert configs.shape == (1, 5)
        assert np.array_equal(configs[0], [1.0, 2.0, 3.0, 4.0, 5.0])

    @patch.object(fit_prior.pd, "read_csv")
    def test_handles_read_exceptions_gracefully(self, mock_read_csv, setup_run_dir, tmp_path):
        """It ignores runs where pd.read_csv throws an Exception."""
        setup_run_dir("run_corrupt")
        mock_read_csv.side_effect = Exception("Corrupt file format")

        with pytest.raises(SystemExit, match="No complete-fill configurations found"):
            fit_prior.collect_complete_configs(tmp_path, time_col="Relative_time")

    @patch.object(fit_prior.pd, "read_csv")
    def test_handles_nan_time_arrays(self, mock_read_csv, setup_run_dir, tmp_path):
        """It bypasses run configurations where time arrays contain all NaNs."""
        setup_run_dir("run_nan")
        mock_read_csv.return_value = pd.DataFrame({"Relative_time": [np.nan, np.nan, np.nan]})

        with pytest.raises(SystemExit, match="No complete-fill configurations found"):
            fit_prior.collect_complete_configs(tmp_path, time_col="Relative_time")

    def test_exits_when_no_directories_found(self, tmp_path):
        """It raises SystemExit if no configurations can be generated from the root."""
        with pytest.raises(SystemExit, match="No complete-fill configurations found"):
            fit_prior.collect_complete_configs(tmp_path, time_col="Relative_time")


class TestMainIntegration:
    """Integration-level checks for the prior fitting script execution."""

    # Use patch.object here as well
    @patch.object(fit_prior, "collect_complete_configs")
    def test_main_fits_and_saves_prior(self, mock_collect, tmp_path, monkeypatch):
        """It successfully delegates array collection, fits the prior, and serializes it to JSON."""
        mock_collect.return_value = np.array([[1.0, 3.0, 5.0, 7.0, 9.0], [1.5, 3.5, 5.5, 7.5, 9.5]])

        out_file = tmp_path / "test_prior.json"

        monkeypatch.setattr("sys.argv", ["fit_prior.py", "--out", str(out_file)])

        fit_prior.main()

        assert out_file.exists()

        data = json.loads(out_file.read_text())
        assert "pairs" in data
        assert "gap" in data
        assert len(data["pairs"]) == 4

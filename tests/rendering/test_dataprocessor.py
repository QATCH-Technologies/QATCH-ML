import numpy as np
import pandas as pd
import pytest

from src.systems.qmodel_7_onyx.rendering.dataprocessor import QModelOnyxDataProcessor as DP


class MockWorker:
    """A dummy worker to simulate buffer extraction."""

    def __init__(self, t_buffer: list, f_buffer: list, d_buffer: list):
        self._t = t_buffer
        self._f = f_buffer
        self._d = d_buffer

    def get_t1_buffer(self, index: int) -> list:
        return self._t

    def get_d1_buffer(self, index: int) -> list:
        return self._f

    def get_d2_buffer(self, index: int) -> list:
        return self._d


class TestConvertToDataframe:
    """Tests for the buffer extraction and truncation logic."""

    def test_extracts_buffers_successfully(self):
        """It correctly extracts equal-length buffers into a DataFrame."""
        worker = MockWorker([0.1, 0.2, 0.3], [100, 101, 102], [1.0, 1.1, 1.2])
        df = DP.convert_to_dataframe(worker)

        # Asserts the basic column structure is correct
        assert len(df) == 3
        assert list(df.columns) == [DP.COL_TIME, DP.COL_FREQ, DP.COL_DISS]
        assert df[DP.COL_TIME].iloc[0] == 0.1

    def test_truncates_uneven_buffers(self):
        """It truncates all arrays to the length of the shortest buffer."""
        # Provide buffers of lengths 4, 5, and 3
        worker = MockWorker([0.1, 0.2, 0.3, 0.4], [100, 101, 102, 103, 104], [1.0, 1.1, 1.2])
        df = DP.convert_to_dataframe(worker)

        # The DataFrame should be truncated to length 3
        assert len(df) == 3
        assert df[DP.COL_FREQ].iloc[-1] == 102

    def test_raises_value_error_on_missing_methods(self):
        """It raises an error if the worker lacks required buffer accessors."""

        class BadWorker:
            def get_t1_buffer(self, index):
                return []

            # Missing d1 and d2 methods

        with pytest.raises(ValueError, match="Worker is missing required method"):
            DP.convert_to_dataframe(BadWorker())

    def test_raises_value_error_on_empty_buffers(self):
        """It raises an error if the minimum buffer length is zero."""
        worker = MockWorker([0.1, 0.2], [], [1.0, 1.1])
        with pytest.raises(ValueError, match="One or more buffers are empty"):
            DP.convert_to_dataframe(worker)


class TestPreprocessDataframe:
    """Tests for the cleaning, interpolation, and smoothing pipeline."""

    @pytest.fixture
    def raw_df(self):
        """Provides a raw DataFrame with extra columns, duplicate times, and gaps."""
        return pd.DataFrame(
            {
                DP.COL_TIME: [0.0, 0.0, 0.02, 0.04],  # Duplicate 0.0, gap to 0.02[cite: 6]
                DP.COL_FREQ: [100.0, 101.0, 102.0, 110.0],
                DP.COL_DISS: [1.0, 1.5, 2.0, 5.0],
                "Temperature": [25.0, 25.1, 25.2, 25.3],  # Should be dropped[cite: 6]
                "Date": ["2026-01-01"] * 4,  # Should be dropped[cite: 6]
            }
        )

    def test_returns_none_if_time_column_missing(self):
        """It safely aborts if the required relative time column is absent."""
        df = pd.DataFrame({"Resonance_Frequency": [100.0]})
        result = DP.preprocess_dataframe(df)
        assert result is None  # [cite: 6]

    def test_drops_unnecessary_columns(self, raw_df):
        """It drops specific metadata columns like Temperature and Date."""
        df = DP.preprocess_dataframe(raw_df)

        assert "Temperature" not in df.columns  # [cite: 6]
        assert "Date" not in df.columns  # [cite: 6]
        assert DP.COL_FREQ in df.columns

    def test_deduplicates_and_reindexes_time_grid(self, raw_df):
        """It drops duplicate times and interpolates to a fixed TIME_STEP grid."""
        df = DP.preprocess_dataframe(raw_df)

        # Max time is 0.04, min is 0.0, step is 0.005[cite: 6]
        # np.arange(0.0, 0.04, 0.005) yields 8 points[cite: 6]
        expected_times = np.arange(0.0, 0.04, DP.TIME_STEP)

        assert len(df) == len(expected_times)
        assert np.allclose(df[DP.COL_TIME].values, expected_times)

        # Original df had a duplicate at 0.0 (values 100.0 and 101.0).
        # keep="first" ensures 100.0 is retained[cite: 6]
        assert df[DP.COL_FREQ].iloc[0] == 100.0

    def test_applies_median_filter(self):
        """It verifies that a scipy median filter is applied to numeric columns."""
        # Using a kernel size of 5[cite: 6]
        # Construct a spike that a median filter of size 5 will flatten
        times = np.arange(0.0, 0.05, 0.005)  # 10 points
        freqs = np.array([10.0, 10.0, 10.0, 100.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        raw = pd.DataFrame({DP.COL_TIME: times, DP.COL_FREQ: freqs})

        df = DP.preprocess_dataframe(raw)

        # The spike of 100.0 should be smoothed out to 10.0 by the size-5 median filter[cite: 6]
        assert df[DP.COL_FREQ].max() == 10.0

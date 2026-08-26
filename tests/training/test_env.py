import os
from pathlib import Path
from unittest.mock import MagicMock

from src.systems.qmodel_7_onyx.training import env


class TestSetupCudaEnv:
    """Tests for the PyTorch CUDA memory environment configuration."""

    def test_sets_default_allocation_conf(self, monkeypatch):
        """It safely applies the expandable_segments config if not already set."""
        # Ensure the environment variable is clean before testing
        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

        env.setup_cuda_env()

        assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"  #

    def test_respects_existing_value(self, monkeypatch):
        """It uses setdefault and does not overwrite user-configured memory settings."""
        # Simulate a user or earlier process setting a custom configuration
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

        env.setup_cuda_env()

        # The existing value should remain untouched
        assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "max_split_size_mb:128"


class TestStageResult:
    """Tests for the StageResult data class."""

    def test_instantiation_without_metrics(self):
        """It correctly instantiates with the default None for metrics."""
        path = Path("/fake/weights/best.pt")
        result = env.StageResult(stage="ch2_zoom", weights_path=path)  #

        assert result.stage == "ch2_zoom"
        assert result.weights_path == path
        assert result.metrics is None  #

    def test_instantiation_with_metrics(self):
        """It stores an extracted metrics dictionary when provided."""
        path = Path("/fake/weights/best.pt")
        metrics = {"mAP50": 0.99, "loss": 0.05}
        result = env.StageResult(stage="fill_classifier", weights_path=path, metrics=metrics)  #

        assert result.metrics == {"mAP50": 0.99, "loss": 0.05}


class TestExtractMetrics:
    """Tests for the robust metric extraction utility."""

    def test_extracts_from_train_return_results_dict(self):
        """It successfully pulls the results_dict from the primary return object."""
        mock_return = MagicMock()
        mock_return.results_dict = {"precision": 0.95}  #

        metrics = env.extract_metrics(train_return=mock_return)
        assert metrics == {"precision": 0.95}

    def test_extracts_from_model_metrics_results_dict(self):
        """It falls back to extracting metrics from the model instance if the return object lacks them."""
        mock_model = MagicMock()
        mock_model.metrics.results_dict = {"recall": 0.92}  #

        # Pass None as the train_return to force fallback
        metrics = env.extract_metrics(train_return=None, model=mock_model)
        assert metrics == {"recall": 0.92}

    def test_extracts_via_to_dict_method(self):
        """It handles Ultralytics versions where the candidate has a to_dict() callable instead."""
        mock_return = MagicMock(results_dict=None)  # Ensure results_dict is missing
        mock_return.to_dict.return_value = {"f1_score": 0.88}  #

        metrics = env.extract_metrics(train_return=mock_return)
        assert metrics == {"f1_score": 0.88}

    def test_handles_to_dict_exception_gracefully(self):
        """It catches arbitrary exceptions from to_dict() and safely falls through to None."""
        mock_return = MagicMock(results_dict=None)
        mock_return.to_dict.side_effect = Exception("Internal Ultralytics error")  #

        metrics = env.extract_metrics(train_return=mock_return)
        # Should not raise the Exception, but instead return None
        assert metrics is None

    def test_returns_none_if_no_metrics_found(self):
        """It returns None when neither candidate object matches the expected shapes."""
        # Neither None nor arbitrary strings have results_dict or to_dict()
        assert env.extract_metrics(train_return=None, model=None) is None
        assert env.extract_metrics(train_return="Unexpected String", model=123) is None

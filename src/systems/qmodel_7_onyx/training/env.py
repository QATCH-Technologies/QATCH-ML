"""Shared training-process environment setup and utility types.

Provides common environment configuration needed before importing PyTorch
to reduce CUDA out-of-memory errors, as well as the standard result type
returned by both training entry points (`train_detectors` and
`train_fill_classifier`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def setup_cuda_env() -> None:
    """Reduces fragmentation-driven OOMs before PyTorch is imported.

    Sets the `PYTORCH_CUDA_ALLOC_CONF` environment variable to
    `expandable_segments:True`. This is harmless if unsupported by the
    installed PyTorch version. It must be called before `torch` or
    `ultralytics` is imported anywhere in the process to take effect.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class StageResult:
    """Outcome of one completed YOLO training stage.

    Attributes:
        stage (str): Tag the stage was trained under (e.g., "ch2_zoom" or
            "fill_classifier").
        weights_path (Path): Location of the stage's best checkpoint.
        metrics (Optional[Dict[str, Any]]): Best-effort final validation
            metrics reported by Ultralytics, or None if none could be
            extracted.
    """

    stage: str
    weights_path: Path
    metrics: Optional[Dict[str, Any]] = None


def extract_metrics(train_return: Any, model: Any = None) -> Optional[Dict[str, Any]]:
    """Extracts a plain-dictionary metrics summary from a training call.

    Ultralytics' `model.train(...)` return shape has varied across
    versions (e.g., a metrics object with a `results_dict`, or `None` with the
    results attached to the model instance instead via `model.metrics`).
    This function safely attempts to locate and extract these metrics without
    raising exceptions. It returns None if nothing recognizable is found
    rather than letting a version mismatch break the training run over a
    reporting nicety.

    Args:
        train_return (Any): The value returned by `model.train(...)`.
        model (Any, optional): The trained model instance, consulted as a
            fallback when `train_return` doesn't carry metrics directly.
            Defaults to None.

    Returns:
        Optional[Dict[str, Any]]: A plain dictionary of metrics, or None if
        none could be successfully extracted from either source.
    """
    for candidate in (train_return, getattr(model, "metrics", None)):
        if candidate is None:
            continue
        results_dict = getattr(candidate, "results_dict", None)
        if isinstance(results_dict, dict):
            return dict(results_dict)
        to_dict = getattr(candidate, "to_dict", None)
        if callable(to_dict):
            try:
                d = to_dict()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    return None

"""
env.py
======

Shared training-process environment setup, plus the small result type both
training entry points (:mod:`.train_detectors`, :mod:`.train_fill_classifier`)
return.

Both training entry points independently need the same CUDA allocator env
var set before importing torch; this module centralizes that into one call
so the two scripts stay in sync.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def setup_cuda_env() -> None:
    """Reduce fragmentation-driven OOMs before torch is imported.

    Harmless if unsupported by the installed torch. Must be called before
    `torch`/`ultralytics` is imported anywhere in the process to take
    effect.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class StageResult:
    """Outcome of one completed YOLO training stage.

    Attributes:
        stage (str): Tag the stage was trained under (e.g. "ch2_zoom" or
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
    """Best-effort extraction of a plain-dict metrics summary from a training call.

    Ultralytics' `model.train(...)` return shape has varied across
    versions (a metrics object with a `results_dict`, or `None` with the
    results attached to the model instance instead via `model.metrics`).
    This never raises; it returns None if nothing recognizable is found
    rather than letting a version mismatch break the training run over a
    reporting nicety.

    Args:
        train_return (Any): The value returned by `model.train(...)`.
        model (Any, optional): The trained model instance, consulted as a
            fallback when `train_return` doesn't carry metrics directly.
            Defaults to None.

    Returns:
        Optional[Dict[str, Any]]: A plain dict of metrics, or None if none
        could be extracted from either source.
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

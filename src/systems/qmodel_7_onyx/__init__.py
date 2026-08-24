"""QATCH sensor-run POI detection pipeline.

The public entry point is :class:`Workspace` + :class:`Pipeline`::

    from src.systems.qmodel_7_onyx import Workspace, Pipeline

    pipeline = Pipeline(Workspace(data_root="path/to/raw"))
    result = pipeline.run()

See :mod:`.pipeline` for the full API, or ``ARCHITECTURE.md`` for how the
individual subpackages (``corpus``, ``tiers``, ``decode``, ``rendering``,
``dataset``, ``training``, ``inference``, ``live``, ``qa``) fit together.
"""

from .pipeline import (
    DatasetBuildResult,
    Pipeline,
    PipelineError,
    PipelineResult,
    PreparationResult,
    TrainingResult,
    Workspace,
)

__all__ = [
    "Workspace",
    "Pipeline",
    "PreparationResult",
    "DatasetBuildResult",
    "TrainingResult",
    "PipelineResult",
    "PipelineError",
]

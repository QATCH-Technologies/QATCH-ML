"""
pipeline.py
===========

The public entry point for this system: ``Workspace`` + ``Pipeline``.

Today, going from "a folder of raw data" to "trained model weights" means
running six CLI scripts by hand in a specific order, with path-wiring
enforced only by convention (every stage happens to default to the same
``paths.py`` constants — override one flag and the chain silently breaks
until a downstream ``FileNotFoundError``):

    decode.fit_prior -> tiers -> dataset.build_detectors ->
    dataset.build_fill_classifier -> training.train_detectors ->
    training.train_fill_classifier

``Workspace`` is the single source of truth for every path a pipeline stage
reads or writes (no ``"v7"``/``"v7_fill"`` literals repeated at the call
site). ``Pipeline`` wraps the underlying stage functions — none of which are
renamed or altered in behavior beyond now returning their results instead of
``None`` — behind three methods (``prepare``, ``build_datasets``, ``train``)
plus a ``run()`` convenience that chains all three:

    from src.systems.qmodel_7_onyx import Workspace, Pipeline

    pipeline = Pipeline(Workspace(data_root="path/to/raw"))
    result = pipeline.run()
    print(result.training.weights)   # {"init": Path(...), ..., "fill_classifier": Path(...)}

Each stage is also usable on its own (``pipeline.prepare()``,
``pipeline.build_datasets(targets=["fill_classifier"])``,
``pipeline.train(detector_stages=["ch2_zoom"])``) for callers who want more
control than the one-shot ``run()``.

``fit_tiers()``/``collect_complete_configs()``/the dataset builders raise
bare ``SystemExit`` today (correct for a CLI, wrong for a library call) —
``Pipeline`` converts those into :class:`PipelineError` so a programmatic
caller gets a normal, catchable exception instead of its process being
killed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Union

import numpy as np

from . import paths
from .corpus import discover_runs
from .dataset.build_detectors import build as _build_detector_dataset
from .dataset.build_fill_classifier import build as _build_fill_dataset
from .decode.fit_prior import collect_complete_configs
from .decode.spacing_prior import SpacingPrior
from .tiers import TierScheme, fit_tiers
from .training.env import StageResult
from .training.train_detectors import STAGE_CHOICES, STAGE_EPOCHS
from .training.train_detectors import train_stage as _train_detector_stage
from .training.train_fill_classifier import DEFAULT_EPOCHS as _FILL_DEFAULT_EPOCHS
from .training.train_fill_classifier import train as _train_fill_classifier

VALID_DATASET_TARGETS = frozenset({"detectors", "fill_classifier"})
VALID_TRAIN_TARGETS = frozenset({"detectors", "fill_classifier"})

_DEFAULT_DETECTOR_BATCH = 16
_DEFAULT_FILL_BATCH = 128
_DEFAULT_DETECTOR_IMGSZ = 1536


class PipelineError(RuntimeError):
    """Raised for any pipeline-stage failure that would otherwise surface as
    a bare ``SystemExit`` (insufficient/missing data, an unknown target
    name, etc.) — always catchable, never kills the caller's process."""


def _normalize_targets(targets: Union[str, Sequence[str]], valid: frozenset) -> Set[str]:
    chosen = {targets} if isinstance(targets, str) else set(targets)
    unknown = chosen - valid
    if unknown:
        raise PipelineError(
            f"unknown target(s) {sorted(unknown)}; valid targets are {sorted(valid)}"
        )
    if not chosen:
        raise PipelineError(f"no targets given; choose from {sorted(valid)}")
    return chosen


def _read_manifest(dataset_dir: Path) -> dict:
    manifest_path = Path(dataset_dir) / "manifest.json"
    return json.loads(manifest_path.read_text())


@dataclass
class Workspace:
    """Every filesystem root a pipeline stage reads or writes, in one place.

    Defaults to this repo's :mod:`paths` constants; override any field to
    point a :class:`Pipeline` at a different data/output location entirely
    (not just via the ``QMODEL_*`` environment variables ``paths.py``
    supports) — e.g. a fresh project working with its own raw-data folder.
    """

    data_root: Path = field(default_factory=lambda: paths.DATA_ROOT)
    datasets_root: Path = field(default_factory=lambda: paths.DATASETS_ROOT)
    runs_root: Path = field(default_factory=lambda: paths.RUNS_ROOT)
    configs_root: Path = field(default_factory=lambda: paths.CONFIGS_ROOT)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.datasets_root = Path(self.datasets_root)
        self.runs_root = Path(self.runs_root)
        self.configs_root = Path(self.configs_root)

    @property
    def tiers_path(self) -> Path:
        return self.configs_root / "tiers.json"

    @property
    def spacing_prior_path(self) -> Path:
        return self.configs_root / "spacing_prior.json"

    @property
    def detector_dataset_dir(self) -> Path:
        return self.datasets_root / "v7"

    @property
    def fill_dataset_dir(self) -> Path:
        return self.datasets_root / "v7_fill"

    @property
    def detector_runs_dir(self) -> Path:
        return self.runs_root / "v7"

    @property
    def fill_runs_dir(self) -> Path:
        return self.runs_root / "v7_fill"


@dataclass
class PreparationResult:
    """What :meth:`Pipeline.prepare` fitted from raw data."""

    tiers: TierScheme
    tiers_path: Path
    prior: SpacingPrior
    prior_path: Path
    n_runs: int


@dataclass
class DatasetBuildResult:
    """What :meth:`Pipeline.build_datasets` built. Fields for a target that
    wasn't requested stay ``None`` rather than being omitted, so callers can
    check ``result.fill_manifest is not None`` uniformly."""

    detector_manifest: Optional[dict] = None
    fill_manifest: Optional[dict] = None
    detector_dataset_dir: Optional[Path] = None
    fill_dataset_dir: Optional[Path] = None


@dataclass
class TrainingResult:
    """What :meth:`Pipeline.train` produced: best-checkpoint path and
    (best-effort) final validation metrics per trained stage, keyed by stage
    name (``"init"``, ``"ch1_zoom"``, ..., ``"fill_classifier"``)."""

    weights: Dict[str, Path] = field(default_factory=dict)
    metrics: Dict[str, Optional[dict]] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """The full output of :meth:`Pipeline.run`."""

    preparation: PreparationResult
    datasets: DatasetBuildResult
    training: TrainingResult


class Pipeline:
    """Orchestrates the fresh-data -> fitted-config -> dataset -> trained-weights
    pipeline over a single :class:`Workspace`, so every stage agrees on where
    everything lives by construction."""

    def __init__(self, workspace: Optional[Workspace] = None, **workspace_overrides: Any):
        if workspace is not None and workspace_overrides:
            raise PipelineError("pass either `workspace` or workspace keyword overrides, not both")
        self.workspace = workspace if workspace is not None else Workspace(**workspace_overrides)

    # ------------------------------------------------------------------
    # Stage 1: fit the viscosity-tier scheme and spacing prior from raw data
    # ------------------------------------------------------------------
    def prepare(
        self,
        *,
        max_tiers: int = 8,
        min_support: int = 40,
        frac_blend: float = 0.5,
        limit: Optional[int] = None,
        time_col: str = "Relative_time",
    ) -> PreparationResult:
        ws = self.workspace
        try:
            runs = discover_runs(ws.data_root)
            viscosities = np.array(
                [r.viscosity_cP for r in runs if r.viscosity_cP is not None], dtype=float
            )
            n_unknown = sum(1 for r in runs if r.viscosity_cP is None)
            tiers = fit_tiers(viscosities, max_tiers=max_tiers, min_support=min_support)
            tiers.n_per_tier[-1] = n_unknown
            ws.configs_root.mkdir(parents=True, exist_ok=True)
            tiers.save(ws.tiers_path)

            configs = collect_complete_configs(ws.data_root, time_col, limit)
            prior = SpacingPrior.fit(configs, frac_blend=frac_blend)
            prior.save(ws.spacing_prior_path)
        except SystemExit as exc:
            raise PipelineError(str(exc)) from exc

        return PreparationResult(
            tiers=tiers,
            tiers_path=ws.tiers_path,
            prior=prior,
            prior_path=ws.spacing_prior_path,
            n_runs=len(runs),
        )

    # ------------------------------------------------------------------
    # Stage 2: render the YOLO datasets
    # ------------------------------------------------------------------
    def build_datasets(
        self,
        targets: Union[str, Sequence[str]] = ("detectors", "fill_classifier"),
        *,
        base_variants: int = 2,
        val_frac: float = 0.15,
        repeat_cap: int = 8,
        seed: int = 7,
        limit: Optional[int] = None,
        cuts_per_class: int = 2,
        hard_cuts: int = 1,
    ) -> DatasetBuildResult:
        chosen = _normalize_targets(targets, VALID_DATASET_TARGETS)
        ws = self.workspace
        result = DatasetBuildResult()
        try:
            if "detectors" in chosen:
                _build_detector_dataset(
                    ws.data_root,
                    ws.tiers_path,
                    ws.detector_dataset_dir,
                    base_variants=base_variants,
                    val_frac=val_frac,
                    repeat_cap=repeat_cap,
                    seed=seed,
                    limit=limit,
                )
                result.detector_manifest = _read_manifest(ws.detector_dataset_dir)
                result.detector_dataset_dir = ws.detector_dataset_dir
            if "fill_classifier" in chosen:
                _build_fill_dataset(
                    ws.data_root,
                    ws.tiers_path,
                    ws.fill_dataset_dir,
                    base_variants=base_variants,
                    cuts_per_class=cuts_per_class,
                    hard_cuts=hard_cuts,
                    val_frac=val_frac,
                    repeat_cap=repeat_cap,
                    seed=seed,
                    limit=limit,
                )
                result.fill_manifest = _read_manifest(ws.fill_dataset_dir)
                result.fill_dataset_dir = ws.fill_dataset_dir
        except SystemExit as exc:
            raise PipelineError(str(exc)) from exc
        return result

    # ------------------------------------------------------------------
    # Stage 3: train
    # ------------------------------------------------------------------
    def train(
        self,
        targets: Union[str, Sequence[str]] = ("detectors", "fill_classifier"),
        *,
        detector_stages: Optional[Sequence[str]] = None,
        size: str = "s",
        epochs: Optional[int] = None,
        batch: Optional[int] = None,
        imgsz: int = _DEFAULT_DETECTOR_IMGSZ,
        device: str = "0",
        seed: int = 7,
        resume: bool = False,
    ) -> TrainingResult:
        chosen = _normalize_targets(targets, VALID_TRAIN_TARGETS)
        ws = self.workspace
        result = TrainingResult()
        try:
            if "detectors" in chosen:
                stages = list(detector_stages) if detector_stages else list(STAGE_CHOICES)
                unknown = set(stages) - set(STAGE_CHOICES)
                if unknown:
                    raise PipelineError(
                        f"unknown detector stage(s) {sorted(unknown)}; "
                        f"valid stages are {STAGE_CHOICES}"
                    )
                for stage in stages:
                    stage_result: StageResult = _train_detector_stage(
                        ws.detector_dataset_dir,
                        stage,
                        size,
                        epochs or STAGE_EPOCHS[stage],
                        ws.detector_runs_dir,
                        batch if batch is not None else _DEFAULT_DETECTOR_BATCH,
                        imgsz,
                        seed,
                        resume,
                        device,
                    )
                    result.weights[stage_result.stage] = stage_result.weights_path
                    result.metrics[stage_result.stage] = stage_result.metrics

            if "fill_classifier" in chosen:
                fill_result: StageResult = _train_fill_classifier(
                    ws.fill_dataset_dir,
                    size,
                    epochs or _FILL_DEFAULT_EPOCHS,
                    ws.fill_runs_dir,
                    batch if batch is not None else _DEFAULT_FILL_BATCH,
                    seed,
                    resume,
                    device,
                )
                result.weights[fill_result.stage] = fill_result.weights_path
                result.metrics[fill_result.stage] = fill_result.metrics
        except SystemExit as exc:
            raise PipelineError(str(exc)) from exc
        return result

    # ------------------------------------------------------------------
    # Convenience: the whole thing in one call
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        prepare_kwargs: Optional[dict] = None,
        build_kwargs: Optional[dict] = None,
        train_kwargs: Optional[dict] = None,
    ) -> PipelineResult:
        """``prepare()`` -> ``build_datasets()`` -> ``train()``, one call.

        Pass per-stage keyword overrides via ``prepare_kwargs``/
        ``build_kwargs``/``train_kwargs`` (each forwarded to the matching
        method) when the defaults aren't right for a particular run.
        """
        preparation = self.prepare(**(prepare_kwargs or {}))
        datasets = self.build_datasets(**(build_kwargs or {}))
        training = self.train(**(train_kwargs or {}))
        return PipelineResult(preparation=preparation, datasets=datasets, training=training)

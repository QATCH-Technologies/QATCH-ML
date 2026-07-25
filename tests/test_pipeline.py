import json
from pathlib import Path

import numpy as np
import pytest

from src.systems.qmodel_7_onyx import pipeline as pl
from src.systems.qmodel_7_onyx.corpus import RunRecord
from src.systems.qmodel_7_onyx.decode.spacing_prior import SpacingPrior
from src.systems.qmodel_7_onyx.tiers import TierScheme
from src.systems.qmodel_7_onyx.training.env import StageResult

# ===========================================================================
#  Workspace
# ===========================================================================


def test_workspace_defaults_match_paths_module():
    from src.systems.qmodel_7_onyx import paths

    ws = pl.Workspace()
    assert ws.data_root == paths.DATA_ROOT
    assert ws.datasets_root == paths.DATASETS_ROOT
    assert ws.runs_root == paths.RUNS_ROOT
    assert ws.configs_root == paths.CONFIGS_ROOT


def test_workspace_derived_paths():
    ws = pl.Workspace(datasets_root="datasets", runs_root="runs", configs_root="configs")
    assert ws.tiers_path == Path("configs/tiers.json")
    assert ws.spacing_prior_path == Path("configs/spacing_prior.json")
    assert ws.detector_dataset_dir == Path("datasets/v7")
    assert ws.fill_dataset_dir == Path("datasets/v7_fill")
    assert ws.detector_runs_dir == Path("runs/v7")
    assert ws.fill_runs_dir == Path("runs/v7_fill")


def test_workspace_normalizes_str_to_path():
    ws = pl.Workspace(data_root="some/string/path")
    assert isinstance(ws.data_root, Path)
    assert ws.data_root == Path("some/string/path")


# ===========================================================================
#  Pipeline construction
# ===========================================================================


def test_pipeline_accepts_explicit_workspace():
    ws = pl.Workspace(data_root="x")
    p = pl.Pipeline(ws)
    assert p.workspace is ws


def test_pipeline_accepts_workspace_overrides():
    p = pl.Pipeline(data_root="y")
    assert p.workspace.data_root == Path("y")


def test_pipeline_rejects_both_workspace_and_overrides():
    ws = pl.Workspace()
    with pytest.raises(pl.PipelineError):
        pl.Pipeline(ws, data_root="y")


# ===========================================================================
#  _normalize_targets
# ===========================================================================


def test_normalize_targets_accepts_single_string():
    assert pl._normalize_targets("detectors", pl.VALID_DATASET_TARGETS) == {"detectors"}


def test_normalize_targets_rejects_unknown():
    with pytest.raises(pl.PipelineError):
        pl._normalize_targets(["bogus"], pl.VALID_DATASET_TARGETS)


def test_normalize_targets_rejects_empty():
    with pytest.raises(pl.PipelineError):
        pl._normalize_targets([], pl.VALID_DATASET_TARGETS)


# ===========================================================================
#  prepare()
# ===========================================================================


def _fake_runs(n=5):
    return [
        RunRecord(run_id=f"{i:05d}", csv_path=None, poi_times={"POI1": 1.0}, viscosity_cP=10.0 + i)
        for i in range(n)
    ]


def test_prepare_writes_tiers_and_prior(tmp_path, monkeypatch):
    ws = pl.Workspace(
        data_root=tmp_path / "raw",
        configs_root=tmp_path / "configs",
        datasets_root=tmp_path / "datasets",
        runs_root=tmp_path / "runs",
    )
    fake_tiers = TierScheme(edges_cp=[1.0, 10.0], n_per_tier=[1, 2, 0])
    fake_prior = SpacingPrior.fit(
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0]] * 5, dtype=float)
    )  # trivially degenerate but fits without error

    monkeypatch.setattr(pl, "discover_runs", lambda raw_root: _fake_runs())
    monkeypatch.setattr(pl, "fit_tiers", lambda v, max_tiers, min_support: fake_tiers)
    monkeypatch.setattr(
        pl, "collect_complete_configs", lambda raw_root, time_col, limit: np.zeros((5, 5))
    )
    monkeypatch.setattr(SpacingPrior, "fit", staticmethod(lambda configs, frac_blend: fake_prior))

    result = pl.Pipeline(ws).prepare()

    assert result.n_runs == 5
    assert result.tiers is fake_tiers
    assert result.prior is fake_prior
    assert ws.tiers_path.exists()
    assert ws.spacing_prior_path.exists()
    assert json.loads(ws.tiers_path.read_text())["edges_cp"] == [1.0, 10.0]


def test_prepare_converts_system_exit_to_pipeline_error(tmp_path, monkeypatch):
    ws = pl.Workspace(data_root=tmp_path / "raw", configs_root=tmp_path / "configs")
    monkeypatch.setattr(pl, "discover_runs", lambda raw_root: [])

    def _raise(*a, **k):
        raise SystemExit("too few runs")

    monkeypatch.setattr(pl, "fit_tiers", _raise)

    with pytest.raises(pl.PipelineError, match="too few runs"):
        pl.Pipeline(ws).prepare()


# ===========================================================================
#  build_datasets()
# ===========================================================================


def _write_manifest(dataset_dir: Path, **fields) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(json.dumps(fields))


def test_build_datasets_reads_back_manifests(tmp_path, monkeypatch):
    ws = pl.Workspace(data_root=tmp_path / "raw", datasets_root=tmp_path / "datasets")

    def fake_build_detectors(raw_root, tiers_path, out_root, **kwargs):
        _write_manifest(out_root, n_train_runs=3)

    def fake_build_fill(raw_root, tiers_path, out_root, **kwargs):
        _write_manifest(out_root, n_train_runs=4)

    monkeypatch.setattr(pl, "_build_detector_dataset", fake_build_detectors)
    monkeypatch.setattr(pl, "_build_fill_dataset", fake_build_fill)

    result = pl.Pipeline(ws).build_datasets()

    assert result.detector_manifest == {"n_train_runs": 3}
    assert result.fill_manifest == {"n_train_runs": 4}
    assert result.detector_dataset_dir == ws.detector_dataset_dir
    assert result.fill_dataset_dir == ws.fill_dataset_dir


def test_build_datasets_respects_target_selection(tmp_path, monkeypatch):
    ws = pl.Workspace(data_root=tmp_path / "raw", datasets_root=tmp_path / "datasets")

    calls = []

    def fake_build_detectors(raw_root, tiers_path, out_root, **kwargs):
        calls.append("detectors")
        _write_manifest(out_root, ok=True)

    def fake_build_fill(raw_root, tiers_path, out_root, **kwargs):
        calls.append("fill_classifier")
        _write_manifest(out_root, ok=True)

    monkeypatch.setattr(pl, "_build_detector_dataset", fake_build_detectors)
    monkeypatch.setattr(pl, "_build_fill_dataset", fake_build_fill)

    result = pl.Pipeline(ws).build_datasets(targets=["detectors"])

    assert calls == ["detectors"]
    assert result.detector_manifest == {"ok": True}
    assert result.fill_manifest is None
    assert result.fill_dataset_dir is None


def test_build_datasets_converts_system_exit(tmp_path, monkeypatch):
    ws = pl.Workspace(data_root=tmp_path / "raw", datasets_root=tmp_path / "datasets")

    def _raise(*a, **k):
        raise SystemExit("no runs under raw root")

    monkeypatch.setattr(pl, "_build_detector_dataset", _raise)

    with pytest.raises(pl.PipelineError, match="no runs under raw root"):
        pl.Pipeline(ws).build_datasets(targets=["detectors"])


# ===========================================================================
#  train()
# ===========================================================================


def test_train_detectors_uses_per_stage_epoch_defaults(tmp_path, monkeypatch):
    ws = pl.Workspace(datasets_root=tmp_path / "datasets", runs_root=tmp_path / "runs")
    seen_epochs = {}

    def fake_train_stage(
        data_root, stage, size, epochs, project, batch, imgsz, seed, resume, device
    ):
        seen_epochs[stage] = epochs
        return StageResult(
            stage=stage, weights_path=project / stage / "best.pt", metrics={"mAP": 0.9}
        )

    monkeypatch.setattr(pl, "_train_detector_stage", fake_train_stage)

    result = pl.Pipeline(ws).train(targets=["detectors"], detector_stages=["init", "ch1_zoom"])

    assert seen_epochs == {
        "init": pl.STAGE_EPOCHS["init"],
        "ch1_zoom": pl.STAGE_EPOCHS["ch1_zoom"],
    }
    assert set(result.weights) == {"init", "ch1_zoom"}
    assert result.metrics["init"] == {"mAP": 0.9}


def test_train_detectors_epochs_override_applies_to_all_stages(tmp_path, monkeypatch):
    ws = pl.Workspace(datasets_root=tmp_path / "datasets", runs_root=tmp_path / "runs")
    seen_epochs = {}

    def fake_train_stage(
        data_root, stage, size, epochs, project, batch, imgsz, seed, resume, device
    ):
        seen_epochs[stage] = epochs
        return StageResult(stage=stage, weights_path=project / stage / "best.pt")

    monkeypatch.setattr(pl, "_train_detector_stage", fake_train_stage)

    pl.Pipeline(ws).train(targets=["detectors"], detector_stages=["init", "ch1"], epochs=5)

    assert seen_epochs == {"init": 5, "ch1": 5}


def test_train_rejects_unknown_detector_stage(tmp_path):
    ws = pl.Workspace(datasets_root=tmp_path / "datasets", runs_root=tmp_path / "runs")
    with pytest.raises(pl.PipelineError):
        pl.Pipeline(ws).train(targets=["detectors"], detector_stages=["not_a_real_stage"])


def test_train_fill_classifier_only(tmp_path, monkeypatch):
    ws = pl.Workspace(datasets_root=tmp_path / "datasets", runs_root=tmp_path / "runs")
    called = {}

    def fake_train_fill(data_root, size, epochs, project, batch, seed, resume, device):
        called["epochs"] = epochs
        return StageResult(stage="fill_classifier", weights_path=project / "best.pt")

    monkeypatch.setattr(pl, "_train_fill_classifier", fake_train_fill)

    result = pl.Pipeline(ws).train(targets=["fill_classifier"])

    assert called["epochs"] == pl._FILL_DEFAULT_EPOCHS
    assert set(result.weights) == {"fill_classifier"}


def test_train_converts_system_exit(tmp_path, monkeypatch):
    ws = pl.Workspace(datasets_root=tmp_path / "datasets", runs_root=tmp_path / "runs")

    def _raise(*a, **k):
        raise SystemExit("missing data.yaml")

    monkeypatch.setattr(pl, "_train_detector_stage", _raise)

    with pytest.raises(pl.PipelineError, match="missing data.yaml"):
        pl.Pipeline(ws).train(targets=["detectors"], detector_stages=["init"])


# ===========================================================================
#  run()
# ===========================================================================


def test_run_chains_all_three_stages_and_forwards_kwargs(tmp_path, monkeypatch):
    ws = pl.Workspace(datasets_root=tmp_path / "datasets", runs_root=tmp_path / "runs")
    p = pl.Pipeline(ws)

    calls = []

    def fake_prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return "PREP"

    def fake_build(**kwargs):
        calls.append(("build_datasets", kwargs))
        return "BUILD"

    def fake_train(**kwargs):
        calls.append(("train", kwargs))
        return "TRAIN"

    monkeypatch.setattr(p, "prepare", fake_prepare)
    monkeypatch.setattr(p, "build_datasets", fake_build)
    monkeypatch.setattr(p, "train", fake_train)

    result = p.run(
        prepare_kwargs={"min_support": 5},
        build_kwargs={"targets": ["detectors"]},
        train_kwargs={"targets": ["detectors"]},
    )

    assert [name for name, _ in calls] == ["prepare", "build_datasets", "train"]
    assert calls[0][1] == {"min_support": 5}
    assert result.preparation == "PREP"
    assert result.datasets == "BUILD"
    assert result.training == "TRAIN"

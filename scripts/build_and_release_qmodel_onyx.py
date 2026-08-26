#!/usr/bin/env python3
"""Builds, trains, releases, and evaluates the QModel Onyx production package.

Runs the end-to-end release pipeline in ordered stages: fetch the current
raw corpus, prepare viscosity tiers and spacing priors, build training
datasets, train selected model stages, deploy their best checkpoints,
optionally clean up completed training runs, and evaluate the resulting
deployment against POI ground truth.

The deployment evaluation is performed against the released package through
the same deployment controller and asset layout used by downstream
consumers. Evaluation results include per-POI timing/index errors, hit rates,
gross-error rates, viscosity-tier breakdowns, and diagnostic plots.

Usage:
    python scripts/build_and_release_qmodel_onyx.py \
        --dropbox-source "D:/Dropbox/QATCH runs"

    python scripts/build_and_release_qmodel_onyx.py \
        --skip-fetch --targets fill_classifier

    python scripts/build_and_release_qmodel_onyx.py \
        --dropbox-source "D:/Dropbox/QATCH runs" \
        --targets detectors --detector-stages ch2_zoom

    python scripts/build_and_release_qmodel_onyx.py \
        --skip-fetch --skip-eval
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _qmodel_onyx_layout import build_model_assets, deploy_subpath, load_assets_map  # noqa: E402

from src.systems.qmodel_7_onyx import paths  # noqa: E402
from src.systems.qmodel_7_onyx.corpus import (  # noqa: E402
    TIER_EDGES,
    RunRecord,
    dedupe_runs,
    discover_runs,
    load_run_filter,
)
from src.systems.qmodel_7_onyx.pipeline import (  # noqa: E402
    Pipeline,
    PipelineError,
    Workspace,
)
from src.systems.qmodel_7_onyx.tiers import TierScheme  # noqa: E402
from src.systems.qmodel_7_onyx.training.train_detectors import STAGE_CHOICES  # noqa: E402
from src.utils.dataset_fetcher import DatasetFetcher  # noqa: E402
from src.utils.logger import configure_logging, get_logger  # noqa: E402

LOG = get_logger("build_and_release_qmodel_onyx")

DEFAULT_RELEASE_DIR = paths.REPO_ROOT / "qmodel_onyx"
DEFAULT_DEPLOYMENT_DIR = paths.REPO_ROOT / "src" / "systems" / "qmodel_7_onyx" / "deployment"
CHAIN_TO_PROD = {"POI1": "POI1", "POI2": "POI2", "POI3": "POI4", "POI4": "POI5", "POI5": "POI6"}
POI_KEYS = list(CHAIN_TO_PROD)
RESULTS_COLUMNS = [
    "run_id",
    "poi",
    "tier",
    "viscosity_cP",
    "true_index",
    "pred_index",
    "index_err",
    "true_t_s",
    "pred_t_s",
    "hit",
    "time_err_s",
    "abs_time_err_s",
]
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
SECONDARY = "#52514e"
PRIMARY = "#0b0b0b"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the release pipeline.

    Returns:
        Parsed command-line options controlling data fetching, workspace
        locations, dataset construction, training, deployment, cleanup, and
        post-release evaluation.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    fetch = ap.add_argument_group("1. fetch")
    fetch.add_argument(
        "--dropbox-source",
        type=str,
        default=None,
        help="Path to the local Dropbox sync folder holding raw run captures. "
        "Falls back to the QMODEL_DROPBOX_SOURCE env var, then to "
        f"{paths.DROPBOX_SOURCE} (this machine's home directory + the "
        "QATCH Dropbox team folder). Pass --skip-fetch to reuse whatever is "
        "already under data/raw instead.",
    )
    fetch.add_argument("--skip-fetch", action="store_true", help="Never contact Dropbox.")
    fetch.add_argument(
        "--num-files", type=int, default=None, help="Cap on new runs copied from Dropbox."
    )

    ws = ap.add_argument_group("workspace")
    ws.add_argument("--data-root", type=Path, default=None, help="Default: data/raw")
    ws.add_argument("--datasets-root", type=Path, default=None, help="Default: datasets/")
    ws.add_argument("--runs-root", type=Path, default=None, help="Default: runs/")
    ws.add_argument("--configs-root", type=Path, default=None, help="Default: configs/")

    train = ap.add_argument_group("3-4. build + train")
    train.add_argument(
        "--targets",
        nargs="+",
        choices=["detectors", "fill_classifier"],
        default=["detectors", "fill_classifier"],
        help="Which model families to (re)build datasets for and train.",
    )
    train.add_argument(
        "--detector-stages",
        nargs="+",
        choices=STAGE_CHOICES,
        default=None,
        help="Subset of detector stages to train (default: all of %(choices)s). "
        "Ignored unless --targets includes detectors.",
    )
    train.add_argument("--size", choices=["n", "s", "m", "l", "xl"], default="s")
    train.add_argument("--epochs", type=int, default=None, help="Override every stage's default.")
    train.add_argument("--batch", type=int, default=None, help="Override every stage's default.")
    train.add_argument("--imgsz", type=int, default=1536, help="Detector render size.")
    train.add_argument("--device", default="0")
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--resume", action="store_true")

    rel = ap.add_argument_group("5-6. release + cleanup")
    rel.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help=f"Where to drop deployable weights. Default: {DEFAULT_RELEASE_DIR}",
    )
    rel.add_argument(
        "--also-update-local-assets",
        action="store_true",
        help="Also copy the same weights into src/systems/qmodel_7_onyx/assets/, "
        "so qa/benchmark.py and local inference pick them up immediately.",
    )
    rel.add_argument(
        "--keep-runs",
        action="store_true",
        help="Don't purge a stage's run directory after a successful deploy. By "
        "default, each stage's raw Ultralytics run tree (checkpoints, logs, plots) "
        "is deleted once its best checkpoint has been copied into --output; only "
        "the stage(s) actually deployed THIS invocation are touched, never runs/ "
        "as a whole, so an unrelated stage's in-progress or --resume-pending run "
        "is left alone.",
    )

    ev = ap.add_argument_group("7. eval")
    ev.add_argument(
        "--skip-eval",
        action="store_true",
        help="Don't run the post-release deployment eval.",
    )
    ev.add_argument(
        "--eval-output",
        type=Path,
        default=paths.ARTIFACTS_ROOT / "eval_onyx_deployment",
        help="Where to write eval results (results_long.csv, summary.csv, plots/).",
    )
    ev.add_argument(
        "--eval-deployment-dir",
        type=Path,
        default=DEFAULT_DEPLOYMENT_DIR,
        help="Directory holding the deployment onyx.py + siblings (the modules under "
        "test) -- NOT the training repo's inference/controller.py.",
    )
    ev.add_argument(
        "--eval-only-runs",
        type=Path,
        default=None,
        help="Restrict eval to run ids in this file (a build_dataset manifest.json -- "
        "its val_ids are used -- or a plain run-id list). Default: the detector "
        "dataset's just-built val split when available -- the only honest eval set "
        "for a model trained on this corpus.",
    )
    ev.add_argument(
        "--eval-n-runs", type=int, default=None, help="Cap the eval corpus to N runs (seeded)."
    )
    ev.add_argument("--eval-seed", type=int, default=1337)
    ev.add_argument("--eval-gross-threshold", type=float, default=2.0, help="Seconds.")
    ev.add_argument(
        "--eval-no-decode-config", action="store_true", help="Disable the configuration decode."
    )
    ev.add_argument(
        "--eval-no-refine-pois", action="store_true", help="Disable zoom-detector refinement."
    )
    ev.add_argument(
        "--eval-restart", action="store_true", help="Ignore existing eval progress and start over."
    )
    ev.add_argument("--eval-log-every", type=int, default=25)

    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def stage_fetch(args: argparse.Namespace, workspace: Workspace) -> int | None:
    """Refreshes the raw run corpus from the configured Dropbox source.

    The existing raw-data directory is removed before fetching so the result
    is a clean mirror of the current source rather than an incremental blend
    with previously fetched runs.

    Args:
        args: Parsed command-line arguments containing fetch configuration.
        workspace: Pipeline workspace defining the raw-data destination.

    Returns:
        Number of run directories retained after the refresh, or `None` if
        fetching was skipped or the configured source was unavailable.
    """
    if args.skip_fetch:
        LOG.info("[1/7] Fetch: skipped (--skip-fetch).")
        return None

    source = Path(args.dropbox_source) if args.dropbox_source else paths.DROPBOX_SOURCE
    if not source.exists():
        LOG.info(
            "[1/7] Fetch: skipped (Dropbox source not found on this machine: {}). "
            "Pass --dropbox-source or set QMODEL_DROPBOX_SOURCE if it lives elsewhere.",
            source,
        )
        return None

    if workspace.data_root.exists():
        LOG.info("[1/7] Fetch: purging stale raw corpus at {}", workspace.data_root)
        shutil.rmtree(workspace.data_root)
    workspace.data_root.mkdir(parents=True, exist_ok=True)

    LOG.info("[1/7] Fetch: copying runs from {} -> {}", source, workspace.data_root)
    fetcher = DatasetFetcher(
        source_dir=str(source),
        target_dir=str(workspace.data_root),
        num_files=args.num_files,
    )
    report_path = paths.ARTIFACTS_ROOT / "dropbox_fetch_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fetcher.run(report_path=report_path)
    n_retained = len(list(workspace.data_root.glob("*")))
    LOG.info(
        "[1/7] Fetch: {} run(s) retained after refresh ({} failure(s) logged to {})",
        n_retained,
        len(fetcher.failures),
        report_path,
    )
    return n_retained


def stage_prepare(pipeline: Pipeline):
    """Fits the viscosity-tier scheme and spacing prior from the raw corpus.

    Args:
        pipeline: Pipeline configured with the current workspace.

    Returns:
        Preparation results produced by :meth:`Pipeline.prepare`, including
        the fitted tier scheme and generated spacing-prior artifact.
    """
    LOG.info("[2/7] Prepare: fitting viscosity tiers + spacing prior from raw data...")
    result = pipeline.prepare()
    LOG.info(
        "[2/7] Prepare: {} run(s) discovered, {} viscosity tier(s) fit -> {}",
        result.n_runs,
        len(result.tiers.labels),
        result.tiers_path,
    )
    return result


def stage_build(pipeline: Pipeline, args: argparse.Namespace):
    """Builds the requested YOLO training and validation datasets.

    Args:
        pipeline: Pipeline configured with the current workspace.
        args: Parsed command-line arguments specifying the model targets to
            build.

    Returns:
        Dataset-build results produced by :meth:`Pipeline.build_datasets`.
    """
    LOG.info("[3/7] Build: rendering YOLO datasets for {}...", args.targets)
    result = pipeline.build_datasets(targets=args.targets)
    if result.detector_manifest is not None:
        m = result.detector_manifest
        LOG.info(
            "[3/7] Build: detectors -> {} train run(s), {} val run(s) -> {}",
            m["n_train_runs"],
            m["n_val_runs"],
            result.detector_dataset_dir,
        )
    if result.fill_manifest is not None:
        m = result.fill_manifest
        LOG.info(
            "[3/7] Build: fill_classifier -> {} train run(s), {} val run(s) -> {}",
            m["n_train_runs"],
            m["n_val_runs"],
            result.fill_dataset_dir,
        )
    return result


def stage_train(pipeline: Pipeline, args: argparse.Namespace):
    """Trains the requested detector and/or fill-classifier stages.

    Training configuration is forwarded from the command line to the
    pipeline, including model size, epochs, batch size, image size, device,
    random seed, selected detector stages, and resume behavior.

    Args:
        pipeline: Pipeline configured with the current workspace.
        args: Parsed command-line arguments containing training configuration.

    Returns:
        Training results produced by :meth:`Pipeline.train`, including the
        best checkpoint path for each successfully trained stage.
    """
    stages = args.detector_stages or STAGE_CHOICES
    if "detectors" in args.targets:
        LOG.info(
            "[4/7] Train: detector stage(s) {} (yolo26{}, imgsz {})",
            stages,
            args.size,
            args.imgsz,
        )
    if "fill_classifier" in args.targets:
        LOG.info("[4/7] Train: fill_classifier (yolo26{}-cls)", args.size)

    result = pipeline.train(
        targets=args.targets,
        detector_stages=args.detector_stages,
        size=args.size,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
    )
    for stage, weights_path in result.weights.items():
        LOG.info("[4/7] Train: {} best checkpoint -> {}", stage, weights_path)
    return result


def _deploy_stage(
    assets_map: dict[str, Any], stage: str, weights_path: Path, *roots: Path
) -> list[Path]:
    """Copies a trained stage checkpoint into one or more deployment roots.

    The destination relative path is resolved from the shared asset map so
    the release layout remains synchronized with the production asset
    configuration.

    Args:
        assets_map: Asset-path configuration describing deployment locations.
        stage: Logical model stage whose checkpoint is being deployed.
        weights_path: Path to the trained checkpoint to copy.
        *roots: Deployment roots beneath which the stage-specific asset path
            is created.

    Returns:
        Paths of the checkpoint files written under each deployment root.
    """
    subpath = deploy_subpath(assets_map, stage)
    written = []
    for root in roots:
        dest = root / subpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights_path, dest)
        written.append(dest)
    return written


def stage_cleanup(args: argparse.Namespace, deployed_run_dirs: list[Path]) -> None:
    """Removes training run directories for stages deployed in this release.

    Cleanup is intentionally scoped to the run directories associated with
    checkpoints successfully deployed during the current invocation. It
    never removes unrelated training runs.

    Args:
        args: Parsed command-line arguments containing the `--keep-runs`
            setting.
        deployed_run_dirs: Training run directories corresponding to
            checkpoints copied during the release stage.
    """
    if args.keep_runs:
        LOG.info("[6/7] Cleanup: skipped (--keep-runs).")
        return
    if not deployed_run_dirs:
        LOG.info("[6/7] Cleanup: skipped (nothing was deployed this run).")
        return
    for run_dir in deployed_run_dirs:
        if run_dir.exists():
            LOG.info(
                "[6/7] Cleanup: purging {} (its best checkpoint is already in --output)",
                run_dir,
            )
            shutil.rmtree(run_dir)


def _load_standalone(alias: str, path: Path) -> types.ModuleType:
    """Loads a Python module from a filesystem path under a supplied alias.

    Args:
        alias: Fully qualified module name to register in `sys.modules`.
        path: Filesystem path to the module source.

    Returns:
        The loaded module object.

    Raises:
        ImportError: If a module specification or loader cannot be created
            for `path`.
    """
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def load_onyx_controller(deployment_dir: Path, model_assets: dict[str, Any]) -> Any:
    """Loads the deployment Onyx controller and its sibling modules.

    Deployment modules are loaded in their required dependency order under
    the dotted module namespace expected by their imports. Existing aliases
    are evicted first so repeated evaluations cannot accidentally reuse
    modules loaded from a different deployment directory.

    Args:
        deployment_dir: Directory containing the deployment `onyx.py` and
            its sibling modules.
        model_assets: Deployment asset mapping passed to the
            `QModelOnyx` constructor.

    Returns:
        An initialized deployment `QModelOnyx` controller.
    """
    ns = "QATCH.QModel.models.qmodel_onyx"
    aliases = [
        f"{ns}.onyx_dataprocessor",
        f"{ns}.onyx_spacing_prior",
        f"{ns}.onyx_decode",
        f"{ns}.onyx_render",
        f"{ns}.onyx_fill_render",
        "onyx_deployment_eval",
    ]
    # _load_standalone caches by dotted alias only, not by source path -- evict any
    # entries a prior in-process call left behind (e.g. a different deployment_dir)
    # so this call always reloads fresh from ITS deployment_dir rather than silently
    # reusing another call's already-executed modules.
    for alias in aliases:
        sys.modules.pop(alias, None)

    # onyx_render has no sibling deps; onyx_fill_render imports onyx_render.py
    # and onyx_dataprocessor.py, so both must be loaded first; onyx.py
    # imports all four (dataprocessor required, the rest optional/try-except).
    _load_standalone(f"{ns}.onyx_dataprocessor", deployment_dir / "onyx_dataprocessor.py")
    _load_standalone(f"{ns}.onyx_spacing_prior", deployment_dir / "onyx_spacing_prior.py")
    _load_standalone(f"{ns}.onyx_decode", deployment_dir / "onyx_decode.py")
    _load_standalone(f"{ns}.onyx_render", deployment_dir / "onyx_render.py")
    _load_standalone(f"{ns}.onyx_fill_render", deployment_dir / "onyx_fill_render.py")
    onyx_mod = _load_standalone("onyx_deployment_eval", deployment_dir / "onyx.py")
    return onyx_mod.QModelOnyx(model_assets)


def _verify_chain_to_prod(controller: Any) -> None:
    """Validates the evaluation POI mapping against the deployment controller.

    Compares the locally maintained chain-space-to-production mapping with
    the controller's `POI_MAP` and `DECODE_ID_TO_NAME` definitions to
    detect numbering drift before any evaluation results are generated.

    Args:
        controller: Loaded deployment `QModelOnyx` controller.

    Raises:
        SystemExit: If the controller exposes POI mappings that disagree with
            `CHAIN_TO_PROD`.
    """
    cls = type(controller)
    poi_map = getattr(cls, "POI_MAP", None)
    decode_map = getattr(cls, "DECODE_ID_TO_NAME", None)
    if poi_map is None or decode_map is None:
        LOG.warning(
            "Deployment QModelOnyx has no POI_MAP/DECODE_ID_TO_NAME to check "
            "CHAIN_TO_PROD against -- proceeding unverified."
        )
        return
    expected = {name: poi_map[pid] for pid, name in decode_map.items()}
    if expected != CHAIN_TO_PROD:
        raise SystemExit(
            f"CHAIN_TO_PROD in build_and_release_qmodel_onyx.py ({CHAIN_TO_PROD}) no longer "
            f"matches the deployment controller's POI_MAP/DECODE_ID_TO_NAME ({expected}). "
            "Update CHAIN_TO_PROD in this file to match before trusting these results."
        )


def _validate_and_prune_assets(model_assets: dict[str, Any]) -> dict[str, Any]:
    """Validates deployment asset paths and disables missing optional assets.

    Missing fill-classifier weights are replaced with `None` so evaluation
    degrades without repeatedly triggering loader errors. Missing detector
    weights and spacing-prior files are reported while retaining their
    configured paths because the deployment controller already handles those
    cases gracefully.

    Args:
        model_assets: Deployment asset mapping to validate and potentially
            modify.

    Returns:
        The validated asset mapping.
    """
    fc = model_assets.get("fill_classifier")
    if fc and not Path(fc).exists():
        LOG.warning(
            "fill_classifier weights not found at {} -- disabling it for this eval "
            "(predictions will assume a full 3-channel fill instead of crashing on "
            "every run). Deploy classifiers/fill_classifier/type_cls.pt to fix this.",
            fc,
        )
        model_assets["fill_classifier"] = None

    missing_detectors = [
        name for name, p in model_assets.get("detectors", {}).items() if p and not Path(p).exists()
    ]
    if missing_detectors:
        LOG.warning(
            "detector weights not found for stage(s): {} -- those POIs will score as "
            "misses for every run (QModelOnyx already degrades gracefully here; no crash).",
            missing_detectors,
        )

    prior = model_assets.get("spacing_prior")
    if prior and not Path(prior).exists():
        LOG.warning(
            "spacing_prior not found at {} -- configuration decode will no-op "
            "(QModelOnyx already degrades gracefully here; no crash).",
            prior,
        )
    return model_assets


def _predicted_positions(
    output: dict[str, Any], time_axis: np.ndarray
) -> dict[str, dict[str, float]]:
    """Extracts valid predicted POI positions from controller output.

    Converts production-space POI identifiers into chain-space names and
    records the first valid predicted index and corresponding time.

    Args:
        output: POI prediction mapping returned by the deployment controller.
        time_axis: Raw signal time axis used to convert predicted indices to
            timestamps.

    Returns:
        Mapping from chain-space POI names to dictionaries containing
        `index` and `t` for each POI the controller successfully placed.
    """
    out: dict[str, dict[str, float]] = {}
    n = len(time_axis)
    for chain, prod in CHAIN_TO_PROD.items():
        rec = output.get(prod, {})
        idxs = rec.get("indices", [-1])
        if not idxs or idxs[0] is None:
            continue
        i = int(idxs[0])
        if 0 <= i < n:
            out[chain] = {"index": i, "t": float(time_axis[i])}
    return out


def load_eval_tier_scheme(tiers_path: Path) -> TierScheme:
    """Loads the viscosity-tier scheme used for deployment evaluation.

    Prefers the fitted `tiers.json` produced during pipeline preparation
    so evaluation stratification matches the current training/validation
    scheme. Falls back to the legacy fixed corpus edges when no fitted tier
    configuration exists.

    Args:
        tiers_path: Path to the persisted fitted tier scheme.

    Returns:
        The loaded or fallback :class:`TierScheme`.
    """
    if tiers_path.exists():
        scheme = TierScheme.load(tiers_path)
        LOG.info(
            "[7/7] Eval: loaded {} viscosity tier(s) from {} (method={})",
            len(scheme.labels),
            tiers_path,
            scheme.method,
        )
        return scheme
    LOG.warning(
        "[7/7] Eval: no fitted tiers.json at {} -- falling back to legacy fixed edges {}",
        tiers_path,
        TIER_EDGES,
    )
    return TierScheme(edges_cp=list(TIER_EDGES))


def _attach_tier_labels(df: pd.DataFrame, tiers: TierScheme) -> pd.DataFrame:
    """Recomputes viscosity-tier labels from stored viscosity measurements.

    This allows summaries and plots to be regenerated using a newly fitted
    tier scheme without rerunning model inference.

    Args:
        df: Evaluation results containing a `viscosity_cP` column.
        tiers: Tier scheme used to classify viscosity values.

    Returns:
        A copy of `df` with its `tier` column replaced by labels derived
        from `tiers`.
    """
    df = df.copy()
    df["tier"] = df["viscosity_cP"].apply(lambda v: tiers.labels[tiers.tier_of(v)])
    return df


def score_run(
    controller: Any,
    run: RunRecord,
    time_col: str,
    decode_config: bool,
    refine_pois: bool,
    tiers: TierScheme,
) -> list[dict[str, Any]]:
    """Scores one deployed run against its POI ground truth.

    Loads the raw run, executes the deployment controller, resolves predicted
    and true POI positions using production index logic, and creates one
    result row for each ground-truth POI present in the run.

    Args:
        controller: Loaded deployment `QModelOnyx` controller.
        run: Corpus record containing the raw CSV, POI times, run identifier,
            and viscosity measurement.
        time_col: Preferred raw-data column containing the time axis.
        decode_config: Whether the controller should perform configuration
            decoding.
        refine_pois: Whether the controller should refine POIs with zoom
            detectors.
        tiers: Viscosity tier scheme used to label the result.

    Returns:
        list of result dictionaries, one for each ground-truth POI reached by
        the run.
    """
    df_raw = pd.read_csv(run.csv_path)
    tcol = time_col if time_col in df_raw.columns else df_raw.columns[0]
    time_axis = pd.to_numeric(df_raw[tcol], errors="coerce").to_numpy(dtype=float)

    output, _num_channels = controller.predict(
        df=df_raw, decode_config=decode_config, refine_pois=refine_pois
    )
    predicted = _predicted_positions(output, time_axis)
    tier = tiers.tier_of(run.viscosity_cP)

    rows: list[dict[str, Any]] = []
    for poi in POI_KEYS:
        true_t = run.poi_times.get(poi)
        if true_t is None:
            continue  # POI not reached in this (possibly partial) run, not scored

        # Reuse production's own index resolution so
        # true_index and pred_index are always computed the same.
        true_index = controller._get_raw_index(df_raw, true_t)
        pred = predicted.get(poi)
        hit = pred is not None
        rows.append(
            dict(
                run_id=run.run_id,
                poi=poi,
                tier=tiers.labels[tier],
                viscosity_cP=run.viscosity_cP,
                true_index=true_index,
                pred_index=pred["index"] if hit else "",
                index_err=(pred["index"] - true_index) if hit else "",
                true_t_s=true_t,
                pred_t_s=pred["t"] if hit else "",
                hit=int(hit),
                time_err_s=(pred["t"] - true_t) if hit else "",
                abs_time_err_s=abs(pred["t"] - true_t) if hit else "",
            )
        )
    return rows


def _check_run_config(output_dir: Path, run_config: dict[str, Any] | None, resume: bool) -> None:
    """Validates evaluation configuration before resuming prior results.

    Prevents results generated with one evaluation configuration from being
    silently mixed with results generated using different deployment,
    decoding, refinement, asset, or tier-scheme settings.

    Args:
        output_dir: Directory containing persisted evaluation state.
        run_config: Configuration describing the current evaluation, or
            `None` to disable configuration checking.
        resume: Whether the caller intends to resume existing evaluation
            progress.

    Raises:
        SystemExit: If persisted configuration differs from the supplied
            configuration while resuming.
    """
    if run_config is None:
        return
    config_path = output_dir / "run_config.json"
    if resume and config_path.exists():
        prior = json.loads(config_path.read_text())
        if prior != run_config:
            raise SystemExit(
                f"{output_dir} already holds results for a different configuration.\n"
                f"  prior: {prior}\n  now:   {run_config}\n"
                "Re-running with a changed --eval-deployment-dir / --output / "
                "--eval-no-decode-config / --eval-no-refine-pois against the same "
                "--eval-output would silently mix results from two configurations. Use a "
                "different --eval-output, or pass --eval-restart to start this "
                "configuration over."
            )
    config_path.write_text(json.dumps(run_config, indent=2))


def run_eval(
    controller: Any,
    runs: list[RunRecord],
    output_dir: Path,
    tiers: TierScheme,
    *,
    time_col: str = "Relative_time",
    decode_config: bool = True,
    refine_pois: bool = True,
    resume: bool = True,
    log_every: int = 25,
    run_config: dict[str, Any] | None = None,
) -> tuple[Path, int]:
    """Runs deployment evaluation with resumable per-run progress tracking.

    Each pending run is scored independently. Successful rows are flushed to
    the long-form CSV before the run is marked complete in the progress
    file, while failed runs are recorded as completed failures so they are
    not retried automatically on subsequent resumes.

    Args:
        controller: Loaded deployment `QModelOnyx` controller.
        runs: Runs to evaluate.
        output_dir: Directory for evaluation results and progress state.
        tiers: Viscosity tier scheme used during scoring.
        time_col: Preferred raw-data time column.
        decode_config: Whether to enable configuration decoding.
        refine_pois: Whether to enable POI refinement.
        resume: Whether to continue from existing progress and results.
        log_every: Number of successfully scored runs between progress logs.
        run_config: Optional configuration fingerprint used to validate
            resumability.

    Returns:
        A tuple containing the path to `results_long.csv` and the number of
        runs that failed during scoring.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_run_config(output_dir, run_config, resume)
    results_path = output_dir / "results_long.csv"
    progress_path = output_dir / "progress.txt"

    done: set = set()
    if resume and progress_path.exists():
        done = {ln.strip() for ln in progress_path.read_text().splitlines() if ln.strip()}
        LOG.info("Resuming: {} run(s) already completed, skipping them", len(done))

    write_header = not (resume and results_path.exists())
    results_f = open(results_path, "a" if resume else "w", newline="", encoding="utf-8")
    progress_f = open(progress_path, "a" if resume else "w", encoding="utf-8")
    results_writer = csv.writer(results_f)
    if write_header:
        results_writer.writerow(RESULTS_COLUMNS)

    pending = [r for r in runs if r.run_id not in done]
    LOG.info("Eval: {} run(s) pending ({} already done)", len(pending), len(done))

    t_start = time.time()
    n_done = 0
    n_failed = 0
    try:
        for run in pending:
            try:
                rows = score_run(controller, run, time_col, decode_config, refine_pois, tiers)
            except Exception as exc:
                n_failed += 1
                LOG.warning("predict failed for run {} ({}); skipping", run.run_id, exc)
                # Marked done not retried on the next --resume rather than left
                # pending forever.
                # A run that fails once (corrupt/moved CSV) is
                # assumed to fail every time; --eval-restart reprocesses everything.
                progress_f.write(run.run_id + "\n")
                progress_f.flush()
                continue

            # Results are flushed before progress.txt is updated in case of a crash in
            # between leaves this run's rows on disk but its id not marked done.
            # The next --resume will rescore it and append a second copy of its
            # rows.
            for r in rows:
                results_writer.writerow([r[c] for c in RESULTS_COLUMNS])
            results_f.flush()
            progress_f.write(run.run_id + "\n")
            progress_f.flush()

            n_done += 1
            if n_done % log_every == 0 or n_done == len(pending):
                elapsed = time.time() - t_start
                rate = elapsed / n_done
                remaining = (len(pending) - n_done) * rate
                LOG.info(
                    "Progress: {}/{} this-session ({}/{} total)  {:.2f}s/run avg  ETA {:.1f} min",
                    n_done,
                    len(pending),
                    len(done) + n_done,
                    len(runs),
                    rate,
                    remaining / 60.0,
                )
    finally:
        results_f.close()
        progress_f.close()

    if n_failed:
        LOG.warning(
            "{} run(s) failed to score and were excluded from every result below "
            "(see warnings above for each one; use --eval-restart to retry them).",
            n_failed,
        )
    return results_path, n_failed


@dataclass
class _POIMetrics:
    """Aggregated accuracy metrics for one POI subset.

    Attributes:
        n_truth: Number of ground-truth POI instances.
        n_hit: Number of instances for which the deployment produced a
            prediction.
        mae_s: Mean absolute timing error in seconds.
        median_ae_s: Median absolute timing error in seconds.
        rmse_s: Root-mean-square timing error in seconds.
        bias_s: Mean signed timing error in seconds.
        gross_rate: Fraction of hits whose absolute timing error exceeds the
            configured gross-error threshold.
        mean_index_err: Mean signed prediction-index error for successful
            predictions.
    """

    n_truth: int = 0
    n_hit: int = 0
    mae_s: float = float("nan")
    median_ae_s: float = float("nan")
    rmse_s: float = float("nan")
    bias_s: float = float("nan")
    gross_rate: float = float("nan")
    mean_index_err: float = float("nan")

    @property
    def hit_rate(self) -> float:
        """Returns the fraction of ground-truth POIs that were predicted.

        Returns:
            Hit rate as a fraction in the range `[0, 1]`, or `NaN` when
            no ground-truth POIs are present.
        """
        return self.n_hit / self.n_truth if self.n_truth else float("nan")


def _summarize(sub: pd.DataFrame, gross_threshold: float) -> _POIMetrics:
    """Computes accuracy metrics for a subset of POI evaluation rows.

    Metrics are calculated from successful predictions for timing and index
    error, while truth and hit counts include all rows in the subset.

    Args:
        sub: Evaluation rows for one POI and optional viscosity tier.
        gross_threshold: Absolute timing-error threshold in seconds used to
            classify gross errors.

    Returns:
        Aggregated :class:`_POIMetrics` for the supplied subset.
    """
    m = _POIMetrics(n_truth=len(sub), n_hit=int(sub["hit"].sum()))
    hit_sub = sub[sub["hit"] == 1]
    if len(hit_sub):
        err = hit_sub["time_err_s"].to_numpy(dtype=float)
        ae = np.abs(err)
        m.mae_s = float(np.mean(ae))
        m.median_ae_s = float(np.median(ae))
        m.rmse_s = float(np.sqrt(np.mean(err**2)))
        m.bias_s = float(np.mean(err))
        m.gross_rate = float(np.mean(ae > gross_threshold))
        m.mean_index_err = float(np.mean(hit_sub["index_err"].to_numpy(dtype=float)))
    return m


def summarize(
    results_path: Path, output_dir: Path, gross_threshold: float, tiers: TierScheme
) -> pd.DataFrame:
    """Aggregates long-form evaluation results and writes summary metrics.

    Duplicate `(run_id, poi)` rows are removed before aggregation to make
    the summary robust to a crash between result and progress-file writes.
    Metrics are produced both across all tiers and separately for each fitted
    viscosity tier.

    Args:
        results_path: Path to the long-form evaluation CSV.
        output_dir: Directory in which `summary.csv` is written.
        gross_threshold: Absolute timing-error threshold in seconds.
        tiers: Tier scheme used to regenerate viscosity labels.

    Returns:
        DataFrame containing overall and per-tier metrics for each POI.
    """
    df = pd.read_csv(results_path)
    # Defends against the duplicate rows.
    df = df.drop_duplicates(subset=["run_id", "poi"], keep="last")
    df["hit"] = df["hit"].astype(int)
    df = _attach_tier_labels(df, tiers)

    overall_rows = []
    for poi in POI_KEYS:
        sub = df[df["poi"] == poi]
        m = _summarize(sub, gross_threshold)
        overall_rows.append(dict(poi=poi, tier="ALL", **vars(m), hit_rate=m.hit_rate))
    for poi in POI_KEYS:
        for tier_label in tiers.labels:
            sub = df[(df["poi"] == poi) & (df["tier"] == tier_label)]
            if len(sub) == 0:
                continue
            m = _summarize(sub, gross_threshold)
            overall_rows.append(dict(poi=poi, tier=tier_label, **vars(m), hit_rate=m.hit_rate))

    summary = pd.DataFrame(overall_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def _print_summary(
    summary: pd.DataFrame,
    gross_threshold: float,
    output_dir: Path,
    tiers: TierScheme,
    n_failed: int = 0,
) -> None:
    """Prints a human-readable deployment evaluation summary.

    Args:
        summary: Aggregated evaluation metrics returned by :func:`summarize`.
        gross_threshold: Timing-error threshold used for the gross-error
            metric.
        output_dir: Evaluation output directory displayed in the report.
        tiers: Tier scheme defining the viscosity-tier ordering.
        n_failed: Number of runs excluded because scoring failed.
    """
    HDR = (
        f"{'POI':<6} {'Tier':<16} {'N':>5} {'Hit%':>7}  "
        f"{'MAE_s':>8} {'Med_s':>8} {'RMSE_s':>8} {'Bias_s':>8}  {'Fail%':>7}  {'MeanIdxErr':>10}"
    )
    SEP = "-" * len(HDR)
    BAR = "=" * (len(HDR) + 4)
    print(f"\n{BAR}")
    print("  Onyx DEPLOYMENT eval -- predicted position vs POI.csv ground truth")
    print(f"  gross > {gross_threshold}s  |  output -> {output_dir}")
    if n_failed:
        print(f"  ** {n_failed} run(s) failed to score and were excluded -- see warnings above **")
    print(f"  {SEP}")
    print(f"  {HDR}")
    print(f"  {SEP}")
    for poi in POI_KEYS:
        sub = summary[summary["poi"] == poi]
        overall = sub[sub["tier"] == "ALL"]
        for _, row in overall.iterrows():
            print(
                "  "
                + f"{row['poi']:<6} {'ALL':<16} {row['n_truth']:>5.0f} {row['hit_rate']:>7.1%}  "
                f"{row['mae_s']:>8.3f} {row['median_ae_s']:>8.3f} {row['rmse_s']:>8.3f} "
                f"{row['bias_s']:>+8.3f}  {row['gross_rate']:>6.1%}  {row['mean_index_err']:>10.1f}"
            )
        for tier_label in tiers.labels:
            trow = sub[sub["tier"] == tier_label]
            if trow.empty:
                continue
            row = trow.iloc[0]
            print(
                "  " + f"{'':<6} {tier_label:<16} {row['n_truth']:>5.0f} {row['hit_rate']:>7.1%}  "
                f"{row['mae_s']:>8.3f} {row['median_ae_s']:>8.3f} {row['rmse_s']:>8.3f} "
                f"{row['bias_s']:>+8.3f}  {row['gross_rate']:>6.1%}  {row['mean_index_err']:>10.1f}"
            )
        print(f"  {SEP}")
    print(f"{BAR}\n")


def _tier_colors(tier_labels: list[str]) -> dict[str, str]:
    """Generates display colors for an ordered set of viscosity tiers.

    Viscosity tiers receive progressively darker blue shades according to
    their order, while the final `unknown` tier, when present, receives a
    neutral color.

    Args:
        tier_labels: Ordered viscosity-tier labels.

    Returns:
        Mapping from tier label to hexadecimal display color.
    """
    import matplotlib.cm as cm
    from matplotlib.colors import rgb2hex

    visc_labels = tier_labels[:-1]  # everything but "unknown"
    n = max(len(visc_labels), 1)
    cmap = cm.get_cmap("Blues")
    colors = {
        label: rgb2hex(cmap(0.35 + 0.55 * (i / max(n - 1, 1))))
        for i, label in enumerate(visc_labels)
    }
    if tier_labels:
        colors[tier_labels[-1]] = MUTED
    return colors


def _setup_style() -> None:
    """Configures the Matplotlib style used by evaluation plots."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "text.color": PRIMARY,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": SECONDARY,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "axes.titlecolor": PRIMARY,
            "font.size": 11,
        }
    )


def _strip_axes(ax) -> None:
    """Applies the shared minimal axis and grid styling to a Matplotlib axis.

    Args:
        ax: Matplotlib axis to style.
    """
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="both", length=0)


def plot_bar_overall(summary: pd.DataFrame, plots_dir: Path) -> None:
    """Creates the overall per-POI mean-absolute-error bar chart.

    Args:
        summary: Aggregated evaluation summary containing `ALL` tier rows.
        plots_dir: Directory in which `bar_overall.png` is written.
    """
    _setup_style()
    import matplotlib.pyplot as plt

    overall = summary[summary["tier"] == "ALL"].set_index("poi").reindex(POI_KEYS)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x = np.arange(len(POI_KEYS))
    ax.bar(x, overall["mae_s"], width=0.56, color="#2a78d6", zorder=3)
    for xi, v, n in zip(x, overall["mae_s"], overall["n_truth"], strict=True):
        if np.isnan(v):
            continue
        ax.text(
            xi, v, f"{v:.3f}s\n(n={int(n)})", ha="center", va="bottom", fontsize=8.5, color=PRIMARY
        )
    ax.set_xticks(x)
    ax.set_xticklabels(POI_KEYS)
    ax.set_ylabel("MAE (seconds)", color=SECONDARY)
    ax.set_title(
        "Onyx deployment -- overall position accuracy",
        color=PRIMARY,
        fontsize=13,
        fontweight="bold",
    )
    _strip_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "bar_overall.png", bbox_inches="tight")
    plt.close(fig)


def plot_bar_by_tier(
    summary: pd.DataFrame,
    plots_dir: Path,
    tier_labels: list[str],
    tier_colors: dict[str, str],
) -> None:
    """Creates a grouped MAE bar chart broken out by viscosity tier.

    Args:
        summary: Aggregated evaluation summary.
        plots_dir: Directory in which `bar_by_tier.png` is written.
        tier_labels: Ordered tier labels to display.
        tier_colors: Display color mapping for each tier.
    """
    _setup_style()
    import matplotlib.pyplot as plt

    tiers_present = [t for t in tier_labels if not summary[summary["tier"] == t].empty]
    if not tiers_present:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    n_poi, n_tier = len(POI_KEYS), len(tiers_present)
    group_w = 0.72
    bar_w = group_w / max(n_tier, 1)
    x = np.arange(n_poi)
    for j, tier_label in enumerate(tiers_present):
        color = tier_colors[tier_label]
        vals = []
        for poi in POI_KEYS:
            row = summary[(summary["poi"] == poi) & (summary["tier"] == tier_label)]
            vals.append(float(row["mae_s"].iloc[0]) if len(row) else np.nan)
        offset = (j - (n_tier - 1) / 2) * bar_w
        ax.bar(x + offset, vals, width=bar_w * 0.92, color=color, label=tier_label, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(POI_KEYS)
    ax.set_ylabel("MAE (seconds)", color=SECONDARY)
    ax.set_title(
        "Onyx deployment -- position accuracy by viscosity tier",
        color=PRIMARY,
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, labelcolor=SECONDARY)
    _strip_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "bar_by_tier.png", bbox_inches="tight")
    plt.close(fig)


def plot_violin_per_poi(
    results: pd.DataFrame,
    plots_dir: Path,
    tier_labels: list[str],
    tier_colors: dict[str, str],
) -> None:
    """Creates per-POI violin plots of absolute timing error by tier.

    Only successfully predicted POIs are included in the distributions.

    Args:
        results: Long-form evaluation results.
        plots_dir: Directory in which one violin plot is written per POI.
        tier_labels: Ordered viscosity-tier labels.
        tier_colors: Display color mapping for each tier.
    """
    _setup_style()
    import matplotlib.pyplot as plt

    hit = results[results["hit"] == 1]
    for poi in POI_KEYS:
        sub = hit[hit["poi"] == poi]
        tiers_present = [t for t in tier_labels if len(sub[sub["tier"] == t])]
        if not tiers_present:
            continue
        data = [
            sub[sub["tier"] == t]["abs_time_err_s"].to_numpy(dtype=float) for t in tiers_present
        ]
        colors = [tier_colors[t] for t in tiers_present]

        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
        positions = list(range(len(tiers_present)))
        parts = ax.violinplot(
            data, positions=positions, showmedians=False, showextrema=False, widths=0.72
        )
        for pc, color in zip(parts["bodies"], colors, strict=True):  # type: ignore
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.45)
        for i, d in zip(positions, data, strict=True):
            if not len(d):
                continue
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            ax.vlines(i, q1, q3, color=SECONDARY, linewidth=3, zorder=3, alpha=0.6)
            ax.scatter(
                [i], [med], color="white", edgecolor=SECONDARY, s=22, zorder=4, linewidth=1.2
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(tiers_present, rotation=15, ha="right")
        ax.set_ylabel("Absolute time error (seconds)", color=SECONDARY)
        ax.set_title(
            f"{poi} -- error distribution by tier", color=PRIMARY, fontsize=13, fontweight="bold"
        )
        _strip_axes(ax)
        fig.tight_layout()
        fig.savefig(plots_dir / f"violin_{poi}.png", bbox_inches="tight")
        plt.close(fig)


def make_all_plots(
    results_path: Path, summary: pd.DataFrame, output_dir: Path, tiers: TierScheme
) -> None:
    """Generates the complete deployment evaluation plot set.

    Produces the overall MAE chart, tier-stratified MAE chart, and one
    timing-error distribution plot for each POI.

    Args:
        results_path: Path to the long-form evaluation results.
        summary: Aggregated evaluation summary.
        output_dir: Root directory for evaluation artifacts.
        tiers: Tier scheme used to label and order viscosity tiers.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = _attach_tier_labels(pd.read_csv(results_path), tiers)
    tier_colors = _tier_colors(tiers.labels)
    plot_bar_overall(summary, plots_dir)
    plot_bar_by_tier(summary, plots_dir, tiers.labels, tier_colors)
    plot_violin_per_poi(results, plots_dir, tiers.labels, tier_colors)
    LOG.info("Plots -> {}", plots_dir)


def stage_eval(args: argparse.Namespace, workspace: Workspace) -> Any | None:
    """Runs the post-release deployment evaluation stage.

    Selects the evaluation corpus, loads the released model assets and
    deployment controller, verifies POI numbering compatibility, runs
    resumable inference, summarizes the results, and generates diagnostic
    plots.

    By default, when available, the detector dataset manifest's validation
    split is used as the evaluation corpus.

    Args:
        args: Parsed command-line arguments containing evaluation options.
        workspace: Pipeline workspace providing data, dataset, and tier paths.

    Returns:
        The summary DataFrame when evaluation completes, or `None` when
        evaluation is skipped or no runs remain after filtering.
    """
    if args.skip_eval:
        LOG.info("[7/7] Eval: skipped (--skip-eval).")
        return None

    tiers = load_eval_tier_scheme(workspace.tiers_path)

    only_runs = args.eval_only_runs
    if only_runs is None:
        candidate = workspace.detector_dataset_dir / "manifest.json"
        if candidate.exists():
            only_runs = candidate
            LOG.info(
                "[7/7] Eval: no --eval-only-runs given; defaulting to the just-built "
                "detector val split -> {}",
                candidate,
            )

    LOG.info("[7/7] Eval: discovering corpus under {}", workspace.data_root)
    runs = dedupe_runs(discover_runs(workspace.data_root))
    if not runs:
        LOG.warning("[7/7] Eval: no runs found under {}; skipping eval.", workspace.data_root)
        return None
    if only_runs is not None:
        keep = load_run_filter(only_runs)
        runs = [r for r in runs if r.run_id in keep]
        LOG.info("[7/7] Eval: run filter -> {} run(s) retained from {}", len(runs), only_runs)
    if args.eval_n_runs is not None and len(runs) > args.eval_n_runs:
        rng = np.random.default_rng(args.eval_seed)
        runs = list(runs)
        rng.shuffle(runs)  # type: ignore
        runs = runs[: args.eval_n_runs]
    if not runs:
        LOG.warning("[7/7] Eval: no runs left to evaluate after filtering.")
        return None
    LOG.info("[7/7] Eval: evaluating over {} run(s)", len(runs))

    assets_map = load_assets_map(paths.ASSETS_PATHS_JSON)
    model_assets = _validate_and_prune_assets(build_model_assets(assets_map, args.output))
    LOG.info("[7/7] Eval: loading deployment Onyx controller from {}", args.eval_deployment_dir)
    controller = load_onyx_controller(args.eval_deployment_dir, model_assets)
    _verify_chain_to_prod(controller)

    decode_config = not args.eval_no_decode_config
    refine_pois = not args.eval_no_refine_pois
    run_config = dict(
        decode_config=decode_config,
        refine_pois=refine_pois,
        assets_root=str(args.output),
        deployment_dir=str(args.eval_deployment_dir),
        tier_edges_cp=tiers.edges_cp,
    )
    results_path, n_failed = run_eval(
        controller,
        runs,
        args.eval_output,
        tiers,
        decode_config=decode_config,
        refine_pois=refine_pois,
        resume=not args.eval_restart,
        log_every=args.eval_log_every,
        run_config=run_config,
    )

    LOG.info("[7/7] Eval: summarizing...")
    summary = summarize(results_path, args.eval_output, args.eval_gross_threshold, tiers)
    _print_summary(summary, args.eval_gross_threshold, args.eval_output, tiers, n_failed=n_failed)

    LOG.info("[7/7] Eval: plotting...")
    make_all_plots(results_path, summary, args.eval_output, tiers)
    LOG.info("[7/7] Eval: done -> {}", args.eval_output)
    return summary


def main() -> None:
    """Executes the complete QModel Onyx release pipeline.

    Constructs the configured workspace and pipeline, executes fetch,
    preparation, dataset-build, and training stages, deploys the resulting
    checkpoints and spacing prior, optionally removes completed training
    directories, and optionally evaluates the released package.

    Pipeline errors are logged and terminate the process with a non-zero exit
    status.
    """
    args = parse_args()
    configure_logging(level=args.log_level.upper())

    ws_overrides = {
        k: v
        for k, v in dict(
            data_root=args.data_root,
            datasets_root=args.datasets_root,
            runs_root=args.runs_root,
            configs_root=args.configs_root,
        ).items()
        if v is not None
    }
    workspace = Workspace(**ws_overrides)
    pipeline = Pipeline(workspace)

    try:
        stage_fetch(args, workspace)
        prep_result = stage_prepare(pipeline)
        stage_build(pipeline, args)
        train_result = stage_train(pipeline, args)
    except PipelineError as exc:
        LOG.error("Pipeline stopped: {}", exc)
        LOG.error(
            "This usually means there isn't enough labeled data yet under {} - "
            "fetch more runs from Dropbox and try again.",
            workspace.data_root,
        )
        sys.exit(1)

    LOG.info("[5/7] Release: deploying trained weights -> {}", args.output)
    assets_map = json.loads(paths.ASSETS_PATHS_JSON.read_text())
    local_assets_root = paths.ASSETS_PATHS_JSON.parent / "assets"
    deployed_run_dirs: list[Path] = []
    for stage, weights_path in train_result.weights.items():
        if not weights_path.exists():
            LOG.warning(
                "[5/7] Release: {} checkpoint missing at {}, skipping deploy", stage, weights_path
            )
            continue
        roots = [args.output] + ([local_assets_root] if args.also_update_local_assets else [])
        _deploy_stage(assets_map, stage, weights_path, *roots)
        # weights_path is always <run_dir>/weights/best.pt (train_detectors.py,
        # train_fill_classifier.py) -- .parent.parent recovers run_dir without
        # re-deriving Ultralytics' project-naming convention here.
        deployed_run_dirs.append(weights_path.parent.parent)

    args.output.mkdir(parents=True, exist_ok=True)
    prior_dest = args.output / "spacing_prior.json"
    shutil.copy2(prep_result.prior_path, prior_dest)
    LOG.info("[5/7] Release: spacing prior -> {}", prior_dest)

    stage_cleanup(args, deployed_run_dirs)
    stage_eval(args, workspace)

    LOG.info("Done.")


if __name__ == "__main__":
    main()

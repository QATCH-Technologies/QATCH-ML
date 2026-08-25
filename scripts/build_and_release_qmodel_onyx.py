#!/usr/bin/env python3
"""
build_and_release_qmodel_onyx.py
=================================

One-command release pipeline for the qmodel_7_onyx system: pulls fresh runs
from Dropbox, (re)builds the YOLO datasets, trains the requested models,
drops the trained weights into a ready-to-ship `qmodel_onyx/` folder, and
evaluates that deployed package against ground truth. There is no longer a
separate `eval_onyx_deployment.py` script -- its logic lives here as the
Eval stage below; drive it with the `--eval-*` flags (or skip it with
`--skip-eval`).

Seven stages, run in order (any can be skipped with a flag - see --help):

  1. Fetch   - copy new runs from a local Dropbox sync folder into data/raw
               (via src/utils/dataset_fetcher.py).
  2. Prepare - fit the viscosity-tier scheme + spacing prior from data/raw.
  3. Build   - render the YOLO datasets (cascade/zoom detectors + fill
               classifier) from the tier scheme.
  4. Train   - train the requested detector stages and/or fill classifier
               (thin wrapper around Ultralytics YOLO training).
  5. Release - copy the best checkpoint of every stage that was just trained
               into:

                   qmodel_onyx/
                     classifiers/fill_classifier/type_cls.pt
                     detectors/init_detector/init.pt
                     detectors/ch1_detector/ch1.pt
                     detectors/ch2_detector/ch2.pt
                     detectors/ch3_detector/ch3.pt
                     detectors/ch1_zoom_detector/ch1_zoom.pt
                     detectors/ch2_zoom_detector/ch2_zoom.pt
                     detectors/ch3_zoom_detector/ch3_zoom.pt
                     spacing_prior.json

               (the folder layout is read straight from assets_paths.json,
               so it can never drift out of sync with what the production
               controller expects).
  6. Cleanup - purge each just-deployed stage's own run directory (its
               Ultralytics checkpoints/logs/plots) now that Release has
               copied out its best checkpoint. Scoped to exactly the stages
               deployed THIS invocation, never the whole runs/ tree --
               --detector-stages lets separate invocations train different
               stages independently, and a sibling stage's still-in-progress
               or --resume-pending run dir may legitimately be sitting right
               next to the ones this invocation just finished.
  7. Eval    - score the qmodel_onyx/ package Release just wrote against
               *_poi.csv ground truth: for each point of interest, how far
               is the deployed model's predicted position from truth,
               broken out by viscosity tier. Not a YOLO-metrics benchmark
               (see qa/benchmark.py for that) -- it measures what a
               deployed run actually produces, loaded exactly as a
               downstream consumer loads it (see the "Deployment eval
               internals" section below for the full rationale). Defaults
               to scoring only the detector dataset's held-out val split --
               the one honest eval set for a model trained on this corpus.

Examples
--------
Full pipeline, everything, defaults::

    python scripts/build_and_release_qmodel_onyx.py \\
        --dropbox-source "D:/Dropbox/QATCH runs"

Skip the Dropbox fetch (data/raw is already up to date), train only the
fill classifier::

    python scripts/build_and_release_qmodel_onyx.py \\
        --skip-fetch --targets fill_classifier

Retrain a single detector stage after adding a few hundred new runs::

    python scripts/build_and_release_qmodel_onyx.py \\
        --dropbox-source "D:/Dropbox/QATCH runs" \\
        --targets detectors --detector-stages ch2_zoom

Train without touching Dropbox or running the post-release eval::

    python scripts/build_and_release_qmodel_onyx.py --skip-fetch --skip-eval
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# scripts/ lives outside the src/ package tree; make `import src...` work
# even without an editable install (`pip install -e .`), and make the
# sibling `_qmodel_onyx_layout` helper importable regardless of whether this
# file is run directly (script dir already on sys.path[0]) or imported as
# `scripts.build_and_release_qmodel_onyx` (it isn't, in that mode).
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

# Chain-space truth name (corpus.py's POI_ORDER) -> production output name
# (QModelOnyx.POI_MAP). POI3 in chain space is the fourth production id
# because production id 3 is a legacy shim row the controller never
# populates (deleted from final_results before formatting) - mirrors
# qa/benchmark.py's identical CHAIN_TO_PROD.
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


def parse_args() -> argparse.Namespace:
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


def stage_fetch(args: argparse.Namespace, workspace: Workspace) -> Optional[int]:
    """Purges `workspace.data_root` and re-copies the full run corpus from
    Dropbox, so every fetch is a clean mirror rather than an incremental
    blend with whatever was fetched on a prior run. Returns the number of
    run directories retained, or None if the stage was skipped."""
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

    # dataset_fetcher only ever *adds* runs (it skips anything whose _poi.csv
    # already exists under target_dir) - it never removes a run that was
    # deleted/superseded upstream. Wipe data_root first so every fetch is a
    # full, clean mirror of what's currently live in Dropbox, never a stale
    # blend of this run's and some earlier run's corpus.
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
    assets_map: Dict[str, Any], stage: str, weights_path: Path, *roots: Path
) -> List[Path]:
    """Copies weights_path to <root>/<deploy_subpath> under every root given.
    Returns the list of destination paths actually written."""
    subpath = deploy_subpath(assets_map, stage)
    written = []
    for root in roots:
        dest = root / subpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights_path, dest)
        written.append(dest)
    return written


def stage_cleanup(args: argparse.Namespace, deployed_run_dirs: List[Path]) -> None:
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
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def load_onyx_controller(deployment_dir: Path, model_assets: Dict[str, Any]) -> Any:
    """Loads the deployment package's `onyx.py` + siblings (in their
    required dependency order) under the exact dotted path they import each
    other by, and constructs `QModelOnyx(model_assets)`."""
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
    """CHAIN_TO_PROD is a hand-maintained mirror of the deployment
    controller's own POI_MAP + DECODE_ID_TO_NAME (see the section docstring
    above: this eval exists to catch onyx.py drifting from the training
    repo). If that numbering ever changes without CHAIN_TO_PROD being
    updated in lockstep, _predicted_positions would silently score every run
    for the drifted POI as a miss instead of raising -- so check for drift
    explicitly, loudly, and once, right after the controller loads."""
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


def _validate_and_prune_assets(model_assets: Dict[str, Any]) -> Dict[str, Any]:
    """QModelOnyx guards every asset EXCEPT fill_classifier against a
    missing-but-non-empty path: detectors and the spacing prior catch (or
    pre-check) a bad path and degrade gracefully, but fill_classifier's
    loader does neither -- a missing file raises FileNotFoundError on
    EVERY single predict() call (never cached as a permanent failure),
    each one caught by predict()'s own outer try/except and printing a
    full traceback. Null it out here if missing, so a partially-deployed
    package degrades the same way the other assets already do instead of
    spamming one traceback per run in the corpus."""
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
    output: Dict[str, Any], time_axis: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """chain-space POI name -> {"index": int, "t": float} for every POI the
    controller actually placed (index >= 0); omits POIs it didn't."""
    out: Dict[str, Dict[str, float]] = {}
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
    """Loads the data-driven viscosity TierScheme that stage_prepare fit from
    raw data (see tiers.py) and saved to configs/tiers.json, so the eval's
    per-tier breakdown always mirrors the tiers actually used for train/val
    stratification -- not the frozen legacy edges in corpus.py. Falls back
    to those legacy fixed edges only when no fitted tiers.json exists yet
    (e.g. running --skip-fetch eval before any prepare stage has run)."""
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
    """(Re)derives the `tier` column from the stored raw `viscosity_cP`
    using the given TierScheme, rather than trusting whatever tier label
    got embedded in results_long.csv at score time. This lets summary.csv
    and the plots be regenerated with an up-to-date stratification after
    tiers.json is refit on new incoming data, without re-scoring every run."""
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
) -> List[Dict[str, Any]]:
    """One run -> one result row per chain-space POI present in truth."""
    df_raw = pd.read_csv(run.csv_path)
    tcol = time_col if time_col in df_raw.columns else df_raw.columns[0]
    time_axis = pd.to_numeric(df_raw[tcol], errors="coerce").to_numpy(dtype=float)

    output, _num_channels = controller.predict(
        df=df_raw, decode_config=decode_config, refine_pois=refine_pois
    )
    predicted = _predicted_positions(output, time_axis)
    tier = tiers.tier_of(run.viscosity_cP)

    rows: List[Dict[str, Any]] = []
    for poi in POI_KEYS:
        true_t = run.poi_times.get(poi)
        if true_t is None:
            continue  # POI not reached in this (possibly partial) run -- not scored
        # Reuse production's own index resolution (not a local reimplementation) so
        # true_index and pred_index are always computed by the identical method.
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


def _check_run_config(output_dir: Path, run_config: Optional[Dict[str, Any]], resume: bool) -> None:
    """Resumability keys on run_id alone, which says nothing about WHICH
    configuration (decode_config, refine_pois, assets_root, deployment_dir)
    produced the rows already on disk. Without this check, re-running with
    a changed flag against the same --eval-output and no --eval-restart
    would find every run_id already "done" and silently report the FIRST
    run's config, mislabeled as the new one. Recorded/compared only when the
    caller supplies run_config (stage_eval() does; direct run_eval() callers
    such as tests may omit it to skip this check entirely)."""
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
    runs: List[RunRecord],
    output_dir: Path,
    tiers: TierScheme,
    *,
    time_col: str = "Relative_time",
    decode_config: bool = True,
    refine_pois: bool = True,
    resume: bool = True,
    log_every: int = 25,
    run_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, int]:
    """Returns (results_path, n_failed) -- n_failed is the count of runs
    whose scoring raised and were excluded (see the per-run except below)."""
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
                # Marked done (not retried on the next --resume) rather than left
                # pending forever -- a run that fails once (corrupt/moved CSV) is
                # assumed to fail every time; --eval-restart reprocesses everything
                # if the underlying cause gets fixed.
                progress_f.write(run.run_id + "\n")
                progress_f.flush()
                continue

            # Results are flushed before progress.txt is updated, so a crash in
            # between leaves this run's rows on disk but its id NOT marked done --
            # the next --resume will rescore it and append a second copy of its
            # rows. summarize() defends against this with drop_duplicates rather
            # than relying on write ordering to be atomic (it isn't).
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
        return self.n_hit / self.n_truth if self.n_truth else float("nan")


def _summarize(sub: pd.DataFrame, gross_threshold: float) -> _POIMetrics:
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
    df = pd.read_csv(results_path)
    # Defends against the duplicate rows a crash between a run's results-flush and
    # its progress.txt write can leave behind (see run_eval's write-order comment)
    # -- keep the LAST occurrence per (run_id, poi) rather than double-counting.
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


GRIDLINE = "#e1e0d9"
MUTED = "#898781"
SECONDARY = "#52514e"
PRIMARY = "#0b0b0b"


def _tier_colors(tier_labels: List[str]) -> Dict[str, str]:
    """Light -> dark blue gradient across the ordered viscosity bins, with a
    fixed neutral grey for the trailing "unknown" bucket. Generated from
    however many tiers the fitted TierScheme produced (fit_tiers' n_bins can
    vary run to run as the corpus grows) rather than a hardcoded 6-tier
    palette sized for the old fixed edges."""
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
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(axis="both", length=0)


def plot_bar_overall(summary: pd.DataFrame, plots_dir: Path) -> None:
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
    summary: pd.DataFrame, plots_dir: Path, tier_labels: List[str], tier_colors: Dict[str, str]
) -> None:
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
    results: pd.DataFrame, plots_dir: Path, tier_labels: List[str], tier_colors: Dict[str, str]
) -> None:
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
        for pc, color in zip(parts["bodies"], colors, strict=True):
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
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = _attach_tier_labels(pd.read_csv(results_path), tiers)
    tier_colors = _tier_colors(tiers.labels)
    plot_bar_overall(summary, plots_dir)
    plot_bar_by_tier(summary, plots_dir, tiers.labels, tier_colors)
    plot_violin_per_poi(results, plots_dir, tiers.labels, tier_colors)
    LOG.info("Plots -> {}", plots_dir)


def stage_eval(args: argparse.Namespace, workspace: Workspace) -> Optional[Any]:
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
        rng.shuffle(runs)
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
        # Ties a resumed --eval-output to the tier scheme active when its rows
        # were scored -- if tiers.json gets refit in between (new incoming
        # data), _check_run_config forces --eval-restart instead of silently
        # mixing old- and new-scheme tier assignments in one summary.
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
    deployed_run_dirs: List[Path] = []
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

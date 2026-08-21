#!/usr/bin/env python3
"""
build_and_release_qmodel_onyx.py
=================================

One-command release pipeline for the qmodel_7_onyx system: pulls fresh runs
from Dropbox, (re)builds the YOLO datasets, trains the requested models,
and drops the trained weights into a ready-to-ship ``qmodel_onyx/`` folder.

Five stages, run in order (any can be skipped with a flag — see --help):

  1. Fetch   - copy new runs from a local Dropbox sync folder into data/raw
               (via src/utils/dataset_fetcher.py).
  2. Prepare - fit the viscosity-tier scheme + spacing prior from data/raw.
  3. Build   - render the YOLO datasets (cascade/zoom detectors + fill
               classifier) from the tier scheme.
  4. Train   - train the requested detector stages and/or fill classifier
               (thin wrapper around Ultralytics YOLO training).
  5. Release - explain each model's validation metrics in plain language,
               then copy the best checkpoint of every stage that was just
               trained into:

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
                     TRAINING_REPORT.md

               (the folder layout is read straight from assets_paths.json,
               so it can never drift out of sync with what the production
               controller expects).

A human-readable ``TRAINING_REPORT.md`` is written into the release folder
and echoed to the console, translating each stage's raw Ultralytics metrics
(precision/recall/mAP, or top-1/top-5 accuracy) into plain-English verdicts
so the results are usable by someone who has never trained a model before.

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
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from _qmodel_onyx_layout import deploy_subpath  # noqa: E402

from src.systems.qmodel_7_onyx import paths  # noqa: E402
from src.systems.qmodel_7_onyx.pipeline import (  # noqa: E402
    Pipeline,
    PipelineError,
    Workspace,
)
from src.systems.qmodel_7_onyx.training.train_detectors import STAGE_CHOICES  # noqa: E402
from src.utils.dataset_fetcher import DatasetFetcher  # noqa: E402
from src.utils.logger import configure_logging, get_logger  # noqa: E402

LOG = get_logger("build_and_release_qmodel_onyx")

DEFAULT_RELEASE_DIR = paths.REPO_ROOT / "qmodel_onyx"

# Candidate Ultralytics results_dict keys, most specific first. Matched by
# exact (case-insensitive) key equality — never substring — so "mAP50" can
# never accidentally match "mAP50-95".
_DETECTOR_METRIC_KEYS = {
    "precision": ["metrics/precision(B)", "metrics/precision"],
    "recall": ["metrics/recall(B)", "metrics/recall"],
    "map50": ["metrics/mAP50(B)", "metrics/mAP50"],
    "map50_95": ["metrics/mAP50-95(B)", "metrics/mAP50-95"],
}
_CLASSIFIER_METRIC_KEYS = {
    "top1": ["metrics/accuracy_top1", "top1"],
    "top5": ["metrics/accuracy_top5", "top5"],
}


# ===========================================================================
#  CLI
# ===========================================================================


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

    rel = ap.add_argument_group("5. release")
    rel.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help=f"Where to drop deployable weights + report. Default: {DEFAULT_RELEASE_DIR}",
    )
    rel.add_argument(
        "--also-update-local-assets",
        action="store_true",
        help="Also copy the same weights into src/systems/qmodel_7_onyx/assets/, "
        "so qa/benchmark.py and local inference pick them up immediately.",
    )

    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


# ===========================================================================
#  Stage 1: fetch
# ===========================================================================


def stage_fetch(args: argparse.Namespace, workspace: Workspace) -> Optional[int]:
    """Copies new runs from Dropbox into ``workspace.data_root``. Returns the
    number of new run directories retained, or None if the stage was skipped."""
    if args.skip_fetch:
        LOG.info("[1/5] Fetch: skipped (--skip-fetch).")
        return None

    source = Path(args.dropbox_source) if args.dropbox_source else paths.DROPBOX_SOURCE
    if not source.exists():
        LOG.info(
            "[1/5] Fetch: skipped (Dropbox source not found on this machine: {}). "
            "Pass --dropbox-source or set QMODEL_DROPBOX_SOURCE if it lives elsewhere.",
            source,
        )
        return None

    LOG.info("[1/5] Fetch: copying new runs from {} -> {}", source, workspace.data_root)
    fetcher = DatasetFetcher(
        source_dir=str(source),
        target_dir=str(workspace.data_root),
        num_files=args.num_files,
    )
    n_before = len(list(workspace.data_root.glob("*"))) if workspace.data_root.exists() else 0
    report_path = paths.ARTIFACTS_ROOT / "dropbox_fetch_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fetcher.run(report_path=report_path)
    n_after = len(list(workspace.data_root.glob("*")))
    n_new = max(0, n_after - n_before)
    LOG.info("[1/5] Fetch: {} new run(s) added ({} failure(s) logged to {})",
              n_new, len(fetcher.failures), report_path)
    return n_new


# ===========================================================================
#  Stages 2-4: prepare, build, train (thin wrappers around Pipeline)
# ===========================================================================


def stage_prepare(pipeline: Pipeline):
    LOG.info("[2/5] Prepare: fitting viscosity tiers + spacing prior from raw data...")
    result = pipeline.prepare()
    LOG.info(
        "[2/5] Prepare: {} run(s) discovered, {} viscosity tier(s) fit -> {}",
        result.n_runs,
        len(result.tiers.labels),
        result.tiers_path,
    )
    return result


def stage_build(pipeline: Pipeline, args: argparse.Namespace):
    LOG.info("[3/5] Build: rendering YOLO datasets for {}...", args.targets)
    result = pipeline.build_datasets(targets=args.targets)
    if result.detector_manifest is not None:
        m = result.detector_manifest
        LOG.info(
            "[3/5] Build: detectors -> {} train run(s), {} val run(s) -> {}",
            m["n_train_runs"], m["n_val_runs"], result.detector_dataset_dir,
        )
    if result.fill_manifest is not None:
        m = result.fill_manifest
        LOG.info(
            "[3/5] Build: fill_classifier -> {} train run(s), {} val run(s) -> {}",
            m["n_train_runs"], m["n_val_runs"], result.fill_dataset_dir,
        )
    return result


def stage_train(pipeline: Pipeline, args: argparse.Namespace):
    stages = args.detector_stages or STAGE_CHOICES
    if "detectors" in args.targets:
        LOG.info(
            "[4/5] Train: detector stage(s) {} (yolo26{}, imgsz {})",
            stages, args.size, args.imgsz,
        )
    if "fill_classifier" in args.targets:
        LOG.info("[4/5] Train: fill_classifier (yolo26{}-cls)", args.size)

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
        LOG.info("[4/5] Train: {} best checkpoint -> {}", stage, weights_path)
    return result


# ===========================================================================
#  Stage 5a: plain-English metric interpretation
# ===========================================================================


def _pick(metrics: Optional[Dict[str, Any]], candidates: List[str]) -> Optional[float]:
    if not metrics:
        return None
    lower = {k.lower(): v for k, v in metrics.items()}
    for candidate in candidates:
        if candidate.lower() in lower:
            try:
                return float(lower[candidate.lower()])
            except (TypeError, ValueError):
                continue
    return None


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _verdict(headline: Optional[float]) -> str:
    if headline is None:
        return "UNKNOWN — no metrics captured; check the run folder's results.png"
    if headline >= 0.90:
        return "GOOD"
    if headline >= 0.75:
        return "USABLE — more labeled data or epochs would likely help"
    return "NEEDS WORK — review the dataset/labels before shipping this checkpoint"


def _interpret(
    stage: str, metrics: Optional[Dict[str, Any]]
) -> Tuple[List[Tuple[str, str, str]], str]:
    """Returns (rows, verdict). rows are (label, value, plain-English meaning)."""
    if stage == "fill_classifier":
        top1 = _pick(metrics, _CLASSIFIER_METRIC_KEYS["top1"])
        top5 = _pick(metrics, _CLASSIFIER_METRIC_KEYS["top5"])
        rows = [
            ("Top-1 accuracy", _pct(top1),
             "how often the model's single best guess at the fill state was correct"),
            ("Top-5 accuracy", _pct(top5),
             "how often the correct fill state was among its top 5 guesses (usually near 100%)"),
        ]
        return rows, _verdict(top1)

    precision = _pick(metrics, _DETECTOR_METRIC_KEYS["precision"])
    recall = _pick(metrics, _DETECTOR_METRIC_KEYS["recall"])
    map50 = _pick(metrics, _DETECTOR_METRIC_KEYS["map50"])
    map50_95 = _pick(metrics, _DETECTOR_METRIC_KEYS["map50_95"])
    rows = [
        ("Precision", _pct(precision),
         "of the boxes the model predicted, the fraction that were actually correct — "
         "low precision means false alarms"),
        ("Recall", _pct(recall),
         "of the true events in the validation data, the fraction the model found — "
         "low recall means missed events"),
        ("mAP@0.5", _pct(map50),
         "overall detection accuracy at a loose position match — the headline number"),
        ("mAP@0.5:0.95", _pct(map50_95),
         "accuracy demanding a tight position match — normal for this to read lower "
         "than mAP@0.5"),
    ]
    return rows, _verdict(map50)


# ===========================================================================
#  Stage 5b: deploy weights into qmodel_onyx/ (layout sourced from
#  assets_paths.json so it can never drift from what the controller expects)
# ===========================================================================


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


# ===========================================================================
#  Stage 5c: report
# ===========================================================================


def build_report(
    *,
    n_new_runs: Optional[int],
    prep_result: Any,
    build_result: Any,
    train_result: Any,
    deployed: Dict[str, List[Path]],
    output_dir: Path,
) -> str:
    lines = []
    lines.append("# QATCH Onyx — Training & Release Report")
    lines.append(f"_generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("## 1. Data")
    if n_new_runs is None:
        lines.append("- Dropbox fetch: skipped")
    else:
        lines.append(f"- Dropbox fetch: {n_new_runs} new run(s) added")
    lines.append(f"- Raw corpus: {prep_result.n_runs} run(s) under `data/raw`")
    lines.append(f"- Viscosity tiers fit: {len(prep_result.tiers.labels)}")
    lines.append("")

    lines.append("## 2. Datasets built")
    if build_result.detector_manifest is not None:
        m = build_result.detector_manifest
        lines.append(
            f"- **detectors**: {m['n_train_runs']} train run(s), {m['n_val_runs']} val run(s)"
        )
    if build_result.fill_manifest is not None:
        m = build_result.fill_manifest
        lines.append(
            f"- **fill_classifier**: {m['n_train_runs']} train run(s), {m['n_val_runs']} val run(s)"
        )
    lines.append("")

    lines.append("## 3. Training results")
    lines.append(
        "_Rule-of-thumb verdicts only (GOOD >=90%, USABLE >=75%, NEEDS WORK below) — "
        "not a guarantee, just a starting point for a novice reading these numbers._"
    )
    lines.append("")
    if not train_result.weights:
        lines.append("_No models were trained this run._")
    for stage, weights_path in train_result.weights.items():
        metrics = train_result.metrics.get(stage)
        rows, verdict = _interpret(stage, metrics)
        lines.append(f"### {stage}")
        lines.append(f"- Checkpoint: `{weights_path}`")
        lines.append(f"- Verdict: **{verdict}**")
        lines.append("")
        lines.append("| Metric | Value | What it means |")
        lines.append("|---|---|---|")
        for label, value, meaning in rows:
            lines.append(f"| {label} | {value} | {meaning} |")
        lines.append("")

    lines.append("## 4. Deployed assets")
    if not deployed:
        lines.append("_Nothing deployed — no stage finished training successfully this run._")
    else:
        for stage, paths_written in deployed.items():
            for p in paths_written:
                lines.append(f"- `{stage}` -> `{p}`")
    lines.append("")

    lines.append("## 5. What to do next")
    needs_work = [
        s
        for s in train_result.weights
        if _interpret(s, train_result.metrics.get(s))[1].startswith("NEEDS")
    ]
    if needs_work:
        lines.append(
            f"- {', '.join(needs_work)} scored low — open `runs/.../weights/../results.png` "
            "and `confusion_matrix.png` for that stage (Ultralytics writes these "
            "automatically) to see exactly what's being missed or confused. More "
            "labeled data for that stage usually helps more than more epochs."
        )
    else:
        lines.append("- All trained stages cleared the USABLE bar.")
    lines.append(
        f"- Deployed weights live under `{output_dir}`. Point the production "
        "controller's asset config at this folder, or copy it over "
        "`src/systems/qmodel_7_onyx/assets/` to test locally."
    )
    lines.append(
        "- For a deeper, corpus-wide accuracy check (not just the validation "
        "split), run `python -m src.systems.qmodel_7_onyx.qa.benchmark` once "
        "the new weights are in place."
    )
    return "\n".join(lines)


# ===========================================================================
#  Main
# ===========================================================================


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
        n_new_runs = stage_fetch(args, workspace)
        prep_result = stage_prepare(pipeline)
        build_result = stage_build(pipeline, args)
        train_result = stage_train(pipeline, args)
    except PipelineError as exc:
        LOG.error("Pipeline stopped: {}", exc)
        LOG.error(
            "This usually means there isn't enough labeled data yet under {} — "
            "fetch more runs from Dropbox and try again.",
            workspace.data_root,
        )
        sys.exit(1)

    LOG.info("[5/5] Release: deploying trained weights -> {}", args.output)
    assets_map = json.loads(paths.ASSETS_PATHS_JSON.read_text())
    local_assets_root = paths.ASSETS_PATHS_JSON.parent / "assets"
    deployed: Dict[str, List[Path]] = {}
    for stage, weights_path in train_result.weights.items():
        if not weights_path.exists():
            LOG.warning(
                "[5/5] Release: {} checkpoint missing at {}, skipping deploy", stage, weights_path
            )
            continue
        roots = [args.output] + ([local_assets_root] if args.also_update_local_assets else [])
        deployed[stage] = _deploy_stage(assets_map, stage, weights_path, *roots)

    args.output.mkdir(parents=True, exist_ok=True)
    prior_dest = args.output / "spacing_prior.json"
    shutil.copy2(prep_result.prior_path, prior_dest)
    LOG.info("[5/5] Release: spacing prior -> {}", prior_dest)

    report_md = build_report(
        n_new_runs=n_new_runs,
        prep_result=prep_result,
        build_result=build_result,
        train_result=train_result,
        deployed=deployed,
        output_dir=args.output,
    )
    report_path = args.output / "TRAINING_REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")
    print("\n" + report_md + "\n")
    LOG.info("Done. Full report -> {}", report_path)


if __name__ == "__main__":
    main()

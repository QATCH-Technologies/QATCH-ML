#!/usr/bin/env python3
"""
eval_onyx_deployment.py
========================

Viscosity-tiered position accuracy eval for the *deployed* Onyx package —
the standalone modules under ``src/systems/qmodel_7_onyx/deployment/``
(``onyx.py`` + siblings), loaded exactly as a downstream consumer loads them
(via the ``QATCH.QModel.models.qmodel_onyx.*`` dotted-import contract those
files hard-code), not the training repo's ``inference/controller.py``. The
two can drift apart — this script exists to catch that drift by testing
what actually ships.

This is deliberately NOT a YOLO detection benchmark (no precision/recall/
mAP — see qa/benchmark.py's decode A/B harness, and the training report
build_and_release_qmodel_onyx.py writes, for that). It measures the one
thing a deployed run cares about: for each point of interest, how far is
the model's predicted position from the position recorded in that run's
``*_poi.csv`` ground truth — in both seconds (the physically meaningful
unit; what a user experiences) and samples (POI.csv's own native unit) —
broken out by viscosity tier, since slow (high-cP) fills are the harder,
lower-margin case and a global average hides that.

Corpus discovery, truth parsing, dedup, and viscosity tiering are reused
verbatim from ``src/systems/qmodel_7_onyx/corpus.py`` (the codebase's single
source of truth for what counts as a valid ground-truth POI mark) rather
than re-derived, so this eval can never drift from what the dataset
builders and qa/benchmark.py already agree "truth" means.

Usage
-----
    python scripts/eval_onyx_deployment.py
        (evaluates qmodel_onyx/ at the repo root — the folder
        build_and_release_qmodel_onyx.py deploys into — against data/raw)

    python scripts/eval_onyx_deployment.py --assets-root src/systems/qmodel_7_onyx/assets

    python scripts/eval_onyx_deployment.py --only-runs datasets/v7/manifest.json
        (score only the held-out val runs — the only honest eval set for a
        model trained on this corpus)

Resumable: progress is checkpointed to ``<output>/progress.txt`` and results
are appended to ``<output>/results_long.csv`` as they're produced, so an
interrupted run continues where it left off (``--restart`` to start over).
Resuming is keyed on run id, not configuration -- re-running against the same
``--output`` with a changed ``--assets-root``/``--deployment-dir``/decode/
refine flag raises immediately (``<output>/run_config.json`` records what
produced the existing results) rather than silently mixing two runs' results.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# scripts/ lives outside the src/ package tree; make `import src...` work
# even without an editable install, and make the sibling
# `_qmodel_onyx_layout` helper importable regardless of invocation mode.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _qmodel_onyx_layout import build_model_assets, load_assets_map  # noqa: E402

from src.systems.qmodel_7_onyx import paths  # noqa: E402
from src.systems.qmodel_7_onyx.corpus import (  # noqa: E402
    TIER_LABELS,
    RunRecord,
    dedupe_runs,
    discover_runs,
    load_run_filter,
    viscosity_tier,
)
from src.utils.logger import configure_logging, get_logger  # noqa: E402

LOG = get_logger("eval_onyx_deployment")

DEFAULT_DEPLOYMENT_DIR = paths.REPO_ROOT / "src" / "systems" / "qmodel_7_onyx" / "deployment"
DEFAULT_ASSETS_ROOT = paths.REPO_ROOT / "qmodel_onyx"

# Chain-space truth name (corpus.py's POI_ORDER) -> production output name
# (QModelOnyx.POI_MAP). POI3 in chain space is the fourth production id
# because production id 3 is a legacy shim row the controller never
# populates (deleted from final_results before formatting) — mirrors
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


# ===========================================================================
#  Standalone deployment-module loading (no real QATCH install, no fake
#  QATCH.common needed -- onyx.py already falls back to a headless Log
#  stub when QATCH.common.logger can't be imported; empirically verified
#  the sys.modules leaf-name trick below resolves onyx.py's sibling
#  imports without any parent package ever existing).
# ===========================================================================


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
    """Loads the deployment package's ``onyx.py`` + siblings (in their
    required dependency order) under the exact dotted path they import each
    other by, and constructs ``QModelOnyx(model_assets)``."""
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
    controller's own POI_MAP + DECODE_ID_TO_NAME (see module docstring:
    this script exists to catch onyx.py drifting from the training repo).
    If that numbering ever changes without CHAIN_TO_PROD being updated in
    lockstep, _predicted_positions would silently score every run for the
    drifted POI as a miss instead of raising -- so check for drift
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
            f"CHAIN_TO_PROD in eval_onyx_deployment.py ({CHAIN_TO_PROD}) no longer matches "
            f"the deployment controller's POI_MAP/DECODE_ID_TO_NAME ({expected}). Update "
            "CHAIN_TO_PROD in this file to match before trusting these results."
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


# ===========================================================================
#  CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--raw-root", type=Path, default=paths.DATA_ROOT)
    ap.add_argument(
        "--deployment-dir",
        type=Path,
        default=DEFAULT_DEPLOYMENT_DIR,
        help="Directory holding the deployment onyx.py + siblings (the modules under "
        "test) -- NOT the training repo's inference/controller.py.",
    )
    ap.add_argument(
        "--assets-root",
        type=Path,
        default=DEFAULT_ASSETS_ROOT,
        help="Root of a qmodel_onyx/-shaped deploy folder (classifiers/, detectors/, "
        "spacing_prior.json) -- what build_and_release_qmodel_onyx.py produces. "
        f"Default: {DEFAULT_ASSETS_ROOT}",
    )
    ap.add_argument(
        "--output", type=Path, default=paths.ARTIFACTS_ROOT / "eval_onyx_deployment"
    )
    ap.add_argument(
        "--only-runs",
        type=Path,
        default=None,
        help="Restrict to run ids in this file (a build_dataset manifest.json -- its "
        "val_ids are used -- or a plain run-id list). The only honest eval set for "
        "a model trained on this corpus.",
    )
    ap.add_argument("--n-runs", type=int, default=None, help="Cap the corpus to N runs (seeded).")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--gross-threshold", type=float, default=2.0, help="Seconds.")
    ap.add_argument(
        "--no-decode-config", action="store_true", help="Disable the configuration decode."
    )
    ap.add_argument(
        "--no-refine-pois", action="store_true", help="Disable zoom-detector refinement."
    )
    ap.add_argument(
        "--restart", action="store_true", help="Ignore existing progress and start over."
    )
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


# ===========================================================================
#  Per-run prediction -> position comparison
# ===========================================================================


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


def score_run(
    controller: Any,
    run: RunRecord,
    time_col: str,
    decode_config: bool,
    refine_pois: bool,
) -> List[Dict[str, Any]]:
    """One run -> one result row per chain-space POI present in truth."""
    df_raw = pd.read_csv(run.csv_path)
    tcol = time_col if time_col in df_raw.columns else df_raw.columns[0]
    time_axis = pd.to_numeric(df_raw[tcol], errors="coerce").to_numpy(dtype=float)

    output, _num_channels = controller.predict(
        df=df_raw, decode_config=decode_config, refine_pois=refine_pois
    )
    predicted = _predicted_positions(output, time_axis)
    tier = viscosity_tier(run.viscosity_cP)

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
                tier=TIER_LABELS[tier],
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


# ===========================================================================
#  Resumable driver
# ===========================================================================


def _check_run_config(output_dir: Path, run_config: Optional[Dict[str, Any]], resume: bool) -> None:
    """Resumability keys on run_id alone, which says nothing about WHICH
    configuration (decode_config, refine_pois, assets_root, deployment_dir)
    produced the rows already on disk. Without this check, re-running with
    a changed flag against the same --output and no --restart would find
    every run_id already "done" and silently report the FIRST run's config,
    mislabeled as the new one. Recorded/compared only when the caller
    supplies run_config (main() does; direct run_eval() callers such as
    tests may omit it to skip this check entirely)."""
    if run_config is None:
        return
    config_path = output_dir / "run_config.json"
    if resume and config_path.exists():
        prior = json.loads(config_path.read_text())
        if prior != run_config:
            raise SystemExit(
                f"{output_dir} already holds results for a different configuration.\n"
                f"  prior: {prior}\n  now:   {run_config}\n"
                "Re-running with a changed --assets-root / --deployment-dir / "
                "--no-decode-config / --no-refine-pois against the same --output would "
                "silently mix results from two configurations. Use a different --output, "
                "or pass --restart to start this configuration over."
            )
    config_path.write_text(json.dumps(run_config, indent=2))


def run_eval(
    controller: Any,
    runs: List[RunRecord],
    output_dir: Path,
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
                rows = score_run(controller, run, time_col, decode_config, refine_pois)
            except Exception as exc:
                n_failed += 1
                LOG.warning("predict failed for run {} ({}); skipping", run.run_id, exc)
                # Marked done (not retried on the next --resume) rather than left
                # pending forever -- a run that fails once (corrupt/moved CSV) is
                # assumed to fail every time; --restart reprocesses everything if
                # the underlying cause gets fixed.
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
            "(see warnings above for each one; use --restart to retry them).",
            n_failed,
        )
    return results_path, n_failed


# ===========================================================================
#  Aggregation
# ===========================================================================


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


def summarize(results_path: Path, output_dir: Path, gross_threshold: float) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    # Defends against the duplicate rows a crash between a run's results-flush and
    # its progress.txt write can leave behind (see run_eval's write-order comment)
    # -- keep the LAST occurrence per (run_id, poi) rather than double-counting.
    df = df.drop_duplicates(subset=["run_id", "poi"], keep="last")
    df["hit"] = df["hit"].astype(int)

    overall_rows = []
    for poi in POI_KEYS:
        sub = df[df["poi"] == poi]
        m = _summarize(sub, gross_threshold)
        overall_rows.append(dict(poi=poi, tier="ALL", **vars(m), hit_rate=m.hit_rate))
    for poi in POI_KEYS:
        for tier_label in TIER_LABELS:
            sub = df[(df["poi"] == poi) & (df["tier"] == tier_label)]
            if len(sub) == 0:
                continue
            m = _summarize(sub, gross_threshold)
            overall_rows.append(dict(poi=poi, tier=tier_label, **vars(m), hit_rate=m.hit_rate))

    summary = pd.DataFrame(overall_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def _print_summary(
    summary: pd.DataFrame, gross_threshold: float, output_dir: Path, n_failed: int = 0
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
        for tier_label in TIER_LABELS:
            trow = sub[sub["tier"] == tier_label]
            if trow.empty:
                continue
            row = trow.iloc[0]
            print(
                "  "
                + f"{'':<6} {tier_label:<16} {row['n_truth']:>5.0f} {row['hit_rate']:>7.1%}  "
                f"{row['mae_s']:>8.3f} {row['median_ae_s']:>8.3f} {row['rmse_s']:>8.3f} "
                f"{row['bias_s']:>+8.3f}  {row['gross_rate']:>6.1%}  {row['mean_index_err']:>10.1f}"
            )
        print(f"  {SEP}")
    print(f"{BAR}\n")


# ===========================================================================
#  Plots (dataviz-skill ordinal ramp: viscosity tiers are ORDERED, so tiers
#  get the single-hue blue sequential ramp, not the unordered categorical
#  palette; "unknown" is pulled out to neutral gray rather than folded in as
#  a false "highest tier". Steps >=250 per the ordinal-ramp floor.)
# ===========================================================================

TIER_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#898781"]
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
SECONDARY = "#52514e"
PRIMARY = "#0b0b0b"


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
        color=PRIMARY, fontsize=13, fontweight="bold",
    )
    _strip_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "bar_overall.png", bbox_inches="tight")
    plt.close(fig)


def plot_bar_by_tier(summary: pd.DataFrame, plots_dir: Path) -> None:
    _setup_style()
    import matplotlib.pyplot as plt

    tiers_present = [t for t in TIER_LABELS if not summary[summary["tier"] == t].empty]
    if not tiers_present:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    n_poi, n_tier = len(POI_KEYS), len(tiers_present)
    group_w = 0.72
    bar_w = group_w / max(n_tier, 1)
    x = np.arange(n_poi)
    for j, tier_label in enumerate(tiers_present):
        color = TIER_COLORS[TIER_LABELS.index(tier_label)]
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
        color=PRIMARY, fontsize=14, fontweight="bold", pad=14,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, labelcolor=SECONDARY)
    _strip_axes(ax)
    fig.tight_layout()
    fig.savefig(plots_dir / "bar_by_tier.png", bbox_inches="tight")
    plt.close(fig)


def plot_violin_per_poi(results: pd.DataFrame, plots_dir: Path) -> None:
    _setup_style()
    import matplotlib.pyplot as plt

    hit = results[results["hit"] == 1]
    for poi in POI_KEYS:
        sub = hit[hit["poi"] == poi]
        tiers_present = [t for t in TIER_LABELS if len(sub[sub["tier"] == t])]
        if not tiers_present:
            continue
        data = [
            sub[sub["tier"] == t]["abs_time_err_s"].to_numpy(dtype=float) for t in tiers_present
        ]
        colors = [TIER_COLORS[TIER_LABELS.index(t)] for t in tiers_present]

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


def make_all_plots(results_path: Path, summary: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(results_path)
    plot_bar_overall(summary, plots_dir)
    plot_bar_by_tier(summary, plots_dir)
    plot_violin_per_poi(results, plots_dir)
    LOG.info("Plots -> {}", plots_dir)


# ===========================================================================
#  Main
# ===========================================================================


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level.upper())

    LOG.info("Discovering corpus under {}", args.raw_root)
    runs = dedupe_runs(discover_runs(args.raw_root))
    if not runs:
        LOG.error("No runs found under {}", args.raw_root)
        return
    if args.only_runs:
        keep = load_run_filter(args.only_runs)
        runs = [r for r in runs if r.run_id in keep]
        LOG.info("Run filter: {} run(s) retained from {}", len(runs), args.only_runs)
        if len(runs) < len(keep):
            LOG.warning(
                "{} requested run id(s) from --only-runs were not found in the discovered/"
                "deduped corpus (already removed as a content-duplicate, or never captured) "
                "and will be skipped.",
                len(keep) - len(runs),
            )
    if args.n_runs is not None and len(runs) > args.n_runs:
        rng = np.random.default_rng(args.seed)
        runs = list(runs)
        rng.shuffle(runs)
        runs = runs[: args.n_runs]
    if not runs:
        LOG.error("No runs left to evaluate after filtering.")
        return
    LOG.info("Evaluating over {} run(s)", len(runs))

    assets_map = load_assets_map(paths.ASSETS_PATHS_JSON)
    model_assets = _validate_and_prune_assets(build_model_assets(assets_map, args.assets_root))
    LOG.info("Loading deployment Onyx controller from {}", args.deployment_dir)
    controller = load_onyx_controller(args.deployment_dir, model_assets)
    _verify_chain_to_prod(controller)

    run_config = dict(
        decode_config=not args.no_decode_config,
        refine_pois=not args.no_refine_pois,
        assets_root=str(args.assets_root),
        deployment_dir=str(args.deployment_dir),
    )
    results_path, n_failed = run_eval(
        controller,
        runs,
        args.output,
        decode_config=run_config["decode_config"],
        refine_pois=run_config["refine_pois"],
        resume=not args.restart,
        log_every=args.log_every,
        run_config=run_config,
    )

    LOG.info("Summarizing...")
    summary = summarize(results_path, args.output, args.gross_threshold)
    _print_summary(summary, args.gross_threshold, args.output, n_failed=n_failed)

    LOG.info("Plotting...")
    make_all_plots(results_path, summary, args.output)
    LOG.info("Done. Output -> {}", args.output)


if __name__ == "__main__":
    main()

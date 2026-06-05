"""
QModel v6 — Benchmark and per-tier evaluation
==============================================

Runs the trained cascade on every discovered run and reports per-POI
time-error statistics. The headline output is a per-viscosity-tier
breakdown so the 150+ cP tail (the v6 / 10k weakness) is visible during
model selection rather than buried in a global aggregate.

Outputs written to ``BENCHMARK_OUTPUT``:
  * ``benchmark_metrics.csv``        — per-POI global aggregate
  * ``benchmark_metrics_by_tier.csv``— per-POI per-tier breakdown
  * ``gross_failures.csv``           — list of |err| > threshold cases

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Version:
    7.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display server
import matplotlib.pyplot as plt

from augmentation import viscosity_tier
from cascade_inference import cascade_predict, load_trained_models
from signal_processing import (
    COL_DIFF,
    COL_DISS,
    COL_FREQ,
    COL_TIME,
    preprocess_dataframe,
)
from config import (
    BENCHMARK_GROSS_THRESHOLD,
    BENCHMARK_N_RUNS,
    BENCHMARK_OUTPUT,
    CASCADE_CHANNELS,
    RNG_SEED,
    RUNS_ROOT,
    TIER_LABELS,
    TRAIN_PROJECT,
    USE_SUBPIXEL_REFINE,
)
from dataset_builder import discover_runs

LOG = logging.getLogger("v6.benchmark")


# ===========================================================================
#  Metric accumulator
# ===========================================================================


@dataclass
class _POIMetrics:
    """Per-POI time-error accumulator."""

    time_errs: List[float] = field(default_factory=list)
    n: int = 0
    mae: float = float("nan")
    rmse: float = float("nan")
    median_ae: float = float("nan")
    bias: float = float("nan")
    max_ae: float = float("nan")
    gross_failure_rate: float = float("nan")

    def record(self, err: float) -> None:
        self.time_errs.append(err)

    def summarize(self, gross_threshold: float) -> None:
        if not self.time_errs:
            return
        e = np.array(self.time_errs)
        ae = np.abs(e)
        self.n = len(e)
        self.mae = float(np.mean(ae))
        self.rmse = float(np.sqrt(np.mean(e**2)))
        self.median_ae = float(np.median(ae))
        self.bias = float(np.mean(e))
        self.max_ae = float(np.max(ae))
        self.gross_failure_rate = float(np.mean(ae > gross_threshold))


# ===========================================================================
#  Pretty printing
# ===========================================================================

POI_KEYS = ["POI1", "POI2", "POI3", "POI4", "POI5"]


def _print_global(
    metrics: Dict[str, _POIMetrics],
    gross_threshold: float,
    n_runs: int,
    refine_used: bool,
    output_dir: Path,
) -> None:
    HDR = (
        f"{'POI':<6} {'N':>5}  {'Bias(s)':>8} {'MAE(s)':>8} {'RMSE(s)':>8} "
        f"{'Med(s)':>8} {'Max(s)':>8} {'Fail%':>7}"
    )
    FMT = "{:<6} {:>5d}  {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>6.1%}"
    SEP = "-" * len(HDR)
    BAR = "=" * (len(HDR) + 4)
    refine_tag = " (sub-pixel refined)" if refine_used else " (coarse YOLO)"

    print(f"\n{BAR}")
    print(f"  v6 Post-Train Time-MAE Benchmark{refine_tag}")
    print(f"  {n_runs} runs  |  gross > {gross_threshold} s")
    print(f"  Output → {output_dir}")
    print(f"  {SEP}")
    print(f"  {HDR}")
    print(f"  {SEP}")
    for poi in POI_KEYS:
        m = metrics.get(poi)
        if m is None or m.n == 0:
            print(f"  {poi:<6} {'-':>5}")
            continue
        print(
            "  "
            + FMT.format(
                poi,
                m.n,
                m.bias,
                m.mae,
                m.rmse,
                m.median_ae,
                m.max_ae,
                m.gross_failure_rate,
            )
        )
    print(f"  {SEP}")
    print(f"{BAR}\n")


def _print_tier(
    tier_metrics: Dict[str, Dict[int, _POIMetrics]],
    gross_threshold: float,
) -> None:
    HDR = (
        f"{'POI':<6} {'Tier':<12} {'N':>5}  {'Bias(s)':>8} {'MAE(s)':>8} "
        f"{'RMSE(s)':>8} {'Max(s)':>8} {'Fail%':>7}"
    )
    FMT = "{:<6} {:<12} {:>5d}  {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>6.1%}"
    SEP = "-" * len(HDR)
    BAR = "=" * (len(HDR) + 4)

    print(f"\n{BAR}")
    print(f"  Per-Tier Time-MAE Breakdown   gross > {gross_threshold} s")
    print(f"  {SEP}")
    print(f"  {HDR}")
    print(f"  {SEP}")
    for poi in POI_KEYS:
        any_printed = False
        for tier_idx, tier_label in enumerate(TIER_LABELS):
            m = tier_metrics.get(poi, {}).get(tier_idx)
            if m is None or m.n == 0:
                continue
            print(
                "  "
                + FMT.format(
                    poi,
                    tier_label,
                    m.n,
                    m.bias,
                    m.mae,
                    m.rmse,
                    m.max_ae,
                    m.gross_failure_rate,
                )
            )
            any_printed = True
        if not any_printed:
            print(f"  {poi:<6} {'-':>12}")
        print(f"  {SEP}")
    print(f"{BAR}\n")


# ===========================================================================
#  Gross-failure plotting
# ===========================================================================

# Each signal gets its own row, drawn from the preprocessed (uniform-dt)
# dataframe so the x-axis matches the time coordinates the cascade used.
_PLOT_SIGNALS = [
    (COL_DISS, "Dissipation"),
    (COL_FREQ, "Resonance_Frequency"),
    (COL_DIFF, "Difference"),
]

# Stable per-POI colours so pred/actual pairs are easy to read across rows.
_POI_COLORS = {
    "POI1": "#e41a1c",
    "POI2": "#377eb8",
    "POI3": "#4daf4a",
    "POI4": "#984ea3",
    "POI5": "#ff7f00",
}


def _render_failure_plot(
    df_raw: pd.DataFrame,
    run_id: str,
    viscosity_cP: float,
    tier_label: str,
    failures: Dict[str, Dict[str, float]],
    out_path: Path,
) -> bool:
    """Render one diagnostic figure for a run with gross failure(s).

    Draws each signal on its own row with a solid line at every failed
    POI's actual time and a dashed line at the predicted time. Only POIs
    that exceeded the gross threshold are marked, so the figure highlights
    what went wrong rather than every prediction.

    Returns True if a figure was written.
    """
    df = preprocess_dataframe(df_raw)
    if df is None or df.empty:
        return False

    t = df[COL_TIME].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        len(_PLOT_SIGNALS),
        1,
        figsize=(13, 8),
        sharex=True,
    )

    for ax, (col, nice) in zip(axes, _PLOT_SIGNALS):
        if col in df.columns:
            ax.plot(t, df[col].to_numpy(dtype=float), color="0.25", lw=0.9)
        ax.set_ylabel(nice, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)

        for poi, info in sorted(failures.items()):
            color = _POI_COLORS.get(poi, "#000000")
            true_t = info.get("true_t")
            pred_t = info.get("pred_t")
            if true_t is not None:
                ax.axvline(true_t, color=color, lw=1.6, alpha=0.9)
            if pred_t is not None:
                ax.axvline(pred_t, color=color, lw=1.6, ls="--", alpha=0.9)

    # One shared legend: solid = actual, dashed = predicted, plus a colour
    # key for the POIs that failed in this run.
    handles = [
        plt.Line2D([0], [0], color="0.25", lw=1.6, label="actual (solid)"),
        plt.Line2D([0], [0], color="0.25", lw=1.6, ls="--", label="predicted (dashed)"),
    ]
    for poi in sorted(failures):
        err = failures[poi].get("error_s")
        err_txt = f"  err={err:+.2f}s" if err is not None else ""
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=_POI_COLORS.get(poi, "#000000"),
                lw=2.4,
                label=f"{poi}{err_txt}",
            )
        )
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.85)

    axes[-1].set_xlabel("Relative_time (s)", fontsize=9)
    n_fail = len(failures)
    fig.suptitle(
        f"Gross failure — run {run_id}   "
        f"({viscosity_cP:.1f} cP, tier {tier_label})   "
        f"{n_fail} POI{'s' if n_fail != 1 else ''} over threshold",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def run_benchmark(
    runs_root: Path = Path(RUNS_ROOT),
    project: Path = Path(TRAIN_PROJECT),
    output_dir: Path = Path(BENCHMARK_OUTPUT),
    n_runs: Optional[int] = BENCHMARK_N_RUNS,
    gross_threshold: float = BENCHMARK_GROSS_THRESHOLD,
    use_subpixel_refine: bool = USE_SUBPIXEL_REFINE,
    n_workers: int = 8,
) -> None:
    """Run the trained cascade on every discovered run and print metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models, channel_map = load_trained_models(CASCADE_CHANNELS, project=project)
    if not models:
        LOG.error("Benchmark: no channel weights loaded — aborting.")
        return

    runs = discover_runs(Path(runs_root), n_workers=n_workers)
    if not runs:
        LOG.error("Benchmark: no runs found under %s.", runs_root)
        return

    if n_runs is not None and len(runs) > n_runs:
        rng = np.random.default_rng(RNG_SEED)
        rng.shuffle(runs)
        runs = runs[:n_runs]
        LOG.info("Benchmark: capped to %d runs", len(runs))

    # Accumulators
    metrics: Dict[str, _POIMetrics] = {p: _POIMetrics() for p in POI_KEYS}
    tier_metrics: Dict[str, Dict[int, _POIMetrics]] = {
        p: {t: _POIMetrics() for t in range(len(TIER_LABELS))} for p in POI_KEYS
    }
    gross_failures: List[Dict[str, Any]] = []

    # Diagnostic plots for runs with at least one gross failure.
    plot_dir = output_dir / "gross_failure_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    n_failure_plots = 0

    LOG.info(
        "Benchmark: running cascade on %d runs (%s)",
        len(runs),
        "sub-pixel refined" if use_subpixel_refine else "coarse YOLO",
    )

    n_processed = 0
    for run in runs:
        try:
            df_raw = pd.read_csv(run.csv_path)
        except Exception as exc:
            LOG.warning("Benchmark: read failed for %s (%s)", run.csv_path, exc)
            continue

        try:
            preds = cascade_predict(
                df_raw,
                models,
                channel_map,
                use_subpixel_refine=use_subpixel_refine,
            )
        except Exception as exc:
            LOG.warning("Benchmark: cascade failed for run %s (%s)", run.run_id, exc)
            continue

        run_tier = viscosity_tier(run.viscosity_cP)
        run_failures: Dict[str, Dict[str, float]] = {}
        for poi in POI_KEYS:
            true_t = run.poi_times.get(poi)
            pred_t = preds.get(poi)
            if true_t is None or pred_t is None:
                continue
            err = pred_t - true_t
            metrics[poi].record(err)
            tier_metrics[poi][run_tier].record(err)
            if abs(err) > gross_threshold:
                gross_failures.append(
                    {
                        "run_id": run.run_id,
                        "poi": poi,
                        "viscosity_cP": run.viscosity_cP,
                        "tier": TIER_LABELS[run_tier],
                        "pred_t": pred_t,
                        "true_t": true_t,
                        "error_s": err,
                    }
                )
                run_failures[poi] = {
                    "pred_t": pred_t,
                    "true_t": true_t,
                    "error_s": err,
                }

        # One diagnostic figure per run that had any gross failure. Rendered
        # here while df_raw is still in scope to avoid re-reading the CSV.
        if run_failures:
            try:
                if _render_failure_plot(
                    df_raw=df_raw,
                    run_id=run.run_id,
                    viscosity_cP=run.viscosity_cP,
                    tier_label=TIER_LABELS[run_tier],
                    failures=run_failures,
                    out_path=plot_dir / f"{run.run_id}.png",
                ):
                    n_failure_plots += 1
            except Exception as exc:
                LOG.warning(
                    "Benchmark: failure-plot render failed for run %s (%s)",
                    run.run_id,
                    exc,
                )

        n_processed += 1
        if n_processed % 100 == 0:
            LOG.info("Benchmark: processed %d / %d", n_processed, len(runs))

    # Summarise + print
    for poi in POI_KEYS:
        metrics[poi].summarize(gross_threshold)
        for t in range(len(TIER_LABELS)):
            tier_metrics[poi][t].summarize(gross_threshold)

    _print_global(metrics, gross_threshold, n_processed, use_subpixel_refine, output_dir)
    _print_tier(tier_metrics, gross_threshold)

    # CSV outputs
    pd.DataFrame(
        [
            dict(
                poi=poi,
                n=metrics[poi].n,
                bias_s=metrics[poi].bias,
                mae_s=metrics[poi].mae,
                rmse_s=metrics[poi].rmse,
                median_ae_s=metrics[poi].median_ae,
                max_ae_s=metrics[poi].max_ae,
                gross_failure_rate=metrics[poi].gross_failure_rate,
                refined=use_subpixel_refine,
            )
            for poi in POI_KEYS
        ]
    ).to_csv(output_dir / "benchmark_metrics.csv", index=False)

    tier_rows = []
    for poi in POI_KEYS:
        for tier_idx in range(len(TIER_LABELS)):
            m = tier_metrics[poi][tier_idx]
            if m.n == 0:
                continue
            tier_rows.append(
                dict(
                    poi=poi,
                    tier_index=tier_idx,
                    tier_label=TIER_LABELS[tier_idx],
                    n=m.n,
                    bias_s=m.bias,
                    mae_s=m.mae,
                    rmse_s=m.rmse,
                    median_ae_s=m.median_ae,
                    max_ae_s=m.max_ae,
                    gross_failure_rate=m.gross_failure_rate,
                    refined=use_subpixel_refine,
                )
            )
    if tier_rows:
        pd.DataFrame(tier_rows).to_csv(
            output_dir / "benchmark_metrics_by_tier.csv",
            index=False,
        )

    if gross_failures:
        pd.DataFrame(gross_failures).to_csv(
            output_dir / "gross_failures.csv",
            index=False,
        )
        LOG.info(
            "Benchmark: %d gross failures → %s",
            len(gross_failures),
            output_dir / "gross_failures.csv",
        )

    if n_failure_plots:
        LOG.info(
            "Benchmark: %d gross-failure plot(s) → %s",
            n_failure_plots,
            plot_dir,
        )

    LOG.info("Benchmark complete. Metrics → %s", output_dir)

"""
QModel v6 — Configuration-decode A/B benchmark
==============================================

Measures the before/after effect of the configuration-prior decode
(`decode_config=True`) against the production greedy cascade, PAIRED on the
same runs and the same YOLO inferences: each run goes through
`QModelV6YOLO.predict(..., decode_config=True)` ONCE, and the cascade's
pre-decode placements are read back from the `_decode.cascade` snapshot, so
both arms come from a single pass.

Headline outputs (mirrors benchmark.py):
  * per-POI global aggregate, side by side (greedy vs decoded), with paired
    deltas, win/loss/tie counts and gross failures FIXED vs INTRODUCED
  * per-viscosity-tier breakdown so the high-cP tail (slow fills — exactly
    where the decode should help) is visible rather than buried
  * decode diagnostics: how often the decode moved a POI, fallback rate, and
    the ORACLE RECALL of the harvested candidate pool — the ceiling on what
    any decoder can achieve. If oracle recall is low, the remaining work is
    harvest-side (detector/augmentation), not decode-side.

Outputs written to ``--output``:
  * ``ab_metrics.csv``          — per-POI global aggregate, both arms
  * ``ab_metrics_by_tier.csv``  — per-POI per-tier breakdown, both arms
  * ``per_run_results.csv``     — one row per (run, POI) with both errors
  * ``regressions.csv``         — cases the decode made grossly worse
  * ``regression_plots/``       — diagnostic figure per regression run

Viscosity tiers are taken from the run's ``*_analyze_out.csv`` (mean of
``viscosity_avg``) when present; runs without one land in the "unknown"
tier rather than being dropped. Corpus discovery, dedup, and truth parsing
live in :mod:`..corpus` — this module is the benchmark CLI + aggregation
layer built on top of it.

Usage
-----
    python -m src.systems.qmodel_7_onyx.qa.benchmark \
        --raw-root data/raw --assets assets_paths.json --prior configs/spacing_prior.json \
        [--n-runs 300] [--gross-threshold 2.0] [--output artifacts/benchmark_decode]

    python -m src.systems.qmodel_7_onyx.qa.benchmark --selftest
        (synthetic corpus + mocked YOLO harvest; exercises the REAL decode
         path and all aggregation/reporting plumbing without model weights)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: write PNGs without a display server
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

from .. import paths
from ..corpus import (
    RunRecord,
    dedupe_runs,
    discover_runs,
    load_run_filter,
    viscosity_tier,
)
from ..decode.spacing_prior import POI_ORDER

LOG = get_logger("qmodel_7_onyx.qa.benchmark")

POI_KEYS = list(POI_ORDER)

# chain name <-> production output name (production POI_MAP id space).
CHAIN_TO_PROD = {"POI1": "POI1", "POI2": "POI2", "POI3": "POI4", "POI4": "POI5", "POI5": "POI6"}

TIER_LABELS = ["<2.66 cP", "2.66-6.16 cP", "6.16-18.14 cP", "18.14-73.4 cP", "73.4+ cP", "unknown"]


# ===========================================================================
#  Metric accumulators
# ===========================================================================


@dataclass
class _POIMetrics:
    """Per-POI time-error accumulator (same fields as benchmark.py)."""

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


@dataclass
class _PairedCounts:
    """Paired A/B counters for one POI (or one POI x tier cell)."""

    wins: int = 0  # decode strictly better (beyond tie band)
    losses: int = 0
    ties: int = 0
    gross_fixed: int = 0  # greedy gross, decoded fine
    gross_introduced: int = 0  # greedy fine, decoded gross
    oracle_dists: List[float] = field(default_factory=list)  # truth -> nearest candidate

    def oracle_recall(self, tol: float) -> float:
        if not self.oracle_dists:
            return float("nan")
        d = np.array(self.oracle_dists)
        return float(np.mean(d <= tol))

    @property
    def oracle_n(self) -> int:
        return len(self.oracle_dists)


# ===========================================================================
#  Pretty printing
# ===========================================================================


def _print_global(
    g: Dict[str, _POIMetrics],
    d: Dict[str, _POIMetrics],
    paired: Dict[str, _PairedCounts],
    gross_threshold: float,
    n_runs: int,
    decode_stats: Dict[str, Any],
    output_dir: Path,
    oracle_tol: float = 1.0,
) -> None:
    HDR = (
        f"{'POI':<6} {'N':>5}  {'MAE_g':>8} {'MAE_d':>8} {'dMAE':>8}  "
        f"{'Med_g':>7} {'Med_d':>7}  {'Fail%_g':>8} {'Fail%_d':>8}  "
        f"{'Fixed':>5} {'Intro':>5}  {'W/L/T':>12}  {'Oracle':>7}"
    )
    FMT = (
        "{:<6} {:>5d}  {:>8.3f} {:>8.3f} {:>+8.3f}  {:>7.3f} {:>7.3f}  "
        "{:>7.1%} {:>8.1%}  {:>5d} {:>5d}  {:>12}  {:>6.1%}"
    )
    SEP = "-" * len(HDR)
    BAR = "=" * (len(HDR) + 4)

    print(f"\n{BAR}")
    print("  v6 Configuration-Decode A/B Benchmark   (greedy cascade vs dp_decode)")
    print(f"  {n_runs} runs  |  gross > {gross_threshold} s  |  paired, single YOLO pass")
    print(f"  Output -> {output_dir}")
    print(f"  {SEP}")
    print(f"  {HDR}")
    print(f"  {SEP}")
    for poi in POI_KEYS:
        mg, md, pc = g.get(poi), d.get(poi), paired.get(poi)
        if mg is None or mg.n == 0:
            print(f"  {poi:<6} {'-':>5}")
            continue
        wlt = f"{pc.wins}/{pc.losses}/{pc.ties}"
        oracle = pc.oracle_recall(oracle_tol)
        print(
            "  "
            + FMT.format(
                poi,
                mg.n,
                mg.mae,
                md.mae,
                md.mae - mg.mae,
                mg.median_ae,
                md.median_ae,
                mg.gross_failure_rate,
                md.gross_failure_rate,
                pc.gross_fixed,
                pc.gross_introduced,
                wlt,
                oracle,
            )
        )
    print(f"  {SEP}")
    print(
        "  decode moved >=1 POI on {moved:.1%} of runs | fallback on {fallback:.1%} | "
        "decode unavailable on {unused} run(s)".format(**decode_stats)
    )
    print(f"  {SEP}")
    print("  Oracle recall (truth within tol of ANY harvested candidate — the decode ceiling):")
    hdr = "  {:<6}".format("POI") + "".join(f" {('<=' + str(t) + 's'):>9}" for t in ORACLE_TOLS)
    print(hdr)
    for poi in POI_KEYS:
        pc = paired.get(poi)
        if pc is None or pc.oracle_n == 0:
            continue
        print("  {:<6}".format(poi) + "".join(f" {pc.oracle_recall(t):>9.2%}" for t in ORACLE_TOLS))
    print(f"{BAR}\n")


def _print_tier(
    g: Dict[str, Dict[int, _POIMetrics]],
    d: Dict[str, Dict[int, _POIMetrics]],
    gross_threshold: float,
) -> None:
    HDR = (
        f"{'POI':<6} {'Tier':<10} {'N':>5}  {'MAE_g':>8} {'MAE_d':>8} {'dMAE':>8}  "
        f"{'Fail%_g':>8} {'Fail%_d':>8}"
    )
    FMT = "{:<6} {:<10} {:>5d}  {:>8.3f} {:>8.3f} {:>+8.3f}  {:>7.1%} {:>8.1%}"
    SEP = "-" * len(HDR)
    BAR = "=" * (len(HDR) + 4)

    print(f"\n{BAR}")
    print(f"  Per-Tier A/B Breakdown   gross > {gross_threshold} s")
    print(f"  {SEP}")
    print(f"  {HDR}")
    print(f"  {SEP}")
    for poi in POI_KEYS:
        any_printed = False
        for tier_idx, tier_label in enumerate(TIER_LABELS):
            mg = g.get(poi, {}).get(tier_idx)
            md = d.get(poi, {}).get(tier_idx)
            if mg is None or mg.n == 0:
                continue
            print(
                "  "
                + FMT.format(
                    poi,
                    tier_label,
                    mg.n,
                    mg.mae,
                    md.mae,
                    md.mae - mg.mae,
                    mg.gross_failure_rate,
                    md.gross_failure_rate,
                )
            )
            any_printed = True
        if not any_printed:
            print(f"  {poi:<6} {'-':>10}")
        print(f"  {SEP}")
    print(f"{BAR}\n")


# ===========================================================================
#  Regression plotting (decode made it grossly worse)
# ===========================================================================

_PLOT_SIGNALS = [("Dissipation", "Dissipation"), ("Resonance_Frequency", "Resonance_Frequency")]
_POI_COLORS = {
    "POI1": "#e41a1c",
    "POI2": "#377eb8",
    "POI3": "#4daf4a",
    "POI4": "#984ea3",
    "POI5": "#ff7f00",
}


def _render_regression_plot(
    df_raw: pd.DataFrame,
    run_id: str,
    tier_label: str,
    regressions: Dict[str, Dict[str, float]],
    out_path: Path,
) -> bool:
    """One figure per run the decode regressed: truth solid, greedy dashed,
    decoded dotted, per failing POI."""
    tcol = "Relative_time" if "Relative_time" in df_raw.columns else df_raw.columns[0]
    t = pd.to_numeric(df_raw[tcol], errors="coerce").to_numpy()
    fig, axes = plt.subplots(len(_PLOT_SIGNALS), 1, figsize=(13, 6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (col, nice) in zip(axes, _PLOT_SIGNALS, strict=True):
        if col in df_raw.columns:
            ax.plot(t, pd.to_numeric(df_raw[col], errors="coerce").to_numpy(), color="0.25", lw=0.9)
        ax.set_ylabel(nice, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        for poi, info in sorted(regressions.items()):
            c = _POI_COLORS.get(poi, "#000000")
            ax.axvline(info["true_t"], color=c, lw=1.6, alpha=0.9)
            ax.axvline(info["greedy_t"], color=c, lw=1.4, ls="--", alpha=0.9)
            ax.axvline(info["decoded_t"], color=c, lw=1.4, ls=":", alpha=0.9)
    handles = [
        plt.Line2D([0], [0], color="0.25", lw=1.6, label="truth (solid)"),
        plt.Line2D([0], [0], color="0.25", lw=1.6, ls="--", label="greedy (dashed)"),
        plt.Line2D([0], [0], color="0.25", lw=1.6, ls=":", label="decoded (dotted)"),
    ]
    for poi in sorted(regressions):
        handles.append(plt.Line2D([0], [0], color=_POI_COLORS.get(poi, "#000"), lw=2.4, label=poi))
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.85)
    axes[-1].set_xlabel("Relative_time (s)", fontsize=9)
    fig.suptitle(f"Decode regression — run {run_id}   (tier {tier_label})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


# ===========================================================================
#  Per-run extraction
# ===========================================================================


def _times_from_output(
    output: Dict[str, Any], df_raw: pd.DataFrame
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(greedy_times, decoded_times) in chain space from one predict() call.

    Decoded times come from the production POI keys; the cascade baseline
    comes from the `_decode.cascade` snapshot. If the decode never ran
    (asset missing etc.), both arms collapse to the cascade output, which
    keeps the benchmark honest rather than crashing."""
    tcol = "Relative_time" if "Relative_time" in df_raw.columns else df_raw.columns[0]
    tv = pd.to_numeric(df_raw[tcol], errors="coerce").to_numpy()

    decoded: Dict[str, float] = {}
    for chain, prod in CHAIN_TO_PROD.items():
        rec = output.get(prod, {})
        idxs = rec.get("indices", [-1])
        if idxs and idxs[0] is not None and int(idxs[0]) >= 0:
            decoded[chain] = float(tv[int(idxs[0])])

    meta = output.get("_decode") or {}
    cascade = meta.get("cascade")
    if cascade:
        greedy = {chain: float(rec["time"]) for chain, rec in cascade.items()}
    else:
        greedy = dict(decoded)
    return greedy, decoded


ORACLE_TOLS = [0.5, 1.0, 2.0, 5.0]


def _oracle_dist(output: Dict[str, Any], chain_poi: str, true_t: float) -> Optional[float]:
    """Distance from truth to the NEAREST harvested candidate (inf if the
    pool is empty). None when no candidate pool was attached at all."""
    cands = (output.get("_candidates") or {}).get(CHAIN_TO_PROD[chain_poi])
    if cands is None:
        return None
    if not cands:
        return float("inf")
    return min(abs(float(c["time"]) - true_t) for c in cands)


# ===========================================================================
#  Benchmark driver
# ===========================================================================


def run_benchmark(
    controller: Any,
    runs: List[RunRecord],
    output_dir: Path,
    n_runs: Optional[int] = None,
    gross_threshold: float = 2.0,
    tie_band: float = 0.05,
    oracle_tol: float = 1.0,
    seed: int = 1337,
    dump_candidates: bool = False,
    refine_pois: bool = True,
) -> Dict[str, Any]:
    """Run the paired A/B benchmark and print/persist metrics.

    controller must expose predict(df=..., decode_config=True) with the
    QModelV6YOLO output contract. Returns the global summary dict (useful
    for asserting in tests / CI gates).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "regression_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Optional dump of every run's candidate pools + truth + cascade picks.
    # This is the input to decode/sweep.py: the decode itself costs ~2 ms, so
    # once the pools are on disk, lambda / margin / blend can be swept across
    # the whole corpus offline in seconds — no YOLO re-runs.
    dump_fh = open(output_dir / "candidates.jsonl", "w") if dump_candidates else None

    if n_runs is not None and len(runs) > n_runs:
        rng = np.random.default_rng(seed)
        runs = list(runs)
        rng.shuffle(runs)
        runs = runs[:n_runs]
        LOG.info("Benchmark: capped to {} runs", len(runs))

    g_metrics = {p: _POIMetrics() for p in POI_KEYS}
    d_metrics = {p: _POIMetrics() for p in POI_KEYS}
    g_tier = {p: {t: _POIMetrics() for t in range(len(TIER_LABELS))} for p in POI_KEYS}
    d_tier = {p: {t: _POIMetrics() for t in range(len(TIER_LABELS))} for p in POI_KEYS}
    paired = {p: _PairedCounts() for p in POI_KEYS}

    per_run_rows: List[Dict[str, Any]] = []
    regression_rows: List[Dict[str, Any]] = []
    refine_stats: Dict[str, Dict[str, Any]] = {}
    n_processed = n_moved = n_fallback = n_decode_unused = n_regression_plots = 0

    for run in runs:
        try:
            df_raw = pd.read_csv(run.csv_path)
        except Exception as exc:
            LOG.warning("Benchmark: read failed for {} ({})", run.csv_path, exc)
            continue
        try:
            output, _ = controller.predict(df=df_raw, decode_config=True, refine_pois=refine_pois)
        except Exception as exc:
            LOG.warning("Benchmark: predict failed for run {} ({})", run.run_id, exc)
            continue

        meta = output.get("_decode") or {}
        refine = output.get("_refine") or {}
        refine_moves = refine.get("moved") or {}
        if not meta.get("used"):
            n_decode_unused += 1
        if meta.get("changed"):
            n_moved += 1
        if meta.get("fallback"):
            n_fallback += 1

        greedy_t, decoded_t = _times_from_output(output, df_raw)
        tier = viscosity_tier(run.viscosity_cP)

        if dump_fh is not None:
            prod_to_chain = {v: k for k, v in CHAIN_TO_PROD.items()}
            pools = {
                prod_to_chain[prod]: [
                    {"time": float(c["time"]), "conf": float(c["conf"])} for c in lst
                ]
                for prod, lst in (output.get("_candidates") or {}).items()
                if prod in prod_to_chain
            }
            dump_fh.write(
                json.dumps(
                    dict(
                        run_id=run.run_id,
                        viscosity_cP=run.viscosity_cP,
                        truth=run.poi_times,
                        present=meta.get("present"),
                        cascade=meta.get("cascade"),
                        pools=pools,
                    )
                )
                + "\n"
            )
        run_regressions: Dict[str, Dict[str, float]] = {}

        for poi in POI_KEYS:
            true_t = run.poi_times.get(poi)
            if true_t is None:
                continue
            odist = _oracle_dist(output, poi, true_t)
            if odist is not None:
                paired[poi].oracle_dists.append(odist)
            gt, dt = greedy_t.get(poi), decoded_t.get(poi)
            if gt is None or dt is None:
                continue  # misses tracked via per_run rows below
            eg, ed = gt - true_t, dt - true_t
            g_metrics[poi].record(eg)
            d_metrics[poi].record(ed)
            g_tier[poi][tier].record(eg)
            d_tier[poi][tier].record(ed)
            delta = abs(eg) - abs(ed)
            if delta > tie_band:
                paired[poi].wins += 1
            elif delta < -tie_band:
                paired[poi].losses += 1
            else:
                paired[poi].ties += 1
            g_gross, d_gross = abs(eg) > gross_threshold, abs(ed) > gross_threshold
            if g_gross and not d_gross:
                paired[poi].gross_fixed += 1
            if d_gross and not g_gross:
                paired[poi].gross_introduced += 1
                regression_rows.append(
                    dict(
                        run_id=run.run_id,
                        poi=poi,
                        viscosity_cP=run.viscosity_cP,
                        tier=TIER_LABELS[tier],
                        true_t=true_t,
                        greedy_t=gt,
                        decoded_t=dt,
                        greedy_err_s=eg,
                        decoded_err_s=ed,
                    )
                )
                run_regressions[poi] = {"true_t": true_t, "greedy_t": gt, "decoded_t": dt}
            # Refinement attribution: _refine.moved is keyed by production
            # names; positive effect = the refiner brought the placement
            # closer to truth than where the decode left it.
            rmove = refine_moves.get(CHAIN_TO_PROD[poi])
            refined = rmove is not None
            refine_effect = (
                abs(float(rmove["from"]) - true_t) - abs(float(rmove["to"]) - true_t)
                if refined
                else None
            )
            if refined:
                rs = refine_stats.setdefault(poi, dict(n=0, helped=0, hurt=0, effects=[]))
                rs["n"] += 1
                rs["effects"].append(refine_effect)
                if refine_effect > tie_band:
                    rs["helped"] += 1
                elif refine_effect < -tie_band:
                    rs["hurt"] += 1
            per_run_rows.append(
                dict(
                    run_id=run.run_id,
                    poi=poi,
                    viscosity_cP=run.viscosity_cP,
                    tier=TIER_LABELS[tier],
                    true_t=true_t,
                    greedy_t=gt,
                    decoded_t=dt,
                    greedy_err_s=eg,
                    decoded_err_s=ed,
                    decode_moved=poi in (meta.get("changed") or []),
                    refined=refined,
                    refine_effect_s=refine_effect,
                    oracle_dist_s=odist,
                )
            )

        if run_regressions:
            try:
                if _render_regression_plot(
                    df_raw,
                    run.run_id,
                    TIER_LABELS[tier],
                    run_regressions,
                    plot_dir / f"{run.run_id}.png",
                ):
                    n_regression_plots += 1
            except Exception as exc:
                LOG.warning("Benchmark: regression plot failed for {} ({})", run.run_id, exc)

        n_processed += 1
        if n_processed % 100 == 0:
            LOG.info("Benchmark: processed {} / {}", n_processed, len(runs))

    for poi in POI_KEYS:
        g_metrics[poi].summarize(gross_threshold)
        d_metrics[poi].summarize(gross_threshold)
        for t in range(len(TIER_LABELS)):
            g_tier[poi][t].summarize(gross_threshold)
            d_tier[poi][t].summarize(gross_threshold)

    if dump_fh is not None:
        dump_fh.close()
        LOG.info("Benchmark: candidate pools -> {}", output_dir / "candidates.jsonl")

    decode_stats = dict(
        moved=n_moved / n_processed if n_processed else float("nan"),
        fallback=n_fallback / n_processed if n_processed else float("nan"),
        unused=n_decode_unused,
    )
    _print_global(
        g_metrics,
        d_metrics,
        paired,
        gross_threshold,
        n_processed,
        decode_stats,
        output_dir,
        oracle_tol,
    )
    _print_tier(g_tier, d_tier, gross_threshold)

    if refine_stats:
        print("  Zoom-refinement attribution (effect = |pre-refine err| - |post-refine err|):")
        for poi in POI_KEYS:
            rs = refine_stats.get(poi)
            if not rs:
                continue
            eff = np.array(rs["effects"], dtype=float)
            print(
                f"    {poi}: moved {rs['n']:>4}  helped {rs['helped']:>4}  "
                f"hurt {rs['hurt']:>4}  mean effect {eff.mean():+.3f}s  "
                f"median {np.median(eff):+.3f}s"
            )
        print()

    # ------------------------------------------------------------- CSVs
    pd.DataFrame(
        [
            dict(
                poi=poi,
                n=g_metrics[poi].n,
                mae_greedy_s=g_metrics[poi].mae,
                mae_decoded_s=d_metrics[poi].mae,
                delta_mae_s=d_metrics[poi].mae - g_metrics[poi].mae,
                median_ae_greedy_s=g_metrics[poi].median_ae,
                median_ae_decoded_s=d_metrics[poi].median_ae,
                rmse_greedy_s=g_metrics[poi].rmse,
                rmse_decoded_s=d_metrics[poi].rmse,
                gross_rate_greedy=g_metrics[poi].gross_failure_rate,
                gross_rate_decoded=d_metrics[poi].gross_failure_rate,
                gross_fixed=paired[poi].gross_fixed,
                gross_introduced=paired[poi].gross_introduced,
                wins=paired[poi].wins,
                losses=paired[poi].losses,
                ties=paired[poi].ties,
                **{
                    f"oracle_recall_{str(t).replace('.', '_')}s": paired[poi].oracle_recall(t)
                    for t in ORACLE_TOLS
                },
            )
            for poi in POI_KEYS
        ]
    ).to_csv(output_dir / "ab_metrics.csv", index=False)

    tier_rows = []
    for poi in POI_KEYS:
        for t in range(len(TIER_LABELS)):
            mg, md = g_tier[poi][t], d_tier[poi][t]
            if mg.n == 0:
                continue
            tier_rows.append(
                dict(
                    poi=poi,
                    tier_index=t,
                    tier_label=TIER_LABELS[t],
                    n=mg.n,
                    mae_greedy_s=mg.mae,
                    mae_decoded_s=md.mae,
                    delta_mae_s=md.mae - mg.mae,
                    gross_rate_greedy=mg.gross_failure_rate,
                    gross_rate_decoded=md.gross_failure_rate,
                )
            )
    if tier_rows:
        pd.DataFrame(tier_rows).to_csv(output_dir / "ab_metrics_by_tier.csv", index=False)
    if per_run_rows:
        pd.DataFrame(per_run_rows).to_csv(output_dir / "per_run_results.csv", index=False)
    if regression_rows:
        pd.DataFrame(regression_rows).to_csv(output_dir / "regressions.csv", index=False)
        LOG.info(
            "Benchmark: {} decode regression(s) -> {} ({} plot(s))",
            len(regression_rows),
            output_dir / "regressions.csv",
            n_regression_plots,
        )

    LOG.info("Benchmark complete. Metrics -> {}", output_dir)
    return dict(
        n_runs=n_processed,
        decode_stats=decode_stats,
        global_greedy={p: g_metrics[p] for p in POI_KEYS},
        global_decoded={p: d_metrics[p] for p in POI_KEYS},
        paired=paired,
    )


# ===========================================================================
#  Self-test: synthetic corpus + mocked YOLO harvest, REAL decode path
# ===========================================================================


def _selftest(tmp_root: Path, output_dir: Path, n_runs: int = 80, seed: int = 11) -> None:
    """Builds a synthetic corpus sampled from the fitted prior, mocks only
    the YOLO harvest (true POIs + jitter + decoys, with the greedy pick
    sometimes hijacked by a confident decoy), and pushes it through the REAL
    controller decode (`_decode_with_prior`) and the full benchmark
    aggregation. Verifies the plumbing end-to-end without model weights."""
    from ..decode.spacing_prior import SpacingPrior
    from ..inference.controller import QModelV6YOLO

    rng = np.random.default_rng(seed)
    prior = SpacingPrior.load(paths.SPACING_PRIOR_JSON)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # ---- corpus
    for i in range(n_runs):
        scale = rng.choice([0.5, 1.0, 2.5, 6.0])  # fast .. very slow fills
        t0 = rng.uniform(3, 12)
        gaps = [
            scale * np.exp(rng.normal(p_.log_mu_sec, 0.4 * p_.log_sd_sec))
            for p_ in (prior.gap[p] for p in prior.pairs)
        ]
        truth = np.concatenate([[t0], t0 + np.cumsum(gaps)])
        end = truth[-1] * 1.3
        t_axis = np.arange(0, end, 0.02)
        idxs = [int(np.searchsorted(t_axis, tt)) for tt in truth]
        run_dir = tmp_root / f"synth_{i:04d}"
        run_dir.mkdir(exist_ok=True)
        diss = np.cumsum(rng.normal(0, 1e-7, len(t_axis))) + 3e-5
        pd.DataFrame(
            {
                "Relative_time": t_axis,
                "Dissipation": diss,
                "Resonance_Frequency": 1.5e7 - np.linspace(0, 500, len(t_axis)),
            }
        ).to_csv(run_dir / f"synth_{i:04d}.csv", index=False)
        # poi rows 0..5 with the legacy row-2 shim duplicated next to row 1
        rows = [idxs[0], idxs[1], idxs[1] + 2, idxs[2], idxs[3], idxs[4]]
        pd.Series(rows).to_csv(run_dir / f"synth_{i:04d}_poi.csv", index=False, header=False)
        cp = {0.5: 3.0, 1.0: 15.0, 2.5: 80.0, 6.0: 220.0}[float(scale)] * rng.uniform(0.8, 1.2)
        pd.DataFrame({"shear_rate": [100.0], "viscosity_avg": [cp]}).to_csv(
            run_dir / f"synth_{i:04d}_analyze_out.csv", index=False
        )

    class MockYOLOController(QModelV6YOLO):
        """Fabricates the harvest; everything downstream (decode, index
        resolution, output contract) is the real production code."""

        def predict(self, df=None, decode_config=False, **kwargs):  # type: ignore[override]
            tcol = "Relative_time"
            tv = df[tcol].to_numpy(dtype=float)
            # recover truth from the time axis is impossible; the selftest
            # writes truth into the harvest via closure below.
            truth = self._truth  # injected per run
            harvested: Dict[int, List[Dict[str, Any]]] = {}
            final_results: Dict[int, Dict[str, Any]] = {}
            for chain, true_t in truth.items():
                pid = self.DECODE_NAME_TO_ID[chain]
                pool = [
                    dict(time=float(true_t + rng.normal(0, 0.1)), conf=float(rng.uniform(0.5, 0.9)))
                ]
                for _ in range(int(rng.integers(1, 5))):
                    pool.append(
                        dict(
                            time=float(rng.uniform(tv[0], tv[-1])),
                            conf=float(rng.uniform(0.05, 0.7)),
                        )
                    )
                # 15%: a decoy outshines the true box -> greedy goes wrong
                if rng.random() < 0.15:
                    pool.append(
                        dict(
                            time=float(rng.uniform(tv[0], tv[-1])),
                            conf=float(rng.uniform(0.91, 0.99)),
                        )
                    )
                for d_ in pool:
                    d_["index"] = self._get_raw_index(df, d_["time"])
                harvested[pid] = pool
                best = max(pool, key=lambda x: x["conf"])
                final_results[pid] = {
                    "index": best["index"],
                    "conf": best["conf"],
                    "time": best["time"],
                }
            num_channels = max(0, len(truth) - 2)
            decode_meta = None
            if decode_config:
                decode_meta = self._decode_with_prior(final_results, harvested, num_channels, df)
            output = self._format_output(final_results)
            cand_out = {}
            for pid, lst in harvested.items():
                cand_out[self.POI_MAP[pid]] = lst
            output["_candidates"] = cand_out
            if decode_meta is not None:
                output["_decode"] = decode_meta
            return output, num_channels

    runs = discover_runs(tmp_root)
    assert runs, "selftest corpus discovery failed"
    ctl = MockYOLOController({"spacing_prior": str(paths.SPACING_PRIOR_JSON)})

    # bind truth per run via a thin wrapper
    class _Bound:
        def __init__(self, ctl, runs):
            self.ctl = ctl
            self.by_path = {str(r.csv_path): r.poi_times for r in runs}
            self._iter = iter(runs)

        def predict(self, df=None, decode_config=False, **kw):
            r = next(self._iter)
            self.ctl._truth = r.poi_times
            return self.ctl.predict(df=df, decode_config=decode_config, **kw)

    summary = run_benchmark(
        _Bound(ctl, runs),
        runs,
        output_dir,
        gross_threshold=2.0,
        oracle_tol=1.0,
        dump_candidates=True,
    )
    # plumbing assertions
    assert summary["n_runs"] == len(runs)
    for poi in POI_KEYS:
        pc = summary["paired"][poi]
        assert pc.oracle_n > 0, "oracle recall never recorded"
    expected = {"ab_metrics.csv", "ab_metrics_by_tier.csv", "per_run_results.csv"}
    present = {p.name for p in Path(output_dir).iterdir()}
    assert expected <= present, f"missing outputs: {expected - present}"
    print("SELFTEST OK — aggregation, A/B pairing, tiers, oracle recall and CSVs verified.")


# ===========================================================================
#  CLI
# ===========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--raw-root", type=Path, default=paths.DATA_ROOT, help="data/raw root of run directories"
    )
    ap.add_argument(
        "--assets",
        type=Path,
        default=paths.ASSETS_PATHS_JSON,
        help="assets_paths.json for the controller",
    )
    ap.add_argument("--prior", type=Path, default=paths.SPACING_PRIOR_JSON)
    ap.add_argument("--output", type=Path, default=paths.ARTIFACTS_ROOT / "benchmark_decode")
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--gross-threshold", type=float, default=2.0)
    ap.add_argument("--tie-band", type=float, default=0.05)
    ap.add_argument("--oracle-tol", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--only-runs",
        type=Path,
        default=None,
        help="restrict to run ids in this file (e.g. the dataset manifest.json, "
        "whose val_ids are the held-out runs — the only honest evaluation set "
        "for a model trained on this corpus)",
    )
    ap.add_argument(
        "--no-refine",
        action="store_true",
        help="disable the zoom-refinement stage (isolates decode-only performance)",
    )
    ap.add_argument(
        "--dump-candidates",
        action="store_true",
        help="write candidates.jsonl (input for decode/sweep.py offline tuning)",
    )
    ap.add_argument("--selftest", action="store_true", help="run synthetic end-to-end self-test")
    args = ap.parse_args()

    if args.selftest:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            _selftest(Path(td) / "corpus", args.output / "selftest")
        return

    if not args.raw_root or not args.assets:
        ap.error("--raw-root and --assets are required (or use --selftest)")

    from ..inference.controller import QModelV6YOLO

    model_assets = json.loads(Path(args.assets).read_text())
    model_assets["spacing_prior"] = str(args.prior)
    controller = QModelV6YOLO(model_assets)

    runs = dedupe_runs(discover_runs(args.raw_root))
    if args.only_runs:
        keep = load_run_filter(args.only_runs)
        runs = [r for r in runs if r.run_id in keep]
        LOG.info("Run filter: {} runs retained from {}", len(runs), args.only_runs)
    if not runs:
        LOG.error("No runs found under {}", args.raw_root)
        return
    LOG.info("Discovered {} runs", len(runs))
    run_benchmark(
        controller,
        runs,
        args.output,
        n_runs=args.n_runs,
        gross_threshold=args.gross_threshold,
        tie_band=args.tie_band,
        oracle_tol=args.oracle_tol,
        seed=args.seed,
        dump_candidates=True,
        refine_pois=not args.no_refine,
    )


if __name__ == "__main__":
    main()

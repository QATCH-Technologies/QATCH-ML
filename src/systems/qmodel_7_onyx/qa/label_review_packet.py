"""Generate the human review packet for persistent offender runs.

When a run misses the same transition at every prefix INCLUDING the full
run, under multiple independent renders and trainings, with miss
confidence rising to saturation, "confidently wrong across all
representations" is the signature of label error or a physically
anomalous fill (partial channel, wicking, bubble) - not of a trainable
representation gap. This module produces the packet a human reviewer
needs to make that call.

For each suspect run given via `--runs` (see :mod:`.triage_offenders`
for how the offender list is derived) this script writes one PNG page:
raw dissipation and resonance frequency, (top) full run with all labeled
POI verticals, and (below) ZOOMED panes around each late POI (default
POI4 and POI5, ±zoom_s), where a genuine transition is unmistakable to a
human eye and a mislabel equally so. Reviewers mark each run: label
correct / label wrong (move or remove POI) / physically anomalous.
Corrections feed the next dataset build; anomalous runs get excluded or
re-classed.

Zero runtime cost - this is offline human review, not a model stage.

Usage
-----
    python -m src.systems.qmodel_7_onyx.qa.label_review_packet --raw-root data/raw \
        --out artifacts/label_review \
        --runs 02311 02810 02938 [--pois POI4 POI5] [--zoom-s 60]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logger import get_logger

from .. import paths
from ..corpus import dedupe_runs, discover_runs

LOG = get_logger("qmodel_7_onyx.qa.label_review_packet")

COL_TIME = "Relative_time"
COL_DISS = "Dissipation"
COL_FREQ = "Resonance_Frequency"
POI_ORDER = ["POI1", "POI2", "POI3", "POI4", "POI5"]
POI_COLOR = {
    "POI1": "#888",
    "POI2": "#888",
    "POI3": "tab:green",
    "POI4": "tab:orange",
    "POI5": "tab:red",
}


def _plot_pair(
    ax_d, ax_f, df: pd.DataFrame, t0: float, t1: float, poi: Dict[str, float], title: str
) -> None:
    """Plot dissipation and resonance-frequency traces over `[t0, t1]`
    onto a pair of axes, with vertical markers for any POI in range.

    Args:
        ax_d: matplotlib axes for the dissipation trace.
        ax_f: matplotlib axes for the resonance-frequency trace.
        df (pd.DataFrame): run dataframe with time/dissipation/frequency
            columns.
        t0 (float): window start time.
        t1 (float): window end time.
        poi (Dict[str, float]): POI name -> time, for markers falling
            inside `[t0, t1]`.
        title (str): subplot title placed on the dissipation axes.
    """
    m = (df[COL_TIME] >= t0) & (df[COL_TIME] <= t1)
    sl = df.loc[m]
    t = sl[COL_TIME].to_numpy(float)
    ax_d.plot(t, sl[COL_DISS].to_numpy(float), lw=0.6, color="tab:red")
    ax_f.plot(t, sl[COL_FREQ].to_numpy(float), lw=0.6, color="tab:green")
    for name in POI_ORDER:
        pt = poi.get(name)
        if pt is None or not (t0 <= pt <= t1):
            continue
        for ax in (ax_d, ax_f):
            ax.axvline(pt, color=POI_COLOR[name], lw=1.0, ls="--", alpha=0.9)
        ax_d.text(
            pt, ax_d.get_ylim()[1], name, fontsize=7, ha="left", va="top", color=POI_COLOR[name]
        )
    ax_d.set_title(title, fontsize=9, loc="left")
    ax_d.set_ylabel("Diss", fontsize=7)
    ax_f.set_ylabel("Freq", fontsize=7)
    for ax in (ax_d, ax_f):
        ax.tick_params(labelsize=6)
        ax.margins(x=0)


def make_page(rec, pois: List[str], zoom_s: float, out: Path) -> None:
    """Render one review PNG page for a single run.

    The page shows the full run followed by a zoomed pane (±`zoom_s`)
    around each requested POI that is present on the run, so a reviewer
    can judge full-run context and transition-level detail together.

    Args:
        rec: a :class:`RunRecord` with `run_id`, `csv_path`,
            `poi_times`, and `viscosity_cP`.
        pois (List[str]): POI names to zoom in on, if present on `rec`.
        zoom_s (float): half-width in seconds of each zoomed pane.
        out (Path): directory to write `{run_id}_review.png` into.
    """
    df = pd.read_csv(rec.csv_path)
    if COL_TIME not in df.columns:
        LOG.warning("{}: no time column", rec.run_id)
        return
    df = df.sort_values(COL_TIME)
    poi = dict(rec.poi_times)
    t0, t1 = float(df[COL_TIME].min()), float(df[COL_TIME].max())
    zoom_targets = [p for p in pois if poi.get(p) is not None]

    rows = 2 * (1 + len(zoom_targets))
    fig, axes = plt.subplots(rows, 1, figsize=(11, 2.1 * rows), sharex=False)
    fig.suptitle(
        f"run {rec.run_id} | visc {rec.viscosity_cP} cP | span {t1 - t0:.0f}s | "
        f"review: are {', '.join(zoom_targets)} real transitions?",
        fontsize=10,
    )
    _plot_pair(axes[0], axes[1], df, t0, t1, poi, "full run")
    for j, name in enumerate(zoom_targets):
        pt = poi[name]
        a, b = axes[2 + 2 * j], axes[3 + 2 * j]
        _plot_pair(
            a,
            b,
            df,
            max(t0, pt - zoom_s),
            min(t1, pt + zoom_s),
            poi,
            f"zoom {name} ±{zoom_s:.0f}s - visible transition? (y / move / remove)",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / f"{rec.run_id}_review.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=paths.DATA_ROOT)
    ap.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="run ids to review (required - see usage above for the current "
        "persistent-offender list)",
    )
    ap.add_argument("--pois", nargs="+", default=["POI4", "POI5"])
    ap.add_argument("--zoom-s", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=paths.ARTIFACTS_ROOT / "label_review")
    args = ap.parse_args()
    if not args.runs:
        ap.error("--runs is required (a list of run ids to generate review pages for)")

    by_id = {r.run_id: r for r in dedupe_runs(discover_runs(args.raw_root))}
    args.out.mkdir(parents=True, exist_ok=True)
    want = [r.zfill(5) for r in args.runs]
    for rid in want:
        rec = by_id.get(rid)
        if rec is None:
            LOG.warning("{}: not found", rid)
            continue
        make_page(rec, args.pois, args.zoom_s, args.out)
        LOG.info("{} -> {}", rid, args.out / (rid + "_review.png"))
    LOG.info("review pages -> {}", args.out)
    LOG.info("mark each: label correct / label wrong (move or remove) / physically anomalous;")
    LOG.info("corrections feed the next build_fill_classifier run, anomalous runs get excluded.")


if __name__ == "__main__":
    main()

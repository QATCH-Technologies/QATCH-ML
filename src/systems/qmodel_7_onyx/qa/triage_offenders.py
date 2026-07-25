"""
triage_offenders.py
===================

Second-stage triage after qa/audit_fill_val.py, built for what that audit
found: the residual errors are NOT diffuse — ~60% sit in ~20 runs, and the
worst population is runs whose ENTIRE post-POI5 region (uniform cuts, hard
cut, full run) reads 2ch. A model that is wrong about a run at every
prefix is not a debounce problem and not an epochs problem; it is one of
exactly three things, and each has a different fix:

  (a) FAINT RIDGE — the POI5 transition is real but its salience in the
      derivative-energy strip is too low at full-run scale (very slow
      transition, very long run). Fix: render lever (a longer timescale in
      the fill render's salience scales) and/or data lever (more warp
      exposure for the offender tier).
  (b) SUSPECT LABEL — there is no salience anywhere near the labeled POI5
      (or strong salience somewhere else). The model may be right and the
      ground truth wrong. Fix: label review; these runs poison training.
  (c) MODEL BLIND — the ridge is plainly there (salience comparable to
      POI3/POI4's) and the model still under-counts. Fix: training-side
      (loss/sampling), the only case where retraining without new
      data/labels is justified.

For every offender run this script prints a salience report — the peak
derivative-energy at each labeled POI as a percentile of the run's whole
energy trace, plus its ratio to the trace median — and writes an annotated
full-run v2 render (POI ground truth overlaid on the strips) so the
(a)/(b)/(c) call takes seconds per run.

It also settles a cheap question with the audit's saved probabilities:
whether the ordinal-tail decision rule (what QModelV7FillClassifier.predict
uses at analysis time) beats raw argmax on the val split at the fitted
temperature — i.e., how many of the low-confidence full-run misses the
decision rule already softens without any retraining.

Usage
-----
    python -m src.systems.qmodel_7_onyx.qa.triage_offenders --raw-root data/raw \
        --misses artifacts/audit_fill/misses.csv \
        [--probs artifacts/audit_fill/val_probs.npz] [--tiers configs/tiers.json] \
        [--temperature 1.45] [--min-misses 3] [--out artifacts/triage_fill]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

from .. import paths
from ..augmentation import dynamic_box_width_sec
from ..corpus import dedupe_runs, discover_runs
from ..rendering.detector_render import derivative_energy
from ..rendering.fill_render import (
    FILL_GEN_H,
    FILL_GEN_W,
    generate_fill_cls_v2,
    step_coincidence_energy,
)
from ..rendering.legacy_dataprocessor import QModelV6YOLO_DataProcessor as DP

LOG = get_logger("qmodel_7_onyx.qa.triage_offenders")

COL_TIME = "Relative_time"
CLASS_NAMES = ["no_fill", "initial_fill", "1ch", "2ch", "3ch"]
POI_ORDER = ["POI1", "POI2", "POI3", "POI4", "POI5"]

# Full-run k per final class under the pre-'f'-tag builder (2 uniform + 1
# hard per achievable state, states in ascending order, full-run appended):
FULLRUN_K = {"no_fill": 2, "initial_fill": 5, "1ch": 8, "2ch": 11, "3ch": 14}


def salience_report(df_p: pd.DataFrame, poi: Dict[str, float]) -> Dict[str, dict]:
    """Peak salience around each labeled POI under BOTH energies — v2
    (curvature derivative-energy, what the current weights were trained
    on) and v3 (step-coincidence, the candidate fix) — as percentile
    within the run's whole trace + ratio to the trace median.

    Verdict calibration note: the first triage pass used v2-only salience
    with an 'ABSENT => suspect label' rule, which over-fired on ~20/25
    offenders — too many to be label errors; it was indicting the render.
    The recalibrated rule: a POI is a LABEL SUSPECT only if it stays
    absent under v3 as well. A POI absent under v2 but clear under v3 is
    the render defect the v3 rebuild fixes; the v2->v3 delta on the
    offender list IS the pre-retrain validation of the render change."""
    t = pd.to_numeric(df_p[COL_TIME], errors="coerce").to_numpy(dtype=float)
    energies = {"v2": derivative_energy(df_p), "v3": step_coincidence_energy(df_p)}
    meds = {k: (float(np.median(e)) or 1e-9) for k, e in energies.items()}
    out: Dict[str, dict] = {}
    for name in POI_ORDER:
        pt = poi.get(name)
        if pt is None:
            continue
        w = max(1.0, 0.75 * dynamic_box_width_sec(df_p, pt))
        m = (t >= pt - w) & (t <= pt + w)
        if not m.any():
            out[name] = dict(peak=None)
            continue
        rec = dict(window_s=float(2 * w))
        for k, e in energies.items():
            peak = float(e[m].max())
            rec[f"{k}_pctile"] = float((e < peak).mean() * 100.0)
            rec[f"{k}_vs_median"] = peak / meds[k]
        rec["peak"] = rec["v2_vs_median"]
        out[name] = rec
    return out


def annotate_render(df_p: pd.DataFrame, poi: Dict[str, float], path: Path) -> None:
    """Full-run v2 render with ground-truth POI verticals + labels — the
    exact pixels the classifier judged, with the answer key drawn on."""
    img = generate_fill_cls_v2(df_p, FILL_GEN_W, FILL_GEN_H)
    t = pd.to_numeric(df_p[COL_TIME], errors="coerce").to_numpy(dtype=float)
    t0, t1 = float(t[0]), float(t[-1])
    span = max(t1 - t0, 1e-9)
    for name in POI_ORDER:
        pt = poi.get(name)
        if pt is None or not (t0 <= pt <= t1):
            continue
        x = int((pt - t0) / span * (FILL_GEN_W - 1))
        cv2.line(img, (x, 0), (x, FILL_GEN_H - 1), (255, 255, 255), 1)
        cv2.putText(
            img,
            name,
            (min(x + 3, FILL_GEN_W - 60), 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), img)


def decision_rule_check(npz_path: Path, temperature: float) -> None:
    """argmax vs ordinal-tail rule on the audit's saved val probabilities
    at the fitted temperature — quantifies the no-retrain rescue rate."""
    from ..live.fill_live import OrdinalEvidence

    data = np.load(npz_path)
    p, true = data["probs"].astype(float), data["true"].astype(int)
    p = np.clip(p, 1e-12, 1.0)
    if temperature != 1.0:
        p = np.power(p, 1.0 / temperature)
        p /= p.sum(axis=1, keepdims=True)
    argmax = p.argmax(axis=1)
    tail = np.array([OrdinalEvidence.decide(row, OrdinalEvidence.CONF_FORWARD) for row in p])
    print(f"\ndecision-rule check on val probs (T={temperature}):")
    print(f"  argmax        top-1 {np.mean(argmax == true):.4%}")
    print(f"  ordinal tail  top-1 {np.mean(tail == true):.4%}")
    changed = tail != argmax
    fixed = int(np.sum(changed & (tail == true) & (argmax != true)))
    broke = int(np.sum(changed & (argmax == true) & (tail != true)))
    print(f"  rule changed {int(changed.sum())} frames: fixed {fixed}, broke {broke}")
    # ordinal distance: an off-by-1 verdict costs the cascade less than off-by-2
    print(
        f"  mean |ordinal error|: argmax {np.mean(np.abs(argmax - true)):.4f}  "
        f"tail {np.mean(np.abs(tail - true)):.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=paths.DATA_ROOT)
    ap.add_argument(
        "--misses", type=Path, default=paths.ARTIFACTS_ROOT / "audit_fill" / "misses.csv"
    )
    ap.add_argument(
        "--probs",
        type=Path,
        default=paths.ARTIFACTS_ROOT / "audit_fill" / "val_probs.npz",
        help="val_probs.npz from the audit",
    )
    ap.add_argument("--tiers", type=Path, default=paths.TIERS_JSON)
    ap.add_argument("--temperature", type=float, default=1.45)
    ap.add_argument("--min-misses", type=int, default=3)
    ap.add_argument("--out", type=Path, default=paths.ARTIFACTS_ROOT / "triage_fill")
    args = ap.parse_args()

    df = pd.read_csv(args.misses)
    df["run"] = df["run"].astype(str).str.zfill(5)
    df["k"] = df["path"].map(
        lambda p: int(re.search(r"_[uhf](\d+)\.png$", str(p).replace("\\", "/")).group(1))
    )
    df["fullrun"] = df.apply(
        lambda r: r["tag"] == "u" and r["k"] == FULLRUN_K.get(r["true"], -9), axis=1
    )

    n_by_run = df["run"].value_counts()
    offenders = sorted(
        set(n_by_run[n_by_run >= args.min_misses].index) | set(df[df["fullrun"]]["run"])
    )
    LOG.info(
        "{} misses | {} probable full-run (analysis-time) misses "
        "| {} offender runs (>= {} misses or any full-run miss)",
        len(df),
        df["fullrun"].sum(),
        len(offenders),
        args.min_misses,
    )

    tiers = None
    if args.tiers is not None and args.tiers.exists():
        from ..tiers import TierScheme

        tiers = TierScheme.load(args.tiers)

    by_id = {r.run_id: r for r in dedupe_runs(discover_runs(args.raw_root))}
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for rid in offenders:
        rec = by_id.get(rid)
        if rec is None:
            LOG.warning("{}: not found under raw root — skipped", rid)
            continue
        try:
            df_p = DP.preprocess_dataframe(pd.read_csv(rec.csv_path))
        except Exception as exc:
            LOG.warning("{}: {}", rid, exc)
            continue
        if df_p is None or df_p.empty:
            continue
        poi = dict(rec.poi_times)
        rep = salience_report(df_p, poi)
        annotate_render(df_p, poi, args.out / f"{rid}_annotated.png")

        dirs = Counter(f"{m.true}->{m.pred}" for m in df[df["run"] == rid].itertuples())
        tier_lbl = tiers.labels[tiers.tier_of(rec.viscosity_cP)] if tiers else "?"
        span = float(df_p[COL_TIME].iloc[-1] - df_p[COL_TIME].iloc[0])
        print(
            f"\n{rid}  visc {rec.viscosity_cP} cP (tier {tier_lbl})  span {span:.0f}s  "
            f"misses {int(n_by_run.get(rid, 0))} {dict(dirs)}"
            f"{'  [FULL-RUN MISS]' if bool(df[df['run'] == rid]['fullrun'].any()) else ''}"
        )
        for name, r in rep.items():
            if r.get("peak") is None:
                print(f"    {name}: no data in window")
                continue
            # Recalibrated verdicts on the v3 energy; the v2 column shows
            # what the CURRENT weights had to work with.
            if r["v3_vs_median"] >= 2.0:
                verdict = "clear under v3" + (
                    " (render fix recovers it)" if r["v2_vs_median"] < 2.0 else ""
                )
            elif r["v3_vs_median"] >= 1.5:
                verdict = "faint under v3 — data lever (tier warp exposure)"
            else:
                verdict = "ABSENT under BOTH — genuine label suspect"
            print(
                f"    {name}: v2 x{r['v2_vs_median']:5.2f} ({r['v2_pctile']:5.1f} pct) | "
                f"v3 x{r['v3_vs_median']:5.2f} ({r['v3_pctile']:5.1f} pct) | "
                f"win {r['window_s']:.1f}s  -> {verdict}"
            )
            rows.append(
                dict(
                    run=rid,
                    poi=name,
                    v2_pctile=r["v2_pctile"],
                    v2_vs_median=r["v2_vs_median"],
                    v3_pctile=r["v3_pctile"],
                    v3_vs_median=r["v3_vs_median"],
                    window_s=r["window_s"],
                    tier=tier_lbl,
                    viscosity_cP=rec.viscosity_cP,
                    span_s=span,
                )
            )

    pd.DataFrame(rows).to_csv(args.out / "salience.csv", index=False)
    LOG.info("annotated renders + salience.csv -> {}", args.out)

    if args.probs is not None and args.probs.exists():
        decision_rule_check(args.probs, args.temperature)

    print(
        "\nreading the verdicts: the v2->v3 delta on this offender list is the pre-retrain "
        "validation of the render fix — 'clear under v3' on previously-missed POIs means the "
        "v3 rebuild+retrain should recover them; 'faint under v3' points at offender-tier "
        "warp exposure; 'ABSENT under BOTH' is now a genuine label suspect worth reviewing "
        "with the lab before the next build. Independently, the zoom cross-check "
        "(inference/crosscheck.py) covers analysis-time under-counts with the current weights."
    )


if __name__ == "__main__":
    main()

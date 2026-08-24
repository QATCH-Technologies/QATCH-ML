"""
sweep.py
========

Offline decode-hyperparameter sweep over candidate pools dumped by
``qa/benchmark.py --dump-candidates``.

The YOLO harvest is the expensive part; the decode is ~2 ms. So tuning
DECODE_LAMBDA / DECODE_MIN_MARGIN / frac_blend by re-running the cascade is
wasteful - dump the pools once, then sweep the whole corpus per setting in
seconds here.

For every (lambda, margin, frac_blend) combination this re-decodes each run,
applies the same accept-margin rule the controller uses, and scores against
ground truth. The ranking objective is regression-averse by default
(production posture: first do no harm):

    1. fewest gross failures INTRODUCED vs the cascade
    2. most gross failures FIXED
    3. lowest total decoded MAE

Outputs:
  * printed ranked table
  * ``sweep_results.csv`` with the full grid

Usage
-----
    python -m src.systems.qmodel_7_onyx.decode.sweep \
        --candidates artifacts/benchmark_decode/candidates.jsonl \
        --prior configs/spacing_prior.json \
        [--lambdas 0.25 0.5 1 2 4] [--margins 0 0.25 0.5 1 2] \
        [--blends 0.5] [--gross-threshold 2.0]
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

from .. import paths
from .dp_decode import Candidate, dp_decode, score_configuration
from .spacing_prior import POI_ORDER, SpacingPrior

LOG = get_logger("qmodel_7_onyx.decode.sweep")


def load_dump(path: Path) -> List[Dict[str, Any]]:
    import json

    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


TIER_EDGES_DEFAULT = [2.66, 6.16, 18.14, 73.4]


def _tier_of(cp, edges) -> int:
    if cp is None or not np.isfinite(cp):
        return len(edges) + 1
    for i, e in enumerate(edges):
        if cp < e:
            return i
    return len(edges)


def tier_weights(rows: List[Dict[str, Any]], edges) -> Dict[int, float]:
    """Inverse-frequency weights per viscosity tier, mean-normalized to 1.
    An unweighted objective is dominated by the easy low-viscosity bulk
    (~60% of the corpus under 6 cP) and will happily trade the rare
    high-viscosity tier - the product's actual hard case - for marginal
    bulk gains. Tier weighting makes each tier count equally."""
    from collections import Counter

    counts = Counter(_tier_of(r.get("viscosity_cP"), edges) for r in rows)
    n_tiers = len(counts)
    total = sum(counts.values())
    return {t: total / (n_tiers * c) for t, c in counts.items()}


def evaluate(
    rows: List[Dict[str, Any]],
    prior: SpacingPrior,
    lam: float,
    margin: float,
    gross_threshold: float,
    weights: Optional[Dict[int, float]] = None,
    edges=None,
) -> Dict[str, Any]:
    """Decode every run at (lam, margin) and aggregate paired stats vs the
    cascade picks recorded in the dump. When `weights` is given, also
    accumulates tier-weighted gross count and MAE."""
    abs_errs: List[float] = []
    n_gross_decoded = n_gross_cascade = fixed = introduced = 0
    n_pairs = 0
    w_gross = 0.0
    w_err_sum = 0.0
    w_sum = 0.0

    for row in rows:
        truth: Dict[str, float] = row["truth"]
        present: List[str] = row.get("present") or [p for p in POI_ORDER if p in truth]
        pools = {
            name: [Candidate(time=c["time"], conf=c["conf"]) for c in lst]
            for name, lst in (row.get("pools") or {}).items()
        }
        cascade_rec = row.get("cascade") or {}
        cascade = {
            name: Candidate(time=rec["time"], conf=rec["conf"])
            for name, rec in cascade_rec.items()
            if name in present
        }
        if not pools:
            continue

        result = dp_decode(pools, present, prior, lam=lam)
        chosen = result.chosen
        # same accept-margin rule as QModelOnyx._decode_with_prior
        if margin > 0 and cascade and chosen and set(chosen.keys()) == set(cascade.keys()):
            if result.total_score < score_configuration(cascade, prior, lam=lam) + margin:
                chosen = cascade

        w = 1.0
        if weights is not None:
            w = weights.get(_tier_of(row.get("viscosity_cP"), edges), 1.0)
        for poi, true_t in truth.items():
            d = chosen.get(poi)
            g = cascade.get(poi)
            if d is None or g is None:
                continue
            ed, eg = abs(d.time - true_t), abs(g.time - true_t)
            abs_errs.append(ed)
            n_pairs += 1
            w_err_sum += w * ed
            w_sum += w
            dg, gg = ed > gross_threshold, eg > gross_threshold
            if dg:
                w_gross += w
            n_gross_decoded += int(dg)
            n_gross_cascade += int(gg)
            if gg and not dg:
                fixed += 1
            if dg and not gg:
                introduced += 1

    return dict(
        n=n_pairs,
        w_gross=float(w_gross),
        w_mae_s=float(w_err_sum / w_sum) if w_sum else float("nan"),
        mae_decoded_s=float(np.mean(abs_errs)) if abs_errs else float("nan"),
        median_ae_decoded_s=float(np.median(abs_errs)) if abs_errs else float("nan"),
        gross_decoded=n_gross_decoded,
        gross_cascade=n_gross_cascade,
        gross_fixed=fixed,
        gross_introduced=introduced,
        net_gross_improvement=fixed - introduced,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--candidates",
        type=Path,
        default=paths.ARTIFACTS_ROOT / "benchmark_decode" / "candidates.jsonl",
    )
    ap.add_argument("--prior", type=Path, default=paths.SPACING_PRIOR_JSON)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--blends", type=float, nargs="+", default=[0.5])
    ap.add_argument(
        "--edge3-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="scale factors applied to lam on the edges touching POI3 "
        "(POI2->POI3 and POI3->POI4); sweeps whether the sharp ch1 event "
        "should be partially freed from the broad gap prior",
    )
    ap.add_argument("--gross-threshold", type=float, default=2.0)
    ap.add_argument(
        "--tiers",
        type=Path,
        default=paths.TIERS_JSON,
        help="TierScheme json for tier-weighted objective (falls back to defaults)",
    )
    ap.add_argument(
        "--out", type=Path, default=paths.ARTIFACTS_ROOT / "benchmark_decode" / "sweep_results.csv"
    )
    args = ap.parse_args()

    rows = load_dump(args.candidates)
    LOG.info("Loaded {} runs from {}", len(rows), args.candidates)
    base_prior = SpacingPrior.load(args.prior)
    edges = TIER_EDGES_DEFAULT
    if args.tiers and Path(args.tiers).exists():
        import json as _json

        edges = _json.loads(Path(args.tiers).read_text())["edges_cp"]
    weights = tier_weights(rows, edges)
    LOG.info(
        "tier weights (inverse frequency): {}",
        {k: round(v, 2) for k, v in sorted(weights.items())},
    )

    results = []
    n_combo = len(args.lambdas) * len(args.margins) * len(args.blends) * len(args.edge3_scales)
    i = 0
    for blend in args.blends:
        prior = deepcopy(base_prior)
        prior.frac_blend = blend
        for lam in args.lambdas:
            for margin in args.margins:
                for e3 in args.edge3_scales:
                    i += 1
                    lam_eff = (
                        lam
                        if e3 == 1.0
                        else {p: lam * (e3 if "POI3" in p else 1.0) for p in prior.pairs}
                    )
                    stats = evaluate(
                        rows,
                        prior,
                        lam_eff,
                        margin,
                        args.gross_threshold,
                        weights=weights,
                        edges=edges,
                    )
                    results.append(
                        dict(frac_blend=blend, lam=lam, margin=margin, edge3_scale=e3, **stats)
                    )
                LOG.info(
                    "[{:>3}/{}] blend={:<4} lam={:<5} margin={:<5} "
                    "MAE={:.3f}s  gross={} (cascade {})  "
                    "fixed={} introduced={} net={:+d}",
                    i,
                    n_combo,
                    blend,
                    lam,
                    margin,
                    stats["mae_decoded_s"],
                    stats["gross_decoded"],
                    stats["gross_cascade"],
                    stats["gross_fixed"],
                    stats["gross_introduced"],
                    stats["net_gross_improvement"],
                )

    df = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    cols = [
        "frac_blend",
        "lam",
        "margin",
        "edge3_scale",
        "mae_decoded_s",
        "gross_decoded",
        "gross_fixed",
        "gross_introduced",
        "net_gross_improvement",
        "w_gross",
        "w_mae_s",
    ]
    # Three objectives, three postures. There is no single "recommended"
    # setting: the conservative ranking minimizes regressions on previously
    # working runs (introduced) and will always favour large margins where
    # the decode barely acts; the gross ranking minimizes TOTAL failures,
    # treating a fixed run and a broken run as equal and opposite; the mae
    # ranking optimizes average accuracy. Pick by production cost model: if
    # a regression costs users about what a fix gains them, use "gross".
    rankings = {
        "conservative (fewest regressions)": df.sort_values(
            ["gross_introduced", "net_gross_improvement", "mae_decoded_s"],
            ascending=[True, False, True],
        ),
        "gross (fewest total failures)": df.sort_values(
            ["gross_decoded", "mae_decoded_s"], ascending=[True, True]
        ),
        "mae (best average accuracy)": df.sort_values("mae_decoded_s"),
        "tier-weighted (balanced across viscosity)": df.sort_values(
            ["w_gross", "w_mae_s"], ascending=[True, True]
        ),
    }
    for label, rk in rankings.items():
        print(f"\nTop settings - {label}:")
        print(rk[cols].head(3).to_string(index=False))
        b = rk.iloc[0]
        print(
            f"  -> DECODE_LAMBDA={b.lam}  DECODE_MIN_MARGIN={b.margin}  frac_blend={b.frac_blend}"
        )
    ga, gc = (
        rankings["gross (fewest total failures)"].iloc[0],
        rankings["conservative (fewest regressions)"].iloc[0],
    )
    if ga.gross_decoded < gc.gross_decoded:
        print(
            f"\nTrade-off between the two extremes: the gross-optimal setting has "
            f"{int(gc.gross_decoded - ga.gross_decoded)} fewer total failures than the "
            f"conservative one, at the cost of {int(ga.gross_introduced - gc.gross_introduced)} "
            f"more regressions on previously-working runs "
            f"(while fixing {int(ga.gross_fixed - gc.gross_fixed)} more broken ones)."
        )
    LOG.info("Full grid -> {}", args.out)


if __name__ == "__main__":
    main()

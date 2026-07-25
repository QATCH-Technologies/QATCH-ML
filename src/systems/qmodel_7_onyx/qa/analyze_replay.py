"""
analyze_replay.py
=================

Decomposes replay_fill.json to answer the question the first replay
raised: live final-state correctness (89.9%) sits ~7 points below the
model's own static full-run accuracy (~96.9%) — WHERE do the extra ~38
runs go? The split that matters:

  * MODEL-LOSS: runs whose full-run frame the model gets wrong statically
    (the persistent offender set). No evidence config recovers these.
  * MACHINERY-LOSS: runs the model gets right statically but the evidence
    layer never confirms — missed confirmations from bars tuned too
    conservative (48 missed vs 11 false forwards is a conservative
    machine's signature). These are free to recover with a threshold
    change, which is what `live/replay.py --sweep` quantifies.

Reads one or more machine configs from the replay JSON (run with --sweep
and/or --legacy for a comparison table), optionally cross-references the
static audit's misses.csv to tag failing runs as known-static-offenders
vs machinery-only.

Usage
-----
    python -m src.systems.qmodel_7_onyx.qa.analyze_replay \
        --replay artifacts/replay/replay_fill.json \
        [--misses artifacts/audit_fill/misses.csv] [--machine ordinal]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import paths

CLASS_NAMES = ["no_fill", "initial_fill", "1ch", "2ch", "3ch"]


def summarize_machine(name: str, per_run: dict) -> dict:
    missed_by_state = Counter()
    failing, false_fwd_runs, backward_runs = [], [], []
    lat_by_state = defaultdict(list)
    for rid, rec in per_run.items():
        s = rec["scores"].get(name)
        if s is None:
            continue
        for cname, v in s["latencies"].items():
            if v is None:
                missed_by_state[cname] += 1
            else:
                lat_by_state[cname].append(v)
        if not s["final_correct"]:
            failing.append(rid)
        if s["false_forward"]:
            false_fwd_runs.append(rid)
        if s["backward"]:
            backward_runs.append(rid)
    return dict(
        missed_by_state=dict(missed_by_state),
        failing=sorted(failing),
        false_fwd_runs=sorted(false_fwd_runs),
        backward_runs=sorted(backward_runs),
        lat_by_state={k: np.array(v) for k, v in lat_by_state.items()},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--replay", type=Path, default=paths.ARTIFACTS_ROOT / "replay" / "replay_fill.json"
    )
    ap.add_argument("--misses", type=Path, default=None, help="static audit misses.csv")
    ap.add_argument("--machine", default="ordinal", help="machine for the deep dive")
    args = ap.parse_args()

    data = json.loads(args.replay.read_text())
    per_run = data["per_run"]
    n_runs = len(per_run)
    machines = sorted({m for rec in per_run.values() for m in rec["scores"]})
    print(f"{n_runs} runs | machines: {machines}\n")

    static_offenders = set()
    static_fullrun_offenders = set()
    if args.misses is not None and args.misses.exists():
        m = pd.read_csv(args.misses)
        m["run"] = m["run"].astype(str).str.zfill(5)
        static_offenders = set(m["run"])
        if "tag" in m.columns and (m["tag"] == "f").any():
            static_fullrun_offenders = set(m.loc[m["tag"] == "f", "run"])

    # ---- comparison table across configs (the sweep decision table) ----
    print(
        f"{'machine':18s} {'final✓':>7s} {'missed':>7s} {'falseF':>7s} {'bwd':>4s} "
        f"{'3ch med/p90 lat':>16s}"
    )
    summaries = {}
    for name in machines:
        s = summarize_machine(name, per_run)
        summaries[name] = s
        n_fail = len(s["failing"])
        lat3 = s["lat_by_state"].get("3ch", np.array([]))
        lat_str = (
            f"{np.median(lat3):5.1f}/{np.percentile(lat3, 90):5.1f}s" if len(lat3) else "  n/a"
        )
        missed = sum(s["missed_by_state"].values())
        print(
            f"{name:18s} {100 * (1 - n_fail / n_runs):6.1f}% {missed:7d} "
            f"{len(s['false_fwd_runs']):7d} {len(s['backward_runs']):4d} {lat_str:>16s}"
        )

    # ---- deep dive on the chosen machine ----
    s = summaries[args.machine]
    print(f"\n=== {args.machine}: decomposition ===")
    print("missed confirmations by state:", s["missed_by_state"])
    if s["false_fwd_runs"]:
        print(
            f"false-forward runs ({len(s['false_fwd_runs'])}): {s['false_fwd_runs']}\n"
            "  (cross-reference with the static over-count offenders — if they match,\n"
            "   these are model errors for the label-review pile, not machinery errors)"
        )
    if s["backward_runs"]:
        print(f"backward-revision runs ({len(s['backward_runs'])}): {s['backward_runs']}")
    fail = set(s["failing"])
    if static_offenders:
        model_loss = sorted(fail & (static_fullrun_offenders or static_offenders))
        machinery_loss = sorted(fail - static_offenders)
        boundary = sorted(fail & static_offenders - set(model_loss))
        print(f"\nfailing runs: {len(fail)}")
        print(f"  MODEL-LOSS (known static full-run offenders):   {len(model_loss)}  {model_loss}")
        if boundary:
            print(f"  static-offender (prefix-only misses):          {len(boundary)}  {boundary}")
        print(f"  MACHINERY-LOSS (statically clean, never confirmed): {len(machinery_loss)}")
        print(f"    {machinery_loss}")
        print(
            "\nMACHINERY-LOSS runs are recoverable by threshold alone — compare their count "
            "across the sweep rows above; the config that zeroes them without inflating "
            "falseF is the shipping config."
        )
    else:
        print(f"\nfailing runs ({len(fail)}): {sorted(fail)}")
        print("(pass --misses to split model-loss vs machinery-loss)")

    # latency tails: which runs confirm absurdly late (the 30 s maxes)
    for cname in ("1ch", "2ch", "3ch"):
        arr = s["lat_by_state"].get(cname, np.array([]))
        if len(arr) and arr.max() > 10:
            slow = [
                (rid, rec["scores"][args.machine]["latencies"].get(cname))
                for rid, rec in per_run.items()
                if (rec["scores"].get(args.machine, {}).get("latencies", {}).get(cname) or 0) > 10
            ]
            slow.sort(key=lambda kv: -kv[1])
            print(
                f"\n{cname} confirmations >10 s ({len(slow)}):",
                [(r, round(v, 1)) for r, v in slow[:10]],
            )


if __name__ == "__main__":
    main()

"""
replay.py
=========

Streaming replay benchmark for the fill classifier - the shipping gate the
static confusion matrix cannot provide. 97.8% frame accuracy says nothing
about the two quantities the live classifier is actually judged on:

  * CONFIRMATION LATENCY - seconds from each ground-truth transition
    (POI1/3/4/5) to the moment the state machine confirms the new state.
    Every second here is a second of delayed operator feedback, and the
    duration-threshold messages ("Data Ready, You Can Stop") key off it.
  * STABILITY - false forward confirmations (a state whose boundary never
    happened), backward flickers, and final-state correctness (the verdict
    that steers the detector cascade at analysis time).

Held-out VAL runs (from the dataset manifest, so the split is identical to
training) are replayed chunk-by-chunk through the exact live decision
stack - preprocess_for_cls -> QModelV7FillClassifier.predict_probs (with
PROB_TEMPERATURE) -> OrdinalEvidence - without the QATCH process plumbing.
Drop gating is emulated with the drop signal at POI1 (the UI fires it at
drop application; POI1 is the closest ground-truth proxy).

`--legacy` additionally runs the SAME probability stream through the v6
decision rule (argmax + symmetric count-of-3 debounce) so the evidence
layer's contribution is isolated from the model's: same model, same
frames, two state machines, one table.

Per-inference wall time is reported for the first vs last quartile of each
run - the flat-cost claim of preprocess_for_cls, verified on real lengths.

Usage
-----
    python -m src.systems.qmodel_7_onyx.live.replay --raw-root data/raw \
        --manifest datasets/v7_fill/manifest.json \
        --weights runs/v7_fill/fill_yolo26s/weights/best.pt \
        [--chunk-s 1.0] [--legacy] [--no-gate] [--limit N] \
        [--out artifacts/replay/replay_fill.json]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

from .. import paths
from ..corpus import dedupe_runs, discover_runs
from .fill_live import OrdinalEvidence, QModelV7FillClassifier, preprocess_for_cls

LOG = get_logger("qmodel_7_onyx.live.replay")

COL_TIME = "Relative_time"
COL_DISS = "Dissipation"
COL_FREQ = "Resonance_Frequency"

CLASS_NAMES = ["no_fill", "initial_fill", "1ch", "2ch", "3ch"]
BOUNDARY_POI = {1: "POI1", 2: "POI3", 3: "POI4", 4: "POI5"}

BASELINE_START, BASELINE_END = 0.5, 2.5  # matches DP / live baseline window


class LegacyDebounce:
    """The v6 rule, reproduced exactly: raw argmax, accept any state change
    (forward OR backward) after 3 identical consecutive predictions."""

    THRESHOLD = 3

    def __init__(self) -> None:
        self.candidate = -99
        self.count = 0

    def update(self, p: np.ndarray, current_state: int) -> int:
        pred = int(np.argmax(p))
        if pred == self.candidate:
            self.count += 1
        else:
            self.candidate = pred
            self.count = 1
        return pred if self.count >= self.THRESHOLD else current_state


def replay_run(
    df_raw: pd.DataFrame,
    poi: Dict[str, float],
    probs_fn: Callable[[pd.DataFrame], Optional[np.ndarray]],
    chunk_s: float,
    gate: bool,
    machines: Dict[str, object],
) -> Optional[dict]:
    """Streams one run through every state machine in `machines` on a
    shared probability stream. Returns per-machine transition logs plus
    inference timing."""
    df_raw = df_raw.sort_values(COL_TIME).reset_index(drop=True)
    t_all = pd.to_numeric(df_raw[COL_TIME], errors="coerce").to_numpy(dtype=float)
    t0, t1 = float(np.nanmin(t_all)), float(np.nanmax(t_all))
    if not np.isfinite(t0) or t1 - t0 < 5.0:
        return None

    drop_t = poi.get("POI1") if gate else None

    states = {name: 0 for name in machines}  # ordinal index; 0 == no_fill
    logs: Dict[str, List[dict]] = {name: [] for name in machines}
    base_f = base_d = None
    infer_ms: List[float] = []

    edges = np.arange(t0 + chunk_s, t1 + chunk_s, chunk_s)
    for t_edge in edges:
        buf = df_raw[t_all < t_edge]
        if len(buf) < 20:
            continue
        now = float(buf[COL_TIME].iloc[-1])

        if base_f is None:
            m = (buf[COL_TIME] >= BASELINE_START) & (buf[COL_TIME] <= BASELINE_END)
            if m.sum() >= 10:
                base_f = float(buf.loc[m, COL_FREQ].mean())
                base_d = float(buf.loc[m, COL_DISS].mean())

        tic = time.perf_counter()
        proc = preprocess_for_cls(
            buf[buf[COL_TIME] > 0.05], baseline_freq=base_f, baseline_diss=base_d
        )
        if proc is None or proc.empty:
            continue
        p = probs_fn(proc)
        infer_ms.append((time.perf_counter() - tic) * 1e3)
        if p is None:
            continue

        gated = drop_t is not None and now < drop_t
        for name, machine in machines.items():
            if gated:
                # pre-drop: state pinned to no_fill; ordinal evidence also
                # holds its accumulator in reset (mirrors the live class).
                if isinstance(machine, OrdinalEvidence):
                    machine.reset()
                states[name] = 0
                continue
            new = machine.update(p, states[name])
            if new != states[name]:
                logs[name].append(dict(t=now, frm=states[name], to=new))
                states[name] = new

    return dict(logs=logs, final={n: s for n, s in states.items()}, infer_ms=infer_ms, t_end=t1)


def score_run(poi: Dict[str, float], log: List[dict], final_state: int) -> dict:
    """Latency per true boundary, false forwards, backward flickers, final
    verdict. Latency counts the FIRST confirmation at-or-above the state
    (a jump 1->3 confirms state 2 and 3 at the same instant)."""
    true_final = 0
    bound_t: Dict[int, float] = {}
    for k in (1, 2, 3, 4):
        pt = poi.get(BOUNDARY_POI[k])
        if pt is not None:
            bound_t[k] = float(pt)
            true_final = k

    first_at_or_above: Dict[int, float] = {}
    false_fwd = 0
    backward = 0
    for ev in log:
        if ev["to"] > ev["frm"]:
            for k in range(ev["frm"] + 1, ev["to"] + 1):
                first_at_or_above.setdefault(k, ev["t"])
                if k not in bound_t:
                    false_fwd += 1
        else:
            backward += 1

    latencies = {
        k: (first_at_or_above[k] - bt if k in first_at_or_above else None)
        for k, bt in bound_t.items()
    }
    return dict(
        latencies=latencies,
        missed=[k for k, v in latencies.items() if v is None],
        false_forward=false_fwd,
        backward=backward,
        final_correct=final_state == true_final,
        true_final=true_final,
    )


def aggregate(name: str, scored: List[dict]) -> dict:
    lat_by_state: Dict[int, List[float]] = defaultdict(list)
    for s in scored:
        for k, v in s["latencies"].items():
            if v is not None:
                lat_by_state[k].append(v)
    out = dict(
        n_runs=len(scored),
        final_correct=float(np.mean([s["final_correct"] for s in scored])),
        false_forward_total=int(sum(s["false_forward"] for s in scored)),
        backward_total=int(sum(s["backward"] for s in scored)),
        missed_confirms=int(sum(len(s["missed"]) for s in scored)),
        latency={},
    )
    print(f"\n=== {name} ===")
    print(
        f"final-state correct: {out['final_correct']:.1%} | "
        f"false forwards: {out['false_forward_total']} | "
        f"backward moves: {out['backward_total']} | "
        f"missed confirmations: {out['missed_confirms']}"
    )
    for k in sorted(lat_by_state):
        arr = np.array(lat_by_state[k])
        out["latency"][CLASS_NAMES[k]] = dict(
            n=len(arr),
            median=float(np.median(arr)),
            p90=float(np.percentile(arr, 90)),
            max=float(arr.max()),
        )
        print(
            f"  {CLASS_NAMES[k]:12s} latency  median {np.median(arr):6.1f} s   "
            f"p90 {np.percentile(arr, 90):6.1f} s   max {arr.max():6.1f} s   (n={len(arr)})"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=paths.DATA_ROOT)
    ap.add_argument(
        "--manifest", type=Path, default=paths.DATASETS_ROOT / "v7_fill" / "manifest.json"
    )
    ap.add_argument(
        "--weights",
        type=Path,
        default=paths.RUNS_ROOT / "v7_fill" / "fill_yolo26s" / "weights" / "best.pt",
    )
    ap.add_argument("--chunk-s", type=float, default=1.0)
    ap.add_argument("--legacy", action="store_true", help="also run the v6 argmax+count-3 rule")
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="run a grid of OrdinalEvidence configs over the SAME probability "
        "stream (zero extra inference): terminal 3ch bar x forward bar. The "
        "48-missed-confirmations vs 11-false-forwards imbalance in the first "
        "replay is exactly what this decides: how much of the live/static "
        "final-state gap is the 0.75 terminal bar vs the model.",
    )
    ap.add_argument("--no-gate", action="store_true", help="disable drop-signal gating")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--out", type=Path, default=paths.ARTIFACTS_ROOT / "replay" / "replay_fill.json"
    )
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    val_ids = set(manifest["val_ids"])
    runs = [r for r in dedupe_runs(discover_runs(args.raw_root)) if r.run_id in val_ids]
    if args.limit:
        runs = runs[: args.limit]
    if not runs:
        raise SystemExit("no val runs matched the manifest")
    LOG.info(
        "replaying {} held-out runs | chunk {}s | gate={}",
        len(runs),
        args.chunk_s,
        not args.no_gate,
    )

    clf = QModelV7FillClassifier(str(args.weights))
    probs_fn = clf.predict_probs

    def make_machines() -> Dict[str, object]:
        machines: Dict[str, object] = {"ordinal": OrdinalEvidence()}
        if True:
            # Terminal-bar column (the false-stop vs missed-3ch trade) and a
            # forward-bar row (overall conservatism). 'ordinal' above is the
            # shipped default (fwd 0.60 / term 0.75) for reference.
            for term in (0.60, 0.65, 0.70):
                machines[f"term{term:.2f}"] = OrdinalEvidence(conf_forward_per_state={4: term})
            machines["fwd0.55_term0.65"] = OrdinalEvidence(
                conf_forward=0.55, conf_forward_per_state={4: 0.65}
            )
        if args.legacy:
            machines["legacy"] = LegacyDebounce()
        return machines

    scored: Dict[str, List[dict]] = defaultdict(list)
    per_run: Dict[str, dict] = {}
    early_ms: List[float] = []
    late_ms: List[float] = []

    for i, rec in enumerate(runs):
        try:
            df_raw = pd.read_csv(rec.csv_path)
        except Exception as exc:
            LOG.warning("skip {}: {}", rec.run_id, exc)
            continue
        machines = make_machines()
        res = replay_run(
            df_raw, dict(rec.poi_times), probs_fn, args.chunk_s, not args.no_gate, machines
        )
        if res is None:
            continue
        run_scores = {}
        for name in machines:
            s = score_run(dict(rec.poi_times), res["logs"][name], res["final"][name])
            scored[name].append(s)
            run_scores[name] = s
        per_run[rec.run_id] = dict(
            scores={
                n: dict(
                    latencies={CLASS_NAMES[k]: v for k, v in s["latencies"].items()},
                    false_forward=s["false_forward"],
                    backward=s["backward"],
                    final_correct=s["final_correct"],
                )
                for n, s in run_scores.items()
            },
            transitions={n: res["logs"][n] for n in machines},
        )
        ms = res["infer_ms"]
        if len(ms) >= 8:
            q = len(ms) // 4
            early_ms.extend(ms[:q])
            late_ms.extend(ms[-q:])
        if (i + 1) % 10 == 0:
            LOG.info("{}/{}", i + 1, len(runs))

    summary = {name: aggregate(name, sc) for name, sc in scored.items()}

    if early_ms and late_ms:
        print(
            f"\nper-inference cost (preprocess+predict): "
            f"first quartile of runs {np.mean(early_ms):.1f} ms, "
            f"last quartile {np.mean(late_ms):.1f} ms "
            f"(flat-cost check: these should be comparable)"
        )
        summary["infer_ms"] = dict(early=float(np.mean(early_ms)), late=float(np.mean(late_ms)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                weights=str(args.weights),
                chunk_s=args.chunk_s,
                gate=not args.no_gate,
                summary=summary,
                per_run=per_run,
            ),
            indent=2,
        )
    )
    LOG.info("full per-run detail -> {}", args.out)


if __name__ == "__main__":
    main()

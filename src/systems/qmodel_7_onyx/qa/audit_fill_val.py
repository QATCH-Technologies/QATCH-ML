"""Post-training triage for the fill classifier's validation misses.

Val-split misses concentrate on adjacent ordinal boundaries; this script
answers the question that decides what (if anything) to do about them:
WHICH frames are missing, and from which population. It runs inference
over the val split and writes per-miss detail for offline review.

The three populations have very different costs:

  * 'f' (full-run, analysis-time) misses are the expensive ones: they
    steer the detector cascade toward the wrong number of cuts. Any
    concentration here is the priority.
  * 'h' (hard-band, just-after-transition) misses are largely expected
    and cheap: they are the frames the live evidence layer exists to
    ride out, and the ridge is by construction barely formed.
  * 'u' (uniform prefix) misses in the middle of a state interval are the
    interesting residual - if they cluster on specific runs/tiers (e.g.
    high-viscosity late flattening), that is a data/augmentation lever,
    not a model lever.

Also fits the probability TEMPERATURE the live evidence layer needs: a
classifier trained to saturation produces raw probabilities that carry
no marginal-vs-certain signal (everything reads near-1.0 confident), so
T is fit by minimizing NLL of p^(1/T) on the val split; write the
result into `live.fill_live.PROB_TEMPERATURE`.

Outputs
-------
  * console: per-tag / per-class / per-direction breakdown, top offender
    runs, fitted temperature
  * <out>/misses.csv: one row per miss (path, run, tag, true, pred, conf)
  * <out>/misses/: the missed images copied for eyeballing

Usage
-----
    python -m src.systems.qmodel_7_onyx.qa.audit_fill_val --data-root datasets/v7_fill \
        --weights runs/v7_fill/fill_yolo26s/weights/best.pt \
        [--out artifacts/audit_fill] [--device 0]
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils.logger import get_logger

from .. import paths

LOG = get_logger("qmodel_7_onyx.qa.audit_fill_val")

CLASS_NAMES = ["no_fill", "initial_fill", "1ch", "2ch", "3ch"]
ORD = {c: i for i, c in enumerate(CLASS_NAMES)}

# {hash}_{rid}_v{v}_{tag}{k}.png - rid may itself contain underscores.
NAME_RE = re.compile(r"^[0-9a-f]{8}_(?P<rid>.+)_v(?P<var>\d+)_(?P<tag>[uhf])(?P<k>\d+)$")

TAG_LABEL = {
    "f": "full-run (analysis-time)",
    "h": "hard band (just confirmed)",
    "u": "uniform prefix",
}


def parse_name(stem: str):
    """Split a dataset image stem into (run id, variant, cut tag).

    Args:
        stem (str): filename stem in `{hash}_{rid}_v{v}_{tag}{k}` form.

    Returns:
        tuple: `(run_id, variant, tag)`, or `(stem, "?", "?")` if the
            stem does not match the expected naming convention.
    """
    m = NAME_RE.match(stem)
    if not m:
        return stem, "?", "?"
    return m.group("rid"), m.group("var"), m.group("tag")


def fit_temperature(prob_rows: np.ndarray, true_idx: np.ndarray) -> float:
    """Fit a scalar probability temperature by 1-D NLL minimization.

    Minimizes the NLL of `p^(1/T)` (renormalized) over a log-spaced
    grid followed by local refinement. Operates on probabilities rather
    than logits, since logits are not exposed uniformly across model
    backends - this is exactly the space `PROB_TEMPERATURE` acts in.

    Args:
        prob_rows (np.ndarray): per-sample class-probability rows,
            shape `(n, n_classes)`.
        true_idx (np.ndarray): true class index per sample, shape
            `(n,)`.

    Returns:
        tuple[float, float, float]: fitted temperature `T`, the NLL at
            `T=1.0`, and the NLL at the fitted `T`.
    """
    p = np.clip(prob_rows, 1e-12, 1.0)

    def nll(T: float) -> float:
        q = np.power(p, 1.0 / T)
        q /= q.sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(q[np.arange(len(q)), true_idx] + 1e-12)))

    Ts = np.geomspace(0.5, 32.0, 60)
    best = min(Ts, key=nll)
    fine = np.linspace(best / 1.5, best * 1.5, 60)
    best = min(fine, key=nll)
    return float(best), nll(1.0), nll(best)


def main() -> None:
    """CLI entry point: run inference over the val split, print the
    tag/class/direction/offender-run breakdown, fit the probability
    temperature, and write `misses.csv` / `val_probs.npz` / the
    copied miss images to the output directory.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=paths.DATASETS_ROOT / "v7_fill")
    ap.add_argument(
        "--weights",
        type=Path,
        default=paths.RUNS_ROOT / "v7_fill" / "fill_yolo26s" / "weights" / "best.pt",
    )
    ap.add_argument("--out", type=Path, default=paths.ARTIFACTS_ROOT / "audit_fill")
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))

    val_root = args.data_root / "val"
    images: List[Path] = sorted(val_root.glob("*/*.png"))
    if not images:
        raise SystemExit(f"no val images under {val_root}")
    LOG.info("{} val images", len(images))

    args.out.mkdir(parents=True, exist_ok=True)
    miss_dir = args.out / "misses"
    miss_dir.mkdir(exist_ok=True)

    misses = []
    all_probs = np.zeros((len(images), len(CLASS_NAMES)), dtype=float)
    all_true = np.zeros(len(images), dtype=int)

    for lo in range(0, len(images), args.batch):
        batch = images[lo : lo + args.batch]
        results = model([str(p) for p in batch], verbose=False, device=args.device)
        for j, (path, res) in enumerate(zip(batch, results, strict=True)):
            true_c = path.parent.name
            probs = res.probs.data
            probs = probs.cpu().numpy() if hasattr(probs, "cpu") else np.asarray(probs)
            # reorder model-name space -> ordinal space
            vec = np.zeros(len(CLASS_NAMES))
            for idx, label in res.names.items():
                if label in ORD:
                    vec[ORD[label]] += float(probs[idx])
            vec /= max(vec.sum(), 1e-12)
            i = lo + j
            all_probs[i] = vec
            all_true[i] = ORD[true_c]
            pred_c = CLASS_NAMES[int(np.argmax(vec))]
            if pred_c != true_c:
                rid, var, tag = parse_name(path.stem)
                misses.append(
                    dict(
                        path=str(path),
                        run=rid,
                        variant=var,
                        tag=tag,
                        true=true_c,
                        pred=pred_c,
                        conf=float(vec.max()),
                        ordinal_step=ORD[pred_c] - ORD[true_c],
                    )
                )
                shutil.copy2(path, miss_dir / f"{true_c}__as__{pred_c}__{path.name}")
        print(f"  {min(lo + args.batch, len(images))}/{len(images)}", end="\r")
    print()

    acc = 1.0 - len(misses) / len(images)
    print(f"\ntop-1 {acc:.4%} | {len(misses)} misses / {len(images)}")

    by_tag = Counter(m["tag"] for m in misses)
    tag_totals = Counter(parse_name(p.stem)[2] for p in images)
    print("\nby cut type (miss / total, rate):")
    for tag in ("f", "h", "u"):
        n, tot = by_tag.get(tag, 0), tag_totals.get(tag, 0)
        rate = n / tot if tot else 0.0
        print(f"  {tag} {TAG_LABEL[tag]:32s} {n:5d} / {tot:6d}  ({rate:.2%})")

    print("\nby direction (ordinal step, pred - true):")
    for step, n in sorted(Counter(m["ordinal_step"] for m in misses).items()):
        print(f"  {step:+d}: {n}  ({'under' if step < 0 else 'over'}-count by {abs(step)})")

    print("\nby confusion pair:")
    for (t, p), n in Counter((m["true"], m["pred"]) for m in misses).most_common():
        print(f"  {t:12s} -> {p:12s} {n}")

    by_run: Dict[str, int] = defaultdict(int)
    for m in misses:
        by_run[m["run"]] += 1
    print("\ntop offender runs:")
    for rid, n in sorted(by_run.items(), key=lambda kv: -kv[1])[:15]:
        dirs = Counter(f"{m['true']}->{m['pred']}" for m in misses if m["run"] == rid)
        print(f"  {rid}: {n}  {dict(dirs)}")

    sat = float(np.mean(all_probs.max(axis=1) > 0.999))
    T, nll1, nllT = fit_temperature(all_probs, all_true)
    print(f"\nsaturation: {sat:.1%} of val frames predicted with p>0.999")
    print(f"fitted PROB_TEMPERATURE = {T:.2f}  (val NLL {nll1:.4f} -> {nllT:.4f})")
    print("-> set live.fill_live.PROB_TEMPERATURE to this value (ships with the weights).")

    with open(args.out / "misses.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["path", "run", "variant", "tag", "true", "pred", "conf", "ordinal_step"]
        )
        w.writeheader()
        w.writerows(misses)
    np.savez(args.out / "val_probs.npz", probs=all_probs, true=all_true)
    LOG.info("misses.csv, val_probs.npz, missed images -> {}", args.out)


if __name__ == "__main__":
    main()

# QATCH-ML

[![Test](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/test.yml/badge.svg)](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/test.yml)
[![Style](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/style.yml/badge.svg)](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/style.yml)
[![Build](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/build.yml/badge.svg)](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/build.yml)

QATCH sensor-run point-of-interest (POI) detection: a pipeline that turns raw
QATCH dissipation/resonance-frequency runs into a signal-domain image
render, feeds a cascade of YOLO detectors + classifiers, joint-decodes the
result against a learned spacing prior, and serves the whole thing live
during data capture. The active implementation is the **`qmodel_7_onyx`**
system under `src/systems/`.

## Install

```
pip install -e ".[dev]"
```

`requires-python >= 3.10`. Core dependencies (numpy, pandas, opencv-python,
scipy, matplotlib, ultralytics, tqdm) are declared in `pyproject.toml`; the
`dev` extra adds `pytest`/`pytest-cov`, and the `tier-discovery` extra adds
optional `scikit-learn` support for GMM-based viscosity tiering. The default
tiering method (`log_uniform`) needs no extra dependency - it bins in
equal-width log10(cP) steps spanning the corpus's actual min..max, which is
what keeps a high-viscosity tail visible instead of collapsing into one
"N+" bucket the way equal-*count* binning (`quantile`, or `gmm`'s BIC
model selection) does on this corpus's right-skewed distribution. Pick a
different method with `--method {gmm,quantile}` on `tiers.py`'s CLI.

Model weights (`*.pt`) are not committed - `assets_paths.json` in the
package describes where the controller expects them, relative to
`src/systems/qmodel_7_onyx/assets/` (gitignored).

## Quickstart

### Python API (recommended)

`Workspace` + `Pipeline` wrap the whole fresh-data -> fitted-config ->
dataset -> trained-weights sequence behind one object, so a fresh dataset
becomes trained results without hand-chaining CLIs or re-discovering where
each stage wrote its output:

```python
from src.systems.qmodel_7_onyx import Workspace, Pipeline

# Workspace defaults to this repo's data/, datasets/, runs/, configs/ roots -
# override any of them to point at a different raw-data folder or output location.
pipeline = Pipeline(Workspace(data_root="path/to/raw"))

result = pipeline.run()  # prepare() -> build_datasets() -> train()
print(result.training.weights)  # {"init": Path(...), ..., "fill_classifier": Path(...)}
print(result.training.metrics)  # best-effort Ultralytics val metrics per stage
```

Each stage is also callable on its own for finer control, sharing the same
`Workspace` so paths never drift out of sync between stages:

```python
pipeline.prepare()  # fit tiers.json + spacing_prior.json
pipeline.build_datasets(targets=["fill_classifier"])  # just the fill-classifier dataset
pipeline.train(targets=["detectors"], detector_stages=["ch2_zoom"], epochs=50)
```

Errors from insufficient/missing data raise a catchable `PipelineError`
rather than killing the process (`SystemExit`, which is what the underlying
CLI stages use). See `pipeline.py` for the full API and every stage's
keyword arguments.

### CLI (equivalent, one command per stage)

```
# discover a corpus, fit the spacing prior and viscosity tiers
python -m src.systems.qmodel_7_onyx.decode.fit_prior --raw-root data/raw --out configs/spacing_prior.json
python -m src.systems.qmodel_7_onyx.tiers --raw-root data/raw --out configs/tiers.json

# build YOLO datasets
python -m src.systems.qmodel_7_onyx.dataset.build_detectors --raw-root data/raw
python -m src.systems.qmodel_7_onyx.dataset.build_fill_classifier --raw-root data/raw

# train
python -m src.systems.qmodel_7_onyx.training.train_detectors
python -m src.systems.qmodel_7_onyx.training.train_fill_classifier

# evaluate the decode layer against the production cascade (no weights needed)
python -m src.systems.qmodel_7_onyx.qa.benchmark --selftest

# run tests
pytest
```

The same commands are also registered as console scripts after install
(`qmodel-build-detectors`, `qmodel-train-detectors`, `qmodel-benchmark`,
etc. - see `pyproject.toml` for the full list).

### One-command release pipeline

`scripts/build_and_release_qmodel_onyx.py` chains fetch → prepare → build →
train → release → cleanup → eval into a single command - pulls fresh runs
from Dropbox, rebuilds the datasets, trains the requested models, copies
the best checkpoints into a ready-to-ship `qmodel_onyx/` folder, purges
each trained stage's now-redundant Ultralytics run directory, and scores
the deployed package's predicted POI positions against `*_poi.csv` ground
truth (not a YOLO-metrics benchmark - see the script's module docstring).
There is no separate eval script; this one command replaces the old
`build_and_release_qmodel_onyx.py` + `eval_onyx_deployment.py` pair.

```
python scripts/build_and_release_qmodel_onyx.py --dropbox-source "D:/Dropbox/QATCH runs"

# skip Dropbox + the post-release eval, retrain one detector stage
python scripts/build_and_release_qmodel_onyx.py \
    --skip-fetch --skip-eval --targets detectors --detector-stages ch2_zoom
```

Any stage can be skipped or reconfigured - see `--help` for the full set
of `--eval-*`/`--keep-runs`/per-stage flags.

## Directory map

```
src/systems/qmodel_7_onyx/
  pipeline.py          Workspace + Pipeline - the public API (see Quickstart above);
                       re-exported from the package's __init__.py
  paths.py            canonical filesystem roots (data/datasets/runs/configs/artifacts),
                       env-var overridable - every module imports its path defaults from here
  corpus.py            run discovery, ground-truth POI parsing, dedup, viscosity tiering (fixed scheme)
  tiers.py              data-driven viscosity TierScheme (log_uniform by default; GMM/quantile opt-in)
  augmentation.py        signal-domain augmentation (time warp, noise, amplitude jitter) + dynamic box sizing

  rendering/            df -> image render contracts (detector cascade + fill classifier)
  decode/                spacing-prior model + DP joint decode over YOLO candidates
  dataset/                YOLO dataset builders (cascade detectors, fill classifier) + shared split/upsample logic
  training/               YOLO training CLIs
  inference/              the production controller (QModelOnyx), config, and the zoom-detector crosscheck
  deployment/             standalone, self-contained QModelOnyx controller (onyx.py + siblings) shipped
                          under the QATCH.QModel.models.qmodel_onyx.* dotted-import contract - a separate
                          copy from inference/, since this one is loaded exactly as a downstream consumer
                          loads it (see scripts/build_and_release_qmodel_onyx.py's Eval stage)
  live/                   streaming inference (base live class, v7 fill-live upgrade, replay benchmark)
  qa/                     offline QA: A/B decode benchmark, val-set audit, offender triage, label review, replay analysis

src/utils/dataset_fetcher.py   Dropbox -> data/raw ingestion CLI (unrelated to qmodel_7_onyx internals)
scripts/
  build_and_release_qmodel_onyx.py   one-command fetch->prepare->build->train->release->cleanup->eval
                                     pipeline (see "One-command release pipeline" above); also holds the
                                     deployed-package eval that used to be a separate eval_onyx_deployment.py
  _qmodel_onyx_layout.py             shared assets_paths.json-derived deploy-layout helper
configs/                        fitted config artifacts (spacing_prior.json, tiers.json) - versioned, not generated-per-run
tests/                          pytest suite, mirrors the package layout above
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline data-flow, module
responsibilities, and path/config conventions in more depth, and
[CONTRIBUTING.md](CONTRIBUTING.md) for how to add to it.

## Pipeline stages

1. **Corpus** (`corpus.py`) - walks `data/raw/<run_id>/` directories, parses
   the `*_poi.csv` ground truth into chain-space POI times, reads viscosity
   from the run's analyze output, dedupes runs with identical POI content
   (directory-name duplicates would otherwise leak across train/val splits).
2. **Rendering** (`rendering/`) - turns a preprocessed run DataFrame into
   the RGB strip images the detector cascade and fill classifier consume.
   Two salience families: v2 (curvature-based `derivative_energy`) and v3
   (step-coincidence), plus the legacy v1 preprocessing/rendering path.
3. **Dataset building** (`dataset/`) - renders labeled YOLO datasets (4
   cascade + 3 zoom-refinement detector sets, and the 5-class ordinal fill
   classifier set), with leakage-proof run-level stratified splitting and
   per-tier upsampling (`dataset/splitting.py`, shared by both builders).
4. **Training** (`training/`) - thin CLIs around Ultralytics YOLO training,
   with all pixel-space augmentation disabled (it happens upstream, in the
   signal domain, via `augmentation.py`) and a documented rect-training
   workaround for an Ultralytics memory-blowup bug.
5. **Decode** (`decode/`) - `SpacingPrior` (a learned log-normal model of
   gaps between consecutive POIs) and `dp_decode` (an exact DP that picks
   one YOLO candidate per POI, jointly, instead of the greedy per-POI
   argmax the cascade uses by default).
6. **Inference** (`inference/`) - `QModelOnyx`, the production controller:
   loads the fill classifier + cascade/zoom detectors, runs the reverse
   cascade, optionally cross-checks the fill-count verdict against the zoom
   detectors (`inference/crosscheck.py`), optionally joint-decodes against
   the spacing prior, optionally refines placements with the zoom detectors.
7. **Live** (`live/`) - streaming variants of the classifier for use during
   data capture: bounded-cost preprocessing, full probability output, and
   an ordinal monotone evidence state machine in place of a simple debounce
   counter. `live/replay.py` replays held-out runs through the exact live
   decision stack for a shipping-quality latency/stability benchmark.
8. **QA** (`qa/`) - the paired A/B benchmark comparing the decode layer to
   the production cascade, post-training confusion/temperature audits,
   second-stage offender triage (faint-ridge vs suspect-label vs
   model-blind), human label-review packet generation, and live-replay
   decomposition (model-loss vs machinery-loss).

# QModel

[![Test](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/test.yml/badge.svg)](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/test.yml)
[![Style](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/style.yml/badge.svg)](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/style.yml)
[![Build](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/build.yml/badge.svg)](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/build.yml)

QModel is a QCM-D sensor-run channel detection system that turns raw
QATCH nanovisQ dissipation/resonance-frequency QCMD data into a signal-domain image
render, feeds a cascade of object detectors and classifiers, and joint-decodes
the result against a learned spacing prior. This repository builds,
trains, and evaluates those models offline.  The trained weights are served
live during data capture by the QATCH Nanovis application, which owns the
streaming/live-inference code path. The active implementation is the **`qmodel_7_onyx`** 
system under `src/systems/`.

## Install

```bash
pip install -e ".[dev]"
```

`requires-python >= 3.10`. Core dependencies (numpy, pandas, opencv-python,
scipy, matplotlib, ultralytics, tqdm) are declared in `pyproject.toml`; the
`dev` extra adds `pytest`/`pytest-cov`, and the `tier-discovery` extra adds
optional `scikit-learn` support for GMM-based viscosity tiering.

Model weights (`*.pt`) are _not_ committed `assets_paths.json` in the
package describes where the controller expects them, relative to
`src/systems/qmodel_7_onyx/assets/`.

## Quickstart

### Python API (recommended)

`Workspace` and `Pipeline` wrap the whole fresh-data, fitted-config,
dataset, trained-weights sequence behind one object, so a fresh dataset
becomes trained results without hand-chaining CLIs or re-discovering where
each stage wrote its output:

```python
from src.systems.qmodel_7_onyx import Workspace, Pipeline

# Workspace defaults to this repo's data/, datasets/, runs/, configs/ roots
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
rather than killing the process.  See `pipeline.py` for the full API and every stage's
keyword arguments.

### CLI (equivalent, one command per stage)

```bash
# discover a corpus, fit the spacing prior and viscosity tiers
python -m src.systems.qmodel_7_onyx.decode.fit_prior --raw-root data/raw --out configs/spacing_prior.json
python -m src.systems.qmodel_7_onyx.tiers --raw-root data/raw --out configs/tiers.json

# build datasets
python -m src.systems.qmodel_7_onyx.dataset.build_detectors --raw-root data/raw
python -m src.systems.qmodel_7_onyx.dataset.build_fill_classifier --raw-root data/raw

# train
python -m src.systems.qmodel_7_onyx.training.train_detectors
python -m src.systems.qmodel_7_onyx.training.train_fill_classifier

# evaluate the decode layer against the production cascade
python -m src.systems.qmodel_7_onyx.qa.benchmark --selftest

# run tests
pytest
```

The same commands are also registered as console scripts after install
(`qmodel-build-detectors`, `qmodel-train-detectors`, `qmodel-benchmark`,
etc. - see `pyproject.toml` for the full list).

### One-command release pipeline

`scripts/build_and_release_qmodel_onyx.py` chains fetch, prepare, build,
train, release, cleanup, and eval into a single command that pulls fresh runs
from Dropbox, rebuilds the datasets, trains the requested models, copies
the best checkpoints into a ready-to-ship `qmodel_onyx/` folder, purges redudant
files, and scores the deployed package's predicted channel positions against 
`*_poi.csv` ground truth events.

```bash
python scripts/build_and_release_qmodel_onyx.py --dropbox-source "Path/To/Raw/Data"

# skip Dropbox + the post-release eval, retrain one detector stage
python scripts/build_and_release_qmodel_onyx.py \
    --skip-fetch --skip-eval --targets detectors --detector-stages ch2_zoom
```

Any stage can be skipped or reconfigured - see `--help` for the full set
of `--eval-*`/`--keep-runs`/per-stage flags.

## Directory map

```text
src/systems/qmodel_7_onyx/
  pipeline.py          Workspace and Pipeline - the public API (see Quickstart above);
                       re-exported from the package's __init__.py
  paths.py            canonical filesystem roots (data/datasets/runs/configs/artifacts),
                       env-var overridable - every module imports its path defaults from here
  corpus.py            run discovery, ground-truth POI parsing, dedup, viscosity tiering (fixed scheme)
  tiers.py              data-driven viscosity TierScheme (log_uniform by default; GMM/quantile opt-in)
  augmentation.py        signal-domain augmentation (time warp, noise, amplitude jitter) + dynamic box sizing

  rendering/            df to image render contracts (detector cascade and fill classifier)
  decode/                spacing-prior model and DP joint decode over YOLO candidates
  dataset/                YOLO dataset builders (cascade detectors, fill classifier) and shared split/upsample logic
  training/               YOLO training CLIs
  inference/              the production controller (QModelOnyx), config, and the zoom-detector crosscheck
  deployment/             standalone, self-contained QModelOnyx controller (onyx.py and siblings) shipped
                          under the QATCH.QModel.models.qmodel_onyx.* dotted-import contract.  This is a separate
                          copy from inference/, since this one is loaded exactly as a downstream consumer
                          loads it (see scripts/build_and_release_qmodel_onyx.py's Eval stage)
  qa/                     offline QA: decode benchmark, val-set audit, offender triage, label review

src/utils/dataset_fetcher.py   Dropbox to data/raw ingestion CLI
scripts/
  build_and_release_qmodel_onyx.py   one-command fetch, prepare, build, train, release, cleanup, eval
                                     pipeline (see "One-command release pipeline" above); also holds the
                                     deployed-package eval that used to be a separate eval_onyx_deployment.py
  _qmodel_onyx_layout.py             shared assets_paths.json-derived deploy-layout helper
configs/                        fitted config artifacts (spacing_prior.json, tiers.json)
tests/                          pytest suite, mirrors the package layout above
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline data-flow, module
responsibilities, and path/config conventions in more depth,
[src/systems/qmodel_7_onyx/DESIGN.md](src/systems/qmodel_7_onyx/DESIGN.md)
for the design rationale behind the system and model, and [CONTRIBUTING.md](CONTRIBUTING.md) 
for how to add to it.

## Pipeline stages

1. **Corpus** (`corpus.py`) - walks `data/raw/<run_id>/` directories, parses
   the `*_poi.csv` ground truth into chain-space POI times, reads viscosity
   from the run's analyze output, dedupes runs with identical POI content.
2. **Rendering** (`rendering/`) - turns a preprocessed run `DataFrame` into
   the RGB strip images the detector cascade and fill classifier consume.
   Preprocessing lives in `dataprocessor.py`.
3. **Dataset building** (`dataset/`) - renders labeled YOLO datasets (4
   cascade and 3 zoom-refinement detector sets, and the 5-class ordinal fill
   classifier set), with leakage-proof run-level stratified splitting and
   per-tier upsampling (`dataset/splitting.py`, shared by both builders).
4. **Training** (`training/`) - thin CLIs around Ultralytics YOLO training,
   with all pixel-space augmentation disabled.
5. **Decode** (`decode/`) - `SpacingPrior` (a learned statistical model of
   gaps between consecutive POIs) and `dp_decode` (an exact dynamic programming 
   algorithm that picks one YOLO candidate per POI, jointly, instead of the greedy per-POI
   argmax the cascade uses by default).
6. **Inference** (`inference/`) - `QModelOnyx`, the production controller:
   loads the fill classifier + cascade/zoom detectors, runs the reverse
   cascade, optionally cross-checks the fill-count verdict against the zoom
   detectors (`inference/crosscheck.py`), optionally joint-decodes against
   the spacing prior, optionally refines placements with the zoom detectors.
7. **QA** (`qa/`) - Production cascade benchmarking for fill classification and
   the detector cascade stages.
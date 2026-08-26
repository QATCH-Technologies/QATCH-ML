# QModel

[![Test](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/test.yml/badge.svg)](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/test.yml)
[![Style](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/style.yml/badge.svg)](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/style.yml)
[![Build](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/build.yml/badge.svg)](https://github.com/QATCH-Technologies/QATCH-ML/actions/workflows/build.yml)

QModel is a QCM-D sensor-run event-detection system. It converts raw
QATCH nanovisQ dissipation and resonance-frequency data into signal-domain
image representations, applies a cascade of object detectors and classifiers,
and optionally combines their predictions with a learned spacing prior to
produce a globally consistent set of channel-event positions.

This repository contains the **offline** data preparation, dataset generation,
model training, evaluation, and deployment-packaging workflow. The trained
weights and controller produced here are consumed by the QATCH Nanovis
application, which owns the streaming/live-inference path.

The active implementation is **QModel Onyx**, located under
`src/systems/qmodel_7_onyx/`.

## Install

```bash
pip install -e ".[dev]"
```

The package requires Python 3.10 or newer. Core dependencies include NumPy,
pandas, OpenCV, SciPy, Matplotlib, Ultralytics, and tqdm; they are declared in
`pyproject.toml`.

The `dev` extra adds the test dependencies (`pytest` and `pytest-cov`).
The optional `tier-discovery` extra adds scikit-learn support for
GMM-based viscosity-tier discovery.

Model weights (`*.pt`) are not committed to the repository. The package's
`assets_paths.json` describes the expected deployed asset locations relative
to `src/systems/qmodel_7_onyx/assets/`.

## Quickstart

### Python API (recommended)

`Workspace` and `Pipeline` provide the primary programmatic interface to the
offline workflow. Together they manage the sequence from raw data and fitted
configuration through dataset construction and model training, while keeping
the paths used by each stage consistent.

```python
from src.systems.qmodel_7_onyx import Workspace, Pipeline

# Workspace defaults to this repository's data/, datasets/, runs/, and configs/
# roots. Override them to use a different raw-data or output location.
pipeline = Pipeline(Workspace(data_root="path/to/raw"))

result = pipeline.run()  # prepare() -> build_datasets() -> train()

print(result.training.weights)
# {"init": Path(...), ..., "fill_classifier": Path(...)}

print(result.training.metrics)
# Best-effort Ultralytics validation metrics for each trained stage.
```

Individual stages can also be invoked independently when finer control is
needed. They share the same `Workspace`, so stage outputs continue to use the
same canonical paths.

```python
pipeline.prepare()  # fit tiers.json and spacing_prior.json
pipeline.build_datasets(targets=["fill_classifier"])
pipeline.train(
    targets=["detectors"],
    detector_stages=["ch2_zoom"],
    epochs=50,
)
```

Conditions such as insufficient or missing input data are reported through
`PipelineError`, which can be caught by callers instead of terminating the
process unexpectedly. See `pipeline.py` for the complete API and stage-level
keyword arguments.

### CLI

The individual workflow stages are also available as module CLIs. The
following commands correspond to the major stages of the Python API:

```bash
# Discover the corpus and fit reusable configuration artifacts.
python -m src.systems.qmodel_7_onyx.decode.fit_prior     --raw-root data/raw     --out configs/spacing_prior.json

python -m src.systems.qmodel_7_onyx.tiers     --raw-root data/raw     --out configs/tiers.json

# Build training datasets.
python -m src.systems.qmodel_7_onyx.dataset.build_detectors     --raw-root data/raw

python -m src.systems.qmodel_7_onyx.dataset.build_fill_classifier     --raw-root data/raw

# Train models.
python -m src.systems.qmodel_7_onyx.training.train_detectors
python -m src.systems.qmodel_7_onyx.training.train_fill_classifier

# Evaluate the decode layer against the production cascade.
python -m src.systems.qmodel_7_onyx.qa.benchmark --selftest

# Run the test suite.
pytest
```

The same operations are registered as console scripts after installation.
See `pyproject.toml` for the complete list, including commands such as
`qmodel-build-detectors`, `qmodel-train-detectors`, and
`qmodel-benchmark`.

### One-command release pipeline

`scripts/build_and_release_qmodel_onyx.py` provides a higher-level workflow
for rebuilding and packaging a QModel Onyx release. It can fetch fresh runs,
prepare fitted configuration, build datasets, train selected models, package
the best checkpoints into a self-contained `qmodel_onyx/` deployment
directory, remove intermediate artifacts according to its cleanup options,
and evaluate the resulting deployment against POI ground truth.

For example:

```bash
python scripts/build_and_release_qmodel_onyx.py     --dropbox-source "Path/To/Raw/Data"

# Skip data fetching and post-release evaluation, and retrain one detector stage.
python scripts/build_and_release_qmodel_onyx.py     --skip-fetch     --skip-eval     --targets detectors     --detector-stages ch2_zoom
```

Individual stages and cleanup behavior can be skipped or reconfigured. Run
`--help` for the complete set of pipeline, evaluation, retention, and
per-stage options.

## Directory map

```text
src/systems/qmodel_7_onyx/
  pipeline.py          Workspace and Pipeline, the primary public API;
                       re-exported from the package's __init__.py
  paths.py             canonical filesystem roots for data, datasets, runs,
                       configs, and artifacts; environment-variable overridable
  corpus.py            run discovery, ground-truth POI parsing, deduplication,
                       and fixed-scheme viscosity tier assignment
  tiers.py             data-driven viscosity TierScheme
                       (log_uniform by default; GMM/quantile opt-in)
  augmentation.py      signal-domain augmentation (time warp, noise,
                       amplitude jitter) and dynamic box sizing

  rendering/            DataFrame-to-image rendering contracts for the
                       detector cascade and fill classifier
  decode/               spacing-prior model and dynamic-programming decode
                       over detector candidates
  dataset/              YOLO dataset builders for cascade detectors and the
                       fill classifier, plus shared splitting/upsampling logic
  training/             model-training CLIs
  inference/            production QModelOnyx controller, configuration,
                       and zoom-detector crosschecks
  deployment/            self-contained QModelOnyx controller packaged for
                       downstream consumption
  qa/                    offline quality-assurance tools, including decode
                       benchmarks, validation-set audits, offender triage,
                       and label review

src/utils/dataset_fetcher.py
                       Dropbox-to-data/raw ingestion CLI

scripts/
  build_and_release_qmodel_onyx.py
                       one-command fetch, prepare, build, train, release,
                       cleanup, and evaluation workflow
                       (also contains deployed-package evaluation)
  _qmodel_onyx_layout.py
                       shared helper that derives the deployed asset layout
                       from assets_paths.json

configs/               fitted configuration artifacts such as
                       spacing_prior.json and tiers.json
tests/                 pytest suite mirroring the package structure
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline data flow, module
responsibilities, and path/configuration conventions.

See [src/systems/qmodel_7_onyx/DESIGN.md](src/systems/qmodel_7_onyx/DESIGN.md)
for the rationale behind the detection and inference design.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and contribution
guidelines.

## Pipeline stages

1. **Corpus** (`corpus.py`) — Discovers raw runs, parses `*_poi.csv` ground
   truth into chain-space POI times, reads viscosity metadata from the run's
   analysis output, and removes duplicate runs according to POI content.

2. **Rendering** (`rendering/`) — Converts a preprocessed run `DataFrame` into
   the RGB strip images consumed by the detector cascade and fill classifier.
   Signal preprocessing is implemented in `dataprocessor.py`.

3. **Dataset building** (`dataset/`) — Creates labeled YOLO datasets for the
   cascade and zoom-refinement detectors, together with the five-class
   ordinal fill classifier. Dataset splitting is performed at the run level
   and stratified using the configured viscosity tier and POI-count metadata;
   underrepresented tiers can then be upsampled. Shared splitting and
   upsampling logic lives in `dataset/splitting.py`.

4. **Training** (`training/`) — Provides thin command-line wrappers around
   Ultralytics YOLO training. Pixel-space augmentation is disabled because
   the rendered image geometry represents the underlying time axis.

5. **Decode** (`decode/`) — Provides `SpacingPrior`, a learned statistical
   model of gaps between consecutive POIs, and `dp_decode`, an exact dynamic
   programming algorithm that jointly selects detector candidates rather than
   independently taking the highest-confidence candidate for each POI.

6. **Inference** (`inference/`) — Provides `QModelOnyx`, the production
   controller. It loads the fill classifier and detector stages, runs the
   reverse cascade, and can optionally cross-check the predicted fill count,
   jointly decode candidates using the spacing prior, and refine event
   placements with zoom detectors.

7. **QA** (`qa/`) — Provides offline evaluation and diagnostic tooling for
   the production cascade, fill-count classification, detector stages, and
   decode behavior.

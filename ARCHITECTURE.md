# Architecture

## Data flow

``mermaid
flowchart TB
    raw["data/raw/&lt;run_id&gt;/\n(*.csv, *_poi.csv, analyze-*.zip)"]
    corpus["corpus.py\ndiscover_runs / dedupe_runs / truth_times"]
    tiers["tiers.py\nTierScheme (log_uniform default; GMM/quantile opt-in)"]
    prior["decode/spacing_prior.py + decode/fit_prior.py\nSpacingPrior"]
    aug["augmentation.py\ntime_warp / inject_noise / dynamic_box_width_sec"]
    render["rendering/\ndetector_render.py, fill_render.py, legacy_dataprocessor.py"]
    dsbuild["dataset/\nbuild_detectors.py, build_fill_classifier.py, splitting.py"]
    train["training/\ntrain_detectors.py, train_fill_classifier.py"]
    weights["*.pt weights\n(gitignored, assets_paths.json)"]
    controller["inference/controller.py\nQModelOnyx"]
    crosscheck["inference/crosscheck.py\nzoom-detector fill-count crosscheck"]
    decode["decode/dp_decode.py\njoint DP decode"]
    deployment["deployment/onyx.py + siblings\nQModelOnyx (standalone, self-contained)"]
    livebase["live/base_live.py\nQModelOnyxLive"]
    fillive["live/fill_live.py\nQModelOnyxFillClassifier, OrdinalEvidence"]
    replay["live/replay.py"]
    qa["qa/\nbenchmark.py, audit_fill_val.py, triage_offenders.py,\nlabel_review_packet.py, analyze_replay.py"]

    raw --> corpus
    corpus --> tiers
    corpus --> prior
    corpus --> dsbuild
    aug --> dsbuild
    render --> dsbuild
    tiers --> dsbuild
    dsbuild --> train
    train --> weights
    weights --> controller
    prior --> controller
    render --> controller
    controller --> crosscheck
    controller --> decode
    weights --> deployment
    prior --> deployment
    weights --> livebase
    livebase --> fillive
    controller --> fillive
    fillive --> replay
    corpus --> replay
    controller --> qa
    corpus --> qa
    replay --> qa
    tiers --> qa
``

## Module responsibility table

| Module | Responsibility | Depends on |
|---|---|---|
| `pipeline.py` | Public API: `Workspace` (path config) + `Pipeline` (`prepare`/`build_datasets`/`train`/`run`) | `corpus`, `tiers`, `decode.*`, `dataset.*`, `training.*` |
| `paths.py` | Canonical, env-overridable filesystem roots | - |
| `corpus.py` | Run discovery, POI truth parsing, dedup, fixed viscosity tiers | `decode.spacing_prior` (POI_ORDER) |
| `tiers.py` | Data-driven `TierScheme` (`log_uniform` default; `gmm`/`quantile` opt-in via `--method`) | `corpus` |
| `augmentation.py` | Signal-domain augmentation, dynamic box sizing | - |
| `rendering/_common.py` | Shared strip-plotting/robust-MAD helpers | `rendering.legacy_dataprocessor` |
| `rendering/legacy_dataprocessor.py` | v1 preprocessing (`preprocess_dataframe`) + legacy render | - |
| `rendering/detector_render.py` | v2 detector-cascade image render, `derivative_energy` salience | `rendering._common`, `rendering.legacy_dataprocessor` |
| `rendering/fill_render.py` | Fill-classifier image render (v2/v3), `step_coincidence_energy` | `rendering._common`, `rendering.legacy_dataprocessor` |
| `decode/spacing_prior.py` | Learned log-normal gap model | - |
| `decode/dp_decode.py` | Exact DP joint decode over YOLO candidates | `decode.spacing_prior` |
| `decode/fit_prior.py` | CLI: fit `SpacingPrior` from complete fills | `corpus`, `decode.spacing_prior` |
| `decode/sweep.py` | CLI: offline decode-hyperparameter sweep | `decode.dp_decode`, `decode.spacing_prior` |
| `dataset/splitting.py` | Leakage-proof stratified split, per-tier upsampling | `corpus`, `tiers` |
| `dataset/build_detectors.py` | CLI: cascade + zoom detector YOLO datasets | `corpus`, `tiers`, `augmentation`, `rendering`, `dataset.splitting` |
| `dataset/build_fill_classifier.py` | CLI: fill-classifier YOLO dataset | same as above |
| `training/env.py` | Shared CUDA allocator env-var setup | - |
| `training/train_detectors.py` | CLI: train cascade + zoom detectors | `training.env` |
| `training/train_fill_classifier.py` | CLI: train fill classifier | `training.env` |
| `inference/config.py` | `QModelOnyxConfig` tunables | - |
| `inference/crosscheck.py` | Zoom-detector fill-count rescue/veto | - |
| `inference/controller.py` | `QModelOnyx` production controller | `inference.config`, `inference.crosscheck`, `decode.*`, `rendering.*` |
| `deployment/onyx_dataprocessor.py` | Raw CSV -> interpolated time series + derived features + rendered signal images | - |
| `deployment/onyx_spacing_prior.py` | Standalone flat, pairwise POI-gap prior loaded from the fitted `spacing_prior.json`; mirror of `decode/spacing_prior.py` | - |
| `deployment/onyx_decode.py` | Joint DP decode over POI candidate detections; mirror of `decode/dp_decode.py` | `deployment.onyx_spacing_prior` |
| `deployment/onyx_render.py` | v2 detector image render (derivative-energy salience); mirror of `rendering/detector_render.py` | - |
| `deployment/onyx_fill_render.py` | v2 fill-classifier image render; mirror of `rendering/fill_render.py` | `deployment.onyx_dataprocessor`, `deployment.onyx_render` |
| `deployment/onyx.py` | `QModelOnyx` - the standalone reverse-cascade controller shipped under the `QATCH.QModel.models.qmodel_onyx.*` dotted-import contract; a separate, self-contained copy from `inference/controller.py`, loaded exactly as a downstream consumer loads it (see `scripts/build_and_release_qmodel_onyx.py`'s Eval stage) | `deployment.onyx_dataprocessor` (required); `deployment.onyx_spacing_prior`, `deployment.onyx_decode`, `deployment.onyx_render`, `deployment.onyx_fill_render` (optional) |
| `deployment/onyx_live.py` | Multiprocessing live fill-classifier wrapper; mirror of `live/fill_live.py` + `live/base_live.py` | `deployment.onyx` (`QModelOnyxConfig`, `QModelOnyxFillClassifier`) |
| `live/base_live.py` | Headless-importable live base class + process wrapper | (optional) QATCH app |
| `live/fill_live.py` | Onyx live classifier: bounded-cost preprocess, `OrdinalEvidence` | `inference.*`, `rendering.fill_render`, `live.base_live` |
| `live/replay.py` | CLI: streaming replay benchmark | `corpus`, `live.fill_live` |
| `qa/benchmark.py` | CLI: paired A/B decode benchmark + selftest | `corpus`, `decode.*`, `inference.controller` |
| `qa/audit_fill_val.py` | CLI: post-training confusion/temperature audit | - (ultralytics only) |
| `qa/triage_offenders.py` | CLI: second-stage offender triage | `corpus`, `augmentation`, `rendering`, `tiers`, `live.fill_live` |
| `qa/label_review_packet.py` | CLI: human label-review PNG packets | `corpus` |
| `qa/analyze_replay.py` | CLI: replay JSON decomposition (model-loss vs machinery-loss) | - |
| `src/utils/dataset_fetcher.py` | CLI: Dropbox source tree -> `data/raw` ingestion | - |
| `scripts/build_and_release_qmodel_onyx.py` | CLI: one-command fetch->prepare->build->train->release->cleanup->eval pipeline; also holds the deployed-package eval (predicted POI position vs `*_poi.csv` ground truth) that used to be the standalone `eval_onyx_deployment.py` | `pipeline`, `corpus`, `dataset_fetcher`, `scripts/_qmodel_onyx_layout.py` |
| `scripts/_qmodel_onyx_layout.py` | Shared `assets_paths.json`-derived deploy-layout helper (write during Release, read back during Eval) | - |

## Path and config conventions

Every stage used to default to a bare `Path("data/raw")`-style relative
path, which only worked if the process happened to be launched from the
repo root. `paths.py` fixes this once: `REPO_ROOT` is resolved from the
package's own file location (`Path(__file__).resolve().parents[3]`), and
every other root (`DATA_ROOT`, `DATASETS_ROOT`, `RUNS_ROOT`, `CONFIGS_ROOT`,
`ARTIFACTS_ROOT`) is derived from it, each overridable via an environment
variable (`QMODEL_DATA_ROOT`, etc.) for deployments that need a different
layout. Every CLI's argparse defaults now import from here instead of
hardcoding a relative path.

Fitted/derived config artifacts (`spacing_prior.json`, `tiers.json`) live
in `configs/` at the repo root - a single canonical location, replacing an
earlier ambiguity where a copy shipped inside the package and a second
CWD-relative default could silently diverge from it. `assets_paths.json`
(the model-weight manifest) stays inside the package, since its paths are
package-relative by design.

Generated QA/audit artifacts (misses.csv, salience reports, replay JSON,
regression plots, benchmark CSVs) write under `artifacts/` at the repo
root by default, not `test/` - `test/` (singular, gitignored) was
previously overloaded as both a pytest-discovery directory name and a
scratch output directory, which is exactly the kind of collision this
refactor's `tests/` (plural, tracked) exists to avoid.

## Deliberately not built: a systems plugin registry

`src/systems/` has held three generations of this pipeline in sequence
(`qmodel_6_2`, `qmodel_7_1`, now `qmodel_7_onyx`), each a self-contained,
unrelated directory - there has never been a shared base class, registry,
or factory that a new generation must implement, and this refactor does not
invent one. Building a plugin abstraction for a currently-single-system
repo would be speculative generality with no second implementation to
validate it against. If a future generation is added, the convention is
simply: a new sibling package under `src/systems/`, structured the same way
`qmodel_7_onyx/` is (by pipeline stage), reusing `src/utils/` where it
genuinely applies. Revisit a shared abstraction only once there are two
real systems to abstract over.

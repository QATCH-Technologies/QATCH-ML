# Graph Report - .  (2026-07-25)

## Corpus Check
- 70 files · ~51,823 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 740 nodes · 1581 edges · 37 communities (36 shown, 1 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 31 edges (avg confidence: 0.59)
- Token cost: 98,242 input · 0 output

## Community Hubs (Navigation)
- DP Decode Engine
- Run Corpus Discovery
- Live Fill Inference & Triage
- QModel V6 Configuration
- Dataset Build Pipelines
- Fill Classifier Model
- Dataset Fetcher
- Data Augmentation
- Fill Verdict Crosscheck
- Core Pipeline Orchestration
- Pipeline Tests
- Training Environment Setup
- Live YOLO Fill Classification
- Fill Classifier Audit & Calibration
- CI/CD & Contributing Guidelines
- README & Entry Points
- Live Base Framework
- Fill Classifier Initialization
- Live Inference Loop & Status
- Workspace Path Management
- Replay Analysis
- Architecture Rationale
- Spacing Prior Fitting
- Path Config & Build Job
- Live YOLO Process Management
- Test Fixtures
- Project Root

## God Nodes (most connected - your core abstractions)
1. `SpacingPrior` - 32 edges
2. `TierScheme` - 32 edges
3. `discover_runs()` - 26 edges
4. `get_logger()` - 25 edges
5. `Candidate` - 21 edges
6. `dp_decode()` - 21 edges
7. `QModelV6YOLO_Live` - 20 edges
8. `OrdinalEvidence` - 20 edges
9. `QModelV6YOLO` - 19 edges
10. `run_benchmark()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `build job: sdist/wheel build + clean-venv install check` --references--> `Workspace`  [EXTRACTED]
  .github/workflows/build.yml → src/systems/qmodel_7_onyx/pipeline.py
- `build job: sdist/wheel build + clean-venv install check` --references--> `Pipeline`  [EXTRACTED]
  .github/workflows/build.yml → src/systems/qmodel_7_onyx/pipeline.py
- `test_prepare_writes_tiers_and_prior()` --indirect_call--> `SpacingPrior`  [INFERRED]
  tests/test_pipeline.py → src/systems/qmodel_7_onyx/decode/spacing_prior.py
- `test_make_monotone_warp_is_strictly_increasing()` --calls--> `make_monotone_warp()`  [EXTRACTED]
  tests/test_augmentation.py → src/systems/qmodel_7_onyx/augmentation.py
- `test_time_warp_preserves_poi_order_and_returns_positive_stretch()` --calls--> `time_warp()`  [EXTRACTED]
  tests/test_augmentation.py → src/systems/qmodel_7_onyx/augmentation.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **CI workflows + dependency maintenance run on every push/PR to main** — github_workflows_build, github_workflows_style, github_workflows_test, github_workflows_codeql, github_dependabot [EXTRACTED 1.00]
- **The 8-stage qmodel_7_onyx pipeline (corpus -> rendering -> dataset -> training -> decode -> inference -> live -> qa)** — src_systems_qmodel_7_onyx_corpus, src_systems_qmodel_7_onyx_rendering, src_systems_qmodel_7_onyx_dataset, src_systems_qmodel_7_onyx_training, src_systems_qmodel_7_onyx_decode, src_systems_qmodel_7_onyx_inference, src_systems_qmodel_7_onyx_live, src_systems_qmodel_7_onyx_qa [EXTRACTED 1.00]
- **Streaming live-inference classes used during data capture** — src_systems_qmodel_7_onyx_live_base_live_qmodelv6yolo_live, src_systems_qmodel_7_onyx_live_fill_live_qmodelv7fillclassifier, src_systems_qmodel_7_onyx_live_fill_live_ordinalevidence [EXTRACTED 1.00]

## Communities (37 total, 1 thin omitted)

### Community 0 - "DP Decode Engine"
Cohesion: 0.06
Nodes (70): Candidate, _clip01(), DecodeResult, dp_decode(), _dp_pass(), greedy_baseline(), _greedy_result(), _lam_between() (+62 more)

### Community 1 - "Run Corpus Discovery"
Cohesion: 0.05
Nodes (67): raw sensor run data (data/raw/<run_id>/), dedupe_runs(), discover_runs(), load_run_filter(), DataFrame, ndarray, Path, corpus.py =========  Shared corpus-discovery library: walks ``data/raw``-style r (+59 more)

### Community 2 - "Live Fill Inference & Triage"
Cohesion: 0.06
Nodes (60): inference package, fill_live.py ============  Live-side upgrades for the fill-type classifier, port, annotate_render(), decision_rule_check(), main(), DataFrame, Path, triage_offenders.py ===================  Second-stage triage after qa/audit_fill (+52 more)

### Community 3 - "QModel V6 Configuration"
Cohesion: 0.05
Nodes (38): Series, QModelV6Config, config.py =========  Configuration constants for the QModel V6 YOLO inference pi, Configuration constants for the QModel V6 YOLO pipeline., Log, Any, DataFrame, QModelV6YOLO (+30 more)

### Community 4 - "Dataset Build Pipelines"
Cohesion: 0.07
Nodes (44): build(), main(), Path, dataset/build_detectors.py ===========================  Builds the per-stage YOL, Returns (cut_time, is_negative). cut_time None => unusable., _sample_cut(), build(), fill_state_at() (+36 more)

### Community 5 - "Fill Classifier Model"
Cohesion: 0.06
Nodes (35): QModelV6YOLO_FillClassifier, Log, OrdinalEvidence, preprocess_for_cls(), DataFrame, ndarray, QModelV7FillClassifier, QModelV7YOLO_Live (+27 more)

### Community 6 - "Dataset Fetcher"
Cohesion: 0.07
Nodes (33): Namespace, DatasetFetcher, FailureRecord, fast_copy(), main(), parse_arguments(), Path, Initializes the DatasetFetcher.          Args:             source_dir (str): (+25 more)

### Community 7 - "Data Augmentation"
Cohesion: 0.10
Nodes (30): amplitude_jitter(), augment_run(), dynamic_box_width_sec(), inject_noise(), make_monotone_warp(), DataFrame, ndarray, augmentation.py ================  Signal-domain augmentation for QModel detector (+22 more)

### Community 8 - "Fill Verdict Crosscheck"
Cohesion: 0.15
Nodes (23): _best_zoom_hit(), CrosscheckResult, DataFrame, crosscheck.py =============  Analysis-time fill-verdict cross-check using the v7, Slides zoom windows over [t_start, t_end], returns the single best     detection, Under-count rescue: climbs the channel count while the next zoom     detector co, Over-count advisory: zoom-inspect the claimed last channel's POI.     Returns th, verify_claimed_poi() (+15 more)

### Community 9 - "Core Pipeline Orchestration"
Cohesion: 0.15
Nodes (18): qmodel_7_onyx =============  QATCH sensor-run POI detection pipeline. The public, DatasetBuildResult, _normalize_targets(), Pipeline, PipelineError, PipelineResult, PreparationResult, Any (+10 more)

### Community 10 - "Pipeline Tests"
Cohesion: 0.11
Nodes (7): _fake_runs(), Path, test_pipeline_accepts_workspace_overrides(), test_prepare_writes_tiers_and_prior(), test_workspace_derived_paths(), test_workspace_normalizes_str_to_path(), _write_manifest()

### Community 11 - "Training Environment Setup"
Cohesion: 0.15
Nodes (16): *.pt model weights (gitignored, described by assets_paths.json), extract_metrics(), Any, env.py ======  Shared training-process environment setup, plus the small result, Reduce fragmentation-driven OOMs before torch is imported.      Harmless if unsu, What one YOLO training stage produced: the tag it was trained under     (e.g. "c, Best-effort extraction of a plain-dict metrics summary from an     Ultralytics `, setup_cuda_env() (+8 more)

### Community 12 - "Live YOLO Fill Classification"
Cohesion: 0.16
Nodes (10): DataFrame, QModelV6YOLO_Live, Manages data buffering and executes predictions for real-time fill classificatio, Captures and caches baseline sensor values from the pre-fill window.          Th, Fires a 'Data Ready, Stop' signal if 3 minutes elapse after Initial Fill, Ingests a new chunk of data into the rolling buffer.          This is the public, Extends the internal data buffer with new time-series data.          Ensures tim, Executes the classification pipeline on the current buffered data.          This (+2 more)

### Community 13 - "Fill Classifier Audit & Calibration"
Cohesion: 0.17
Nodes (14): fit_temperature(), main(), parse_name(), ndarray, audit_fill_val.py =================  Post-training triage for the fill classifie, 1-D NLL minimization of p^(1/T) (renormalized) over a log-spaced     grid + loca, configure_logging(), get_logger() (+6 more)

### Community 14 - "CI/CD & Contributing Guidelines"
Cohesion: 0.18
Nodes (12): Duck-typed stand-ins let pure-logic tests avoid model weights/ultralytics, pyupgrade (UP*) rules deliberately disabled — Optional/List/Dict is a style choice, not a lint finding, GitHub Actions dependency updates (weekly, grouped), Pip dependency updates (weekly, grouped into one PR), Build workflow (build.yml), CodeQL workflow (codeql.yml), analyze job (python, security-extended queries, weekly + on-push), Style workflow (style.yml) (+4 more)

### Community 15 - "README & Entry Points"
Cohesion: 0.19
Nodes (11): Console scripts registered via pyproject.toml (qmodel-build-detectors, qmodel-train-detectors, qmodel-benchmark, etc.), dataset package, live package, rendering package, training package, main(), _make_rect_trainer(), Path (+3 more)

### Community 16 - "Live Base Framework"
Cohesion: 0.17
Nodes (7): Convention: adding a module to an existing pipeline stage, NamedTuple, DropEpochSignal, Log, live/base_live.py  This module provides the infrastructure for running a YOLO-ba, Sentinel put into the forecaster input queue by the UI when the drop is     dete, # NOTE: `run()` always constructs the classifier with

### Community 17 - "Fill Classifier Initialization"
Cohesion: 0.20
Nodes (7): Queue, RuntimeError, Initializes the Fill Classifier with the provided model weights.          Args:, QModelV6YOLO_FillClassifier, Initializes the live fill classifier with a model and buffer settings., Initializes the LiveProcess with queue handles and buffer configuration., Placeholder base so ``QModelV6YOLO_Live`` remains a valid class         definiti

### Community 18 - "Live Inference Loop & Status"
Cohesion: 0.18
Nodes (6): Seeds the fill epoch from the UI-detected drop-application timestamp.          C, Returns the pending on-display message and clears it atomically.          The me, Retrieves the human-readable string representation of the current prediction., Executes the main inference loop for the live fill classification process., Any, Convert raw buffer data from a worker into a pandas DataFrame.          Retrieve

### Community 19 - "Workspace Path Management"
Cohesion: 0.33
Nodes (3): Path, Every filesystem root a pipeline stage reads or writes, in one place.      Defau, Workspace

### Community 20 - "Replay Analysis"
Cohesion: 0.40
Nodes (8): main(), analyze_replay.py =================  Decomposes replay_fill.json to answer the q, summarize_machine(), _rec(), test_summarize_machine_counts_failing_runs(), test_summarize_machine_ignores_runs_without_this_machine(), test_summarize_machine_missed_confirmation_when_latency_is_none(), test_summarize_machine_tracks_false_forward_and_backward_runs()

### Community 21 - "Architecture Rationale"
Cohesion: 0.29
Nodes (6): configs/ at repo root as single canonical location for fitted config artifacts, paths.py centralizes env-overridable roots, fixing CWD-relative path fragility, Deliberately not building a systems plugin registry for src/systems/, tests/ (plural, tracked) vs test/ (singular, gitignored scratch dir) split, Convention: adding a brand-new pipeline-stage subpackage, qmodel_7_onyx system package

### Community 22 - "Spacing Prior Fitting"
Cohesion: 0.43
Nodes (6): collect_complete_configs(), _find_run_csv(), main(), ndarray, Path, fit_prior.py ============  Fit the SpacingPrior from data/raw complete-fill conf

### Community 23 - "Path Config & Build Job"
Cohesion: 0.40
Nodes (5): build job: sdist/wheel build + clean-venv install check, ASSETS_PATHS_JSON, _env_path(), Path, paths.py ========  Canonical filesystem roots for the qmodel_7_onyx pipeline, re

### Community 24 - "Live YOLO Process Management"
Cohesion: 0.33
Nodes (4): QModelV6YOLO_LiveProcess, Dedicated process for running real-time YOLO fill state predictions.      This c, Checks whether the process is still executing.          Returns:             boo, Signals the process to terminate gracefully.          Sets the internal exit eve

### Community 25 - "Test Fixtures"
Cohesion: 0.33
Nodes (5): complete_poi_times(), Path, Shared fixtures for the qmodel_7_onyx test suite.  ``make_run`` builds a synthet, A well-separated, strictly-ascending complete-fill configuration., _write_run()

## Knowledge Gaps
- **7 isolated node(s):** `qatch-ml`, `Pip dependency updates (weekly, grouped into one PR)`, `GitHub Actions dependency updates (weekly, grouped)`, `analyze job (python, security-extended queries, weekly + on-push)`, `raw sensor run data (data/raw/<run_id>/)` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_logger()` connect `Fill Classifier Audit & Calibration` to `DP Decode Engine`, `Run Corpus Discovery`, `Live Fill Inference & Triage`, `Dataset Build Pipelines`, `Dataset Fetcher`, `Training Environment Setup`, `README & Entry Points`, `Live Base Framework`, `Live Inference Loop & Status`, `Spacing Prior Fitting`?**
  _High betweenness centrality (0.114) - this node is a cross-community bridge._
- **Why does `SpacingPrior` connect `DP Decode Engine` to `Core Pipeline Orchestration`, `Pipeline Tests`, `Spacing Prior Fitting`, `Run Corpus Discovery`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `QModelV6YOLO` connect `QModel V6 Configuration` to `DP Decode Engine`, `Run Corpus Discovery`, `README & Entry Points`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `SpacingPrior` (e.g. with `Candidate` and `DecodeResult`) actually correct?**
  _`SpacingPrior` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `TierScheme` (e.g. with `DatasetBuildResult` and `Pipeline`) actually correct?**
  _`TierScheme` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `qatch-ml`, `Pip dependency updates (weekly, grouped into one PR)`, `GitHub Actions dependency updates (weekly, grouped)` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `DP Decode Engine` be split into smaller, more focused modules?**
  _Cohesion score 0.05533279871692061 - nodes in this community are weakly interconnected._
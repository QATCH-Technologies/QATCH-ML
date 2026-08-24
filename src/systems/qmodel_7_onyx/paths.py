"""
Canonical filesystem roots for the qmodel_7_onyx pipeline, resolved from this
package's own location rather than the current working directory. Every
stage (dataset build, training, decode, live, QA) that used to default to a
bare `Path("data/raw")`-style relative path now imports its default from
here, so every CLI behaves the same regardless of the directory it is
launched from.

Each root is overridable via an environment variable, so a deployment can
point at a different data/artifacts location without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../QATCH-ML/src/systems/qmodel_7_onyx/paths.py -> parents[3] == QATCH-ML
REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_path(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else default


DATA_ROOT = _env_path("QMODEL_DATA_ROOT", REPO_ROOT / "data" / "raw")
DATASETS_ROOT = _env_path("QMODEL_DATASETS_ROOT", REPO_ROOT / "datasets")
RUNS_ROOT = _env_path("QMODEL_RUNS_ROOT", REPO_ROOT / "runs")
CONFIGS_ROOT = _env_path("QMODEL_CONFIGS_ROOT", REPO_ROOT / "configs")
ARTIFACTS_ROOT = _env_path("QMODEL_ARTIFACTS_ROOT", REPO_ROOT / "artifacts")

# Local Dropbox sync folder holding raw run captures, used by the fetch
# stage (dataset_fetcher.py / build_and_release_qmodel_onyx.py). Derived from
# the current user's home directory rather than a hardcoded profile path, so
# the same default works on any machine/account with this Dropbox team
# folder synced. Overridable via QMODEL_DROPBOX_SOURCE for machines where the
# sync folder lives somewhere else (or isn't synced at all).
DROPBOX_SOURCE = _env_path(
    "QMODEL_DROPBOX_SOURCE",
    Path.home() / "QATCH Dropbox" / "QATCH Team Folder" / "Production Notes",
)

# Fitted/derived config artifacts (single canonical location - replaces the
# old ambiguity between a CWD-relative default and a copy inside the package).
TIERS_JSON = CONFIGS_ROOT / "tiers.json"
SPACING_PRIOR_JSON = CONFIGS_ROOT / "spacing_prior.json"

# Model-weight manifest ships alongside the package code (paths inside it are
# package-relative, not CWD-relative).
ASSETS_PATHS_JSON = Path(__file__).resolve().parent / "assets_paths.json"

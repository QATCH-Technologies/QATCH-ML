"""Defines the shared deployment-file layout for the QModel Onyx package.

Derives deployed asset paths from `assets_paths.json` so release and
evaluation code use the same layout as the production controller without
duplicating path conventions.

The helpers in this module support two complementary operations: resolving
the deployment-relative path for an individual model stage and constructing
the complete `model_assets` mapping expected by `QModelOnyx`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def deploy_subpath(assets_map: Dict[str, Any], stage: str) -> Path:
    """Resolves a model stage's path relative to the deployment asset root.

    The path is derived from the corresponding entry in the shared asset
    configuration rather than from a hard-coded deployment layout.

    Args:
        assets_map: Parsed `assets_paths.json` mapping containing classifier
            and detector asset paths.
        stage: Model stage name. `"fill_classifier"` selects the fill
            classifier; all other values are resolved from the detector
            mapping.

    Returns:
        Path to the stage's weights file relative to the `assets` directory.

    Raises:
        KeyError: If `stage` is not present in the configured asset mapping.
        ValueError: If the configured asset path does not contain an
            `assets` path component.
    """
    raw = (
        assets_map["fill_classifier"]
        if stage == "fill_classifier"
        else assets_map["detectors"][stage]
    )
    parts = Path(raw).parts
    return Path(*parts[parts.index("assets") + 1 :])


def build_model_assets(assets_map: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Builds the deployment asset mapping expected by `QModelOnyx`.

    Paths are constructed relative to the supplied deployment root using the
    shared asset configuration. File existence is intentionally not checked,
    allowing callers to construct partially deployed configurations and
    letting the controller handle missing assets according to its own
    loading behavior.

    Args:
        assets_map: Parsed `assets_paths.json` mapping containing classifier
            and detector asset paths.
        root: Root directory of the deployed `qmodel_onyx` package.

    Returns:
        A `model_assets` dictionary containing the fill-classifier path,
        detector-stage paths, and spacing-prior path.
    """
    return {
        "fill_classifier": str(root / deploy_subpath(assets_map, "fill_classifier")),
        "detectors": {
            stage: str(root / deploy_subpath(assets_map, stage))
            for stage in assets_map["detectors"]
        },
        "spacing_prior": str(root / "spacing_prior.json"),
    }


def load_assets_map(assets_paths_json: Path) -> Dict[str, Any]:
    return json.loads(Path(assets_paths_json).read_text())

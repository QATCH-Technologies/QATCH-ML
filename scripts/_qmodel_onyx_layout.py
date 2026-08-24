"""
_qmodel_onyx_layout.py
=======================

Shared helper for where a deployed qmodel_onyx package's files live, used by
both roles ``build_and_release_qmodel_onyx.py`` plays: writing the layout
(its Release stage) and reading it back to build a ``QModelOnyx``
``model_assets`` dict (its Eval stage). Derives the layout from
``assets_paths.json`` rather than hard-coding it twice, so it can never
drift out of sync with what the production controller expects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def deploy_subpath(assets_map: Dict[str, Any], stage: str) -> Path:
    """The path of one stage's weights file relative to an assets root, e.g.
    ``detectors/ch1_zoom_detector/ch1_zoom.pt`` or
    ``classifiers/fill_classifier/type_cls.pt``. Derived from
    ``assets_paths.json``'s own paths (everything after their ``assets/``
    segment) so it's always exactly what ``build_model_assets`` below, and
    the production controller, agree on."""
    raw = (
        assets_map["fill_classifier"]
        if stage == "fill_classifier"
        else assets_map["detectors"][stage]
    )
    parts = Path(raw).parts
    return Path(*parts[parts.index("assets") + 1:])


def build_model_assets(assets_map: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Builds a ``QModelOnyx.__init__``-ready ``model_assets`` dict pointing
    every stage at ``root / deploy_subpath(...)``. Does not check existence -
    callers loading a partially-deployed package (e.g. detectors only) get a
    dict with those paths regardless; ``QModelOnyx`` lazy-loads each asset
    and only fails when a requested stage is actually used."""
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

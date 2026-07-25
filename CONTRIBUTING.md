# Contributing

## Setup

```
pip install -e ".[dev]"
```

## Running tests

```
pytest                     # full suite
pytest tests/decode/       # one subpackage
pytest -k dp_decode        # by name
```

The suite covers every pure-logic module (corpus discovery, spacing prior,
DP decode, tier fitting, augmentation, rendering salience math, dataset
splitting, the crosscheck rescue/veto, the ordinal-evidence live state
machine) without requiring model weights or `ultralytics` — YOLO-dependent
code is tested via duck-typed stand-ins (see `tests/inference/test_crosscheck.py`
for the pattern) or exercised end-to-end with a synthetic corpus via
`qa/benchmark.py --selftest`. Thin CLI wrappers around Ultralytics training
(`training/train_detectors.py`, `training/train_fill_classifier.py`) are not
unit-tested beyond import/argument-parsing, since their behavior is
Ultralytics'.

Run `python -m src.systems.qmodel_7_onyx.qa.benchmark --selftest` for a full
integration smoke test of corpus discovery → decode → controller → benchmark
aggregation with zero model weights, using a synthetic corpus sampled from
the fitted spacing prior.

## Style

[Ruff](https://docs.astral.sh/ruff/) handles both linting and formatting;
config lives in `pyproject.toml`'s `[tool.ruff]`/`[tool.ruff.lint]`.

```
ruff check .            # lint
ruff check --fix .      # lint, auto-fixing what's safe to
ruff format .           # format
ruff format --check .   # format, check only (what CI runs)
```

Note `pyupgrade` (`UP*`) rules are deliberately not enabled — this codebase
consistently uses `typing.Optional`/`List`/`Dict` rather than the PEP 604/585
`X | None` / `list[X]` syntax; that's a style decision, not a lint finding.

## CI

Three workflows run on every push to `main` and every pull request (see
`.github/workflows/`):

- **test.yml** — the full `pytest` suite across Python 3.10/3.11/3.12 on
  both Ubuntu and Windows (this codebase went through a deliberate
  CWD/path-fragility cleanup earlier — the Windows leg is what actually
  verifies that, not just code review).
- **style.yml** — `ruff check` + `ruff format --check`.
- **build.yml** — builds the sdist/wheel and installs it into a clean venv
  to catch packaging regressions (e.g. a new module `[tool.setuptools.packages.find]`
  doesn't pick up, or a non-`.py` data file like `assets_paths.json` not
  declared in `[tool.setuptools.package-data]`) that passing tests alone
  wouldn't catch, since tests run against the editable install, not a built
  wheel.

`codeql.yml` runs a weekly + on-push security scan; `dependabot.yml`
proposes weekly dependency-update PRs for both pip and the workflow actions
themselves.

## Adding to an existing pipeline stage

Add the new module inside the relevant subpackage (`rendering/`, `decode/`,
`dataset/`, `training/`, `inference/`, `live/`, or `qa/`) and:

- Import path defaults from `paths.py` rather than hardcoding a relative
  `Path(...)` — every CLI must work regardless of the launch directory.
- Use plain relative imports for other `qmodel_7_onyx` modules
  (`from ..corpus import ...`, `from .config import ...`). The old
  `try/except` bare-import fallback for standalone script execution
  (`python some_script.py` from inside the flat directory) is gone — this is
  a package now.
- Preserve the try/except pattern only for genuinely optional external
  dependencies (the QATCH application, `ultralytics`, `scikit-learn`) — see
  `live/base_live.py` for the shape: import failure sets an availability
  sentinel, module import never raises, only construction of the
  app-dependent class raises a clear error.
- Add a corresponding test module under `tests/<subpackage>/`.

## Adding a new pipeline stage

Create a new subpackage under `src/systems/qmodel_7_onyx/` with its own
`__init__.py`, mirroring the existing stage packages' shape (a shared
library module plus thin CLI entry points with `main()` functions). Register
any new CLI as a console script in `pyproject.toml`'s `[project.scripts]` if
it's a common entry point. Add the mirrored `tests/<new_subpackage>/`
directory.

Do not build a cross-system plugin/registry abstraction for `src/systems/`
— see [ARCHITECTURE.md](ARCHITECTURE.md#deliberately-not-built-a-systems-plugin-registry)
for why that's deferred until a second system actually needs it.

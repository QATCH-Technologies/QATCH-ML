# History
## 2.0.1 (09-11-2024)
### Added
- Bounding and adjustment for POI 2
- `__init__` files for `file_io`, `models`, `tests`, and `utils` directories

### Changed
- Refactored `QModel` directory into `src` and `test` directories.
- `src/models` now contains model soruce code and untilities.
- `QTest.py` is now `q_validation.py`
- `qmodel` has been reverted to `QModel`

### Fixed
- Fixed test directory structure (unit tests are still broken!)

### Deprecated
- No deprecations

### Removed
- No removals

## 2.0.0 (09-09-2024)
### Added
- Significant Refactorted changes to the module structure.
- Added setup script.
- Added AUTHORS.md, CODE_OF_CONDUCT.md, MAKEFILE, `test_requriments.txt`, `.readthedocs.yaml`
- Added skeleton test suite (currently not working).
- `q_log.py` module in place for all future logging (not yet fully integrated)
- Added `__init__.py` for all submodules

### Changed
- Signicant refactoring to qmodel structure.
- `QModel` is now `qmodel`.
- Submodules of qmodel are renamed in AutoPep8 snake case
- `QDataPipeline` -> `q_data_pipeline`
- `QImageClusterer` -> `q_image_clusterer`
- `QTrainMulti` -> `q_train_multi`
- `FileIO` has been renamed to `file_io`
- Other has been renamed to utils
- `file_io` and utils have been moved inside of main source package QModel.

### Fixed
- Corrected bug in q_data_pipeline for `__init__()` error reporting.

### Deprecated
- No deprecations

### Removed
- Clutter has been removed. Check repo deletions.
- All pycache directories have been removed.

# History
## 2.0.11 (02-07-2025)
### Added
- Zoning restrictions for POI4-6 leading to major reductions to MSE and RMSE as well as POI 4 accuracy.
- New POI 6 post process ~27 median MAE.

### Changed
- POI 5 using MD POI 5 for initial guesses.

### Fixed
- Ordering prevents out of order predictions.

### Deprecated
- No deprecations

### Removed
- No removals
  
## 2.0.8 (11-08-2024)
### Added
- Improved restrictions for POI1-3 to prevent unreasonable prediction estimates.
- Additional sorting and restrictions to the placements of POI 4-6

### Changed
- Upgraded `q_log` utility; currently not in use.

### Fixed
- No fixes

### Deprecated
- No deprecations

### Removed
- No removals
## 2.0.7 (11-04-2024)
### Added
- No additions

### Changed
- Major improvements to POI4 bringing average case MAE to ~80
- Minor improvements to ordering on tail classification for POI 6 bringing average case MAE to ~29

### Fixed
- No fixes

### Deprecated
- No deprecations

### Removed
- No removals

## 2.0.7 (11-04-2024)
### Added
- Siginificant improvements to POI 6 predictions bringing average case MAE to ~35 for POI 6

### Changed
- No changes

### Fixed
- No fixes

### Deprecated
- No deprecations

### Removed
- No removals

## 2.0.6 (10-18-2024)
### Added
- Added downsampling functionality for `q_data_pipeline`.  Currently broken and working to integrate it but the endpoint exists.

### Changed
- No changes

### Fixed
- `q_validation` now handles return from q_multi_model correctly

### Deprecated
- No deprecations

### Removed
- No removals

## 2.0.5 (10-02-2024)
### Added
- No Additions

### Changed
- `q_multi_model` predictor does not return POI list.

### Fixed
- No fixes

### Deprecated
- No deprecations

### Removed
- No removals
## 2.0.4 (10-01-2024)
### Added
- Confidence reporting per candidate POI in candidate list.

### Changed
- `q_multi_model` predictor now returns a best point along with a candidate list that is now a list of tuples where each tuple contains a list of ca

### Fixed
- No fixes

### Deprecated
- No deprecations

### Removed
- No removals

## 2.0.3 (09-30-2024)
### Added
- POI 6 adjustment in place by adding difference and QModel signal's together and taking maxima.
- Added functionality for updating QModel but not yet a fully integrated system.
- Added new datasets pulled from Production Data.

### Changed
- Adjusted POI cannot be reported in as duplicate values in the candidtate lists 

### Fixed
- Candidate lists of points are now returned correctly.
- Each guessed POI value is added to the head of each respective candidate list.
- Added updates for edge case errors that occured during adjustment phase of testing.

### Deprecated
- No deprecations

### Removed
- No removals

## 2.0.2 (09-13-2024)
### Added
- Working on adjustment for POI 6.  Very broken at the moment!

### Changed
- No changes

### Fixed
- No fixes

### Deprecated
- No deprecations

### Removed
- No removals

## 2.0.1 (09-11-2024)
### Added
- Bounding and adjustment for POI 2 within 2.01 IQR MAE from true.
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
- Significant Refactorted changes to the modutructure.
- Added setup script.
- Added `AUTHORS.md`, `CODE_OF_CONDUCT.md`, `MAKEFILE`, `test_requriments.txt`, `.readthedocs.yaml`
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

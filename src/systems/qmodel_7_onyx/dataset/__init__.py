"""Dataset-building subpackage for the QATCH POI detection pipeline.

Turns discovered runs (:func:`..corpus.discover_runs`) into on-disk training
datasets: :mod:`.build_detectors` renders the per-stage YOLO detection
datasets for the cascade detectors, and :mod:`.build_fill_classifier` renders
the fill-type classification dataset. Both builders share their run-level
train/val split and per-tier upsampling logic from :mod:`.splitting`.
"""

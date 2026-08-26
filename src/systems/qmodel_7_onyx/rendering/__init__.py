"""Image-rendering utilities for the QModel Onyx pipeline.

Turns preprocessed sensor dataframes into the RGB images consumed by the
point-of-interest detector and fill-type classifier. The legacy v1 render
lives in :mod:`.legacy_dataprocessor`; the newer derivative-energy renders
live in :mod:`.detector_render` and :mod:`.fill_render`; geometry and
robust-statistics helpers shared between the renderers live in
:mod:`._common`. Each renderer exposes a version flag so that model
weights keep seeing the exact render they were trained on while newer
renders roll out independently.
"""

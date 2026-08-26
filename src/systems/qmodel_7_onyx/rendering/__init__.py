"""Image-rendering utilities for the QModel Onyx pipeline.

Turns preprocessed sensor dataframes into the RGB images consumed by the
point-of-interest detector and fill-type classifier. Preprocessing (raw CSV
-> interpolated, median-filtered time series) lives in :mod:`.dataprocessor`;
the derivative-energy and step-coincidence-energy salience renders live in
:mod:`.detector_render` and :mod:`.fill_render`; geometry and
robust-statistics helpers shared between the renderers live in
:mod:`._common`.
"""

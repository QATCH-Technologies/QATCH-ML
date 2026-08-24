"""Configuration constants for the QModel V6 YOLO inference pipeline.

Centralizes the tunable knobs consumed by the reverse-cascade controller,
the configuration-prior decode layer, and the zoom refinement stage, so
detector/render code and sweep scripts read from a single source of truth
instead of hardcoding thresholds.
"""

from __future__ import annotations

from typing import Dict, Optional


class QModelOnyxConfig:
    """Configuration constants for the QModel V6 YOLO pipeline.

    A plain namespace of class-level constants grouped by pipeline stage
    (detector, fill classifier, configuration-prior decode, zoom
    refinement); see the inline comments beside each group for rationale
    on individual values.

    Attributes:
        IMG_WIDTH (int): Width, in pixels, of the rendered detection image.
        IMG_HEIGHT (int): Height, in pixels, of the rendered detection image.
        MIN_SLICE_LENGTH (int): Minimum number of samples a cascade slice
            must contain to be considered detectable.
        CONF_THRESHOLD (float): Minimum YOLO detection confidence retained
            before candidate filtering.
        FILL_INFERENCE_W (int): Width, in pixels, of the image fed to the
            fill classifier at inference time.
        FILL_INFERENCE_H (int): Height, in pixels, of the image fed to the
            fill classifier at inference time.
        FILL_GEN_W (int): Width, in pixels, at which fill-classifier
            training images are generated.
        FILL_GEN_H (int): Height, in pixels, at which fill-classifier
            training images are generated.
        FILL_CLASS_MAP (Dict[str, int]): Maps a fill-classifier label to
            the number of channels the controller should search for.
        DECODE_LAMBDA (float): Default weight of the spacing log-likelihood
            relative to detection confidence in configuration-prior decode.
        DECODE_LAMBDA_PAIRS (Optional[Dict[str, float]]): Per-edge override
            of `DECODE_LAMBDA`, keyed by `"POI_i->POI_j"`; edges not
            listed fall back to `DECODE_LAMBDA`.
        DECODE_CONF_WEIGHT (float): Weight on summed (clipped) detection
            confidence in the decode objective.
        DECODE_FEAS_SLACK (float): Multiplicative slack applied to the
            learned hard gap bounds during decode.
        DECODE_MAX_CANDIDATES (int): Per-POI cap (top-K by confidence) on
            the decode lattice width.
        REFINE_WINDOW_S (float): Width, in seconds, of the zoom window
            re-rendered around each decoded channel POI during refinement.
        REFINE_MIN_CONF (float): Minimum zoom-detector confidence required
            to accept a refinement move.
        REFINE_MAX_SHIFT_FRAC (float): Maximum move the refiner may apply,
            as a fraction of the window; larger moves indicate the refiner
            latched onto a different event.
        RENDER_VERSION (int): Detection-image renderer version; must match
            the render the deployed detector weights were trained on.
        DECODE_MIN_MARGIN (float): Minimum score margin, under the decode
            objective, that a decoded configuration must beat the cascade
            configuration by before it replaces the cascade's result.
        PROG_LOAD_DATA (int): Progress-signal value emitted after raw data
            load.
        PROG_CLASSIFY (int): Progress-signal value emitted after fill
            classification.
        PROG_CONFIG (int): Progress-signal value emitted after
            configuration-prior decode.
        PROG_COMPLETE (int): Progress-signal value emitted on pipeline
            completion.
    """

    # Detector Settings
    IMG_WIDTH: int = 2560
    IMG_HEIGHT: int = 384
    MIN_SLICE_LENGTH: int = 20
    CONF_THRESHOLD: float = 0.01

    # Fill Classifier Settings
    FILL_INFERENCE_W: int = 224
    FILL_INFERENCE_H: int = 224
    FILL_GEN_W: int = 640
    FILL_GEN_H: int = 640

    # Maps YOLO classification labels to the number of channels to detect.
    # The Controller uses this Int to decide how many 'cuts' to make.
    FILL_CLASS_MAP: Dict[str, int] = {
        "no_fill": -1,
        "initial_fill": 0,
        "1ch": 1,
        "2ch": 2,
        "3ch": 3,
    }

    # Configuration-Prior Decode Settings
    # Weight of the spacing log-likelihood relative to detection confidence.
    # Scalar, or set DECODE_LAMBDA_PAIRS to weight edges individually - the
    # prior's value is not uniform across the chain: on sharp well-detected
    # events (POI2->POI3) a broad gap prior mostly drags correct detections,
    # while on ambiguous late events it is the main defence. Sweep with
    # sweep_decode.py --edge3-scales; do not hand-pick.
    DECODE_LAMBDA: float = 0.25

    # e.g. {"POI2->POI3": 0.5} - unlisted pairs default to DECODE_LAMBDA.
    DECODE_LAMBDA_PAIRS: Optional[Dict[str, float]] = {
        "POI1->POI2": 0.0,  # analytic init exclusion
        "POI2->POI3": 0.125,  # edge3_scale 0.5 x base lambda
        "POI3->POI4": 0.125,
    }
    # Weight on summed detection confidence.
    DECODE_CONF_WEIGHT: float = 1.0

    # Multiplicative slack on the learned hard gap bounds.
    DECODE_FEAS_SLACK: float = 1.5

    # Per-POI cap (top-K by confidence) on the decode lattice width.
    DECODE_MAX_CANDIDATES: int = 10

    # Zoom Refinement Settings
    # Post-decode refinement: re-render a window around each decoded channel
    # POI and re-detect with a zoom-trained detector. Targets the 1-5 s
    # localization band (candidates exist near truth but the full-run render
    # is too coarse for slow transitions). No-ops when zoom detector assets
    # are absent.
    REFINE_WINDOW_S: float = 24.0
    REFINE_MIN_CONF: float = 0.20

    # Maximum move the refiner may apply, as a fraction of the window; moves
    # larger than this indicate the refiner latched onto a different event.
    REFINE_MAX_SHIFT_FRAC: float = 0.45

    # Detection-image renderer version. MUST match the render the deployed
    # detector weights were trained on: 1 = legacy (diss/freq/difference
    # strips), 2 = v7 (diss/freq/derivative-energy salience strips, from
    # v7_render). Weights and render version ship together.
    RENDER_VERSION: int = 2

    # Hysteresis ("the decode must earn the move"): the decoded configuration
    # is only accepted if its score beats the cascade configuration's score -
    # under the SAME objective - by at least this margin. 0.0 disables the
    # guard (always accept the decode optimum). Raising it trades a few
    # missed fixes for fewer regressions on runs where the cascade was
    # already right; tune it with sweep_decode.py, not by hand.
    DECODE_MIN_MARGIN: float = 0.25

    # Progress Signal Steps
    PROG_LOAD_DATA: int = 10
    PROG_CLASSIFY: int = 30
    PROG_CONFIG: int = 40
    PROG_COMPLETE: int = 100

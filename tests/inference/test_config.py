"""Invariant checks for QModelOnyxConfig.

QModelOnyxConfig is a plain namespace of tunables, so restating each literal
value as an assertion would just be a change-detector with no regression
value. These tests instead check the structural invariants other modules
rely on implicitly (ordinal axis coverage, monotonic progress signals, valid
ranges for weights/fractions) - the kind of thing a typo or copy-paste error
in a future edit would actually break.
"""

from src.systems.qmodel_7_onyx.inference.config import QModelOnyxConfig as cfg


class TestFillClassMap:
    def test_covers_the_full_ordinal_channel_axis(self):
        """Values must be exactly {-1, 0, 1, 2, 3} - the fill classifier's
        ordinal state axis (no_fill through 3-channel)."""
        assert set(cfg.FILL_CLASS_MAP.values()) == {-1, 0, 1, 2, 3}

    def test_keys_are_the_expected_label_set(self):
        assert set(cfg.FILL_CLASS_MAP) == {"no_fill", "initial_fill", "1ch", "2ch", "3ch"}

    def test_no_fill_maps_below_the_channel_range(self):
        """no_fill must sort below every real channel count for downstream
        `if num_channels >= k` gating to treat it as "nothing detected"."""
        assert cfg.FILL_CLASS_MAP["no_fill"] < min(
            v for k, v in cfg.FILL_CLASS_MAP.items() if k != "no_fill"
        )


class TestDecodeLambdaPairs:
    def test_keys_follow_the_poi_chain_edge_naming_convention(self):
        for key in cfg.DECODE_LAMBDA_PAIRS:
            lhs, sep, rhs = key.partition("->")
            assert sep == "->", f"{key!r} is not a 'POIx->POIy' edge name"
            assert lhs.startswith("POI") and rhs.startswith("POI")

    def test_values_are_non_negative(self):
        assert all(v >= 0 for v in cfg.DECODE_LAMBDA_PAIRS.values())


class TestThresholdsAndWeights:
    def test_confidence_and_weight_knobs_are_non_negative(self):
        assert cfg.CONF_THRESHOLD >= 0
        assert cfg.DECODE_LAMBDA >= 0
        assert cfg.DECODE_CONF_WEIGHT >= 0
        assert cfg.DECODE_MIN_MARGIN >= 0

    def test_refine_min_conf_is_a_valid_probability(self):
        assert 0.0 <= cfg.REFINE_MIN_CONF <= 1.0

    def test_refine_max_shift_frac_is_a_valid_fraction_of_the_window(self):
        assert 0.0 < cfg.REFINE_MAX_SHIFT_FRAC <= 1.0

    def test_decode_feas_slack_can_only_widen_the_learned_bounds(self):
        """A slack < 1.0 would shrink the learned hard gap bounds rather than
        loosen them, defeating the point of a "slack" multiplier."""
        assert cfg.DECODE_FEAS_SLACK >= 1.0

    def test_decode_max_candidates_is_a_positive_cap(self):
        assert cfg.DECODE_MAX_CANDIDATES > 0

    def test_min_slice_length_is_a_positive_sample_count(self):
        assert cfg.MIN_SLICE_LENGTH > 0


class TestImageGeometry:
    def test_detector_image_dimensions_are_positive(self):
        assert cfg.IMG_WIDTH > 0
        assert cfg.IMG_HEIGHT > 0

    def test_fill_generation_resolution_is_at_least_the_inference_resolution(self):
        """The generated render is downscaled to inference size (never
        upscaled), so generation dimensions must dominate."""
        assert cfg.FILL_GEN_W >= cfg.FILL_INFERENCE_W
        assert cfg.FILL_GEN_H >= cfg.FILL_INFERENCE_H

    def test_refine_window_is_positive(self):
        assert cfg.REFINE_WINDOW_S > 0


class TestProgressSignals:
    def test_progress_signals_are_strictly_increasing(self):
        stages = [cfg.PROG_LOAD_DATA, cfg.PROG_CLASSIFY, cfg.PROG_CONFIG, cfg.PROG_COMPLETE]
        assert stages == sorted(stages)
        assert len(set(stages)) == len(stages)

    def test_progress_signals_are_within_the_0_to_100_range(self):
        for value in (cfg.PROG_LOAD_DATA, cfg.PROG_CLASSIFY, cfg.PROG_CONFIG, cfg.PROG_COMPLETE):
            assert 0 <= value <= 100

    def test_completion_signal_is_exactly_100(self):
        assert cfg.PROG_COMPLETE == 100

import math

import pytest

from src.systems.qmodel_7_onyx.decode import sweep
from src.systems.qmodel_7_onyx.decode.spacing_prior import GapStat, SpacingPrior


class TestLoadDump:
    """Tests for the JSONL candidate pool loading function."""

    def test_load_dump(self, tmp_path):
        """It correctly parses line-delimited JSON objects into a list."""
        dump_file = tmp_path / "candidates.jsonl"
        dump_file.write_text(
            '{"run_id": "A", "truth": {"POI1": 1.0}}\n\n{"run_id": "B", "truth": {"POI1": 2.0}}\n'
        )

        rows = sweep.load_dump(dump_file)

        assert len(rows) == 2
        assert rows[0]["run_id"] == "A"
        assert rows[1]["run_id"] == "B"


class TestTierLogic:
    """Tests for viscosity tier categorization and inverse-frequency weighting."""

    @pytest.fixture
    def edges(self):
        return sweep.TIER_EDGES_DEFAULT  # [2.66, 6.16, 18.14, 73.4]

    @pytest.mark.parametrize(
        "cp, expected_tier",
        [
            (1.0, 0),
            (5.0, 1),
            (10.0, 2),
            (50.0, 3),
            (100.0, 4),
        ],
    )
    def test_tier_of_valid_values(self, edges, cp, expected_tier):
        """It correctly assigns the integer index based on boundary edges."""
        assert sweep._tier_of(cp, edges) == expected_tier

    def test_tier_of_invalid_values(self, edges):
        """None or non-finite values fall into an out-of-bounds bucket."""
        expected = len(edges) + 1
        assert sweep._tier_of(None, edges) == expected
        assert sweep._tier_of(float("inf"), edges) == expected
        assert sweep._tier_of(float("nan"), edges) == expected

    def test_tier_weights(self, edges):
        """It calculates mean-normalized inverse-frequency weights."""
        rows = [
            {"viscosity_cP": 1.0},  # Tier 0
            {"viscosity_cP": 2.0},  # Tier 0
            {"viscosity_cP": 10.0},  # Tier 2
            {"viscosity_cP": 100.0},  # Tier 4
        ]

        weights = sweep.tier_weights(rows, edges)

        # Expected formula: total / (n_tiers * c)
        assert weights[0] == 4 / (3 * 2)
        assert weights[2] == 4 / (3 * 1)
        assert weights[4] == 4 / (3 * 1)


class TestEvaluate:
    """Tests for the offline decode-hyperparameter evaluation logic."""

    @pytest.fixture
    def real_prior(self):
        """Constructs a real SpacingPrior with 10-second optimal gaps."""
        prior = SpacingPrior(
            pairs=["POI1->POI2", "POI2->POI3", "POI3->POI4", "POI4->POI5"],
            frac_blend=0.0,  # Disable frac blend for predictable absolute seconds math
        )
        for pair in prior.pairs:
            prior.gap[pair] = GapStat(
                log_mu_sec=math.log(10.0),  # Ideal gap is exactly 10.0s
                log_sd_sec=0.1,
                log_mu_frac=math.log(0.25),
                log_sd_frac=0.1,
                min_gap_sec=1.0,  # Wide feasibility bounds
                max_gap_sec=30.0,
                n=100,
            )
        return prior

    @pytest.fixture
    def sample_row(self):
        """Provides a run configuration with pools that bracket the truth."""
        return {
            "viscosity_cP": 10.0,
            "truth": {"POI1": 10.0, "POI2": 20.0},
            "present": ["POI1", "POI2"],
            "pools": {
                # 10.0/20.0 creates a perfect 10s gap, maximizing prior score
                "POI1": [{"time": 9.5, "conf": 0.9}, {"time": 10.0, "conf": 0.95}],
                "POI2": [{"time": 19.5, "conf": 0.8}, {"time": 20.0, "conf": 0.95}],
            },
            "cascade": {
                # Cascade gap is 17.0s (heavily penalized by prior)
                "POI1": {"time": 8.0, "conf": 0.95},  # Error = 2.0
                "POI2": {"time": 25.0, "conf": 0.85},  # Error = 5.0 (Gross failure)
            },
        }

    def test_evaluate_basic_metrics(self, sample_row, real_prior):
        """It computes accurate improvements when decoding beats the cascade."""
        stats = sweep.evaluate(
            rows=[sample_row], prior=real_prior, lam=1.0, margin=0.0, gross_threshold=2.0
        )

        # Decoder selects {POI1: 10.0, POI2: 20.0} due to optimal 10s gap and high conf
        # Decoded errors: 0.0, 0.0. Cascade errors: 2.0 (ok), 5.0 (gross).
        assert stats["n"] == 2
        assert stats["mae_decoded_s"] == 0.0
        assert stats["gross_decoded"] == 0
        assert stats["gross_cascade"] == 1
        assert stats["gross_fixed"] == 1  # 5.0 cascade error was fixed
        assert stats["gross_introduced"] == 0
        assert stats["net_gross_improvement"] == 1

    def test_evaluate_margin_override(self, sample_row, real_prior):
        """It reverts to the cascade baseline if the margin rule overrides the DP decoder."""
        # A massive margin ensures the decode score never beats cascade + margin
        huge_margin = 1000.0

        stats = sweep.evaluate(
            rows=[sample_row], prior=real_prior, lam=1.0, margin=huge_margin, gross_threshold=2.0
        )

        # Forced to fallback to cascade picks (8.0 and 25.0)
        # Truth is 10.0 and 20.0, so errors are 2.0 and 5.0
        assert stats["mae_decoded_s"] == 3.5  # (2.0 + 5.0) / 2
        assert stats["gross_decoded"] == 1  # 5.0 > 2.0 threshold
        assert stats["gross_cascade"] == 1
        assert stats["gross_fixed"] == 0  # No fixes because we reverted to cascade

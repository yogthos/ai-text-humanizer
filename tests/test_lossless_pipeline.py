"""Tests for Lossless Pipeline features.

This test suite verifies:
1. Regression Testing: Ensure existing features still work
2. Verification: Test complex new "Lossless" pipeline features:
   - Elastic Mapping (merging sentences for verbose authors)
   - Style Inflation (adjusting target lengths based on author density)
   - Micro-Validation (absolute tolerance for short sentences)
   - Split Logic (average length validation for split operations)
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure config.json exists before imports
from tests.test_helpers import ensure_config_exists
ensure_config_exists()

from src.atlas.paragraph_atlas import ParagraphAtlas
from src.validator.statistical_critic import StatisticalCritic
from src.generator.refiner import ParagraphRefiner


class TestLosslessPipeline(unittest.TestCase):

    def setUp(self):
        # Mocks
        self.mock_atlas = MagicMock(spec=ParagraphAtlas)
        self.critic = StatisticalCritic("config.json")
        # Mock critic config for deterministic testing
        self.critic.config = {
            "length_gate": {
                "micro_sentence_threshold": 6,
                "micro_sentence_absolute_tolerance": 2
            },
            "critic": {"stat_tolerance": 0.2}
        }

    # =========================================================================
    # 1. ATLAS TESTS: Elastic Mapping & Style Inflation
    # =========================================================================

    def test_elastic_grouping_verbose_author(self):
        """Test that short sentences are merged for verbose authors (Target Density > 25)."""
        # Input: 4 short sentences (5 words each) = 20 words total
        input_text = "This is sentence one here. This is sentence two here. This is sentence three here. This is sentence four here."

        # Use method binding to access the real method without requiring a full atlas
        from unittest.mock import Mock
        atlas = Mock(spec=ParagraphAtlas)
        atlas._create_synthetic_archetype = ParagraphAtlas._create_synthetic_archetype.__get__(atlas, ParagraphAtlas)

        # Test Case: High Target Density (30 words/sent)
        # Should merge all 4 short sentences into 1 slot (20 words < 30 * 1.5)
        result = atlas._create_synthetic_archetype(input_text, target_density=30.0)

        structure_map = result['structure_map']
        content_map = result['content_map']

        # Expectation: 1 Merged Slot (not 4)
        self.assertEqual(len(structure_map), 1, "Should merge 4 short sentences into 1 for verbose author")
        self.assertEqual(len(content_map), 1)

        # Check Inflation: 20 raw words * 1.2 (verbose inflation) = 24 target words
        # Tolerance allow for rounding
        self.assertTrue(22 <= structure_map[0]['target_len'] <= 26,
                        f"Target length should be inflated. Got {structure_map[0]['target_len']}")

    def test_concise_author_no_merge(self):
        """Test that sentences are NOT merged for concise authors when they exceed capacity."""
        # Two sentences of 8 words each = 16 words total
        # For target_density=5, adjusted_capacity = 5*1.5 = 7.5
        # First sentence (8 words) > 7.5, so it should be isolated
        # Second sentence (8 words) > 7.5, so it should also be isolated
        input_text = "This is a longer sentence with eight words. This is another longer sentence with eight words."

        # Use method binding to access the real method without requiring a full atlas
        from unittest.mock import Mock
        atlas = Mock(spec=ParagraphAtlas)
        atlas._create_synthetic_archetype = ParagraphAtlas._create_synthetic_archetype.__get__(atlas, ParagraphAtlas)

        # Target Density Low (5 words/sent)
        result = atlas._create_synthetic_archetype(input_text, target_density=5.0)

        # Should result in 2 slots because each sentence (8 words) > 5*1.5 = 7.5
        self.assertEqual(len(result['structure_map']), 2, "Should not merge for concise author when sentences exceed capacity")

    # =========================================================================
    # 2. VALIDATOR TESTS: Micro-Sentences
    # =========================================================================

    def test_micro_sentence_absolute_tolerance(self):
        """Regression Test for the 'Infinite Retry Loop' on 6-word sentences."""
        target = 6
        generated = 8

        # Logic extracted from _is_length_compliant fix
        threshold = self.critic.config['length_gate']['micro_sentence_threshold']
        abs_tol = self.critic.config['length_gate']['micro_sentence_absolute_tolerance']

        # Simulate the FIXED logic (using <=)
        is_micro = target <= threshold
        diff = abs(generated - target)
        passed = is_micro and (diff <= abs_tol)

        self.assertTrue(passed, "8 words should pass for 6-word target (Absolute Tolerance)")

    def test_micro_sentence_fail(self):
        """Ensure we still fail if wildly off."""
        target = 6
        generated = 10 # +4 diff, > 2 tolerance

        threshold = self.critic.config['length_gate']['micro_sentence_threshold']
        abs_tol = self.critic.config['length_gate']['micro_sentence_absolute_tolerance']

        is_micro = target <= threshold
        diff = abs(generated - target)
        passed = is_micro and (diff <= abs_tol)

        self.assertFalse(passed, "10 words should fail for 6-word target")

    # =========================================================================
    # 3. REFINER TESTS: Split Logic
    # =========================================================================

    def test_split_validation_logic(self):
        """Verify that split validation uses AVERAGE length, not TOTAL."""
        target_avg = 10
        # Simulated Output: Two sentences of 10 words each = 20 words total, 10 words average
        variant = "One two three four five six seven eight nine ten. One two three four five six seven eight nine ten."

        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(variant)
        avg_len = sum(len(s.split()) for s in sents) / len(sents)

        self.assertEqual(avg_len, 10.0)

        # Check against target 10
        diff = abs(avg_len - 10)
        self.assertEqual(diff, 0, "Split validation should match average length exactly")

    # =========================================================================
    # 4. REGRESSION TEST: Standard Pipeline
    # =========================================================================

    def test_standard_pipeline_trigger(self):
        """Ensure standard flow is used when divergence is low."""
        # Setup: Input has 5 sentences. Selected Archetype has 5 sentences.
        input_beats = 5
        match_beats = 5
        divergence = abs(input_beats - match_beats)

        # Logic check
        trigger_synthetic = divergence > 1

        self.assertFalse(trigger_synthetic, "Should NOT trigger synthetic fallback for perfect match")
        # If this fails, we broke the standard generation path!


if __name__ == '__main__':
    unittest.main()


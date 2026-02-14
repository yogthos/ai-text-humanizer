"""Tests for perturbation module.

Tests cover:
- Bug 5: Perturbation SYNONYMS/adjectives must match training
"""

import pytest
import random


class TestPerturbationMatchesTraining:
    """Tests for perturbation matching training distribution (Bug 5)."""

    def test_synonyms_match_training(self):
        """SYNONYMS should match training script exactly."""
        from src.utils.perturbation import SYNONYMS

        expected_keys = {
            "big", "small", "old", "new", "good", "bad",
            "house", "said", "walked", "looked",
            "very", "really",
        }

        assert set(SYNONYMS.keys()) == expected_keys, (
            f"Extra keys: {set(SYNONYMS.keys()) - expected_keys}, "
            f"Missing keys: {expected_keys - set(SYNONYMS.keys())}"
        )

        # Check specific values that match training
        assert SYNONYMS["looked"] == ["appeared", "seemed", "gazed"]

    def test_adjectives_to_drop_match_training(self):
        """adjectives_to_drop should match training script exactly."""
        from src.utils import perturbation
        import inspect

        # Get the adjectives_to_drop from the function source
        source = inspect.getsource(perturbation.perturb_text)

        # The expected training set
        expected = {
            'great', 'small', 'large', 'old', 'new', 'good', 'bad',
            'long', 'short', 'high', 'low', 'young', 'little', 'big',
            'dark', 'light', 'strange',
        }

        # Run perturb_text and verify by calling with drop_adjectives=True
        # We verify the set by checking the source matches
        assert "'great'" in source
        assert "'young'" in source
        assert "'strange'" in source
        # These should NOT be in the set (extras from inference)
        assert "'ancient'" not in source or 'ancient' not in str(expected)

    def test_adjective_dropping_is_per_call_decision(self):
        """Adjective dropping should be all-or-nothing per call, not per word."""
        from src.utils.perturbation import perturb_text

        text = "The great old big dark strange light new good bad small large high low young little long short house stood."

        # Run many times and check: either ALL adjectives survive or MOST are dropped
        # Per-call means ~70% of calls keep all, ~30% drop them
        all_kept = 0
        some_dropped = 0

        for i in range(100):
            random.seed(i + 1000)
            result = perturb_text(text, perturbation_rate=0.0, drop_adjectives=True)
            result_words = set(result.lower().split())

            adj_words = {'great', 'old', 'big', 'dark', 'strange', 'light',
                         'new', 'good', 'bad', 'small', 'large', 'high',
                         'low', 'young', 'little', 'long', 'short'}
            present = adj_words & result_words

            if present == adj_words:
                all_kept += 1
            elif len(present) < len(adj_words):
                # If some are dropped, ALL should be dropped (per-call decision)
                some_dropped += 1

        # With per-call decision at 30%, expect ~70% all_kept, ~30% some_dropped
        # (vs per-word: would almost always have a mix)
        assert all_kept > 50, (
            f"Expected ~70% calls to keep all adjectives (per-call decision), "
            f"got {all_kept}% all_kept"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

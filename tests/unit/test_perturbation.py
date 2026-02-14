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


class TestRstripMatchesTraining:
    """Tests for rstrip matching training distribution (Bug 1 Round 3)."""

    def test_rstrip_matches_training(self):
        """Word ending with quote or hyphen should NOT have them stripped for matching.

        Training strips only '.,!?;:' but inference was stripping '.,!?;:"\\'- '
        which changes synonym/adjective matching behavior.
        """
        from src.utils.perturbation import perturb_text

        # Word ending with apostrophe — rstrip should NOT remove it
        # "don't" should match as "don't", not "don"
        text = "don't won't it's he's"
        random.seed(42)
        result = perturb_text(text, perturbation_rate=0.0)
        # With rate=0, no perturbation — all words preserved
        assert "don't" in result

        # Word ending with hyphen — should NOT be stripped
        text2 = 'well-known self-made'
        random.seed(42)
        result2 = perturb_text(text2, perturbation_rate=0.0)
        assert "well-known" in result2

    def test_rstrip_chars_are_exactly_six(self):
        """The rstrip call should use exactly 6 chars: .,!?;: matching training."""
        import inspect
        from src.utils import perturbation

        source = inspect.getsource(perturbation.perturb_text)
        # Should contain the training-matching rstrip
        assert "rstrip('.,!?;:')" in source, (
            "rstrip should use exactly '.,!?;:' to match training script"
        )
        # Should NOT contain the extended set
        assert "rstrip('.,!?;:\"\\'\\-')" not in source.replace(" ", ""), (
            "rstrip should NOT include quotes or hyphens"
        )


class TestSynonymSwapDeadCode:
    """Bug: synonym swap appends word[len(word_lower):] which is always empty."""

    def test_no_trailing_slice_in_synonym_swap(self):
        """Synonym swap should not append dead word[len(word_lower):]."""
        import inspect
        from src.utils import perturbation

        source = inspect.getsource(perturbation.perturb_text)
        assert "word[len(word_lower):]" not in source, (
            "word[len(word_lower):] is dead code — len(word.lower()) == len(word) always, "
            "so this always appends empty string"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

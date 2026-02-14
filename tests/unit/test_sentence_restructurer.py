"""Tests for sentence_restructurer module.

Tests cover:
- Bug 2: Hardcoded Lovecraft inversion prefixes
"""

import pytest


class TestInversionPrefixes:
    """Tests for inversion prefix content (Bug 2)."""

    def test_default_prefixes_are_generic(self):
        """INVERSION_PREFIXES should not contain cosmic horror vocabulary."""
        from src.vocabulary.sentence_restructurer import SentenceRestructurer

        cosmic_horror_words = [
            "mortal", "nameless", "unknowable", "eldritch", "cosmic",
            "terrible", "impenetrable", "feeble", "blasphemous",
        ]

        for prefix in SentenceRestructurer.INVERSION_PREFIXES:
            prefix_lower = prefix.lower()
            for word in cosmic_horror_words:
                assert word not in prefix_lower, (
                    f"Cosmic horror word '{word}' found in prefix: {prefix}"
                )

    def test_choose_inversion_prefix_returns_generic(self):
        """_choose_inversion_prefix should not return author-specific content."""
        from src.vocabulary.sentence_restructurer import SentenceRestructurer
        import random

        restructurer = SentenceRestructurer()
        cosmic_horror_words = [
            "mortal comprehension", "nameless things", "unknowable",
            "light fears to tread", "terrible expanse",
            "feeble understanding", "impenetrable darkness",
        ]

        # Test with various noun sets and sentences
        test_cases = [
            (["universe", "cosmos"], "The universe is vast."),
            (["limit", "boundary"], "The limit was reached."),
            (["experience", "knowledge"], "Experience teaches us."),
            (["dark", "shadow"], "The dark corner loomed."),
            (["table", "chair"], "The table was set."),
        ]

        random.seed(42)
        for nouns, sent in test_cases:
            for _ in range(10):
                prefix = restructurer._choose_inversion_prefix(nouns, sent)
                prefix_lower = prefix.lower()
                for phrase in cosmic_horror_words:
                    assert phrase not in prefix_lower, (
                        f"Cosmic horror phrase '{phrase}' found in prefix: {prefix}"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

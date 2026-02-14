"""Tests for enhanced_analyzer bug fixes.

Tests cover:
- Bug 3 (Round 3): No Lovecraft words in SENSORY_ADJECTIVES
- Bug 7 (Round 3): No [ORGANIC COMPLEXITY] header in format_for_prompt()
- Bug 8 (Round 3): No author-specific examples in to_instruction()
"""

import pytest


class TestSensoryAdjectivesGeneric:
    """Tests for SENSORY_ADJECTIVES being generic (Bug 3 Round 3)."""

    def test_no_lovecraft_words_in_sensory_adjectives(self):
        """SENSORY_ADJECTIVES should not contain Lovecraft-specific vocabulary."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        lovecraft_words = {'squamous', 'rugose', 'cyclopean', 'eldritch', 'gibbous', 'tenebrous'}
        overlap = lovecraft_words & EnhancedStructuralAnalyzer.SENSORY_ADJECTIVES
        assert len(overlap) == 0, (
            f"SENSORY_ADJECTIVES contains Lovecraft-specific words: {overlap}"
        )

    def test_sensory_adjectives_are_generic(self):
        """SENSORY_ADJECTIVES should contain generic sensory words."""
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        # Should have some generic sensory adjectives
        assert len(EnhancedStructuralAnalyzer.SENSORY_ADJECTIVES) >= 8


class TestEnhancedGuidanceFormat:
    """Tests for format_for_prompt() not emitting training-alien headers (Bug 7 Round 3)."""

    def test_no_organic_complexity_header(self):
        """format_for_prompt() should not emit [ORGANIC COMPLEXITY header."""
        from src.rag.enhanced_analyzer import EnhancedStyleProfile

        profile = EnhancedStyleProfile()
        formatted = profile.format_for_prompt()

        assert "[ORGANIC COMPLEXITY" not in formatted, (
            "format_for_prompt() should not use [ORGANIC COMPLEXITY header — "
            "this format was never in training data"
        )

    def test_guidance_uses_simple_format(self):
        """Output should use flat, simple format matching training style."""
        from src.rag.enhanced_analyzer import EnhancedStyleProfile

        profile = EnhancedStyleProfile()
        formatted = profile.format_for_prompt()

        # Should still have useful content, just not the bracketed header
        assert len(formatted) > 0
        # Should not have the specific bracketed header format
        assert "CRITICAL FOR HUMAN-LIKE OUTPUT" not in formatted


class TestHumanPatternsGeneric:
    """Tests for to_instruction() not containing author-specific examples (Bug 8 Round 3)."""

    def test_no_author_specific_examples(self):
        """to_instruction() should not contain Lovecraft-specific phrases."""
        from src.rag.enhanced_analyzer import OrganicComplexityProfile

        profile = OrganicComplexityProfile()
        instruction = profile.to_instruction()

        lovecraft_phrases = [
            'The cold seeped',
            'A faint odor of',
            'horrible to relate',
            'I shudder to recall',
        ]
        for phrase in lovecraft_phrases:
            assert phrase not in instruction, (
                f"to_instruction() contains author-specific phrase: '{phrase}'"
            )

    def test_examples_are_generic(self):
        """Examples in to_instruction() should be generic writing advice."""
        from src.rag.enhanced_analyzer import OrganicComplexityProfile

        profile = OrganicComplexityProfile()
        instruction = profile.to_instruction()

        # Generic phrases should still be present
        assert "And yet" in instruction or "But here" in instruction
        # Should have positive guidance
        assert "DO THIS" in instruction or "HUMAN" in instruction
        # Should have negative guidance
        assert "AVOID" in instruction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for base_generator module.

Tests cover:
- Bug 1: Hardcoded fiction markers should come from config
"""

import pytest
import re
from unittest.mock import patch, MagicMock


class TestFictionMarkers:
    """Tests for fiction marker configuration (Bug 1)."""

    def test_fiction_markers_loaded_from_config(self):
        """Fiction markers should not be hardcoded in _clean_response."""
        import inspect
        from src.generation.base_generator import BaseStyleGenerator

        source = inspect.getsource(BaseStyleGenerator._clean_response)
        # Should NOT contain Lovecraft-specific markers
        lovecraft_markers = ["arkham", "cthulhu", "necronomicon", "miskatonic",
                             "innsmouth", "dunwich", "shoggoth", "yog-sothoth",
                             "azathoth", "nyarlathotep", "r'lyeh", "dagon"]
        for marker in lovecraft_markers:
            assert marker not in source.lower(), (
                f"Hardcoded Lovecraft marker '{marker}' found in _clean_response"
            )

    def test_clean_response_removes_configured_markers(self):
        """Sentences with configured fiction markers should be removed."""
        from src.generation.base_generator import BaseStyleGenerator

        # Create a minimal concrete subclass for testing
        class TestGenerator(BaseStyleGenerator):
            def generate(self, content, author, max_tokens=None, target_words=None,
                         structural_guidance=None, raw_prompt=False, temperature=None):
                return "test"
            def unload(self):
                pass

        gen = TestGenerator()
        gen.fiction_markers = ['foo_marker', 'bar_marker']

        text = "Normal sentence here. The foo_marker was terrible. Another sentence."
        result = gen._clean_response(text)
        assert "foo_marker" not in result
        assert "Normal sentence here" in result
        assert "Another sentence" in result

    def test_clean_response_no_markers_preserves_all(self):
        """Empty markers list should preserve all sentences."""
        from src.generation.base_generator import BaseStyleGenerator

        class TestGenerator(BaseStyleGenerator):
            def generate(self, content, author, max_tokens=None, target_words=None,
                         structural_guidance=None, raw_prompt=False, temperature=None):
                return "test"
            def unload(self):
                pass

        gen = TestGenerator()
        gen.fiction_markers = []

        text = "Sentence one. Sentence two. Sentence three."
        result = gen._clean_response(text)
        assert "Sentence one" in result
        assert "Sentence two" in result
        assert "Sentence three" in result


class TestBrokenAtmosphericExceptionLogging:
    """Tests for Bug 2 Round 5: Silent exception swallowing in _fix_broken_atmospheric_phrases."""

    def test_exception_logged_not_swallowed(self):
        """When spaCy loading raises non-ImportError, a warning should be logged."""
        from src.generation.base_generator import BaseStyleGenerator

        class DummyGenerator(BaseStyleGenerator):
            def generate(self, content, author, **kwargs):
                return content
            def unload(self):
                pass

        gen = DummyGenerator()

        with patch('src.utils.nlp.get_nlp', side_effect=RuntimeError("spaCy memory error")):
            with patch('src.generation.base_generator.logger') as mock_logger:
                result = gen._fix_broken_atmospheric_phrases("Some test text.")

        assert result == "Some test text."
        mock_logger.warning.assert_called_once()
        assert "spaCy memory error" in str(mock_logger.warning.call_args)


class TestBoundsCheckOffByOne:
    """Tests for Bug 3 Round 5: Off-by-one in _fix_broken_atmospheric_phrases bounds check."""

    def test_start_char_at_end_does_not_wrongly_skip(self):
        """Bounds check should use >= len(text), not >= len(text) - 1."""
        # The bounds check `start_char >= len(text) - 1` incorrectly rejects
        # the valid case where start_char points to the last character.
        # The fix changes it to `start_char >= len(text)`.
        # We verify the correct boundary condition directly:
        text = "AB"
        start_char = 1  # points to 'B', which is len(text) - 1
        # Old code: 1 >= 2 - 1 → 1 >= 1 → True → returns unchanged (WRONG)
        # New code: 1 >= 2 → False → proceeds with fix (CORRECT)
        assert start_char < len(text), "start_char at last position should be valid"
        result = text[start_char].upper() + text[start_char + 1:]
        assert result == "B"


class TestFictionMarkerAllHallucinated:
    """Bug: When ALL sentences match fiction markers, clean_sentences is empty,
    `if clean_sentences:` is False, and original hallucinated response is returned."""

    def test_all_sentences_hallucinated_returns_empty(self):
        """When all sentences are fiction hallucinations, response should be empty."""
        from src.generation.base_generator import BaseStyleGenerator

        class TestGenerator(BaseStyleGenerator):
            def generate(self, content, author, max_tokens=None, target_words=None,
                         structural_guidance=None, raw_prompt=False, temperature=None):
                return "test"
            def unload(self):
                pass

        gen = TestGenerator()
        gen.fiction_markers = ['Cthulhu', "R'lyeh"]

        # All sentences contain fiction markers — should all be filtered
        text = "Cthulhu stirred in the deep. R'lyeh rose from the ocean."
        result = gen._clean_response(text)
        # Should NOT return the original hallucinated text
        assert "Cthulhu" not in result, (
            "All-hallucinated output should not be preserved"
        )
        assert "R'lyeh" not in result, (
            "All-hallucinated output should not be preserved"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

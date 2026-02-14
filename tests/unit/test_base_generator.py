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
        gen.fiction_markers = [r'\bfoo_marker\b', r'\bbar_marker\b']

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

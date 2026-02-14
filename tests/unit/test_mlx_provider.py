"""Tests for mlx_provider module.

Tests cover:
- Bug 8: Infinite recursion in _neutralize_chunked for long single sentences
"""

import pytest
from unittest.mock import patch, MagicMock


class TestNeutralizeChunked:
    """Tests for _neutralize_chunked infinite recursion (Bug 8)."""

    def test_long_single_sentence_no_recursion(self):
        """A 400-word sentence with no periods should not recurse infinitely."""
        from src.llm.mlx_provider import RTTNeutralizer

        # Create a mock instance
        provider = RTTNeutralizer.__new__(RTTNeutralizer)
        provider._model = None
        provider._tokenizer = None

        # 400-word sentence with no sentence-ending punctuation
        long_sentence = " ".join(["word"] * 400)

        call_count = 0
        original_neutralize = provider.neutralize.__func__ if hasattr(provider.neutralize, '__func__') else None

        def mock_neutralize(text, max_retries=2, monotone=False):
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                raise RecursionError("Infinite recursion detected!")
            # For chunks that are still >300 words, should go to _do_neutralize
            if len(text.split()) > 300:
                return provider._neutralize_chunked(text, max_retries, monotone)
            return f"neutralized: {text[:50]}"

        def mock_do_neutralize(text, max_retries=2, monotone=False):
            return f"directly neutralized: {text[:50]}"

        provider.neutralize = mock_neutralize
        provider._do_neutralize = mock_do_neutralize

        # Should not raise RecursionError
        result = provider._neutralize_chunked(long_sentence, max_retries=2, monotone=False)
        assert result is not None
        assert call_count <= 5  # Should not recurse excessively

    def test_chunk_over_300_words_handled(self):
        """Chunks >300 words should go to _do_neutralize directly."""
        from src.llm.mlx_provider import RTTNeutralizer

        provider = RTTNeutralizer.__new__(RTTNeutralizer)
        provider._model = None
        provider._tokenizer = None

        # Text that produces a chunk >300 words (no sentence boundaries)
        long_text = " ".join(["word"] * 400)

        do_neutralize_called = False

        def mock_do_neutralize(text, max_retries=2, monotone=False):
            nonlocal do_neutralize_called
            do_neutralize_called = True
            return "neutralized text"

        provider._do_neutralize = mock_do_neutralize

        result = provider._neutralize_chunked(long_text, max_retries=2, monotone=False)
        assert do_neutralize_called, "_do_neutralize should be called for chunks >300 words"
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

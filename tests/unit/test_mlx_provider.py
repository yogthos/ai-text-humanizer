"""Tests for mlx_provider module.

Tests cover:
- Bug 8: Infinite recursion in _neutralize_chunked for long single sentences
- Parity: MLX and DeepSeek neutralizers share _extract_entities / _restore_entities
  / _monotone_flatten byte-for-byte. The parity tests pin that equivalence so a
  later Round 2 refactor (extracting a BaseRTTNeutralizer) is a no-op on behavior.
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


class TestRttOnceHelper:
    """_rtt_once: single pass through Mandarin→English, shared by neutralize()
    and _do_neutralize() to eliminate the ~60-line duplicated loop body."""

    def _provider(self):
        from src.llm.mlx_provider import RTTNeutralizer
        provider = RTTNeutralizer.__new__(RTTNeutralizer)
        provider._model = None
        provider._tokenizer = None
        return provider

    def test_empty_mandarin_returns_none(self):
        provider = self._provider()
        provider._generate = MagicMock(side_effect=["", "unused"])

        result = provider._rtt_once("some English text here.", word_count=4)
        assert result is None

    def test_empty_english_returns_none(self):
        provider = self._provider()
        provider._generate = MagicMock(side_effect=["mandarin output", ""])

        result = provider._rtt_once("some English text here.", word_count=4)
        assert result is None

    def test_chinese_residue_returns_none(self):
        """If the Mandarin→English step leaks Chinese characters, pass failed."""
        provider = self._provider()
        provider._generate = MagicMock(
            side_effect=["some mandarin", "mixed with 中文 characters here"]
        )

        result = provider._rtt_once("source text.", word_count=3)
        assert result is None

    def test_strips_code_fences(self):
        """Leading/trailing ``` fences should be stripped from the output."""
        provider = self._provider()
        provider._generate = MagicMock(
            side_effect=["some mandarin text", "```\nplain english output\n```"]
        )

        result = provider._rtt_once("source.", word_count=2)
        assert result == "plain english output"

    def test_success_returns_cleaned_english(self):
        provider = self._provider()
        provider._generate = MagicMock(
            side_effect=["mandarin translation", "  clean english output  "]
        )

        result = provider._rtt_once("source here.", word_count=3)
        assert result == "clean english output"


class TestNeutralizerSharedMethodParity:
    """Pin that MLX and DeepSeek neutralizers agree on the pure helper methods.

    This is the Round 1 safety net for Round 2's BaseRTTNeutralizer extraction:
    the three methods below (_extract_entities, _restore_entities, _monotone_flatten)
    are currently duplicated verbatim across the two classes. After extraction,
    both subclasses must still produce identical output for the same input — these
    tests fail loudly if the refactor drifts.
    """

    @pytest.fixture
    def mlx_neutralizer(self):
        from src.llm.mlx_provider import RTTNeutralizer
        obj = RTTNeutralizer.__new__(RTTNeutralizer)
        obj._model = None
        obj._tokenizer = None
        return obj

    @pytest.fixture
    def ds_neutralizer(self):
        from src.llm.mlx_provider import DeepSeekRTTNeutralizer
        obj = DeepSeekRTTNeutralizer.__new__(DeepSeekRTTNeutralizer)
        # Don't call __init__ — avoids the API key requirement.
        return obj

    # Samples exercise: multi-word names, single caps, sentence-start caps,
    # embedded punctuation, short + long inputs, already-placeholdered text.
    SAMPLES = [
        "Jervas Dudley walked through New England toward Squire Brewster Hyde.",
        "The shadows lengthened as Cthulhu stirred in R'lyeh beneath the waves.",
        "However, although Paris fell in June, the resistance endured.",
        "Short text.",
        "Already __ENT0__ masked text with Paris inside.",
        "Punctuation: commas, semicolons; dashes — and (parentheticals).",
    ]

    @pytest.mark.parametrize("text", SAMPLES)
    def test_extract_entities_parity(self, mlx_neutralizer, ds_neutralizer, text):
        mlx_masked, mlx_map = mlx_neutralizer._extract_entities(text)
        ds_masked, ds_map = ds_neutralizer._extract_entities(text)
        assert mlx_masked == ds_masked
        assert mlx_map == ds_map

    @pytest.mark.parametrize("text", SAMPLES)
    def test_restore_entities_parity(self, mlx_neutralizer, ds_neutralizer, text):
        masked, entity_map = mlx_neutralizer._extract_entities(text)
        assert mlx_neutralizer._restore_entities(masked, entity_map) == \
            ds_neutralizer._restore_entities(masked, entity_map)

    MONOTONE_SAMPLES = [
        "The old man walked slowly down the lane. He saw many things.",
        "She came in, she sat down (quietly), she smiled; the room brightened.",
        "A very long sentence that rambles and meanders and connects many clauses with conjunctions and never quite ends properly.",
        "Short. Fragments. Here.",
        "No punctuation at all just words trailing",
    ]

    @pytest.mark.parametrize("text", MONOTONE_SAMPLES)
    def test_monotone_flatten_parity(self, mlx_neutralizer, ds_neutralizer, text):
        assert mlx_neutralizer._monotone_flatten(text) == \
            ds_neutralizer._monotone_flatten(text)

    def test_extract_entities_round_trip(self, mlx_neutralizer):
        """Masking then restoring must recover the original text exactly."""
        text = "Jervas Dudley walked through New England at dawn."
        masked, entity_map = mlx_neutralizer._extract_entities(text)
        assert mlx_neutralizer._restore_entities(masked, entity_map) == text

    def test_extract_entities_skips_sentence_start_single_caps(self, mlx_neutralizer):
        """Single capitalized word at sentence start is not masked (pattern 2 requires
        preceding punctuation/whitespace)."""
        masked, entity_map = mlx_neutralizer._extract_entities("The door opened.")
        assert "__ENT" not in masked
        assert entity_map == {}


class TestDeadActiveWorkersRemoved:
    """Regression guard for the dead `active_workers`/`workers_lock` counter that
    was written but never read. Removing them lets the ThreadPoolExecutor join
    naturally and cuts one class of future drift (a reader adding logic that
    relies on the stale counter)."""

    def test_no_active_workers_counter(self):
        import inspect
        from src.llm import mlx_provider

        source = inspect.getsource(mlx_provider)
        assert "active_workers" not in source, (
            "active_workers counter was dead code — never read. "
            "ThreadPoolExecutor.futures already tracks completion."
        )
        assert "workers_lock" not in source, (
            "workers_lock guarded only the dead active_workers counter."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

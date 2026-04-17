"""Tests for the GrammarCorrector module."""

import pytest
from unittest.mock import MagicMock, patch


class TestGrammarCorrectorConfig:
    """Tests for GrammarCorrectorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from src.vocabulary.grammar_corrector import GrammarCorrectorConfig

        config = GrammarCorrectorConfig()

        assert config.language == "en-US"
        assert "STYLE" in config.skip_categories
        assert "CASING" in config.skip_categories
        assert "PASSIVE_VOICE" in config.skip_rules
        assert "TOO_LONG_SENTENCE" in config.skip_rules
        assert config.fix_only_categories == set()  # Empty by default (fix all)

    def test_custom_language(self):
        """Test custom language setting."""
        from src.vocabulary.grammar_corrector import GrammarCorrectorConfig

        config = GrammarCorrectorConfig(language="en-GB")

        assert config.language == "en-GB"


class TestGrammarStats:
    """Tests for GrammarStats dataclass."""

    def test_default_stats(self):
        """Test default stats values."""
        from src.vocabulary.grammar_corrector import GrammarStats

        stats = GrammarStats()

        assert stats.total_matches == 0
        assert stats.filtered_matches == 0
        assert stats.corrections_applied == 0
        assert len(stats.categories_found) == 0
        assert len(stats.rules_found) == 0


class TestGrammarCorrector:
    """Tests for GrammarCorrector class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        assert corrector.config is not None
        assert corrector.config.language == "en-US"

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        from src.vocabulary.grammar_corrector import GrammarCorrector, GrammarCorrectorConfig

        config = GrammarCorrectorConfig(language="en-GB")
        corrector = GrammarCorrector(config)

        assert corrector.config.language == "en-GB"

    def test_correct_short_text_skipped(self):
        """Test that very short text is skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        # Mock to avoid loading LanguageTool
        corrector._available = True
        corrector._tool = MagicMock()

        text = "Hi."  # Less than min_words (3)
        result, stats = corrector.correct(text)

        assert result == text
        assert stats.corrections_applied == 0
        # Should not have called the tool
        corrector._tool.check.assert_not_called()

    def test_correct_when_unavailable(self):
        """Test that correction is skipped when LanguageTool is unavailable."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        corrector._available = False

        text = "This is a test sentence with grammar errors."
        result, stats = corrector.correct(text)

        assert result == text
        assert stats.corrections_applied == 0

    def test_should_skip_style_category(self):
        """Test that STYLE category is skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        mock_match = MagicMock()
        mock_match.category = "STYLE"
        mock_match.rule_id = "SOME_RULE"

        assert corrector._should_skip_match(mock_match) is True

    def test_should_skip_casing_category(self):
        """Test that CASING category is skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        mock_match = MagicMock()
        mock_match.category = "CASING"
        mock_match.rule_id = "SOME_RULE"

        assert corrector._should_skip_match(mock_match) is True

    def test_should_skip_passive_voice_rule(self):
        """Test that PASSIVE_VOICE rule is skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        mock_match = MagicMock()
        mock_match.category = "GRAMMAR"  # Would normally be fixed
        mock_match.rule_id = "PASSIVE_VOICE"

        assert corrector._should_skip_match(mock_match) is True

    def test_should_not_skip_grammar_category(self):
        """Test that GRAMMAR category is NOT skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        mock_match = MagicMock()
        mock_match.category = "GRAMMAR"
        mock_match.rule_id = "SOME_GRAMMAR_RULE"

        assert corrector._should_skip_match(mock_match) is False

    def test_should_not_skip_typos_category(self):
        """Test that TYPOS category is NOT skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()

        mock_match = MagicMock()
        mock_match.category = "TYPOS"
        mock_match.rule_id = "SOME_TYPO_RULE"

        assert corrector._should_skip_match(mock_match) is False

    def test_apply_corrections_empty_matches(self):
        """Test applying corrections with no matches."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        text = "This is a test."

        result, applied = corrector._apply_corrections(text, [])

        assert result == text
        assert applied == 0

    def test_apply_corrections_single_match(self):
        """Test applying a single correction."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        text = "The horror were lurking."

        mock_match = MagicMock()
        mock_match.offset = 11  # Position of "were"
        mock_match.error_length = 4  # Length of "were"
        mock_match.replacements = ["was"]

        result, applied = corrector._apply_corrections(text, [mock_match])

        assert result == "The horror was lurking."
        assert applied == 1

    def test_apply_corrections_multiple_matches(self):
        """Test applying multiple corrections in correct order."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        text = "She dont like hte food."

        # "dont" -> "doesn't" at position 4
        match1 = MagicMock()
        match1.offset = 4
        match1.error_length = 4
        match1.replacements = ["doesn't"]

        # "hte" -> "the" at position 14
        match2 = MagicMock()
        match2.offset = 14
        match2.error_length = 3
        match2.replacements = ["the"]

        result, applied = corrector._apply_corrections(text, [match1, match2])

        assert result == "She doesn't like the food."
        assert applied == 2

    def test_analyze_empty_when_unavailable(self):
        """Test that analyze returns empty list when unavailable."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        corrector._available = False

        result = corrector.analyze("This is a test.")

        assert result == []

    def test_correct_handles_exception(self):
        """Test that exceptions during correction are handled gracefully."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        corrector._available = True
        corrector._tool = MagicMock()
        corrector._tool.check.side_effect = Exception("Test error")

        text = "This is a test sentence."
        result, stats = corrector.correct(text)

        # Should return original text on error
        assert result == text
        assert stats.corrections_applied == 0


class TestGrammarCorrectorIntegration:
    """Integration tests that may require LanguageTool."""

    @pytest.fixture
    def corrector(self):
        """Create a grammar corrector for testing."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        # Check if LanguageTool is available
        if not corrector.available:
            pytest.skip("LanguageTool not installed")
        return corrector

    def test_correct_spelling_error(self, corrector):
        """Test correction of spelling errors."""
        # This test only runs if LanguageTool is installed
        text = "The cat sat onn the mat."
        result, stats = corrector.correct(text)

        # Should fix "onn" -> "on"
        assert "on the mat" in result or stats.corrections_applied > 0

    def test_preserve_archaic_language(self, corrector):
        """Test that archaic language is preserved."""
        text = "Whereupon the nameless dread descended upon the village."
        result, stats = corrector.correct(text)

        # Should preserve "Whereupon" - not change it
        assert "Whereupon" in result or "whereupon" in result.lower()

    def test_preserve_long_sentences(self, corrector):
        """Test that long sentences are not flagged."""
        text = (
            "The ancient and crumbling edifice stood silently upon the hill, "
            "its weathered stones bearing witness to countless eons of solitude, "
            "while the eldritch mists swirled about its foundation in patterns "
            "that seemed to defy the natural laws of our comprehension."
        )
        original_words = len(text.split())
        result, stats = corrector.correct(text)

        # Should not drastically change length (TOO_LONG_SENTENCE should be skipped)
        result_words = len(result.split())
        assert abs(result_words - original_words) < 5


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_grammar_corrector_singleton(self):
        """get_grammar_corrector() returns the same corrector across calls,
        now backed by the default Services container."""
        from src.services import default_services
        from src.vocabulary.grammar_corrector import get_grammar_corrector

        with default_services():  # fresh container, per-thread isolation
            c1 = get_grammar_corrector()
            c2 = get_grammar_corrector()
            assert c1 is c2

    def test_correct_grammar_function(self):
        """Test the convenience correct_grammar function."""
        from src.services import default_services
        from src.vocabulary.grammar_corrector import correct_grammar

        with default_services():
            # Just verify it doesn't crash (LanguageTool may not be installed)
            result = correct_grammar("This is a test.")
            assert isinstance(result, str)


class TestDefaultSkipLists:
    """Tests for the default skip categories and rules."""

    def test_default_skip_categories_complete(self):
        """Test that default skip categories include expected values."""
        from src.vocabulary.grammar_corrector import DEFAULT_SKIP_CATEGORIES

        expected = {"STYLE", "CASING", "MISC", "TYPOGRAPHY", "REDUNDANCY"}
        assert expected.issubset(DEFAULT_SKIP_CATEGORIES)

    def test_default_skip_rules_complete(self):
        """Test that default skip rules include expected values."""
        from src.vocabulary.grammar_corrector import DEFAULT_SKIP_RULES

        expected = {
            "PASSIVE_VOICE",
            "TOO_LONG_SENTENCE",
            "SENTENCE_FRAGMENT",
            "EN_QUOTES",
        }
        assert expected.issubset(DEFAULT_SKIP_RULES)


class TestGrammarCorrectorDefaults:
    """Tests for grammar corrector default values (Bug 17)."""

    def test_fix_only_categories_default_empty(self):
        """Default fix_only_categories should be empty (fix all categories)."""
        from src.vocabulary.grammar_corrector import GrammarCorrectorConfig

        config = GrammarCorrectorConfig()
        assert config.fix_only_categories == set(), (
            f"fix_only_categories should default to empty set, got {config.fix_only_categories}"
        )


class TestReplacementGuards:
    """Tests for H12: guard suspicious replacements.

    LanguageTool sometimes suggests replacements that would rewrite entire
    sentences or replace text with an empty string. Applying these blindly
    destroys authorial voice, which is the opposite of what this module is for.
    """

    def test_empty_replacement_rejected(self):
        """Replacement that is an empty string should be skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        text = "The horror lurks in the shadows."

        mock_match = MagicMock()
        mock_match.offset = 4
        mock_match.error_length = 6  # "horror"
        mock_match.replacements = [""]  # Empty replacement

        result, applied = corrector._apply_corrections(text, [mock_match])
        assert result == text, "Empty replacement should not delete text"
        assert applied == 0

    def test_suspiciously_long_replacement_rejected(self):
        """Replacement dramatically longer than the error should be skipped."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        text = "The horror were lurking in the shadows."

        mock_match = MagicMock()
        mock_match.offset = 11
        mock_match.error_length = 4  # "were"
        # LanguageTool occasionally suggests whole-sentence rewrites
        mock_match.replacements = [
            "was lurking, its presence an indescribable shadow upon the night"
        ]

        result, applied = corrector._apply_corrections(text, [mock_match])
        assert result == text, (
            f"Suspiciously long replacement should be skipped, got: {result}"
        )
        assert applied == 0

    def test_normal_replacement_still_applied(self):
        """Short, sane replacements (e.g., were -> was) should still apply."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        text = "The horror were lurking."

        mock_match = MagicMock()
        mock_match.offset = 11
        mock_match.error_length = 4  # "were"
        mock_match.replacements = ["was"]

        result, applied = corrector._apply_corrections(text, [mock_match])
        assert result == "The horror was lurking."
        assert applied == 1

    def test_slightly_longer_replacement_applied(self):
        """Slightly longer replacement (dont -> doesn't) should be allowed."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        text = "She dont know."

        mock_match = MagicMock()
        mock_match.offset = 4
        mock_match.error_length = 4  # "dont"
        mock_match.replacements = ["doesn't"]  # 7 chars — < 3x

        result, applied = corrector._apply_corrections(text, [mock_match])
        assert result == "She doesn't know."
        assert applied == 1


class TestCorrectionsAppliedCounterAccuracy:
    """Review finding: stats.corrections_applied was set to len(filtered),
    but _apply_corrections skips matches with no replacements and matches
    rejected by the safety guard. The counter was lying about work done."""

    def _make_match(self, offset, error_length, replacements):
        m = MagicMock()
        m.offset = offset
        m.error_length = error_length
        m.replacements = replacements
        m.category = "GRAMMAR"
        m.rule_id = "RULE"
        return m

    def test_counter_excludes_empty_replacement_matches(self):
        """A match with [''] replacements gets skipped but used to be counted."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        corrector._available = True

        safe_match = self._make_match(0, 3, ["was"])  # "The" -> "was" (applied)
        empty_match = self._make_match(10, 6, [""])  # skipped

        mock_tool = MagicMock()
        mock_tool.check.return_value = [safe_match, empty_match]
        corrector._tool = mock_tool

        _, stats = corrector.correct("The horror were lurking.")

        assert stats.corrections_applied == 1, (
            "Counter should reflect replacements actually written, "
            f"not the pre-filter match count. Got {stats.corrections_applied}"
        )

    def test_counter_excludes_missing_replacements(self):
        """A match with [] replacements gets skipped but used to be counted."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        corrector._available = True

        no_repl_match = self._make_match(0, 5, [])  # skipped: no replacement

        mock_tool = MagicMock()
        mock_tool.check.return_value = [no_repl_match]
        corrector._tool = mock_tool

        _, stats = corrector.correct("Some text here for testing.")

        assert stats.corrections_applied == 0

    def test_counter_excludes_unsafe_long_replacements(self):
        """A match the safety guard rejects gets skipped but used to be counted."""
        from src.vocabulary.grammar_corrector import GrammarCorrector

        corrector = GrammarCorrector()
        corrector._available = True

        unsafe_match = self._make_match(
            0, 3,
            ["a sprawling whole-sentence rewrite that would rewrite all of this"],
        )

        mock_tool = MagicMock()
        mock_tool.check.return_value = [unsafe_match]
        corrector._tool = mock_tool

        _, stats = corrector.correct("The quick brown fox jumps over.")

        assert stats.corrections_applied == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

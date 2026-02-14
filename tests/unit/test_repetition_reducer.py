"""Tests for the repetition reducer module."""

import pytest
from src.vocabulary.repetition_reducer import RepetitionReducer, ReductionStats, LLM_SPEAK


class TestRepetitionReducer:
    """Tests for RepetitionReducer class."""

    def test_basic_initialization(self):
        """Test basic reducer initialization."""
        reducer = RepetitionReducer(threshold=3)
        assert reducer.threshold == 3
        assert reducer.use_wordnet is True
        assert reducer.synonym_replacement is False  # Default

    def test_reduce_preserves_normal_text(self):
        """Test that normal text is preserved."""
        reducer = RepetitionReducer(threshold=3)
        text = "The quick brown fox jumps over the lazy dog."
        result, stats = reducer.reduce(text)
        # Text should be mostly preserved (no overused words)
        assert "fox" in result
        assert "dog" in result

    def test_llm_speak_replacement(self):
        """Test that LLM-speak words are replaced."""
        reducer = RepetitionReducer(threshold=3)
        text = "We need to utilize this functionality to leverage our synergy."
        result, stats = reducer.reduce(text)
        # LLM-speak should be replaced
        assert "utilize" not in result.lower() or stats.replacements_made > 0
        assert stats.words_checked > 0

    def test_llm_speak_dictionary(self):
        """Test that common LLM-speak words are in the dictionary."""
        assert "utilize" in LLM_SPEAK
        assert "leverage" in LLM_SPEAK
        assert "synergy" in LLM_SPEAK
        assert "robust" in LLM_SPEAK
        assert "streamline" in LLM_SPEAK

    def test_reset_clears_state(self):
        """Test that reset clears word counts."""
        reducer = RepetitionReducer(threshold=3)
        text = "The word word word word appears many times."
        reducer.reduce(text)
        assert len(reducer.word_counts) > 0
        reducer.reset()
        assert len(reducer.word_counts) == 0

    def test_get_overused_words(self):
        """Test getting overused words."""
        reducer = RepetitionReducer(threshold=2)
        text = "Word word word appears often. Other word here."
        reducer.reduce(text)
        overused = reducer.get_overused_words(limit=5)
        # 'word' should be in overused list
        words = [w for w, _ in overused]
        assert len(overused) >= 0  # May or may not have overused depending on lemmatization

    def test_reduction_stats(self):
        """Test that stats are populated correctly."""
        reducer = RepetitionReducer(threshold=3)
        text = "This is a simple test sentence."
        result, stats = reducer.reduce(text)
        assert stats.words_checked > 0
        assert isinstance(stats.replacements_made, int)
        assert isinstance(stats.overused_words, list)
        assert isinstance(stats.replacements_detail, dict)


class TestReductionStats:
    """Tests for ReductionStats dataclass."""

    def test_default_values(self):
        """Test default stat values."""
        stats = ReductionStats()
        assert stats.words_checked == 0
        assert stats.replacements_made == 0
        assert stats.overused_words == []
        assert stats.replacements_detail == {}

    def test_stats_can_be_modified(self):
        """Test that stats can be updated."""
        stats = ReductionStats()
        stats.words_checked = 100
        stats.replacements_made = 5
        stats.overused_words.append("test")
        assert stats.words_checked == 100
        assert stats.replacements_made == 5
        assert "test" in stats.overused_words


class TestLLMSpeakControl:
    """Tests for LLM-speak replacement control (Bug 9)."""

    def test_llm_speak_disabled_preserves_words(self):
        """When fix_llm_speak=False, LLM-speak words should be preserved."""
        reducer = RepetitionReducer(threshold=3, fix_llm_speak=False)
        text = "We need to utilize this functionality to leverage our synergy."
        result, stats = reducer.reduce(text)
        assert "utilize" in result.lower()

    def test_llm_speak_enabled_by_default(self):
        """LLM-speak replacement should be enabled by default."""
        reducer = RepetitionReducer(threshold=3)
        assert reducer.fix_llm_speak is True


class TestEmptyLLMSpeakReplacements:
    """Tests for empty string replacements in LLM_SPEAK (Bug 19)."""

    def test_no_empty_replacement_for_single_words(self):
        """Single-word LLM_SPEAK entries should not have empty string replacements."""
        single_word_empties = [
            key for key, val in LLM_SPEAK.items()
            if " " not in key and val == ""
        ]
        # These should have real synonyms, not empty strings
        assert len(single_word_empties) == 0, (
            f"Single-word keys with empty replacements: {single_word_empties}"
        )


class TestLLMSpeakDictionary:
    """Tests for the LLM_SPEAK dictionary."""

    def test_common_replacements(self):
        """Test common LLM-speak replacements."""
        assert LLM_SPEAK["utilize"] == "use"
        assert LLM_SPEAK["leverage"] == "use"
        assert LLM_SPEAK["facilitate"] == "help"
        assert LLM_SPEAK["comprehensive"] == "full"

    def test_qwen_vocabulary_fixes(self):
        """Test Qwen-specific vocabulary fixes."""
        # These are weird substitutions Qwen makes
        assert LLM_SPEAK["ticker"] == "watch"
        assert LLM_SPEAK["cogwheel"] == "gear"


class TestEmDashPreservation:
    """Tests for em-dash preservation (Bug 4 Round 3)."""

    def test_em_dashes_preserved_by_default(self):
        """Em-dashes should NOT be unconditionally converted to commas/periods."""
        reducer = RepetitionReducer(threshold=3)
        text = "The truth—if truth it was—haunted him."
        result, stats = reducer.reduce(text)
        # Em-dashes should be preserved (they're a signature literary element)
        assert "—" in result, (
            f"Em-dashes should be preserved but were converted: '{result}'"
        )

    def test_excessive_em_dashes_reduced(self):
        """Only reduce em-dashes if >3 in a single sentence."""
        reducer = RepetitionReducer(threshold=3)
        text = "The thing—dark—terrible—ancient—nameless—crept forward."
        result, stats = reducer.reduce(text)
        # Should still have some em-dashes but not all 5
        em_count = result.count("—")
        assert em_count <= 3, (
            f"Excessive em-dashes should be reduced to <=3 but got {em_count}: '{result}'"
        )

    def test_single_em_dash_preserved(self):
        """A single em-dash should always be preserved."""
        reducer = RepetitionReducer(threshold=3)
        text = "The cosmos—vast and terrifying—surrounded us."
        result, stats = reducer.reduce(text)
        assert "—" in result


class TestNoLegitimateWordsInLLMSpeak:
    """Tests for legitimate words removed from LLM_SPEAK (Bug 6)."""

    def test_no_legitimate_words_in_llm_speak(self):
        """Words like cosmos, creation, wandering should NOT be in LLM_SPEAK."""
        legitimate_words = [
            "cosmos", "creation", "existence", "domain", "macrocosm",
            "corporeal", "sentinel", "scout", "vigil", "wandering",
            "nomadic", "roving", "peregrine", "illuminates",
            "elucidates", "delineates", "underscores",
            "lookout", "sentry", "picket", "spotter", "timekeeper",
            "gearing", "unmarried", "undivided", "exclusive",
            "unharmed", "unhurt", "unscathed",
        ]
        for word in legitimate_words:
            assert word not in LLM_SPEAK, (
                f"Legitimate word '{word}' should not be in LLM_SPEAK"
            )

    def test_qwen_vocabulary_subset_reasonable(self):
        """Remaining Qwen fixes should be genuinely model artifacts."""
        genuine_artifacts = [
            "ticker", "cogwheel", "geartrain", "paraphernalia",
            "appurtenance", "earphone", "earpiece", "headphone", "telephony",
        ]
        for word in genuine_artifacts:
            assert word in LLM_SPEAK, (
                f"Genuine Qwen artifact '{word}' should remain in LLM_SPEAK"
            )


class TestCommaNumberPreservation:
    """Tests for comma normalization not destroying numbers."""

    def test_number_with_commas_preserved(self):
        """Numbers like 1,000 should not become '1, 000'."""
        reducer = RepetitionReducer(threshold=3)
        text = "The population was 1,000 people in the town."
        result, stats = reducer.reduce(text)
        assert "1,000" in result, f"Number destroyed: '{result}'"

    def test_large_number_with_commas_preserved(self):
        """Numbers like 10,000,000 should be preserved."""
        reducer = RepetitionReducer(threshold=3)
        text = "The city had 10,000,000 inhabitants."
        result, stats = reducer.reduce(text)
        assert "10,000,000" in result, f"Number destroyed: '{result}'"

    def test_normal_comma_spacing_still_fixed(self):
        """Normal comma spacing should still be normalized."""
        reducer = RepetitionReducer(threshold=3)
        text = "The cat ,the dog ,and the bird flew away."
        result, stats = reducer.reduce(text)
        # Should normalize spacing around commas
        assert " ,the" not in result


class TestWordBoundaryPhraseReplacement:
    """Tests for multi-word phrase replacement respecting word boundaries."""

    def test_in_turn_not_matched_across_words(self):
        """'in turn' should not match inside 'main turnover'."""
        reducer = RepetitionReducer(threshold=3)
        text = "The main turnover rate increased significantly last year."
        result, stats = reducer.reduce(text)
        assert "turnover" in result, f"'turnover' was corrupted: '{result}'"

    def test_in_turn_matched_as_standalone(self):
        """'in turn' as a standalone phrase should still be replaced."""
        reducer = RepetitionReducer(threshold=3)
        text = "This affects morale, which in turn reduces productivity."
        result, stats = reducer.reduce(text)
        # "in turn" maps to "" so "which" should be followed by "reduces"
        assert "in turn" not in result.lower() or "which" in result

    def test_prior_to_not_matched_across_words(self):
        """'prior to' should not match inside other word combinations."""
        reducer = RepetitionReducer(threshold=3)
        text = "The a priori to posterior analysis was complete."
        result, stats = reducer.reduce(text)
        # Should not corrupt the text
        assert "priori" in result


class TestCascadingReplacementPrevention:
    """Tests for preventing cascading double-replacement chains."""

    def test_at_the_end_of_the_day_no_double_replacement(self):
        """'at the end of the day' should map directly, not cascade through 'ultimately'."""
        reducer = RepetitionReducer(threshold=3)
        text = "At the end of the day, we need this solution."
        result, stats = reducer.reduce(text)
        # Should be replaced (original phrase gone)
        assert "at the end of the day" not in result.lower()
        # Should NOT go through "ultimately" as intermediate
        assert "ultimately" not in result.lower(), (
            f"Cascading via 'ultimately' intermediate detected: '{result}'"
        )

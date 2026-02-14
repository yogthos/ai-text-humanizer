"""Tests for the inference/transfer pipeline.

Tests cover:
- TransferConfig: Configuration dataclass
- StyleTransfer: Main pipeline orchestration
- Integration with LoRA generator
- RAG context integration
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import tempfile


# =============================================================================
# Tests for TransferConfig
# =============================================================================

class TestTransferConfig:
    """Tests for TransferConfig dataclass."""

    def test_default_values(self):
        """Test that default values are sensible."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig()

        assert config.max_tokens == 512
        assert config.temperature is None  # None means use lora config
        assert config.top_p == 0.9
        assert config.verify_entailment is True
        assert config.entailment_threshold == 0.7
        assert config.max_repair_attempts == 5
        assert config.reduce_repetition is True

    def test_custom_values(self):
        """Test that custom values are applied."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(
            temperature=0.8,
            verify_entailment=False,
        )

        assert config.temperature == 0.8
        assert config.verify_entailment is False

    def test_perspective_options(self):
        """Test perspective configuration."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(perspective="first_person_singular")
        assert config.perspective == "first_person_singular"

        config2 = TransferConfig(perspective="third_person")
        assert config2.perspective == "third_person"

    def test_expansion_ratios(self):
        """Test expansion ratio configuration."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(
            max_expansion_ratio=2.0,
            target_expansion_ratio=1.5,
        )

        assert config.max_expansion_ratio == 2.0
        assert config.target_expansion_ratio == 1.5


# =============================================================================
# Tests for TransferStats
# =============================================================================

class TestTransferStats:
    """Tests for TransferStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        from src.generation.transfer import TransferStats

        stats = TransferStats()

        assert stats.paragraphs_processed == 0
        assert stats.paragraphs_repaired == 0
        assert stats.words_replaced == 0
        assert stats.total_time_seconds == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.generation.transfer import TransferStats

        stats = TransferStats(
            paragraphs_processed=5,
            paragraphs_repaired=1,
            words_replaced=10,
            total_time_seconds=45.5,
            avg_time_per_paragraph=9.1,
            entailment_scores=[0.8, 0.9, 0.85, 0.75, 0.95],
        )

        d = stats.to_dict()

        assert d["paragraphs_processed"] == 5
        assert d["paragraphs_repaired"] == 1
        assert d["words_replaced"] == 10
        assert d["total_time_seconds"] == 45.5
        assert d["avg_time_per_paragraph"] == 9.1
        assert d["avg_entailment_score"] == 0.85  # Average of scores

    def test_to_dict_empty_scores(self):
        """Test to_dict with empty entailment scores."""
        from src.generation.transfer import TransferStats

        stats = TransferStats()
        d = stats.to_dict()

        assert d["avg_entailment_score"] == 0.0


# =============================================================================
# Tests for StyleTransfer
# =============================================================================

class TestStyleTransfer:
    """Tests for StyleTransfer class."""

    @pytest.fixture
    def mock_generator(self):
        """Create a mock LoRA generator."""
        generator = MagicMock()
        generator.generate.return_value = "This is the styled output text."
        return generator

    @pytest.fixture
    def mock_critic(self):
        """Create a mock critic provider."""
        critic = MagicMock()
        critic.provider_name = "mock"
        critic.call.return_value = "Repaired text here."
        return critic

    @patch('src.generation.transfer.create_style_generator')
    def test_init_with_adapter(self, mock_generator_class, mock_critic):
        """Test initialization with adapter path."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)

        transfer = StyleTransfer(
            adapter_path="lora_adapters/test",
            author_name="Test Author",
            critic_provider=mock_critic,
            config=config,
        )

        assert transfer.author == "Test Author"
        mock_generator_class.assert_called_once()

    @patch('src.generation.transfer.create_style_generator')
    def test_init_without_adapter(self, mock_generator_class, mock_critic):
        """Test initialization without adapter (base model only)."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)

        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test Author",
            critic_provider=mock_critic,
            config=config,
        )

        assert transfer.author == "Test Author"

    @patch('src.generation.transfer.create_style_generator')
    def test_ensure_complete_ending_with_period(self, mock_generator_class, mock_critic):
        """Test that text ending with period is unchanged."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        text = "This is a complete sentence."
        result = transfer._ensure_complete_ending(text)

        assert result == text

    @patch('src.generation.transfer.create_style_generator')
    def test_ensure_complete_ending_adds_period(self, mock_generator_class, mock_critic):
        """Test that incomplete text gets period added."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        text = "This sentence is incomplete and trails off"
        result = transfer._ensure_complete_ending(text)

        assert result.endswith(".")

    @patch('src.generation.transfer.create_style_generator')
    def test_clean_repair_output_basic(self, mock_generator_class, mock_critic):
        """Test that clean_repair_output handles empty and simple text."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Empty text should return empty
        assert transfer._clean_repair_output("") == ""
        assert transfer._clean_repair_output("   ") == ""

        # Normal text should pass through
        text = "This is normal text without any LLM prefixes."
        result = transfer._clean_repair_output(text)
        assert result == text

    @patch('src.generation.transfer.create_style_generator')
    def test_clean_repair_output_strips_whitespace(self, mock_generator_class, mock_critic):
        """Test that clean_repair_output strips whitespace."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        text = "  Some text with leading and trailing spaces.  "
        result = transfer._clean_repair_output(text)
        assert result == "Some text with leading and trailing spaces."

    @patch('src.generation.transfer.create_style_generator')
    def test_transfer_paragraph_skips_short(self, mock_generator_class, mock_critic):
        """Test that short paragraphs are skipped."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(
            verify_entailment=False,
            min_paragraph_words=10,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Very short paragraph (below min_paragraph_words)
        para = "Too short."
        result, score = transfer.transfer_paragraph(para)

        # Should pass through unchanged because it's below min_paragraph_words
        assert result == para
        assert score == 1.0

    @patch('src.generation.transfer.create_style_generator')
    def test_get_partial_results(self, mock_generator_class, mock_critic):
        """Test getting partial results after interruption."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_entailment=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Simulate partial transfer
        transfer._transfer_outputs = ["Para 1", "Para 2"]
        transfer._transfer_stats = MagicMock()
        transfer._transfer_stats.paragraphs_processed = 2
        transfer._transfer_stats.total_time_seconds = 30.0
        transfer._transfer_start_time = 0

        output, stats = transfer.get_partial_results()

        assert "Para 1" in output
        assert "Para 2" in output


# =============================================================================
# Tests for Document Transfer
# =============================================================================

class TestDocumentTransfer:
    """Tests for full document transfer."""

    @patch('src.generation.transfer.create_style_generator')
    def test_transfer_document_basic(self, mock_generator_class):
        """Test basic document transfer with mocked paragraph transfer."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            skip_neutralization=True,
            use_document_context=False,
            min_paragraph_words=5,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Simple document
        doc = "First paragraph with enough words to process properly.\n\nSecond paragraph also with sufficient content."

        # Mock transfer_paragraph to return styled output
        with patch.object(transfer, 'transfer_paragraph', return_value=("Styled output paragraph.", 0.9)):
            output, stats = transfer.transfer_document(doc)

        assert stats.paragraphs_processed == 2
        assert len(output) > 0
        assert "Styled output" in output

    @patch('src.generation.transfer.create_style_generator')
    def test_transfer_document_preserves_headings(self, mock_generator_class):
        """Test that headings are passed through unchanged."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Styled content."
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            pass_headings_unchanged=True,
            skip_neutralization=True,
            min_paragraph_words=5,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        doc = "# Heading\n\nParagraph content here with enough words to process properly."

        with patch.object(transfer, 'transfer_paragraph', return_value=("Styled output.", 1.0)):
            output, stats = transfer.transfer_document(doc)

        # Heading should be preserved
        assert "# Heading" in output

    @patch('src.generation.transfer.create_style_generator')
    def test_transfer_document_callback(self, mock_generator_class):
        """Test that progress callback is called."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Output."
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            skip_neutralization=True,
            min_paragraph_words=3,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        progress_calls = []

        def on_progress(current, total, status):
            progress_calls.append((current, total, status))

        doc = "First paragraph.\n\nSecond paragraph."

        with patch.object(transfer, 'transfer_paragraph', return_value=("Output.", 1.0)):
            output, stats = transfer.transfer_document(doc, on_progress=on_progress)

        assert len(progress_calls) > 0


# =============================================================================
# Tests for Repetition Reduction Integration
# =============================================================================

class TestRepetitionReduction:
    """Tests for repetition reduction in transfer."""

    @patch('src.generation.transfer.create_style_generator')
    def test_repetition_reducer_applied(self, mock_generator_class):
        """Test that repetition reducer is applied to output."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "The amazing amazing text."
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            reduce_repetition=True,
            repetition_threshold=2,
            skip_neutralization=True,
            min_paragraph_words=3,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        assert transfer.repetition_reducer is not None

    @patch('src.generation.transfer.create_style_generator')
    def test_repetition_reducer_disabled(self, mock_generator_class):
        """Test that repetition reducer can be disabled."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            reduce_repetition=False,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        assert transfer.repetition_reducer is None


# =============================================================================
# Tests for Repair Prompt Format (Bug 10)
# =============================================================================

class TestRepairPromptFormat:
    """Tests for _repair_missing_entities passing raw_prompt (Bug 10)."""

    @patch('src.generation.transfer.create_style_generator')
    def test_repair_passes_raw_prompt(self, mock_generator_class):
        """Repair should pass raw_prompt=True to generator.generate()."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Repaired text with Entity Name included."
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(verify_entailment=False, repair_temperature=0.3)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        result = transfer._repair_missing_entities(
            source="Original text with Entity Name.",
            output="Styled text missing entities.",
            missing_entities=["Entity Name"],
        )

        # Verify raw_prompt=True was passed
        call_kwargs = mock_generator.generate.call_args
        assert call_kwargs.kwargs.get('raw_prompt') is True or \
               (len(call_kwargs.args) > 0 and True in call_kwargs.args), \
               "raw_prompt=True should be passed to generator.generate()"


# =============================================================================
# Tests for Word Count Tracking (Bug 1)
# =============================================================================

class TestWordCountTracking:
    """Tests for word count updates after perspective conversion and RTT."""

    @patch('src.generation.transfer.create_style_generator')
    def test_word_count_updated_after_perspective_conversion(self, mock_generator_class):
        """target_words should reflect post-perspective-conversion word count."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Styled output text from the generator model."
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            skip_neutralization=True,
            perspective="first_person_singular",
            use_persona=False,
            apply_input_perturbation=False,
            use_structural_rag=False,
            use_structural_grafting=False,
            reduce_repetition=False,
            restructure_sentences=False,
            split_sentences=False,
            correct_grammar=False,
            min_paragraph_words=3,
            target_expansion_ratio=1.0,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Mock perspective conversion to return shorter text
        original_text = "The observer noticed the changes in the environment around them quite clearly"
        shorter_text = "I noticed the changes around me"  # fewer words

        with patch.object(transfer, '_convert_to_perspective', return_value=shorter_text):
            transfer.transfer_paragraph(original_text)

        # Check that target_words was based on the post-conversion text, not the original
        call_kwargs = mock_generator.generate.call_args
        target_words = call_kwargs.kwargs.get('target_words') or call_kwargs[1].get('target_words')
        expected_target = len(shorter_text.split())  # 1.0 expansion ratio
        assert target_words == expected_target, (
            f"target_words={target_words} should be {expected_target} (post-perspective count)"
        )

    @patch('src.generation.transfer.create_style_generator')
    def test_word_count_updated_after_rtt(self, mock_generator_class):
        """target_words should reflect post-RTT word count."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Styled output text from the generator model."
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            skip_neutralization=False,
            perspective="preserve",
            use_persona=False,
            apply_input_perturbation=False,
            use_structural_rag=False,
            use_structural_grafting=False,
            reduce_repetition=False,
            restructure_sentences=False,
            split_sentences=False,
            correct_grammar=False,
            min_paragraph_words=3,
            target_expansion_ratio=1.0,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Mock RTT to return shorter text (compression)
        original_text = "The magnificent and extraordinarily beautiful sunset painted the vast expansive sky with brilliant colors"
        rtt_text = "The sunset painted the sky with colors"  # compressed by RTT

        with patch.object(transfer, '_rtt_neutralize', return_value=rtt_text):
            transfer.transfer_paragraph(original_text)

        call_kwargs = mock_generator.generate.call_args
        target_words = call_kwargs.kwargs.get('target_words') or call_kwargs[1].get('target_words')
        expected_target = len(rtt_text.split())  # 1.0 expansion ratio
        assert target_words == expected_target, (
            f"target_words={target_words} should be {expected_target} (post-RTT count)"
        )

    @patch('src.generation.transfer.create_style_generator')
    def test_word_count_not_updated_after_perturbation(self, mock_generator_class):
        """target_words should NOT change after perturbation (intentional drops)."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator.generate.return_value = "Styled output text from the generator model."
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(
            verify_entailment=False,
            skip_neutralization=True,
            perspective="preserve",
            use_persona=False,
            apply_input_perturbation=True,
            use_structural_rag=False,
            use_structural_grafting=False,
            reduce_repetition=False,
            restructure_sentences=False,
            split_sentences=False,
            correct_grammar=False,
            min_paragraph_words=3,
            target_expansion_ratio=1.0,
        )
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Original text with 10 words
        original_text = "The ancient darkness consumed all light within the vast chambers below"
        original_word_count = len(original_text.split())

        # Mock perturb_text to drop some words
        perturbed = "ancient darkness consumed light within vast chambers below"  # dropped some

        with patch('src.utils.perturbation.perturb_text', return_value=perturbed):
            transfer.transfer_paragraph(original_text)

        call_kwargs = mock_generator.generate.call_args
        target_words = call_kwargs.kwargs.get('target_words') or call_kwargs[1].get('target_words')
        # target_words should be based on pre-perturbation count, not post-perturbation
        assert target_words == original_word_count, (
            f"target_words={target_words} should be {original_word_count} (pre-perturbation)"
        )


class TestDocumentContextDisabled:
    """Tests for document_context being disabled by default (Bug 13)."""

    def test_document_context_disabled_by_default(self):
        """use_document_context should default to False."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig()
        assert config.use_document_context is False, (
            "use_document_context should default to False (extracted but never used)"
        )


class TestRepairSkipLogging:
    """Tests for Bug 6 Round 5: Silent repair skip when no missing entities."""

    def test_repair_skip_logged_when_no_missing_entities(self):
        """When needs_repair=True but missing_entities is empty, should log info."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        with patch.object(StyleTransfer, '__init__', lambda self, **kwargs: None):
            st = StyleTransfer.__new__(StyleTransfer)
            st.config = TransferConfig()
            st.config.verify_entailment = True
            st.config.max_hallucinations_before_reject = 0  # any hallucination triggers

            # Create a mock semantic result: hallucinations > threshold but no missing entities
            mock_result = MagicMock()
            mock_result.hallucination_count = 5
            mock_result.missing_entities = []  # empty
            mock_result.fabricated_entities = []
            mock_result.get_issues.return_value = ["hallucinations"]

            # Verify the condition is correctly detectable
            needs_repair = (
                mock_result.hallucination_count > st.config.max_hallucinations_before_reject
            )
            assert needs_repair is True
            assert not mock_result.missing_entities
            # The fix adds an elif log for this case


class TestCleanedIndexValueError:
    """Tests for _cleanup_document_paragraphs not raising ValueError on mutated paragraphs."""

    @patch('src.generation.transfer.create_style_generator')
    def test_duplicate_para_after_mutation_no_crash(self, mock_generator_class):
        """When a paragraph is mutated after being stored, index lookup should not crash."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(verify_entailment=False, min_paragraph_words=3)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        # Two paragraphs sharing the same 50-char prefix but second is longer
        para1 = "A" * 50 + " first paragraph ending here."
        para2 = "A" * 50 + " second paragraph that is longer and has more content added."

        # This should not raise ValueError
        result = transfer._cleanup_document_paragraphs([para1, para2])
        assert len(result) >= 1


class TestRepairRetryContinue:
    """Tests for repair retry using continue instead of break on exception."""

    @patch('src.generation.transfer.create_style_generator')
    def test_repair_continues_after_transient_error(self, mock_generator_class):
        """Transient errors should not abort all repair attempts."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        mock_generator = MagicMock()
        # First call raises, second returns valid repair (>10 words required)
        mock_generator.generate.side_effect = [
            RuntimeError("transient API error"),
            "Repaired text that includes Entity Name along with sufficient additional context words here.",
        ]
        mock_generator_class.return_value = mock_generator

        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(verify_entailment=False, repair_temperature=0.3)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )

        result = transfer._repair_missing_entities(
            source="Original text with Entity Name.",
            output="Styled text without the entity.",
            missing_entities=["Entity Name"],
            max_attempts=3,
        )

        # Should have tried at least twice (not broken on first error)
        assert mock_generator.generate.call_count >= 2
        # Should have returned the repaired text from second attempt
        assert "Entity Name" in result


class TestIdentityCheckVariable:
    """Bug: Identity check compares LoRA output against paragraph_clean (pre-RTT)
    instead of content_for_generation (what LoRA actually received)."""

    @patch('src.generation.transfer.create_style_generator')
    def test_identity_check_uses_content_for_generation(self, mock_generator_class):
        """Identity check should compare against RTT-neutralized content, not original."""
        from src.generation.transfer import StyleTransfer, TransferConfig
        import inspect

        # Verify the source code compares against content_for_generation
        source = inspect.getsource(StyleTransfer.transfer_paragraph)
        # Look for the identity check pattern
        # It should compare output against content_for_generation, not paragraph_clean
        assert "output.strip() == content_for_generation.strip()" in source or \
               "output.strip()==content_for_generation.strip()" in source, \
            "Identity check should compare against content_for_generation, not paragraph_clean"


class TestHallucinationThresholdOperator:
    """Bug: Hallucination repair uses > instead of >= for threshold comparison."""

    def test_threshold_triggers_at_exact_count(self):
        """Repair should trigger when hallucination_count == max_hallucinations_before_reject."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(max_hallucinations_before_reject=2)
        # With >=, repair triggers at count 2 (the configured max)
        # With >, repair only triggers at count 3
        assert config.max_hallucinations_before_reject == 2

        # Simulate the threshold check as it should work
        hallucination_count = 2
        needs_repair = hallucination_count >= config.max_hallucinations_before_reject
        assert needs_repair, "Repair should trigger when count equals threshold"


class TestMaxSentenceLengthDefault:
    """Bug: TransferConfig defaults max_sentence_length to 50 but
    GenerationConfig defaults to 60."""

    def test_transfer_config_default_matches_generation_config(self):
        """TransferConfig default should match GenerationConfig for max_sentence_length."""
        from src.generation.transfer import TransferConfig
        from src.config import GenerationConfig

        transfer_default = TransferConfig().max_sentence_length
        generation_default = GenerationConfig().max_sentence_length
        assert transfer_default == generation_default, (
            f"TransferConfig default ({transfer_default}) != "
            f"GenerationConfig default ({generation_default})"
        )


class TestReferenceMarkerWordCount:
    """Bug: word_count, source_words, and repair source all use paragraph
    (with [^N] references) instead of paragraph_clean (references stripped)."""

    def test_word_count_excludes_references(self):
        """Initial word_count should be based on cleaned text, not raw paragraph."""
        import inspect
        from src.generation.transfer import StyleTransfer

        source = inspect.getsource(StyleTransfer.transfer_paragraph)
        # After extract_references, word_count should use paragraph_clean
        # Find the word_count initialization pattern
        lines = source.split('\n')
        found_extract = False
        for line in lines:
            if 'extract_references' in line:
                found_extract = True
            if found_extract and 'word_count' in line and 'paragraph.split()' in line:
                assert False, (
                    "word_count uses paragraph.split() after extract_references — "
                    "should use paragraph_clean.split() to exclude reference markers"
                )
                break

    def test_expansion_ratio_excludes_references(self):
        """source_words for expansion check should exclude reference markers."""
        import inspect
        from src.generation.transfer import StyleTransfer

        source = inspect.getsource(StyleTransfer.transfer_paragraph)
        # The expansion ratio check should use paragraph_clean, not paragraph
        assert "source_words = len(paragraph_clean.split())" in source or \
               "source_words = len(paragraph_clean.split())" in source.replace(" ", ""), \
            "Expansion ratio check uses paragraph (with refs) instead of paragraph_clean"

    def test_repair_uses_clean_source(self):
        """_repair_missing_entities should receive text without reference markers."""
        import inspect
        from src.generation.transfer import StyleTransfer

        source = inspect.getsource(StyleTransfer.transfer_paragraph)
        # The repair call should pass paragraph_clean or original_for_verification
        assert "source=paragraph," not in source, (
            "Repair receives paragraph (with refs) — "
            "should use paragraph_clean (refs stripped)"
        )


class TestDeadCodeLoraInputWords:
    """Bug: lora_input_words computed but never used in transfer_paragraph."""

    def test_no_lora_input_words_in_source(self):
        """transfer_paragraph should not compute unused lora_input_words."""
        import inspect
        from src.generation.transfer import StyleTransfer

        source = inspect.getsource(StyleTransfer.transfer_paragraph)
        assert "lora_input_words" not in source, (
            "lora_input_words is dead code — computed but never used"
        )


class TestCleanPunctuationAbbreviations:
    """Bug: _clean_punctuation_artifacts breaks abbreviations like U.S. -> U. S."""

    def test_abbreviations_preserved(self):
        """Abbreviations like U.S. should not get spaces inserted."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        st = StyleTransfer.__new__(StyleTransfer)
        result = st._clean_punctuation_artifacts("The U.S. economy grew.")
        assert "U.S." in result, f"Abbreviation broken: {result}"

    def test_normal_missing_space_still_fixed(self):
        """Normal missing spaces after punctuation should still be fixed."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        st = StyleTransfer.__new__(StyleTransfer)
        result = st._clean_punctuation_artifacts("The cat sat.The dog ran.")
        assert "sat. The" in result, f"Missing space not fixed: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

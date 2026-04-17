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

        assert config.temperature is None  # None means use lora config
        assert config.verify_semantic_fidelity is True

    def test_sampling_params_live_on_generation_config_not_transfer(self):
        """max_tokens / top_p belong on GenerationConfig. TransferConfig used to
        carry parallel unused copies — C5 removed them. Regression guard so
        they can't sneak back in."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig()
        assert not hasattr(config, "max_tokens"), (
            "TransferConfig.max_tokens was unused dead weight; "
            "sampling params live on GenerationConfig only."
        )
        assert not hasattr(config, "top_p"), (
            "TransferConfig.top_p was unused dead weight; "
            "sampling params live on GenerationConfig only."
        )

    def test_custom_values(self):
        """Test that custom values are applied."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(
            temperature=0.8,
            verify_semantic_fidelity=False,
        )

        assert config.temperature == 0.8
        assert config.verify_semantic_fidelity is False

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
        assert stats.total_time_seconds == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.generation.transfer import TransferStats

        stats = TransferStats(
            paragraphs_processed=5,
            total_time_seconds=45.5,
            avg_time_per_paragraph=9.1,
            entailment_scores=[0.8, 0.9, 0.85, 0.75, 0.95],
        )

        d = stats.to_dict()

        assert d["paragraphs_processed"] == 5
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

        config = TransferConfig(verify_semantic_fidelity=False)

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

        config = TransferConfig(verify_semantic_fidelity=False)

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

        config = TransferConfig(verify_semantic_fidelity=False)
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

        config = TransferConfig(verify_semantic_fidelity=False)
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
    def test_transfer_paragraph_skips_short(self, mock_generator_class, mock_critic):
        """Test that short paragraphs are skipped."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(
            verify_semantic_fidelity=False,
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
    def test_init_uses_default_services_when_none_passed(self, mock_generator_class, mock_critic):
        """Without a services arg, StyleTransfer falls back to the process default."""
        from src.generation.transfer import StyleTransfer, TransferConfig
        from src.services import get_default_services

        config = TransferConfig(verify_semantic_fidelity=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
        )
        assert transfer.services is get_default_services()

    @patch('src.generation.transfer.create_style_generator')
    def test_init_accepts_injected_services(self, mock_generator_class, mock_critic):
        """An explicit Services instance is stored on the transfer and used
        instead of the module-level default — the primary test seam."""
        from src.generation.transfer import StyleTransfer, TransferConfig
        from src.services import Services

        services = Services()
        config = TransferConfig(verify_semantic_fidelity=False)
        transfer = StyleTransfer(
            adapter_path=None,
            author_name="Test",
            critic_provider=mock_critic,
            config=config,
            services=services,
        )
        assert transfer.services is services

    @patch('src.generation.transfer.create_style_generator')
    def test_get_partial_results(self, mock_generator_class, mock_critic):
        """Test getting partial results after interruption."""
        from src.generation.transfer import StyleTransfer, TransferConfig

        config = TransferConfig(verify_semantic_fidelity=False)
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


class TestStyleTransferNestedDIPropagation:
    """Fix #3: self.services must propagate through nested collaborators.

    StyleTransfer previously stored self.services but never pushed it onto
    the default-services stack, so nested calls like get_structural_rag()
    and get_structural_grafter() resolved against the process-wide default
    container instead of the injected one. The get_nlp() helper used by
    split_into_sentences / is_heading / is_sentence_incomplete had the same
    problem. These tests pin the invariant that an injected Services
    container reaches every collaborator StyleTransfer constructs.
    """

    @pytest.fixture
    def mock_critic(self):
        critic = MagicMock()
        critic.provider_name = "mock"
        return critic

    @patch('src.generation.transfer.create_style_generator')
    def test_structural_rag_uses_injected_services(
        self, mock_generator_class, mock_critic
    ):
        """StructuralRAG collaborator reads analyzers from the injected
        Services, not the process-wide default."""
        from src.generation.transfer import StyleTransfer, TransferConfig
        from src.services import Services
        from src.rag import structural_rag as sr_module

        # Clear cache to guarantee a fresh construction against our container
        sr_module._rag_cache.clear()

        sentinel_analyzer = MagicMock()
        sentinel_enhanced = MagicMock()
        sentinel_indexer = MagicMock()
        # Prevent load_patterns from blowing up on a real ChromaDB call
        sentinel_indexer.get_random_chunks.return_value = []

        custom_svc = Services(
            structural_analyzer=sentinel_analyzer,
            enhanced_analyzer=sentinel_enhanced,
            indexer=sentinel_indexer,
        )

        config = TransferConfig(
            verify_semantic_fidelity=False,
            use_structural_rag=True,
            use_structural_grafting=False,
        )

        transfer = StyleTransfer(
            adapter_path=None,
            author_name="DIPropagationTestAuthor",
            critic_provider=mock_critic,
            config=config,
            services=custom_svc,
        )

        # Regardless of whether structural_rag loaded any patterns, the
        # load_patterns() path inside StyleTransfer.__init__ must have hit
        # OUR sentinel indexer. If it hit the process-wide default indexer,
        # the sentinel's get_random_chunks is never called.
        assert sentinel_indexer.get_random_chunks.called, (
            "StructuralRAG must call get_random_chunks on the injected "
            "services' indexer, not the process-wide default"
        )

    @patch('src.generation.transfer.create_style_generator')
    def test_structural_grafter_uses_injected_services(
        self, mock_generator_class, mock_critic
    ):
        """StructuralGrafter collaborator reads its indexer from the injected
        Services, not the process-wide default."""
        from src.generation.transfer import StyleTransfer, TransferConfig
        from src.services import Services
        from src.rag import structural_grafter as sg_module

        sg_module._grafter_cache.clear()

        sentinel_indexer = MagicMock()
        custom_svc = Services(indexer=sentinel_indexer)

        config = TransferConfig(
            verify_semantic_fidelity=False,
            use_structural_rag=False,
            use_structural_grafting=True,
        )

        transfer = StyleTransfer(
            adapter_path=None,
            author_name="DIGrafterTestAuthor",
            critic_provider=mock_critic,
            config=config,
            services=custom_svc,
        )

        assert transfer.structural_grafter is not None
        assert transfer.structural_grafter.indexer is sentinel_indexer, (
            "StructuralGrafter.indexer must come from the injected "
            "Services container, not the process-wide default"
        )

    @patch('src.generation.transfer.create_style_generator')
    def test_cached_rag_isolated_per_services_container(
        self, mock_generator_class, mock_critic
    ):
        """Two StyleTransfer instances built for the same author but with
        DIFFERENT Services containers must get DIFFERENT StructuralRAGs.
        Otherwise the first container's collaborators leak into the second."""
        from src.generation.transfer import StyleTransfer, TransferConfig
        from src.services import Services
        from src.rag import structural_rag as sr_module

        sr_module._rag_cache.clear()

        indexer_a = MagicMock()
        indexer_a.get_random_chunks.return_value = []
        indexer_b = MagicMock()
        indexer_b.get_random_chunks.return_value = []

        svc_a = Services(
            structural_analyzer=MagicMock(),
            enhanced_analyzer=MagicMock(),
            indexer=indexer_a,
        )
        svc_b = Services(
            structural_analyzer=MagicMock(),
            enhanced_analyzer=MagicMock(),
            indexer=indexer_b,
        )

        config = TransferConfig(
            verify_semantic_fidelity=False,
            use_structural_rag=True,
            use_structural_grafting=False,
        )

        t_a = StyleTransfer(
            adapter_path=None,
            author_name="SharedAuthorForCacheTest",
            critic_provider=mock_critic,
            config=config,
            services=svc_a,
        )
        t_b = StyleTransfer(
            adapter_path=None,
            author_name="SharedAuthorForCacheTest",
            critic_provider=mock_critic,
            config=config,
            services=svc_b,
        )

        # The two transfers must see their own container's indexer, not
        # whichever one was constructed first.
        if t_a.structural_rag is not None:
            assert t_a.structural_rag.indexer is indexer_a
        if t_b.structural_rag is not None:
            assert t_b.structural_rag.indexer is indexer_b
        if t_a.structural_rag is not None and t_b.structural_rag is not None:
            assert t_a.structural_rag is not t_b.structural_rag, (
                "Per-author StructuralRAG cache must not be shared across "
                "different Services containers"
            )


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
            verify_semantic_fidelity=False,
            skip_neutralization=True,
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
            verify_semantic_fidelity=False,
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
            verify_semantic_fidelity=False,
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
            verify_semantic_fidelity=False,
            skip_neutralization=True,
            perspective="first_person_singular",
            use_persona=False,
            apply_input_perturbation=False,
            use_structural_rag=False,
            use_structural_grafting=False,
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
            verify_semantic_fidelity=False,
            skip_neutralization=False,
            perspective="preserve",
            use_persona=False,
            apply_input_perturbation=False,
            use_structural_rag=False,
            use_structural_grafting=False,
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
            verify_semantic_fidelity=False,
            skip_neutralization=True,
            perspective="preserve",
            use_persona=False,
            apply_input_perturbation=True,
            use_structural_rag=False,
            use_structural_grafting=False,
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

        config = TransferConfig(verify_semantic_fidelity=False, min_paragraph_words=3)
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


class TestPersonaStartupValidation:
    """A typo in config.worldview should fail fast at StyleTransfer init, not
    mid-document when the first paragraph tries to load the persona file."""

    @patch('src.generation.transfer.create_style_generator')
    def test_init_raises_on_missing_persona_file(self, mock_generator_class, tmp_path):
        """StyleTransfer.__init__ should raise FileNotFoundError when worldview
        points to a missing file and use_persona is True."""
        from src.generation.transfer import StyleTransfer, TransferConfig
        from src.persona.prompt_builder import _load_persona_file

        _load_persona_file.cache_clear()
        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(verify_semantic_fidelity=False, use_persona=True)

        # Point the worldview lookup at a filename that doesn't exist.
        with patch(
            'src.persona.prompt_builder._get_worldview_filename',
            return_value='totally_missing_persona_file.txt',
        ):
            with pytest.raises(FileNotFoundError, match="totally_missing_persona_file"):
                StyleTransfer(
                    adapter_path="lora_adapters/test",
                    author_name="Test",
                    critic_provider=mock_critic,
                    config=config,
                )

    @patch('src.generation.transfer.create_style_generator')
    def test_init_skips_persona_validation_when_use_persona_false(
        self, mock_generator_class, tmp_path
    ):
        """When use_persona is False, init should succeed even if the worldview
        file would be missing — the persona path is never taken."""
        from src.generation.transfer import StyleTransfer, TransferConfig
        from src.persona.prompt_builder import _load_persona_file

        _load_persona_file.cache_clear()
        mock_critic = MagicMock()
        mock_critic.provider_name = "mock"

        config = TransferConfig(verify_semantic_fidelity=False, use_persona=False)

        with patch(
            'src.persona.prompt_builder._get_worldview_filename',
            return_value='totally_missing_persona_file.txt',
        ):
            # Should NOT raise — persona is disabled
            transfer = StyleTransfer(
                adapter_path="lora_adapters/test",
                author_name="Test",
                critic_provider=mock_critic,
                config=config,
            )
            assert transfer.author == "Test"


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

"""Tests for the Services dependency container.

Services replaces module-global singletons (get_nlp(), get_grammar_corrector(),
get_semantic_verifier(), …) with an explicit container that can be injected
for testing. Each property lazy-loads on first access and caches thereafter.

The scaffold here pins the shape of the container. Per-singleton lazy-load
tests live alongside each migration step.
"""

import pytest


ALL_SLOTS = [
    "nlp",
    "grammar_corrector",
    "semantic_verifier",
    "nli_model",
    "chromadb",
    "embedding_model",
    "indexer",
    "structural_analyzer",
    "style_analyzer",
    "enhanced_analyzer",
]


class TestServicesScaffold:
    """Container shape — every migration target has a slot, injection works."""

    def test_services_class_exists(self):
        from src.services import Services

        assert Services is not None

    def test_services_can_be_instantiated_no_args(self):
        from src.services import Services

        services = Services()
        assert services is not None

    @pytest.mark.parametrize("slot", ALL_SLOTS)
    def test_every_slot_exists_as_property(self, slot):
        """Each singleton we plan to migrate must have a named slot on Services."""
        from src.services import Services

        # We look on the class, not the instance, so the property descriptor
        # is visible even before any instance is created.
        assert hasattr(Services, slot), (
            f"Services.{slot} is missing — add a slot for this singleton"
        )

    @pytest.mark.parametrize("slot", ALL_SLOTS)
    def test_injected_value_is_returned_verbatim(self, slot):
        """When an object is passed to Services(slot=obj), property returns it
        without calling any loader. This is the primary test seam."""
        from src.services import Services

        sentinel = object()
        services = Services(**{slot: sentinel})
        assert getattr(services, slot) is sentinel

    @pytest.mark.parametrize("slot", ALL_SLOTS)
    def test_injected_value_is_cached(self, slot):
        """Repeat access returns the same injected object."""
        from src.services import Services

        sentinel = object()
        services = Services(**{slot: sentinel})
        first = getattr(services, slot)
        second = getattr(services, slot)
        assert first is second

    def test_constructor_is_keyword_only(self):
        """Positional args would make the constructor brittle as slots grow.
        Force keyword-only so `Services(some_mock)` can't accidentally populate
        the wrong slot."""
        from src.services import Services

        with pytest.raises(TypeError):
            Services(object())  # type: ignore[misc]

    def test_unknown_kwarg_raises(self):
        """Typos in slot names should surface loudly, not silently no-op."""
        from src.services import Services

        with pytest.raises(TypeError):
            Services(nonexistent_slot=object())  # type: ignore[call-arg]


class TestServicesDefaultAccessor:
    """get_default_services() returns a process-wide default container that
    existing module-level get_*() helpers can delegate to during migration."""

    def test_default_services_returns_services_instance(self):
        from src.services import Services, get_default_services

        services = get_default_services()
        assert isinstance(services, Services)

    def test_default_services_is_singleton(self):
        from src.services import get_default_services

        a = get_default_services()
        b = get_default_services()
        assert a is b

    def test_set_default_services_swaps_the_default(self):
        """Tests should be able to swap the default for isolation."""
        from src.services import Services, get_default_services, set_default_services

        original = get_default_services()
        try:
            replacement = Services()
            set_default_services(replacement)
            assert get_default_services() is replacement
        finally:
            set_default_services(original)


class TestGrammarCorrectorMigration:
    """Services.grammar_corrector lazy-loads on first access and
    get_grammar_corrector() delegates through get_default_services()."""

    def test_lazy_loads_grammar_corrector(self):
        from src.services import Services
        from src.vocabulary.grammar_corrector import GrammarCorrector

        services = Services()
        corrector = services.grammar_corrector
        assert isinstance(corrector, GrammarCorrector)

    def test_lazy_load_is_cached(self):
        from src.services import Services

        services = Services()
        first = services.grammar_corrector
        second = services.grammar_corrector
        assert first is second

    def test_get_grammar_corrector_delegates_to_default_services(self):
        """Module-level get_grammar_corrector() returns the Services instance
        so every call site migrates in one edit."""
        from src.services import Services, get_default_services, set_default_services
        from src.vocabulary.grammar_corrector import GrammarCorrector, get_grammar_corrector

        # Inject a sentinel via a fresh default services container
        sentinel = GrammarCorrector()
        original = get_default_services()
        try:
            set_default_services(Services(grammar_corrector=sentinel))
            assert get_grammar_corrector() is sentinel
        finally:
            set_default_services(original)

    def test_set_default_services_resets_corrector(self):
        """Swapping default services is the replacement for `module._corrector = None`."""
        from src.services import Services, get_default_services, set_default_services
        from src.vocabulary.grammar_corrector import get_grammar_corrector

        original = get_default_services()
        try:
            # First Services instance has its own corrector
            set_default_services(Services())
            c1 = get_grammar_corrector()

            # Swap to a fresh Services — new corrector
            set_default_services(Services())
            c2 = get_grammar_corrector()

            assert c1 is not c2
        finally:
            set_default_services(original)


class TestSemanticVerifierMigration:
    """Services.semantic_verifier lazy-loads, get_semantic_verifier() delegates,
    and kwargs now yield an uncached new instance (fixing a latent bug where
    subsequent kwargs were silently ignored by the old module singleton)."""

    def test_lazy_loads_semantic_verifier(self):
        from src.services import Services
        from src.validation.semantic_verifier import SemanticVerifier

        services = Services()
        assert isinstance(services.semantic_verifier, SemanticVerifier)

    def test_lazy_load_is_cached(self):
        from src.services import Services

        services = Services()
        assert services.semantic_verifier is services.semantic_verifier

    def test_get_semantic_verifier_delegates_to_default_services(self):
        from src.services import Services, get_default_services, set_default_services
        from src.validation.semantic_verifier import SemanticVerifier, get_semantic_verifier

        sentinel = SemanticVerifier()
        original = get_default_services()
        try:
            set_default_services(Services(semantic_verifier=sentinel))
            assert get_semantic_verifier() is sentinel
        finally:
            set_default_services(original)

    def test_get_semantic_verifier_with_kwargs_returns_new_uncached_instance(self):
        """Kwargs were previously ignored after the first call. Now they yield
        a new uncached instance, leaving the default container untouched."""
        from src.services import Services, get_default_services, set_default_services
        from src.validation.semantic_verifier import get_semantic_verifier

        original = get_default_services()
        try:
            set_default_services(Services())
            v1 = get_semantic_verifier(grounding_threshold=0.8)
            v2 = get_semantic_verifier(grounding_threshold=0.5)
            assert v1 is not v2
            assert v1.grounding_threshold == 0.8
            assert v2.grounding_threshold == 0.5
        finally:
            set_default_services(original)


class TestNliModelMigration:
    """Services.nli_model holds the CrossEncoder. Loading is expensive so tests
    inject a sentinel rather than triggering a real load."""

    def test_injected_nli_model_is_returned(self):
        from src.services import Services

        sentinel = object()
        services = Services(nli_model=sentinel)
        assert services.nli_model is sentinel

    def test_get_nli_model_delegates_to_services(self):
        from src.services import Services, get_default_services, set_default_services
        from src.validation.semantic_verifier import _get_nli_model

        sentinel = object()
        original = get_default_services()
        try:
            set_default_services(Services(nli_model=sentinel))
            assert _get_nli_model() is sentinel
        finally:
            set_default_services(original)


class TestCorpusIndexerMigration:
    """Services.chromadb / embedding_model / indexer replace three module
    singletons in src/rag/corpus_indexer.py. Only injection is exercised
    here — the real loaders require chromadb + sentence-transformers."""

    def test_injected_chromadb_is_returned(self):
        from src.services import Services

        sentinel = object()
        services = Services(chromadb=sentinel)
        assert services.chromadb is sentinel

    def test_injected_embedding_model_is_returned(self):
        from src.services import Services

        sentinel = object()
        services = Services(embedding_model=sentinel)
        assert services.embedding_model is sentinel

    def test_injected_indexer_is_returned(self):
        from src.services import Services

        sentinel = object()
        services = Services(indexer=sentinel)
        assert services.indexer is sentinel

    def test_get_chromadb_delegates_to_services(self):
        from src.services import Services, get_default_services, set_default_services
        from src.rag.corpus_indexer import get_chromadb

        sentinel = object()
        original = get_default_services()
        try:
            set_default_services(Services(chromadb=sentinel))
            assert get_chromadb() is sentinel
        finally:
            set_default_services(original)

    def test_get_embedding_model_delegates_to_services(self):
        from src.services import Services, get_default_services, set_default_services
        from src.rag.corpus_indexer import get_embedding_model

        sentinel = object()
        original = get_default_services()
        try:
            set_default_services(Services(embedding_model=sentinel))
            assert get_embedding_model() is sentinel
        finally:
            set_default_services(original)

    def test_get_indexer_no_arg_delegates_to_services(self):
        """get_indexer() with no persist_dir returns the shared Services indexer."""
        from src.services import Services, get_default_services, set_default_services
        from src.rag.corpus_indexer import get_indexer

        sentinel = object()
        original = get_default_services()
        try:
            set_default_services(Services(indexer=sentinel))
            assert get_indexer() is sentinel
        finally:
            set_default_services(original)

    def test_get_indexer_with_persist_dir_returns_new_instance(self):
        """Explicit persist_dir bypasses the Services cache (new uncached instance)."""
        from src.services import Services, get_default_services, set_default_services
        from src.rag.corpus_indexer import CorpusIndexer, get_indexer

        sentinel = CorpusIndexer("/tmp/unused")
        original = get_default_services()
        try:
            set_default_services(Services(indexer=sentinel))
            other = get_indexer(persist_dir="/tmp/other")
            assert other is not sentinel
            assert isinstance(other, CorpusIndexer)
            assert other.persist_dir == "/tmp/other"
        finally:
            set_default_services(original)


class TestStyleAnalyzerMigration:
    """Services.style_analyzer lazy-loads a StyleAnalyzer, and
    get_style_analyzer() delegates through the default Services container."""

    def test_lazy_loads_style_analyzer(self):
        from src.services import Services
        from src.rag.style_analyzer import StyleAnalyzer

        services = Services()
        assert isinstance(services.style_analyzer, StyleAnalyzer)

    def test_lazy_load_is_cached(self):
        from src.services import Services

        services = Services()
        assert services.style_analyzer is services.style_analyzer

    def test_get_style_analyzer_delegates_to_services(self):
        from src.services import Services, get_default_services, set_default_services
        from src.rag.style_analyzer import StyleAnalyzer, get_style_analyzer

        sentinel = StyleAnalyzer()
        original = get_default_services()
        try:
            set_default_services(Services(style_analyzer=sentinel))
            assert get_style_analyzer() is sentinel
        finally:
            set_default_services(original)


class TestNlpMigration:
    """Services.nlp holds the spaCy model. Loading is very expensive so most
    tests inject a sentinel rather than triggering a real load."""

    def test_injected_nlp_is_returned(self):
        from src.services import Services

        sentinel = object()
        services = Services(nlp=sentinel)
        assert services.nlp is sentinel

    def test_get_nlp_delegates_to_services(self):
        from src.services import Services, get_default_services, set_default_services
        from src.utils.nlp import get_nlp

        sentinel = object()
        original = get_default_services()
        try:
            set_default_services(Services(nlp=sentinel))
            assert get_nlp() is sentinel
        finally:
            set_default_services(original)

    def test_nlp_lazy_load_is_cached(self):
        """Once loaded, the spaCy model is cached on the Services instance."""
        from src.services import Services

        services = Services()
        first = services.nlp
        second = services.nlp
        assert first is second


class TestStructuralAnalyzerMigration:
    """Services.structural_analyzer lazy-loads, get_structural_analyzer()
    delegates through the default Services container."""

    def test_lazy_loads_structural_analyzer(self):
        from src.services import Services
        from src.rag.structural_analyzer import StructuralAnalyzer

        services = Services()
        assert isinstance(services.structural_analyzer, StructuralAnalyzer)

    def test_lazy_load_is_cached(self):
        from src.services import Services

        services = Services()
        assert services.structural_analyzer is services.structural_analyzer

    def test_get_structural_analyzer_delegates_to_services(self):
        from src.services import Services, get_default_services, set_default_services
        from src.rag.structural_analyzer import StructuralAnalyzer, get_structural_analyzer

        sentinel = StructuralAnalyzer()
        original = get_default_services()
        try:
            set_default_services(Services(structural_analyzer=sentinel))
            assert get_structural_analyzer() is sentinel
        finally:
            set_default_services(original)


class TestEnhancedAnalyzerMigration:
    """Services.enhanced_analyzer lazy-loads, get_enhanced_analyzer()
    delegates through the default Services container."""

    def test_lazy_loads_enhanced_analyzer(self):
        from src.services import Services
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer

        services = Services()
        assert isinstance(services.enhanced_analyzer, EnhancedStructuralAnalyzer)

    def test_lazy_load_is_cached(self):
        from src.services import Services

        services = Services()
        assert services.enhanced_analyzer is services.enhanced_analyzer

    def test_get_enhanced_analyzer_delegates_to_services(self):
        from src.services import Services, get_default_services, set_default_services
        from src.rag.enhanced_analyzer import EnhancedStructuralAnalyzer, get_enhanced_analyzer

        sentinel = EnhancedStructuralAnalyzer()
        original = get_default_services()
        try:
            set_default_services(Services(enhanced_analyzer=sentinel))
            assert get_enhanced_analyzer() is sentinel
        finally:
            set_default_services(original)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

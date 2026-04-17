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


class TestLazyLoadCaching:
    """Lazy-load slots must cache the result even when the loader legitimately
    returns None (e.g. optional dependency missing). Using `None` as the
    'unset' sentinel would re-run the loader forever in that case."""

    def test_nli_model_loader_called_once_when_it_returns_none(self, monkeypatch):
        from src import services as services_mod
        from src.services import Services

        call_count = [0]

        def fake_loader():
            call_count[0] += 1
            return None  # simulate sentence-transformers not installed

        monkeypatch.setattr(
            "src.validation.semantic_verifier._load_nli_model",
            fake_loader,
        )

        s = Services()
        first = s.nli_model
        second = s.nli_model
        third = s.nli_model

        assert first is None and second is None and third is None
        assert call_count[0] == 1, (
            f"loader ran {call_count[0]} times — None must be cached just like "
            "any other loaded value; using None as 'unset' sentinel is a bug"
        )

    def test_grammar_corrector_loader_called_once_when_it_returns_none(self, monkeypatch):
        """Same invariant for any slot — use grammar_corrector as a spot-check."""
        from src.services import Services

        call_count = [0]

        class FakeGrammarCorrector:
            def __init__(self):
                call_count[0] += 1

        monkeypatch.setattr(
            "src.vocabulary.grammar_corrector.GrammarCorrector",
            FakeGrammarCorrector,
        )

        s = Services()
        _ = s.grammar_corrector
        _ = s.grammar_corrector
        assert call_count[0] == 1

    def test_chromadb_loader_called_once_when_it_returns_none(self, monkeypatch):
        """Fix #5: _load_chromadb must return None (not raise) when the
        dependency is missing, so the Services cache stores the None and
        subsequent accesses return without re-running the failing import.

        Pre-fix: _load_chromadb raises ImportError, the slot stays _UNSET,
        every access retries the import. Asymmetric with _load_nli_model."""
        from src.services import Services

        call_count = [0]

        def fake_loader():
            call_count[0] += 1
            return None  # simulate chromadb not installed

        monkeypatch.setattr(
            "src.rag.corpus_indexer._load_chromadb",
            fake_loader,
        )

        s = Services()
        first = s.chromadb
        second = s.chromadb
        third = s.chromadb

        assert first is None and second is None and third is None
        assert call_count[0] == 1, (
            f"loader ran {call_count[0]} times — None must be cached; "
            "raising on every call defeats the cache"
        )

    def test_embedding_model_loader_called_once_when_it_returns_none(self, monkeypatch):
        """Fix #5: _load_embedding_model must return None (not raise) when
        sentence-transformers is missing, matching _load_nli_model."""
        from src.services import Services

        call_count = [0]

        def fake_loader():
            call_count[0] += 1
            return None  # simulate sentence-transformers not installed

        monkeypatch.setattr(
            "src.rag.corpus_indexer._load_embedding_model",
            fake_loader,
        )

        s = Services()
        _ = s.embedding_model
        _ = s.embedding_model
        _ = s.embedding_model

        assert call_count[0] == 1, (
            f"loader ran {call_count[0]} times — None must be cached; "
            "raising on every call defeats the cache"
        )


class TestLoaderMissingDependencyContract:
    """Fix #5: All optional-dependency loaders must return None on missing
    deps, not raise. The Services cache cannot store a raised exception,
    so raising defeats the single-import guarantee."""

    def test_load_chromadb_returns_none_when_import_fails(self, monkeypatch):
        """With `import chromadb` unavailable, _load_chromadb returns None."""
        import sys
        from src.rag import corpus_indexer

        # Force ImportError on `import chromadb`
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "chromadb" or name.startswith("chromadb."):
                raise ImportError("simulated missing chromadb")
            return real_import(name, *args, **kwargs)

        # Purge any already-imported chromadb so the import actually runs
        for mod_name in list(sys.modules):
            if mod_name == "chromadb" or mod_name.startswith("chromadb."):
                monkeypatch.delitem(sys.modules, mod_name, raising=False)

        monkeypatch.setattr("builtins.__import__", blocked_import)

        result = corpus_indexer._load_chromadb()
        assert result is None, (
            "_load_chromadb must return None on missing dep, not raise — "
            "Services cache uses None as a legitimate cached value"
        )

    def test_load_embedding_model_returns_none_when_import_fails(self, monkeypatch):
        """With `sentence_transformers` unavailable, loader returns None."""
        import sys
        from src.rag import corpus_indexer

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "sentence_transformers" or name.startswith("sentence_transformers."):
                raise ImportError("simulated missing sentence_transformers")
            return real_import(name, *args, **kwargs)

        for mod_name in list(sys.modules):
            if mod_name == "sentence_transformers" or mod_name.startswith("sentence_transformers."):
                monkeypatch.delitem(sys.modules, mod_name, raising=False)

        monkeypatch.setattr("builtins.__import__", blocked_import)

        result = corpus_indexer._load_embedding_model()
        assert result is None, (
            "_load_embedding_model must return None on missing dep, not raise"
        )


class TestSentenceSplitterMigration:
    """Fix #7: the sentence splitter lived as a module-global singleton in
    src/vocabulary/sentence_splitter.py. Pull it onto the Services container
    so test isolation uses the same mechanism as every other slot."""

    def test_injected_sentence_splitter_is_returned(self):
        """An explicit splitter on Services is the canonical test seam."""
        from src.services import Services

        sentinel = object()
        s = Services(sentence_splitter=sentinel)
        assert s.sentence_splitter is sentinel

    def test_sentence_splitter_lazy_loads(self, monkeypatch):
        """With no injection, the first access constructs and caches."""
        from src.services import Services

        call_count = [0]

        class FakeSplitter:
            def __init__(self, config=None):
                call_count[0] += 1

        monkeypatch.setattr(
            "src.vocabulary.sentence_splitter.SentenceSplitter",
            FakeSplitter,
        )

        s = Services()
        first = s.sentence_splitter
        second = s.sentence_splitter

        assert first is second
        assert call_count[0] == 1, (
            f"SentenceSplitter constructed {call_count[0]} times — "
            "the slot must cache"
        )

    def test_get_sentence_splitter_delegates_to_services(self):
        """get_sentence_splitter() must read from the default Services
        container so the module-global _splitter is no longer the source
        of truth."""
        from src.services import Services, set_default_services, get_default_services
        from src.vocabulary.sentence_splitter import get_sentence_splitter

        sentinel = object()
        original = get_default_services()
        try:
            set_default_services(Services(sentence_splitter=sentinel))
            assert get_sentence_splitter() is sentinel
        finally:
            set_default_services(original)


class TestLazyLoadThreadSafety:
    """Services is accessed from worker threads in several places (see
    mlx_provider.ThreadPoolExecutor usage). The lazy-load slots must be
    safe against double-initialization races — two threads hitting an
    unloaded slot should result in the loader running exactly once."""

    def test_concurrent_access_runs_loader_once(self, monkeypatch):
        import threading
        import time
        from src.services import Services

        call_count = [0]
        count_lock = threading.Lock()

        class SlowStyleAnalyzer:
            def __init__(self):
                # Small sleep inside the loader widens the race window so a
                # missing container lock would produce multiple instantiations.
                time.sleep(0.01)
                with count_lock:
                    call_count[0] += 1

        monkeypatch.setattr(
            "src.rag.style_analyzer.StyleAnalyzer",
            SlowStyleAnalyzer,
        )

        s = Services()
        results = [None] * 8
        # All threads rendezvous before touching the property so they all
        # race into the lazy-load path at once.
        barrier = threading.Barrier(8)

        def worker(idx):
            barrier.wait()
            results[idx] = s.style_analyzer

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count[0] == 1, (
            f"loader ran {call_count[0]} times across 8 threads — "
            "lazy-load slot is not thread-safe"
        )
        # All threads received the same cached instance
        assert all(r is results[0] for r in results)

    def test_get_default_services_concurrent_calls_create_one_instance(self, monkeypatch):
        import threading
        from src import services as services_mod
        from src.services import Services

        # Force the lazy default to re-init for this test
        monkeypatch.setattr(services_mod, "_default_services", None)

        # Count Services() constructions so we can tell the difference
        # between "all threads saw the same instance because the FIRST
        # construction won the race" and "Services() was actually called
        # only once". Without this, a racy implementation that builds
        # multiple instances but last-write-wins would still pass.
        construct_count = [0]
        real_init = Services.__init__

        def counting_init(self, *args, **kwargs):
            construct_count[0] += 1
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(Services, "__init__", counting_init)

        barrier = threading.Barrier(8)
        results = [None] * 8

        def worker(idx):
            barrier.wait()
            results[idx] = services_mod.get_default_services()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = results[0]
        assert isinstance(first, Services)
        assert all(r is first for r in results), (
            "get_default_services() produced more than one container under concurrent access"
        )
        assert construct_count[0] == 1, (
            f"Services() ran {construct_count[0]} times under concurrent access — "
            "double-checked locking is broken; extra instances were built even "
            "if only one survived"
        )


class TestNestedSlotAccessNoDeadlock:
    """A loader may legitimately touch another Services slot during
    construction — e.g. CorpusIndexer pulls the shared style_analyzer.
    Both accesses run on the same thread while the container lock is
    held, so the lock MUST be reentrant."""

    def test_loader_can_access_another_slot(self, monkeypatch):
        """Simulate the CorpusIndexer pattern: outer slot's loader reads
        an inner slot from the same Services instance."""
        import threading
        from src.services import Services

        s = Services()

        class InnerAnalyzer:
            pass

        class OuterIndexer:
            def __init__(self):
                # Touches another slot while the container lock is held
                # by the outer slot's loader — would deadlock on a plain
                # threading.Lock.
                self.analyzer = s.style_analyzer

        monkeypatch.setattr(
            "src.rag.style_analyzer.StyleAnalyzer",
            InnerAnalyzer,
        )
        monkeypatch.setattr(
            "src.rag.corpus_indexer._load_default_indexer",
            lambda services=None: OuterIndexer(),
        )

        # Run in a worker thread with a hard timeout so a deadlock shows
        # up as a timeout rather than hanging the whole suite.
        done = threading.Event()
        result = {}

        def go():
            result["indexer"] = s.indexer
            done.set()

        t = threading.Thread(target=go, daemon=True)
        t.start()
        done.wait(timeout=5.0)
        assert done.is_set(), "nested slot access deadlocked the container lock"
        assert isinstance(result["indexer"], OuterIndexer)
        assert isinstance(result["indexer"].analyzer, InnerAnalyzer)


class TestDefaultServicesContextManagerConcurrency:
    """Concurrent `with default_services()` blocks on different threads must
    each isolate their own override without corrupting each other — the
    classic "two context managers racing replace/restore" bug."""

    def test_per_thread_override_does_not_leak(self):
        """Two threads each enter default_services() with their own payload.
        Inside the block each must see its own. After both exit, the process
        default must equal what it was before any override."""
        import threading
        from src.services import (
            Services,
            default_services,
            get_default_services,
        )

        before = get_default_services()

        seen_a: list = []
        seen_b: list = []
        release = threading.Event()

        def worker(marker: str, seen: list):
            mine = Services(nlp=f"nlp-{marker}")
            with default_services(mine):
                # Verify this thread sees its own override.
                seen.append(get_default_services())
                # Hold both blocks open simultaneously to surface any
                # process-wide "original" bookkeeping race.
                release.wait(timeout=5.0)
                seen.append(get_default_services())

        t_a = threading.Thread(target=worker, args=("a", seen_a))
        t_b = threading.Thread(target=worker, args=("b", seen_b))
        t_a.start()
        t_b.start()
        # Give both threads a chance to enter their `with` block.
        import time as _time
        _time.sleep(0.1)
        release.set()
        t_a.join(timeout=5.0)
        t_b.join(timeout=5.0)

        assert not t_a.is_alive() and not t_b.is_alive()
        assert len(seen_a) == 2 and len(seen_b) == 2
        # Each thread saw its own override on entry AND after the barrier.
        assert seen_a[0] is seen_a[1]
        assert seen_b[0] is seen_b[1]
        assert seen_a[0] is not seen_b[0], (
            "threads leaked each other's override — default_services() is "
            "not per-thread isolated"
        )
        # Neither thread saw the other's override.
        assert seen_a[0].nlp == "nlp-a"
        assert seen_b[0].nlp == "nlp-b"
        # Original default restored.
        assert get_default_services() is before


class TestNestedInjectionPropagatesToLoaders:
    """When a caller injects `Services(style_analyzer=fake)` and then accesses
    `services.indexer`, the CorpusIndexer that gets lazy-loaded must pull
    the analyzer from THAT container, not from the process-wide default."""

    def test_indexer_loader_uses_injected_style_analyzer(self):
        from src.services import Services

        class FakeAnalyzer:
            pass

        fake = FakeAnalyzer()
        svc = Services(style_analyzer=fake)
        # Do NOT install svc as the default — the bug manifests when the
        # injected container is not the global default.
        assert svc.indexer._analyzer is fake, (
            "nested DI: indexer loader must read style_analyzer from the "
            "container being loaded, not from the process-wide default"
        )


class TestDefaultServicesContextManager:
    """`default_services(services)` swaps the process default, yields it, and
    restores the original on exit — a clean replacement for the try/finally
    pattern test code was repeating everywhere."""

    def test_context_manager_swaps_and_restores(self):
        from src.services import (
            Services,
            default_services,
            get_default_services,
        )

        before = get_default_services()
        with default_services() as fresh:
            assert isinstance(fresh, Services)
            assert get_default_services() is fresh
            assert fresh is not before
        assert get_default_services() is before

    def test_context_manager_restores_on_exception(self):
        from src.services import default_services, get_default_services

        before = get_default_services()
        with pytest.raises(RuntimeError):
            with default_services():
                raise RuntimeError("boom")
        assert get_default_services() is before, (
            "default services must be restored even when the block raised"
        )

    def test_context_manager_accepts_explicit_services(self):
        from src.services import Services, default_services, get_default_services

        custom = Services(nlp="fake-nlp")
        with default_services(custom) as yielded:
            assert yielded is custom
            assert get_default_services() is custom


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

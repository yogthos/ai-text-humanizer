"""Dependency container that replaces module-global singletons.

Historically, shared state (spaCy model, grammar corrector, semantic verifier,
ChromaDB handle, …) has been held in module-level variables and accessed through
`get_*()` helpers. That pattern makes tests brittle — they reach into private
module globals (e.g. `sv._verifier = None`) to reset state between runs — and
couples everything to a single process-wide instance.

`Services` is an explicit container for these dependencies. Each slot can be:

- pre-populated via the keyword-only constructor (test injection), or
- lazy-loaded on first property access (production default).

The legacy `get_*()` helpers delegate to `get_default_services()` during the
migration so existing callers keep working; new code is encouraged to accept a
`services: Services` argument directly.
"""

import contextlib
import threading
from typing import Any, Iterator, Optional


# Sentinel distinct from None. A loader may legitimately return None (e.g. an
# optional dependency is missing) — using None as "not loaded" would re-run
# the loader forever in that case.
_UNSET: Any = object()


class Services:
    """Lazy dependency container.

    All constructor arguments are keyword-only to keep slot assignment
    unambiguous as the container grows. Unknown kwargs raise TypeError.

    Each lazy-load property is guarded by a per-container lock with
    double-checked locking so concurrent access from worker threads
    (see e.g. ThreadPoolExecutor in `mlx_provider`) runs the loader
    exactly once. The lock is an RLock because some loaders legitimately
    touch other Services slots during construction — e.g.
    `CorpusIndexer.__init__` pulls the shared `style_analyzer` — which
    would re-enter the same lock on the same thread.
    """

    def __init__(
        self,
        *,
        nlp: Any = _UNSET,
        grammar_corrector: Any = _UNSET,
        semantic_verifier: Any = _UNSET,
        nli_model: Any = _UNSET,
        chromadb: Any = _UNSET,
        embedding_model: Any = _UNSET,
        indexer: Any = _UNSET,
        structural_analyzer: Any = _UNSET,
        style_analyzer: Any = _UNSET,
        enhanced_analyzer: Any = _UNSET,
        sentence_splitter: Any = _UNSET,
    ):
        self._lock = threading.RLock()
        self._nlp = nlp
        self._grammar_corrector = grammar_corrector
        self._semantic_verifier = semantic_verifier
        self._nli_model = nli_model
        self._chromadb = chromadb
        self._embedding_model = embedding_model
        self._indexer = indexer
        self._structural_analyzer = structural_analyzer
        self._style_analyzer = style_analyzer
        self._enhanced_analyzer = enhanced_analyzer
        self._sentence_splitter = sentence_splitter

    @property
    def nlp(self) -> Any:
        if self._nlp is _UNSET:
            with self._lock:
                if self._nlp is _UNSET:
                    from .utils.nlp import _load_spacy_nlp
                    self._nlp = _load_spacy_nlp()
        return self._nlp

    @property
    def grammar_corrector(self) -> Any:
        if self._grammar_corrector is _UNSET:
            with self._lock:
                if self._grammar_corrector is _UNSET:
                    from .vocabulary.grammar_corrector import GrammarCorrector
                    self._grammar_corrector = GrammarCorrector()
        return self._grammar_corrector

    @property
    def semantic_verifier(self) -> Any:
        if self._semantic_verifier is _UNSET:
            with self._lock:
                if self._semantic_verifier is _UNSET:
                    from .validation.semantic_verifier import SemanticVerifier
                    self._semantic_verifier = SemanticVerifier()
        return self._semantic_verifier

    @property
    def nli_model(self) -> Any:
        if self._nli_model is _UNSET:
            with self._lock:
                if self._nli_model is _UNSET:
                    from .validation.semantic_verifier import _load_nli_model
                    self._nli_model = _load_nli_model()
        return self._nli_model

    @property
    def chromadb(self) -> Any:
        if self._chromadb is _UNSET:
            with self._lock:
                if self._chromadb is _UNSET:
                    from .rag.corpus_indexer import _load_chromadb
                    self._chromadb = _load_chromadb()
        return self._chromadb

    @property
    def embedding_model(self) -> Any:
        if self._embedding_model is _UNSET:
            with self._lock:
                if self._embedding_model is _UNSET:
                    from .rag.corpus_indexer import _load_embedding_model
                    self._embedding_model = _load_embedding_model()
        return self._embedding_model

    @property
    def indexer(self) -> Any:
        if self._indexer is _UNSET:
            with self._lock:
                if self._indexer is _UNSET:
                    from .rag.corpus_indexer import _load_default_indexer
                    # Pass self so the indexer's StyleAnalyzer comes from
                    # THIS container, not from the process-wide default.
                    self._indexer = _load_default_indexer(self)
        return self._indexer

    @property
    def structural_analyzer(self) -> Any:
        if self._structural_analyzer is _UNSET:
            with self._lock:
                if self._structural_analyzer is _UNSET:
                    from .rag.structural_analyzer import StructuralAnalyzer
                    self._structural_analyzer = StructuralAnalyzer()
        return self._structural_analyzer

    @property
    def style_analyzer(self) -> Any:
        if self._style_analyzer is _UNSET:
            with self._lock:
                if self._style_analyzer is _UNSET:
                    from .rag.style_analyzer import StyleAnalyzer
                    self._style_analyzer = StyleAnalyzer()
        return self._style_analyzer

    @property
    def enhanced_analyzer(self) -> Any:
        if self._enhanced_analyzer is _UNSET:
            with self._lock:
                if self._enhanced_analyzer is _UNSET:
                    from .rag.enhanced_analyzer import EnhancedStructuralAnalyzer
                    self._enhanced_analyzer = EnhancedStructuralAnalyzer()
        return self._enhanced_analyzer

    @property
    def sentence_splitter(self) -> Any:
        if self._sentence_splitter is _UNSET:
            with self._lock:
                if self._sentence_splitter is _UNSET:
                    from .vocabulary.sentence_splitter import SentenceSplitter
                    self._sentence_splitter = SentenceSplitter()
        return self._sentence_splitter


_default_services: Optional[Services] = None
_default_services_lock = threading.Lock()
# Per-thread override stack for `default_services()`. Using thread-local
# state means two threads that each enter `with default_services(...)`
# concurrently don't race on a shared "restore original" slot.
_override = threading.local()


def _override_stack() -> list:
    stack = getattr(_override, "stack", None)
    if stack is None:
        stack = []
        _override.stack = stack
    return stack


def get_default_services() -> Services:
    """Return the active Services container for this thread.

    If the calling thread is inside a `default_services()` block, return
    the override on top of its stack. Otherwise return the process-wide
    default (created lazily).
    """
    stack = _override_stack()
    if stack:
        return stack[-1]
    global _default_services
    if _default_services is None:
        with _default_services_lock:
            if _default_services is None:
                _default_services = Services()
    return _default_services


def set_default_services(services: Services) -> None:
    """Swap the process-wide default container.

    This is the legacy test seam — it affects every thread that isn't
    inside its own `default_services()` block. Prefer `default_services()`
    for per-test isolation.
    """
    global _default_services
    with _default_services_lock:
        _default_services = services


@contextlib.contextmanager
def default_services(services: Optional[Services] = None) -> Iterator[Services]:
    """Temporarily install a Services container for the calling thread.

    Each thread has its own stack of overrides, so two threads that each
    enter `with default_services(...)` concurrently do not race on a
    shared "restore original" slot. Workers that don't call
    `default_services()` themselves continue to see the process-wide
    default set by `set_default_services()`.

    With no argument, a fresh empty Services is installed.
    """
    replacement = services if services is not None else Services()
    stack = _override_stack()
    stack.append(replacement)
    try:
        yield replacement
    finally:
        popped = stack.pop()
        assert popped is replacement, (
            "default_services() stack corrupted — nested exits out of order"
        )

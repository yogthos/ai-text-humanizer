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
    exactly once.
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
    ):
        self._lock = threading.Lock()
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
                    self._indexer = _load_default_indexer()
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


_default_services: Optional[Services] = None
_default_services_lock = threading.Lock()


def get_default_services() -> Services:
    """Return the process-wide default Services container, creating it lazily.

    Legacy get_*() helpers delegate here during the migration. Tests that need
    isolation should use the `default_services()` context manager.
    """
    global _default_services
    if _default_services is None:
        with _default_services_lock:
            if _default_services is None:
                _default_services = Services()
    return _default_services


def set_default_services(services: Services) -> None:
    """Swap the default container — primarily a test seam."""
    global _default_services
    _default_services = services


@contextlib.contextmanager
def default_services(services: Optional[Services] = None) -> Iterator[Services]:
    """Temporarily swap the default Services container for the body of a
    `with` block, restoring the previous container (even on exception).

    Replaces the try/finally pattern test code was repeating everywhere:

        original = get_default_services()
        try:
            set_default_services(Services(...))
            ...
        finally:
            set_default_services(original)

    becomes simply:

        with default_services(Services(...)):
            ...

    With no argument, a fresh empty Services is installed.
    """
    original = get_default_services()
    replacement = services if services is not None else Services()
    set_default_services(replacement)
    try:
        yield replacement
    finally:
        set_default_services(original)

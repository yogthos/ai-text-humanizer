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

from typing import Any, Optional


class Services:
    """Lazy dependency container.

    All constructor arguments are keyword-only to keep slot assignment
    unambiguous as the container grows. Unknown kwargs raise TypeError.
    """

    def __init__(
        self,
        *,
        nlp: Any = None,
        grammar_corrector: Any = None,
        semantic_verifier: Any = None,
        nli_model: Any = None,
        chromadb: Any = None,
        embedding_model: Any = None,
        indexer: Any = None,
        structural_analyzer: Any = None,
        style_analyzer: Any = None,
        enhanced_analyzer: Any = None,
    ):
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

    # Properties return the stored slot. Per-singleton migrations wire up
    # lazy loaders here; until a slot is migrated, callers continue using its
    # legacy get_*() helper. An uninjected, unmigrated slot returns None.

    @property
    def nlp(self) -> Any:
        if self._nlp is None:
            from .utils.nlp import _load_spacy_nlp
            self._nlp = _load_spacy_nlp()
        return self._nlp

    @property
    def grammar_corrector(self) -> Any:
        if self._grammar_corrector is None:
            from .vocabulary.grammar_corrector import GrammarCorrector
            self._grammar_corrector = GrammarCorrector()
        return self._grammar_corrector

    @property
    def semantic_verifier(self) -> Any:
        if self._semantic_verifier is None:
            from .validation.semantic_verifier import SemanticVerifier
            self._semantic_verifier = SemanticVerifier()
        return self._semantic_verifier

    @property
    def nli_model(self) -> Any:
        if self._nli_model is None:
            from .validation.semantic_verifier import _load_nli_model
            self._nli_model = _load_nli_model()
        return self._nli_model

    @property
    def chromadb(self) -> Any:
        if self._chromadb is None:
            from .rag.corpus_indexer import _load_chromadb
            self._chromadb = _load_chromadb()
        return self._chromadb

    @property
    def embedding_model(self) -> Any:
        if self._embedding_model is None:
            from .rag.corpus_indexer import _load_embedding_model
            self._embedding_model = _load_embedding_model()
        return self._embedding_model

    @property
    def indexer(self) -> Any:
        if self._indexer is None:
            from .rag.corpus_indexer import _load_default_indexer
            self._indexer = _load_default_indexer()
        return self._indexer

    @property
    def structural_analyzer(self) -> Any:
        if self._structural_analyzer is None:
            from .rag.structural_analyzer import StructuralAnalyzer
            self._structural_analyzer = StructuralAnalyzer()
        return self._structural_analyzer

    @property
    def style_analyzer(self) -> Any:
        if self._style_analyzer is None:
            from .rag.style_analyzer import StyleAnalyzer
            self._style_analyzer = StyleAnalyzer()
        return self._style_analyzer

    @property
    def enhanced_analyzer(self) -> Any:
        if self._enhanced_analyzer is None:
            from .rag.enhanced_analyzer import EnhancedStructuralAnalyzer
            self._enhanced_analyzer = EnhancedStructuralAnalyzer()
        return self._enhanced_analyzer


_default_services: Optional[Services] = None


def get_default_services() -> Services:
    """Return the process-wide default Services container, creating it lazily.

    Legacy get_*() helpers delegate here during the migration. Tests that need
    isolation should use `set_default_services(Services(...))` (remember to
    restore the original in a `finally` block).
    """
    global _default_services
    if _default_services is None:
        _default_services = Services()
    return _default_services


def set_default_services(services: Services) -> None:
    """Swap the default container — primarily a test seam."""
    global _default_services
    _default_services = services

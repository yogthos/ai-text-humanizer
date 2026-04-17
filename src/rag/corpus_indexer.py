"""Corpus indexer for Style RAG.

Indexes author corpus files into ChromaDB with style metrics.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..utils.logging import get_logger
from ..utils.nlp import split_into_paragraphs
from .style_analyzer import StyleAnalyzer, StyleMetrics

logger = get_logger(__name__)

def _load_chromadb():
    """Import and return the chromadb module. Called once per Services
    container (cached on `Services._chromadb`)."""
    try:
        import chromadb
        return chromadb
    except ImportError:
        raise ImportError("chromadb required. Install with: pip install chromadb")


def _load_embedding_model():
    """Load the shared SentenceTransformer with stdout/stderr/tqdm suppressed.
    Called once per Services container (cached on `Services._embedding_model`)."""
    try:
        import sys
        import os
        import warnings
        import logging
        from io import StringIO

        # Disable tqdm before importing sentence_transformers
        os.environ["TQDM_DISABLE"] = "1"
        from sentence_transformers import SentenceTransformer

        # Suppress noisy warnings and stdout during model loading
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            st_logger = logging.getLogger("sentence_transformers")
            tf_logger = logging.getLogger("transformers")
            old_st_level = st_logger.level
            old_tf_level = tf_logger.level
            st_logger.setLevel(logging.ERROR)
            tf_logger.setLevel(logging.ERROR)
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            try:
                model = SentenceTransformer("all-MiniLM-L6-v2")
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
                st_logger.setLevel(old_st_level)
                tf_logger.setLevel(old_tf_level)

        logger.debug("Loaded sentence transformer: all-MiniLM-L6-v2")
        return model
    except ImportError:
        raise ImportError(
            "sentence-transformers required. Install with: pip install sentence-transformers"
        )


def _default_indexer_path() -> str:
    """Project-local default ChromaDB path."""
    return str(Path(__file__).parent.parent.parent / "data" / "rag_index")


def _load_default_indexer() -> "CorpusIndexer":
    """Construct the shared default indexer pointed at the project data dir."""
    return CorpusIndexer(_default_indexer_path())


def get_chromadb():
    """Return the chromadb module from the default Services container."""
    from ..services import get_default_services
    return get_default_services().chromadb


def get_embedding_model():
    """Return the shared SentenceTransformer from Services."""
    from ..services import get_default_services
    return get_default_services().embedding_model


@dataclass
class IndexedChunk:
    """A chunk of text with its metadata."""

    id: str
    text: str
    author: str
    source_file: str
    metrics: StyleMetrics


class CorpusIndexer:
    """Indexes author corpora into ChromaDB for style retrieval."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        *,
        style_analyzer: Optional[StyleAnalyzer] = None,
    ):
        """Initialize the indexer.

        Args:
            persist_dir: Directory to persist ChromaDB. If None, uses in-memory.
            style_analyzer: Injected StyleAnalyzer. Defaults to the one held by
                the shared Services container (`get_default_services().style_analyzer`).
        """
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        if style_analyzer is None:
            from ..services import get_default_services
            style_analyzer = get_default_services().style_analyzer
        self._analyzer = style_analyzer
        self._embedding_model = None

    @property
    def client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            chromadb = get_chromadb()
            settings = chromadb.Settings(anonymized_telemetry=False)
            if self.persist_dir:
                os.makedirs(self.persist_dir, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self.persist_dir, settings=settings)
                logger.info(f"Using persistent ChromaDB at: {self.persist_dir}")
            else:
                self._client = chromadb.Client(settings=settings)
                logger.info("Using in-memory ChromaDB")
        return self._client

    @property
    def collection(self):
        """Get or create the style collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name="style_chunks",
                metadata={"description": "Author style chunks for RAG retrieval"}
            )
            count = self._collection.count()
            if count == 0:
                logger.warning("ChromaDB collection 'style_chunks' is empty — no authors indexed. Run load_corpus.py first.")
            else:
                logger.info(f"Collection 'style_chunks' has {count} chunks")
        return self._collection

    @property
    def embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def _chunk_id(self, author: str, source: str, index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{author}:{source}:{index}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _split_into_chunks(
        self,
        text: str,
        min_words: int = 30,
        max_words: int = 200
    ) -> List[str]:
        """Split text into paragraph-sized chunks.

        Args:
            text: Raw corpus text.
            min_words: Minimum words per chunk.
            max_words: Maximum words per chunk.

        Returns:
            List of text chunks.
        """
        paragraphs = split_into_paragraphs(text)
        chunks = []

        current_chunk = []
        current_words = 0

        for para in paragraphs:
            para_words = len(para.split())

            # Skip very short paragraphs (likely headers)
            if para_words < 10:
                continue

            # If this paragraph alone exceeds max, split it
            if para_words > max_words:
                # Flush current
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_words = 0

                # Split large paragraph by sentences
                sentences = para.replace(". ", ".\n").split("\n")
                sent_chunk = []
                sent_words = 0
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    sw = len(sent.split())
                    if sent_words + sw > max_words and sent_chunk:
                        chunks.append(" ".join(sent_chunk))
                        sent_chunk = [sent]
                        sent_words = sw
                    else:
                        sent_chunk.append(sent)
                        sent_words += sw
                if sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                continue

            # Normal paragraph - accumulate
            if current_words + para_words > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [para]
                current_words = para_words
            else:
                current_chunk.append(para)
                current_words += para_words

        # Flush remaining
        if current_chunk and current_words >= min_words:
            chunks.append(" ".join(current_chunk))

        return chunks

    def index_corpus(
        self,
        corpus_path: str,
        author: str,
        clear_existing: bool = False
    ) -> int:
        """Index a corpus file into ChromaDB.

        Args:
            corpus_path: Path to corpus text file.
            author: Author name for this corpus.
            clear_existing: If True, delete existing chunks for this author.

        Returns:
            Number of chunks indexed.
        """
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        logger.info(f"Indexing corpus: {corpus_path} for author: {author}")

        # Read corpus
        text = path.read_text(encoding="utf-8")
        source_file = path.name

        # Clear existing if requested
        if clear_existing:
            self._delete_author_chunks(author)

        # Split into chunks
        chunks = self._split_into_chunks(text)
        logger.info(f"Split into {len(chunks)} chunks")

        if not chunks:
            logger.warning("No chunks extracted from corpus")
            return 0

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)

        # Analyze style metrics
        metrics_list = self._analyzer.analyze_batch(chunks)

        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embedding_list = []

        for i, (chunk, metrics, embedding) in enumerate(
            zip(chunks, metrics_list, embeddings)
        ):
            chunk_id = self._chunk_id(author, source_file, i)
            ids.append(chunk_id)
            documents.append(chunk)

            # Build metadata
            meta = metrics.to_dict()
            meta["author"] = author
            meta["source_file"] = source_file
            meta["chunk_index"] = i
            metadatas.append(meta)

            embedding_list.append(embedding.tolist())

        # Upsert to collection in batches (ChromaDB limit is 5461)
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embedding_list[start:end],
            )

        logger.info(f"Indexed {len(chunks)} chunks for {author}")
        return len(chunks)

    def _delete_author_chunks(self, author: str) -> int:
        """Delete all chunks for an author.

        Args:
            author: Author name.

        Returns:
            Number of chunks deleted.
        """
        # Get existing chunks for this author
        results = self.collection.get(
            where={"author": author},
            include=["metadatas"]
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} existing chunks for {author}")
            return len(results["ids"])

        return 0

    def get_authors(self) -> List[str]:
        """Get list of indexed authors.

        Returns:
            List of author names.
        """
        # Get unique authors from metadata
        results = self.collection.get(include=["metadatas"])
        authors = set()
        for meta in results.get("metadatas", []):
            if meta and "author" in meta:
                authors.add(meta["author"])
        return sorted(authors)

    def get_chunk_count(self, author: Optional[str] = None) -> int:
        """Get number of indexed chunks.

        Args:
            author: If provided, count only for this author.

        Returns:
            Number of chunks.
        """
        if author:
            results = self.collection.get(where={"author": author})
            return len(results.get("ids", []))
        return self.collection.count()

    def get_random_chunks(self, author: str, n: int = 50) -> List[str]:
        """Get random chunks from an author's corpus.

        Args:
            author: Author name.
            n: Number of chunks to retrieve.

        Returns:
            List of chunk texts.
        """
        import random

        results = self.collection.get(
            where={"author": author},
            include=["documents"]
        )

        documents = results.get("documents", [])
        if not documents:
            logger.warning(f"No chunks found for author: {author}")
            return []

        # Random sample
        if len(documents) <= n:
            return documents

        return random.sample(documents, n)

    def retrieve_similar(
        self,
        author: str,
        query_text: str,
        n: int = 3,
        fallback_to_random: bool = True
    ) -> List[dict]:
        """Retrieve chunks semantically similar to query text.

        Args:
            author: Author name to search within.
            query_text: Text to find similar chunks for.
            n: Number of results to return.
            fallback_to_random: If True, fall back to random chunks when
                semantic search returns empty but chunks exist for the author.

        Returns:
            List of dicts with 'text', 'skeleton', and 'distance' keys.
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query_text], show_progress_bar=False)[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            where={"author": author},
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            # Diagnose why no results were found
            chunk_count = self.get_chunk_count(author)
            if chunk_count == 0:
                available = self.get_authors()
                if available:
                    logger.warning(
                        f"No chunks indexed for author '{author}'. "
                        f"Available authors: {', '.join(available)}. "
                        f"Run: python scripts/load_corpus.py --input <corpus> --author \"{author}\""
                    )
                else:
                    logger.warning(
                        f"No chunks indexed for author '{author}' "
                        f"(ChromaDB is empty — no authors indexed). "
                        f"Run: python scripts/load_corpus.py --input <corpus> --author \"{author}\""
                    )
                return []
            else:
                # Chunks exist but semantic search returned nothing — unexpected
                logger.warning(
                    f"Semantic search returned no results for author '{author}' "
                    f"despite {chunk_count} indexed chunks. This is unexpected."
                )
                if fallback_to_random:
                    logger.info(f"Falling back to random chunks for '{author}'")
                    random_chunks = self.get_random_chunks(author, n=n)
                    return [
                        {"text": text, "skeleton": "", "distance": -1.0}
                        for text in random_chunks
                    ]
                return []

        # Build result list
        chunks = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append({
                "text": doc,
                "skeleton": meta.get("skeleton", ""),
                "distance": dist,
            })

        return chunks


def get_indexer(persist_dir: Optional[str] = None) -> CorpusIndexer:
    """Return a CorpusIndexer.

    With no persist_dir, returns the shared indexer owned by the default
    Services container (pointed at the project data directory). With an
    explicit persist_dir, returns a new uncached instance.
    """
    if persist_dir is not None:
        return CorpusIndexer(persist_dir)
    from ..services import get_default_services
    return get_default_services().indexer

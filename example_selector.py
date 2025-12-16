"""
Example Selector Module

Selects contextually relevant paragraphs from sample text to use as
few-shot examples for style transfer. Uses both semantic similarity
(via sentence embeddings) and structural similarity to find the most
appropriate examples for each input paragraph.

This replaces the static "first 3 paragraphs" approach with dynamic,
context-aware selection that utilizes the full sample text.
"""

import sqlite3
import json
import hashlib
import re
import pickle
import numpy as np
import spacy
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SampleParagraph:
    """A paragraph from the sample text with precomputed features."""
    text: str
    index: int
    embedding: Optional[np.ndarray] = None  # Semantic embedding
    sentence_count: int = 0
    word_count: int = 0
    avg_sentence_length: float = 0.0
    opener_type: str = "other"  # 'contrary_to', 'hence', 'the', etc.
    structural_role: str = "body"  # 'section_opener', 'paragraph_opener', 'body', 'conclusion'
    has_citations: bool = False
    has_quotes: bool = False
    discourse_markers: List[str] = field(default_factory=list)


class ExampleSelector:
    """
    Selects contextually relevant examples from sample text.

    Features:
    - Semantic similarity via sentence-transformers embeddings
    - Structural similarity (sentence count, length, opener type)
    - SQLite caching for embeddings
    - Configurable weights for semantic vs structural similarity
    """

    DB_PATH = Path(__file__).parent / "example_cache.db"

    # Opener type patterns
    OPENER_PATTERNS = {
        'contrary_to': r'^contrary to\b',
        'hence': r'^hence[,\s]',
        'thus': r'^thus[,\s]',
        'therefore': r'^therefore[,\s]',
        'the_method': r'^the \w+ method\b',
        'however': r'^however[,\s]',
        'further': r'^further[,\s]',
        'it_is': r'^it is\b',
        'this': r'^this\b',
        'such': r'^such\b',
        'moreover': r'^moreover[,\s]',
        'consequently': r'^consequently[,\s]',
        'in_this': r'^in this\b',
        'the': r'^the\b',
    }

    # Discourse markers to detect
    DISCOURSE_MARKERS = [
        'hence', 'therefore', 'thus', 'consequently', 'accordingly',
        'moreover', 'furthermore', 'however', 'nevertheless', 'nonetheless',
        'contrary to', 'in contrast', 'on the other hand',
        'for example', 'for instance', 'such as',
        'in this connection', 'it follows that', 'this means that'
    ]

    def __init__(self, sample_text: str, config: Optional[Dict] = None):
        """
        Initialize the example selector.

        Args:
            sample_text: Full sample text to extract examples from
            config: Configuration dict with example_selection settings
        """
        self.sample_text = sample_text
        self.sample_hash = self._get_hash(sample_text)
        self.paragraphs: List[SampleParagraph] = []

        # Load config
        self.config = config or {}
        es_config = self.config.get('example_selection', {})
        self.num_examples = es_config.get('num_examples', 3)
        self.semantic_weight = es_config.get('semantic_weight', 0.4)
        self.structural_weight = es_config.get('structural_weight', 0.6)
        self.min_word_count = es_config.get('min_word_count', 5)
        self.max_word_count = es_config.get('max_word_count', 10000)

        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize embedder (with fallback)
        self.embedder = None
        self._init_embedder()

        # Initialize database
        self._init_db()

        # Analyze and cache sample
        self._analyze_sample()

    def _get_hash(self, text: str) -> str:
        """Generate hash of text for cache lookup."""
        return hashlib.md5(text.encode()).hexdigest()

    def _init_embedder(self):
        """Initialize sentence-transformers embedder with fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight but effective model
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("  [ExampleSelector] Using sentence-transformers for semantic similarity")
        except ImportError:
            print("  [ExampleSelector] WARNING: sentence-transformers not available")
            print("  [ExampleSelector] Install with: pip install sentence-transformers")
            print("  [ExampleSelector] Falling back to structural similarity only")
            self.embedder = None

    def _init_db(self):
        """Initialize SQLite database for caching embeddings."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sample_paragraphs (
                sample_hash TEXT,
                para_index INTEGER,
                text TEXT,
                embedding BLOB,
                sentence_count INTEGER,
                word_count INTEGER,
                avg_sentence_length REAL,
                opener_type TEXT,
                structural_role TEXT,
                has_citations INTEGER,
                has_quotes INTEGER,
                discourse_markers TEXT,
                PRIMARY KEY (sample_hash, para_index)
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sample_hash
            ON sample_paragraphs(sample_hash)
        ''')

        conn.commit()
        conn.close()

    def _load_from_cache(self) -> bool:
        """Try to load paragraphs from cache. Returns True if successful."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            'SELECT COUNT(*) FROM sample_paragraphs WHERE sample_hash = ?',
            (self.sample_hash,)
        )
        count = cursor.fetchone()[0]

        if count == 0:
            conn.close()
            return False

        cursor.execute('''
            SELECT para_index, text, embedding, sentence_count, word_count,
                   avg_sentence_length, opener_type, structural_role,
                   has_citations, has_quotes, discourse_markers
            FROM sample_paragraphs
            WHERE sample_hash = ?
            ORDER BY para_index
        ''', (self.sample_hash,))

        rows = cursor.fetchall()
        conn.close()

        self.paragraphs = []
        for row in rows:
            embedding = pickle.loads(row[2]) if row[2] else None
            markers = json.loads(row[10]) if row[10] else []

            para = SampleParagraph(
                text=row[1],
                index=row[0],
                embedding=embedding,
                sentence_count=row[3],
                word_count=row[4],
                avg_sentence_length=row[5],
                opener_type=row[6],
                structural_role=row[7],
                has_citations=bool(row[8]),
                has_quotes=bool(row[9]),
                discourse_markers=markers
            )
            self.paragraphs.append(para)

        print(f"  [ExampleSelector] Loaded {len(self.paragraphs)} paragraphs from cache")
        return True

    def _save_to_cache(self):
        """Save analyzed paragraphs to cache."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        # Clear existing data for this sample
        cursor.execute(
            'DELETE FROM sample_paragraphs WHERE sample_hash = ?',
            (self.sample_hash,)
        )

        for para in self.paragraphs:
            embedding_blob = pickle.dumps(para.embedding) if para.embedding is not None else None
            markers_json = json.dumps(para.discourse_markers)

            cursor.execute('''
                INSERT INTO sample_paragraphs
                (sample_hash, para_index, text, embedding, sentence_count, word_count,
                 avg_sentence_length, opener_type, structural_role, has_citations,
                 has_quotes, discourse_markers)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.sample_hash,
                para.index,
                para.text,
                embedding_blob,
                para.sentence_count,
                para.word_count,
                para.avg_sentence_length,
                para.opener_type,
                para.structural_role,
                int(para.has_citations),
                int(para.has_quotes),
                markers_json
            ))

        conn.commit()
        conn.close()
        print(f"  [ExampleSelector] Cached {len(self.paragraphs)} paragraphs")

    def _analyze_sample(self):
        """Analyze sample text and extract paragraph features."""
        # Try loading from cache first
        if self._load_from_cache():
            return

        print("  [ExampleSelector] Analyzing sample text...")

        # Split into paragraphs
        raw_paragraphs = [p.strip() for p in self.sample_text.split('\n\n') if p.strip()]

        # Analyze each paragraph
        self.paragraphs = []
        texts_for_embedding = []

        for i, text in enumerate(raw_paragraphs):
            word_count = len(text.split())

            # Skip paragraphs outside acceptable range
            if word_count < self.min_word_count or word_count > self.max_word_count:
                continue

            # Analyze with spaCy
            doc = self.nlp(text)
            sentences = list(doc.sents)
            sentence_count = len(sentences)
            avg_sent_len = word_count / max(sentence_count, 1)

            # Detect opener type
            opener_type = self._detect_opener_type(text)

            # Detect structural role
            structural_role = self._detect_structural_role(text, i, len(raw_paragraphs))

            # Check for citations and quotes
            has_citations = bool(re.search(r'\[\^?\d+\]|\(\d{4}\)', text))
            has_quotes = bool(re.search(r'"[^"]+"|\'[^\']+\'', text))

            # Detect discourse markers
            markers = self._detect_discourse_markers(text)

            para = SampleParagraph(
                text=text,
                index=len(self.paragraphs),
                sentence_count=sentence_count,
                word_count=word_count,
                avg_sentence_length=avg_sent_len,
                opener_type=opener_type,
                structural_role=structural_role,
                has_citations=has_citations,
                has_quotes=has_quotes,
                discourse_markers=markers
            )
            self.paragraphs.append(para)
            texts_for_embedding.append(text)

        # Compute embeddings in batch if embedder available
        if self.embedder and texts_for_embedding:
            print(f"  [ExampleSelector] Computing embeddings for {len(texts_for_embedding)} paragraphs...")
            embeddings = self.embedder.encode(texts_for_embedding, show_progress_bar=False)
            for i, para in enumerate(self.paragraphs):
                para.embedding = embeddings[i]

        # Save to cache
        self._save_to_cache()
        print(f"  [ExampleSelector] Analyzed {len(self.paragraphs)} qualifying paragraphs")

    def _detect_opener_type(self, text: str) -> str:
        """Detect the opener pattern type of a paragraph."""
        text_lower = text.lower()

        for opener_type, pattern in self.OPENER_PATTERNS.items():
            if re.match(pattern, text_lower, re.IGNORECASE):
                return opener_type

        return "other"

    def _detect_structural_role(self, text: str, index: int, total: int) -> str:
        """Detect the structural role of a paragraph."""
        text_lower = text.lower()

        # Section openers typically have contrastive patterns
        if re.match(r'^contrary to\b', text_lower):
            return "section_opener"
        if re.match(r'^the principal features\b', text_lower):
            return "section_opener"

        # Conclusion markers
        if any(m in text_lower for m in ['such is', 'this means that', 'it follows that']):
            return "conclusion"

        # Paragraph openers have transition markers at start
        opener_markers = ['hence', 'thus', 'therefore', 'consequently', 'further', 'moreover']
        for marker in opener_markers:
            if re.match(rf'^{marker}[,\s]', text_lower):
                return "paragraph_opener"

        # Position-based heuristic
        if index == 0:
            return "section_opener"
        if index == total - 1:
            return "conclusion"

        return "body"

    def _detect_discourse_markers(self, text: str) -> List[str]:
        """Detect which discourse markers are present in the text."""
        text_lower = text.lower()
        found = []

        for marker in self.DISCOURSE_MARKERS:
            if marker in text_lower:
                found.append(marker)

        return found

    def _analyze_input_paragraph(self, text: str) -> SampleParagraph:
        """Analyze an input paragraph to create a comparable SampleParagraph."""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        word_count = len(text.split())
        sentence_count = len(sentences)

        # Compute embedding if available
        embedding = None
        if self.embedder:
            embedding = self.embedder.encode(text, show_progress_bar=False)

        return SampleParagraph(
            text=text,
            index=-1,
            embedding=embedding,
            sentence_count=sentence_count,
            word_count=word_count,
            avg_sentence_length=word_count / max(sentence_count, 1),
            opener_type=self._detect_opener_type(text),
            structural_role=self._detect_structural_role(text, 0, 1),
            has_citations=bool(re.search(r'\[\^?\d+\]|\(\d{4}\)', text)),
            has_quotes=bool(re.search(r'"[^"]+"|\'[^\']+\'', text)),
            discourse_markers=self._detect_discourse_markers(text)
        )

    def _calculate_similarity(self, input_para: SampleParagraph,
                              sample_para: SampleParagraph) -> float:
        """
        Calculate combined semantic + structural similarity.

        Returns:
            Similarity score between 0 and 1
        """
        # === Semantic Similarity ===
        semantic_sim = 0.5  # Default if no embeddings
        if input_para.embedding is not None and sample_para.embedding is not None:
            # Cosine similarity
            dot = np.dot(input_para.embedding, sample_para.embedding)
            norm = np.linalg.norm(input_para.embedding) * np.linalg.norm(sample_para.embedding)
            semantic_sim = dot / max(norm, 1e-10)
            # Normalize to 0-1 range (cosine can be negative)
            semantic_sim = (semantic_sim + 1) / 2

        # === Structural Similarity ===

        # Sentence count similarity (penalize large differences)
        sent_diff = abs(input_para.sentence_count - sample_para.sentence_count)
        sent_sim = max(0, 1 - sent_diff / 10)

        # Word count similarity
        word_diff = abs(input_para.word_count - sample_para.word_count)
        word_sim = max(0, 1 - word_diff / 150)

        # Opener type match (strong signal)
        opener_match = 1.0 if input_para.opener_type == sample_para.opener_type else 0.3

        # Structural role match
        role_match = 1.0 if input_para.structural_role == sample_para.structural_role else 0.4

        # Average sentence length similarity
        len_diff = abs(input_para.avg_sentence_length - sample_para.avg_sentence_length)
        len_sim = max(0, 1 - len_diff / 20)

        # Citation/quote feature match
        feature_match = 0.5
        if input_para.has_citations == sample_para.has_citations:
            feature_match += 0.25
        if input_para.has_quotes == sample_para.has_quotes:
            feature_match += 0.25

        # Discourse marker overlap (Jaccard similarity)
        input_markers = set(input_para.discourse_markers)
        sample_markers = set(sample_para.discourse_markers)
        if input_markers or sample_markers:
            marker_sim = len(input_markers & sample_markers) / len(input_markers | sample_markers)
        else:
            marker_sim = 0.5  # Both empty = neutral

        # Combine structural components
        structural_sim = (
            sent_sim * 0.15 +
            word_sim * 0.15 +
            opener_match * 0.25 +
            role_match * 0.20 +
            len_sim * 0.10 +
            feature_match * 0.05 +
            marker_sim * 0.10
        )

        # === Combined Score ===
        if input_para.embedding is not None and sample_para.embedding is not None:
            # Use configured weights
            return (self.semantic_weight * semantic_sim +
                    self.structural_weight * structural_sim)
        else:
            # No embeddings - use structural only
            return structural_sim

    def select_examples(self, input_text: str, k: Optional[int] = None,
                        exclude_indices: Optional[List[int]] = None) -> List[str]:
        """
        Select the k most relevant sample paragraphs for the input.

        Args:
            input_text: The paragraph being transformed
            k: Number of examples to return (default from config)
            exclude_indices: Paragraph indices to exclude (e.g., already used)

        Returns:
            List of selected paragraph texts
        """
        if k is None:
            k = self.num_examples

        if not self.paragraphs:
            return []

        # Analyze input
        input_para = self._analyze_input_paragraph(input_text)

        # Calculate similarities
        scores = []
        for para in self.paragraphs:
            if exclude_indices and para.index in exclude_indices:
                continue

            score = self._calculate_similarity(input_para, para)
            scores.append((score, para))

        # Sort by score (descending)
        scores.sort(key=lambda x: x[0], reverse=True)

        # Return top k
        return [para.text for score, para in scores[:k]]

    def select_diverse_examples(self, input_text: str, k: Optional[int] = None) -> List[str]:
        """
        Select examples that are similar to input but diverse among themselves.

        Uses maximal marginal relevance (MMR) style selection to balance
        relevance with diversity.

        Args:
            input_text: The paragraph being transformed
            k: Number of examples to return

        Returns:
            List of selected paragraph texts
        """
        if k is None:
            k = self.num_examples

        if not self.paragraphs:
            return []

        # Analyze input
        input_para = self._analyze_input_paragraph(input_text)

        # Calculate relevance scores
        candidates = []
        for para in self.paragraphs:
            score = self._calculate_similarity(input_para, para)
            candidates.append({'para': para, 'relevance': score})

        # Sort by relevance
        candidates.sort(key=lambda x: x['relevance'], reverse=True)

        # MMR-style selection
        selected = []
        selected_texts = set()

        while len(selected) < k and candidates:
            best_idx = 0
            best_mmr = -1

            for i, cand in enumerate(candidates):
                if cand['para'].text in selected_texts:
                    continue

                # Relevance to input
                relevance = cand['relevance']

                # Max similarity to already selected (diversity penalty)
                max_sim_to_selected = 0
                for sel in selected:
                    sim = self._calculate_similarity(sel['para'], cand['para'])
                    max_sim_to_selected = max(max_sim_to_selected, sim)

                # MMR score: relevance - lambda * max_sim_to_selected
                lambda_param = 0.5  # Balance between relevance and diversity
                mmr = relevance - lambda_param * max_sim_to_selected

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(candidates[best_idx])
            selected_texts.add(candidates[best_idx]['para'].text)
            candidates.pop(best_idx)

        return [s['para'].text for s in selected]

    def get_stats(self) -> Dict:
        """Get statistics about the analyzed sample."""
        if not self.paragraphs:
            return {'total_paragraphs': 0}

        opener_counts = {}
        role_counts = {}

        for para in self.paragraphs:
            opener_counts[para.opener_type] = opener_counts.get(para.opener_type, 0) + 1
            role_counts[para.structural_role] = role_counts.get(para.structural_role, 0) + 1

        return {
            'total_paragraphs': len(self.paragraphs),
            'with_embeddings': sum(1 for p in self.paragraphs if p.embedding is not None),
            'avg_word_count': sum(p.word_count for p in self.paragraphs) / len(self.paragraphs),
            'avg_sentence_count': sum(p.sentence_count for p in self.paragraphs) / len(self.paragraphs),
            'opener_types': opener_counts,
            'structural_roles': role_counts,
            'has_citations': sum(1 for p in self.paragraphs if p.has_citations),
            'has_quotes': sum(1 for p in self.paragraphs if p.has_quotes),
        }


# Test function
if __name__ == '__main__':
    from pathlib import Path

    sample_path = Path(__file__).parent / "prompts" / "sample.txt"
    if sample_path.exists():
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()

        print("=== Example Selector Test ===\n")

        selector = ExampleSelector(sample_text)

        print("\n=== Sample Stats ===")
        stats = selector.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n=== Testing Selection ===")

        # Test input paragraph
        test_input = """Human experience reinforces the rule of finitude. The biological cycle
        of birth, life, and decay defines our reality. Every object we touch eventually breaks.
        Every star burning in the night sky eventually succumbs to erosion."""

        print(f"\nInput paragraph ({len(test_input.split())} words):")
        print(f"  {test_input[:100]}...")

        print("\n--- Standard Selection (top 3) ---")
        examples = selector.select_examples(test_input, k=3)
        for i, ex in enumerate(examples, 1):
            print(f"\nExample {i} ({len(ex.split())} words):")
            print(f"  {ex[:150]}...")

        print("\n--- Diverse Selection (top 3) ---")
        diverse_examples = selector.select_diverse_examples(test_input, k=3)
        for i, ex in enumerate(diverse_examples, 1):
            print(f"\nExample {i} ({len(ex.split())} words):")
            print(f"  {ex[:150]}...")
    else:
        print("No sample.txt found. Create prompts/sample.txt first.")


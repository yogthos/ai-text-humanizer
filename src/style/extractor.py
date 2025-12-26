"""Extract style profiles from corpus text.

All extraction is data-driven using spaCy - NO hardcoded patterns.
"""

import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np

from ..utils.nlp import get_nlp, split_into_sentences
from ..utils.logging import get_logger
from .profile import (
    SentenceLengthProfile,
    TransitionProfile,
    RegisterProfile,
    DeltaProfile,
    AuthorStyleProfile,
)

logger = get_logger(__name__)


# Transition word categories (used for classification, not prescription)
TRANSITION_CATEGORIES = {
    "causal": {
        "therefore", "thus", "hence", "consequently", "so", "because",
        "since", "as a result", "accordingly", "for this reason",
    },
    "adversative": {
        "however", "but", "yet", "although", "though", "nevertheless",
        "nonetheless", "on the other hand", "in contrast", "while",
        "whereas", "despite", "in spite of", "conversely",
    },
    "additive": {
        "also", "moreover", "furthermore", "additionally", "besides",
        "in addition", "likewise", "similarly", "not only", "as well as",
    },
    "temporal": {
        "then", "finally", "subsequently", "meanwhile", "first",
        "second", "third", "next", "afterwards", "before", "after",
        "during", "when", "while",
    },
}


class StyleProfileExtractor:
    """Extract complete style profile from corpus."""

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp()
        return self._nlp

    def extract(
        self,
        paragraphs: List[str],
        author_name: str,
    ) -> AuthorStyleProfile:
        """Extract complete style profile from corpus paragraphs.

        Args:
            paragraphs: List of paragraph texts.
            author_name: Author name.

        Returns:
            AuthorStyleProfile with all extracted metrics.
        """
        logger.info(f"Extracting style profile for {author_name} from {len(paragraphs)} paragraphs")

        # Collect all sentences
        all_sentences = []
        for para in paragraphs:
            sentences = split_into_sentences(para)
            all_sentences.extend(sentences)

        if not all_sentences:
            raise ValueError("No sentences found in corpus")

        # Extract component profiles
        length_profile = self._extract_length_profile(all_sentences)
        transition_profile = self._extract_transition_profile(all_sentences)
        register_profile = self._extract_register_profile(all_sentences, paragraphs)
        delta_profile = self._extract_delta_profile(all_sentences)

        # Calculate totals
        word_count = sum(len(s.split()) for s in all_sentences)

        logger.info(f"Extracted profile: {len(all_sentences)} sentences, {word_count} words")
        logger.info(f"  Length: {length_profile.mean:.1f} +/- {length_profile.std:.1f}")
        logger.info(f"  Burstiness: {length_profile.burstiness:.3f}")
        logger.info(f"  No-transition ratio: {transition_profile.no_transition_ratio:.2f}")

        return AuthorStyleProfile(
            author_name=author_name,
            corpus_word_count=word_count,
            corpus_sentence_count=len(all_sentences),
            length_profile=length_profile,
            transition_profile=transition_profile,
            register_profile=register_profile,
            delta_profile=delta_profile,
        )

    def _extract_length_profile(self, sentences: List[str]) -> SentenceLengthProfile:
        """Extract sentence length profile with Markov transitions."""
        lengths = [len(s.split()) for s in sentences]

        if not lengths:
            return SentenceLengthProfile(
                mean=20.0, std=10.0, min_length=3, max_length=60
            )

        # Basic statistics
        mean = np.mean(lengths)
        std = np.std(lengths)
        min_len = min(lengths)
        max_len = max(lengths)

        # Percentiles
        percentiles = {
            10: int(np.percentile(lengths, 10)),
            25: int(np.percentile(lengths, 25)),
            50: int(np.percentile(lengths, 50)),
            75: int(np.percentile(lengths, 75)),
            90: int(np.percentile(lengths, 90)),
        }

        # Burstiness (coefficient of variation)
        burstiness = std / mean if mean > 0 else 0.0

        # Short/long ratios
        short_ratio = sum(1 for l in lengths if l < 10) / len(lengths)
        long_ratio = sum(1 for l in lengths if l > 30) / len(lengths)

        # Build Markov transition matrix
        length_transitions = self._build_length_markov(lengths, percentiles)

        return SentenceLengthProfile(
            mean=float(mean),
            std=float(std),
            min_length=min_len,
            max_length=max_len,
            percentiles=percentiles,
            burstiness=float(burstiness),
            short_ratio=float(short_ratio),
            long_ratio=float(long_ratio),
            length_transitions=length_transitions,
        )

    def _build_length_markov(
        self,
        lengths: List[int],
        percentiles: Dict[int, int],
    ) -> Dict[str, Dict[str, float]]:
        """Build Order-1 Markov chain for length transitions.

        Research shows bigram (Order-1) is sufficient for sentence lengths.
        """
        p25 = percentiles.get(25, 12)
        p75 = percentiles.get(75, 25)

        def categorize(length: int) -> str:
            if length < p25:
                return "short"
            elif length > p75:
                return "long"
            else:
                return "medium"

        # Count transitions
        transitions = defaultdict(Counter)
        categories = [categorize(l) for l in lengths]

        for i in range(len(categories) - 1):
            current = categories[i]
            next_cat = categories[i + 1]
            transitions[current][next_cat] += 1

        # Normalize to probabilities
        markov = {}
        for current, next_counts in transitions.items():
            total = sum(next_counts.values())
            markov[current] = {
                cat: count / total
                for cat, count in next_counts.items()
            }

        # Ensure all categories have entries
        for cat in ["short", "medium", "long"]:
            if cat not in markov:
                # Default uniform distribution
                markov[cat] = {"short": 0.33, "medium": 0.34, "long": 0.33}

        return markov

    def _extract_transition_profile(self, sentences: List[str]) -> TransitionProfile:
        """Extract transition word usage patterns."""
        # Count transitions per category
        category_counts = defaultdict(Counter)
        total_sentences = len(sentences)
        sentences_with_transition = 0
        transitions_at_start = 0
        total_transitions = 0

        all_transitions = set()
        for cat_words in TRANSITION_CATEGORIES.values():
            all_transitions.update(cat_words)

        for sentence in sentences:
            words = sentence.lower().split()
            if not words:
                continue

            found_transition = False
            first_words = " ".join(words[:3])  # Check first 3 words

            for category, cat_words in TRANSITION_CATEGORIES.items():
                for trans in cat_words:
                    if trans in sentence.lower():
                        category_counts[category][trans] += 1
                        total_transitions += 1
                        found_transition = True

                        # Check if at start
                        if trans in first_words:
                            transitions_at_start += 1

            if found_transition:
                sentences_with_transition += 1

        # Calculate ratios
        no_transition_ratio = 1.0 - (sentences_with_transition / total_sentences) if total_sentences > 0 else 0.5
        start_position_ratio = transitions_at_start / total_transitions if total_transitions > 0 else 0.0
        transition_per_sentence = total_transitions / total_sentences if total_sentences > 0 else 0.0

        # Normalize category frequencies
        def normalize_category(counts: Counter) -> Dict[str, float]:
            total = sum(counts.values())
            if total == 0:
                return {}
            return {word: count / total for word, count in counts.most_common(10)}

        return TransitionProfile(
            no_transition_ratio=float(no_transition_ratio),
            start_position_ratio=float(start_position_ratio),
            transition_per_sentence=float(transition_per_sentence),
            causal=normalize_category(category_counts["causal"]),
            adversative=normalize_category(category_counts["adversative"]),
            additive=normalize_category(category_counts["additive"]),
            temporal=normalize_category(category_counts["temporal"]),
        )

    def _extract_register_profile(
        self,
        sentences: List[str],
        paragraphs: List[str],
    ) -> RegisterProfile:
        """Extract register/formality features."""
        total_sentences = len(sentences)
        total_paragraphs = len(paragraphs)

        if total_sentences == 0:
            return RegisterProfile()

        # Count punctuation
        semicolons = sum(s.count(";") for s in sentences)
        colons = sum(s.count(":") for s in sentences)
        dashes = sum(s.count("—") + s.count("--") for s in sentences)
        parentheticals = sum(s.count("(") for s in sentences)
        questions = sum(1 for s in sentences if s.strip().endswith("?"))

        # Count passive voice and imperatives using spaCy
        passive_count = 0
        imperative_count = 0

        for sentence in sentences[:200]:  # Limit for performance
            doc = self.nlp(sentence)
            for token in doc:
                # Passive detection
                if token.dep_ == "nsubjpass":
                    passive_count += 1
                    break
                # Imperative detection (verb at start with no subject)
                if token.i == 0 and token.pos_ == "VERB":
                    has_subject = any(
                        child.dep_ in ("nsubj", "nsubjpass")
                        for child in token.children
                    )
                    if not has_subject:
                        imperative_count += 1
                        break

        # Calculate formality score (heuristic based on features)
        # Higher formality: more semicolons, colons, longer sentences, fewer contractions
        avg_length = sum(len(s.split()) for s in sentences) / total_sentences
        contraction_count = sum(
            1 for s in sentences
            if any(c in s.lower() for c in ["n't", "'re", "'ve", "'ll", "'d"])
        )

        formality_indicators = [
            min(1.0, semicolons / total_sentences * 5),  # Semicolons boost formality
            min(1.0, avg_length / 30),  # Longer sentences = more formal
            1.0 - min(1.0, contraction_count / total_sentences),  # Fewer contractions = formal
        ]
        formality_score = sum(formality_indicators) / len(formality_indicators)

        return RegisterProfile(
            formality_score=float(formality_score),
            narrative_ratio=0.0,  # Would need more analysis
            question_frequency=float(questions / total_paragraphs) if total_paragraphs > 0 else 0.0,
            imperative_frequency=float(imperative_count / total_paragraphs) if total_paragraphs > 0 else 0.0,
            passive_voice_ratio=float(passive_count / min(total_sentences, 200)),
            semicolon_per_sentence=float(semicolons / total_sentences),
            colon_per_sentence=float(colons / total_sentences),
            dash_per_sentence=float(dashes / total_sentences),
            parenthetical_per_sentence=float(parentheticals / total_sentences),
        )

    def _extract_delta_profile(self, sentences: List[str]) -> DeltaProfile:
        """Extract Burrows' Delta profile.

        Uses 300 most frequent words per research recommendations.
        """
        # Tokenize and count all words
        word_counts = Counter()
        total_words = 0

        for sentence in sentences:
            words = re.findall(r'\b[a-z]+\b', sentence.lower())
            word_counts.update(words)
            total_words += len(words)

        if total_words == 0:
            return DeltaProfile()

        # Get top 300 MFW
        top_300 = word_counts.most_common(300)

        # Calculate frequencies
        mfw_frequencies = {
            word: count / total_words
            for word, count in top_300
        }

        # For z-scores, we need corpus mean and std
        # In a full implementation, this would be calculated across multiple authors
        # For now, use the author's own frequencies as baseline
        mfw_zscores = {}
        corpus_mean = {}
        corpus_std = {}

        for word, freq in mfw_frequencies.items():
            # Simplified: z-score is 0 for the author's own text
            # In practice, you'd compare to a reference corpus
            mfw_zscores[word] = 0.0
            corpus_mean[word] = freq
            corpus_std[word] = freq * 0.5  # Rough estimate

        return DeltaProfile(
            mfw_frequencies=mfw_frequencies,
            mfw_zscores=mfw_zscores,
            corpus_mean=corpus_mean,
            corpus_std=corpus_std,
        )


def extract_author_profile(
    paragraphs: List[str],
    author_name: str,
) -> AuthorStyleProfile:
    """Convenience function to extract author profile."""
    extractor = StyleProfileExtractor()
    return extractor.extract(paragraphs, author_name)

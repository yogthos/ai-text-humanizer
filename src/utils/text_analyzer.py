"""Lightweight text analyzer for quick statistical checks.

This module provides fast methods to analyze text statistics without
running the full heavy style profiler analysis.
"""

import re
from typing import Dict, Any
from src.utils.nlp_manager import NLPManager


class TextAnalyzer:
    """Lightweight text analyzer for quick statistical checks."""

    @staticmethod
    def quick_scan(text: str) -> Dict[str, Any]:
        """Quickly analyze text to extract key statistics.

        This is a lightweight version that only calculates essential metrics
        needed for style validation, without running the full style profiler.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with:
            - avg_words_per_sentence: Average number of words per sentence
            - dashes_per_100: Number of em-dashes per 100 words
            - semicolons_per_100: Number of semicolons per 100 words
        """
        if not text or not text.strip():
            return {
                "avg_words_per_sentence": 0.0,
                "dashes_per_100": 0.0,
                "semicolons_per_100": 0.0
            }

        # Use spaCy for sentence splitting and word counting
        nlp = NLPManager.get_nlp()
        doc = nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return {
                "avg_words_per_sentence": 0.0,
                "dashes_per_100": 0.0,
                "semicolons_per_100": 0.0
            }

        # Calculate average words per sentence (excluding punctuation tokens)
        sentence_lengths = []
        for sent in sentences:
            # Count words (tokens that are not punctuation)
            words = [token for token in sent if not token.is_punct and not token.is_space]
            sentence_lengths.append(len(words))

        avg_words_per_sentence = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0

        # Count total words (for per-100-word calculations)
        total_words = sum(sentence_lengths)

        # Count punctuation marks
        semicolons = sum(1 for token in doc if token.text == ';')
        # Count em-dashes (—), en-dashes (–), and hyphens (-) when used as punctuation
        dashes = sum(1 for token in doc if token.text in ['—', '–', '-'] and token.pos_ == 'PUNCT')

        # Calculate per 100 words
        if total_words > 0:
            multiplier = 100.0 / total_words
            dashes_per_100 = dashes * multiplier
            semicolons_per_100 = semicolons * multiplier
        else:
            dashes_per_100 = 0.0
            semicolons_per_100 = 0.0

        return {
            "avg_words_per_sentence": round(avg_words_per_sentence, 2),
            "dashes_per_100": round(dashes_per_100, 2),
            "semicolons_per_100": round(semicolons_per_100, 2)
        }


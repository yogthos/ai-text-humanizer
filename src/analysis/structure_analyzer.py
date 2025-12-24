"""
Structure Analyzer for extracting and comparing structural skeletons of sentences.

This module provides functionality to extract "load-bearing" structural tokens
(punctuation, conjunctions, negations, auxiliaries) and calculate similarity
between sentence structures.
"""

import spacy
from difflib import SequenceMatcher
from typing import List, Optional


class StructureAnalyzer:
    """Extracts and compares structural skeletons of sentences."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the structure analyzer with spaCy model.

        Args:
            model_name: Name of the spaCy model to load
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # Fallback: try to load from shared location or raise
            raise ImportError(
                f"spaCy model '{model_name}' not found. "
                "Please install it with: python -m spacy download {model_name}"
            )

    def extract_skeleton_tokens(self, text: str) -> List[str]:
        """
        Extracts the 'load-bearing' structural tokens from text:
        - Punctuation (., ;, ?, !)
        - Conjunctions (and, but, or, however)
        - Negations (not, no, never)
        - Modals/Auxiliaries (must, should, is, are)

        Args:
            text: Input text to analyze

        Returns:
            List of structural tokens in order of appearance
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        skeleton = []

        for token in doc:
            # Keep Punctuation
            if token.pos_ == "PUNCT":
                skeleton.append(token.text.lower())

            # Keep Coordinating and Subordinating Conjunctions
            elif token.pos_ in ["CCONJ", "SCONJ"]:
                skeleton.append(token.text.lower())

            # Keep Negations
            elif token.dep_ == "neg":  # 'not', 'no', 'never'
                skeleton.append(token.text.lower())

            # Keep Auxiliaries/Modals that define rhythm
            elif token.lemma_.lower() in ["be", "have", "do", "must", "should", "can", "will", "may"]:
                skeleton.append(token.text.lower())

        return skeleton

    def calculate_fidelity(self, draft: str, reference: str) -> float:
        """
        Calculates how well the Draft matches the Reference's structure.

        Uses SequenceMatcher to compare structural skeletons, which handles
        slight reordering or missing items gracefully.

        Args:
            draft: The generated sentence to evaluate
            reference: The reference sentence to match against

        Returns:
            Similarity score between 0.0 and 1.0 (1.0 = perfect match)
        """
        if not reference or not reference.strip():
            return 1.0  # No structure to match

        if not draft or not draft.strip():
            return 0.0  # Empty draft has no structure

        skel_ref = self.extract_skeleton_tokens(reference)
        skel_draft = self.extract_skeleton_tokens(draft)

        if not skel_ref:
            return 1.0  # No structure to match

        if not skel_draft:
            return 0.0  # Draft has no structure

        # SequenceMatcher calculates a similarity ratio (0.0 to 1.0)
        # It handles slight reordering or missing items gracefully
        similarity = SequenceMatcher(None, skel_ref, skel_draft).ratio()
        return similarity

    def extract_dynamic_skeleton(self, text: str) -> str:
        """
        Extracts a human-readable skeleton representation for prompting.

        This creates a "Mad Libs" style template showing the structure
        while preserving key structural markers.

        Args:
            text: Input text to analyze

        Returns:
            String representation of the skeleton (e.g., "It is not [X], but [Y].")
        """
        if not text or not text.strip():
            return ""

        doc = self.nlp(text)
        skeleton_parts = []

        for token in doc:
            # Keep structural tokens as-is
            if token.pos_ == "PUNCT":
                skeleton_parts.append(token.text)
            elif token.pos_ in ["CCONJ", "SCONJ"]:
                skeleton_parts.append(token.text)
            elif token.dep_ == "neg":
                skeleton_parts.append(token.text)
            elif token.lemma_.lower() in ["be", "have", "do", "must", "should", "can", "will", "may"]:
                skeleton_parts.append(token.text)
            # Replace content words with placeholders
            elif token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"]:
                if not skeleton_parts or skeleton_parts[-1] != "[...]":
                    skeleton_parts.append("[...]")

        # Join with spaces, but handle punctuation spacing
        result = []
        for i, part in enumerate(skeleton_parts):
            if part in [".", ",", ";", ":", "!", "?"]:
                # Punctuation attaches to previous word
                if result:
                    result[-1] += part
                else:
                    result.append(part)
            else:
                result.append(part)

        return " ".join(result)


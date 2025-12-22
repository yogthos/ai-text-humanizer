"""Vocabulary Budget: Tracks restricted word usage across document generation.

This module implements a "Power Word Budget" system that:
- Tracks usage of restricted vocabulary words (e.g., "profound", "vast", "monumental")
- Enforces per-chapter limits on word usage
- Detects clustering violations (words appearing too close together)
- Provides dynamic constraints for prompt injection
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class VocabularyBudget:
    """Tracks restricted vocabulary word usage and enforces budget constraints."""

    def __init__(
        self,
        restricted_words: List[str],
        max_per_chapter: int = 2,
        clustering_distance: int = 50
    ):
        """Initialize vocabulary budget tracker.

        Args:
            restricted_words: List of words to track (case-insensitive)
            max_per_chapter: Maximum allowed uses per word per chapter/document
            clustering_distance: Minimum word distance to avoid clustering (in words)
        """
        # Normalize words to lowercase for case-insensitive matching
        self.restricted_words = [word.lower() for word in restricted_words]
        self.max_per_chapter = max_per_chapter
        self.clustering_distance = clustering_distance

        # Track usage: word -> list of (position, text_context) tuples
        # Position is word index in the document (0-based)
        self._word_positions: Dict[str, List[int]] = defaultdict(list)

        # Track total word count in document for position calculation
        self._total_words = 0

    def check_word_allowed(self, word: str, current_position: Optional[int] = None) -> Tuple[bool, str]:
        """Check if a word is allowed to be used.

        Args:
            word: Word to check (case-insensitive)
            current_position: Current word position in document (optional, for clustering check)

        Returns:
            Tuple of (allowed: bool, reason: str)
            Reason can be: "ALLOWED", "FORBIDDEN" (over budget), "CLUSTERING_VIOLATION"
        """
        word_lower = word.lower()

        # Check if word is in restricted list
        if word_lower not in self.restricted_words:
            return True, "ALLOWED"

        # Check if over budget
        current_count = len(self._word_positions[word_lower])
        if current_count >= self.max_per_chapter:
            return False, "FORBIDDEN"

        # Check clustering if position provided
        if current_position is not None:
            positions = self._word_positions[word_lower]
            for prev_position in positions:
                distance = abs(current_position - prev_position)
                if distance < self.clustering_distance:
                    return False, "CLUSTERING_VIOLATION"

        return True, "ALLOWED"

    def record_word(self, word: str, position: int, text: str = "") -> None:
        """Record word usage with position.

        Args:
            word: Word that was used (case-insensitive)
            position: Word position in document (0-based word index)
            text: Optional text context (for debugging)
        """
        word_lower = word.lower()
        if word_lower in self.restricted_words:
            self._word_positions[word_lower].append(position)

    def get_forbidden_words(self) -> List[str]:
        """Get words that are over budget (forbidden).

        Returns:
            List of words that have reached max_per_chapter limit
        """
        forbidden = []
        for word in self.restricted_words:
            if len(self._word_positions[word]) >= self.max_per_chapter:
                forbidden.append(word)
        return forbidden

    def get_warning_words(self) -> List[str]:
        """Get words that are close to budget (should be used sparingly).

        Returns:
            List of words that have used at least 1 but less than max_per_chapter
        """
        warning = []
        for word in self.restricted_words:
            count = len(self._word_positions[word])
            if 0 < count < self.max_per_chapter:
                warning.append(word)
        return warning

    def check_clustering(self, text: str, word: str) -> Tuple[bool, str]:
        """Check if adding word to text would violate clustering rule.

        Args:
            text: Text to check (will be analyzed for word positions)
            word: Word to check for clustering violations

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        word_lower = word.lower()
        if word_lower not in self.restricted_words:
            return True, "ALLOWED"

        # Find all occurrences of the word in the text
        words = text.lower().split()
        text_positions = []
        for i, w in enumerate(words):
            # Use word boundary matching
            if re.search(r'\b' + re.escape(word_lower) + r'\b', w):
                text_positions.append(self._total_words + i)

        # Check against existing positions
        existing_positions = self._word_positions[word_lower]
        for text_pos in text_positions:
            for existing_pos in existing_positions:
                distance = abs(text_pos - existing_pos)
                if distance < self.clustering_distance:
                    return False, f"CLUSTERING_VIOLATION: '{word}' appears within {distance} words of previous occurrence"

        return True, "ALLOWED"

    def reset(self) -> None:
        """Reset tracking for new document."""
        self._word_positions.clear()
        self._total_words = 0

    def get_usage_stats(self) -> Dict[str, int]:
        """Get current usage counts for all restricted words.

        Returns:
            Dictionary mapping word -> usage count
        """
        return {
            word: len(self._word_positions[word])
            for word in self.restricted_words
        }

    def update_total_words(self, additional_words: int) -> None:
        """Update total word count in document.

        Args:
            additional_words: Number of words to add to total count
        """
        self._total_words += additional_words

    def find_restricted_words(self, text: str) -> List[Tuple[str, int]]:
        """Find all restricted words in text with their positions.

        Args:
            text: Text to search

        Returns:
            List of (word, position) tuples where position is word index in text
        """
        found = []
        words = text.split()

        for i, word_token in enumerate(words):
            # Normalize word token (remove punctuation, lowercase)
            word_clean = re.sub(r'[^\w]', '', word_token).lower()

            # Check against restricted words with word boundary matching
            for restricted_word in self.restricted_words:
                if re.search(r'\b' + re.escape(restricted_word) + r'\b', word_clean):
                    found.append((restricted_word, i))
                    break  # Only count once per word token

        return found

    def validate_vocabulary(self, text: str) -> List[str]:
        """Validate text for restricted vocabulary violations.

        When max_per_chapter is 0, performs a simple presence check (stateless).
        Otherwise, uses budget logic to check if words are allowed.

        Args:
            text: Text to validate

        Returns:
            List of violating words (empty if no violations)
        """
        violations = []

        # If max_per_chapter is 0, use simple presence check (stateless)
        if self.max_per_chapter == 0:
            text_lower = text.lower()
            for word in self.restricted_words:
                # Use word boundary matching to avoid partial matches
                if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                    violations.append(word)
        else:
            # Use budget logic: check each found word
            found_words = self.find_restricted_words(text)
            for word, position in found_words:
                # Calculate absolute position in document
                absolute_position = self._total_words + position
                # Check if allowed
                allowed, reason = self.check_word_allowed(word, absolute_position)
                if not allowed:
                    violations.append(word)

        return violations


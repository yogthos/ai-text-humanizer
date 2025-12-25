"""Global style state tracker to prevent repetitive phrasing and enforce variety."""

import collections
import re
from typing import List, Set, Deque, Dict, Any, Optional

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class GlobalStyleTracker:
    """
    Tracks the usage of stylistic elements (phrases, connectors, structures)
    across a document to prevent repetition and enforce variety.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the style tracker with configuration.

        Args:
            config: Full config dictionary (will extract 'style_state' section)
        """
        self.config = config.get('style_state', {}) if config else {}

        # Configuration
        self.history_window = self.config.get('history_window', 5)
        self.connector_threshold = self.config.get('connector_filter_threshold', 1)
        self.opener_method = self.config.get('opener_extraction_method', 'first_clause')
        self.enabled = self.config.get('enabled', True)

        # State (Sliding Windows)
        self.phrase_history: Deque[str] = collections.deque(maxlen=self.history_window)
        self.connector_history: Deque[str] = collections.deque(maxlen=self.history_window * 3)  # Track more connectors
        self.structure_history: Deque[str] = collections.deque(maxlen=self.history_window)

        # Load NLP for connector extraction (lightweight usage)
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            except (OSError, IOError):
                # spaCy model not available, will use fallback
                self.nlp = None

    def reset(self):
        """Reset history state (call at start of new document)."""
        self.phrase_history.clear()
        self.connector_history.clear()
        self.structure_history.clear()

    def register_usage(
        self,
        text: str,
        connectors: Optional[List[str]] = None,
        structure: Optional[str] = None,
        opener: Optional[str] = None
    ):
        """
        Register a generated sentence's style features into the history.

        Args:
            text: Generated text
            connectors: List of connectors used (if None, will extract from text)
            structure: Logical signature (CONTRAST, DEFINITION, etc.)
            opener: Opening phrase (if None, will extract from text)
        """
        if not self.enabled:
            return

        # 1. Register Opener
        if not opener and text:
            opener = self._extract_opener(text)

        if opener:
            self.phrase_history.append(opener.lower().strip())

        # 2. Register Structure
        if structure:
            self.structure_history.append(structure)

        # 3. Register Connectors
        # If connectors aren't provided explicitly, try to extract them
        if connectors is None and text:
            connectors = self._extract_connectors_from_text(text)

        if connectors:
            for c in connectors:
                self.connector_history.append(c.lower().strip())

    def filter_available_connectors(self, candidates: List[str], signature: str) -> List[str]:
        """
        Filter a list of candidate connectors based on recent usage history.

        Args:
            candidates: List of candidate connector strings
            signature: Logical signature (for context, not currently used)

        Returns:
            List of allowable connectors (filtered by usage)
        """
        if not self.enabled or not candidates:
            return candidates

        available = []
        for c in candidates:
            clean_c = c.lower().strip()
            # Count occurrences in history
            count = self.connector_history.count(clean_c)

            # If used less than threshold, it's available
            if count < self.connector_threshold:
                available.append(c)

        # If we filtered everything (too strict), return original list to avoid breaking generation
        # Better to repeat than to have no connectors
        return available if available else candidates

    def get_forbidden_openers(self) -> List[str]:
        """Return list of recently used sentence openers.

        Returns:
            Unique list of recently used opener phrases
        """
        if not self.enabled:
            return []
        return list(set(self.phrase_history))

    def _extract_opener(self, text: str) -> str:
        """
        Extract the opening phrase of a sentence.
        Method: Take the first 4 words to capture structural patterns.

        Args:
            text: Sentence text

        Returns:
            Normalized opener phrase (first 4 words, lowercase, stripped)
        """
        if not text:
            return ""

        # Take first 4 words to capture structural patterns like "It is not a"
        # This prevents repetition of patterns like "It is not..." appearing twice
        words = text.split()

        # Take first 4 words
        opener_words = words[:4] if len(words) >= 4 else words

        # Join and normalize: lowercase, strip punctuation
        opener = " ".join(opener_words)
        # Remove all punctuation, keep only alphanumeric and spaces
        opener = "".join(c for c in opener if c.isalnum() or c.isspace())
        opener = opener.lower().strip()

        # Remove trailing punctuation
        opener = re.sub(r'[.,;:!?]+$', '', opener)

        return opener

    def _extract_connectors_from_text(self, text: str) -> List[str]:
        """
        Heuristic extraction of connectors from text.
        Checks against a common list of logical transition words.

        Args:
            text: Generated text to extract connectors from

        Returns:
            List of found connector words/phrases
        """
        # Common connector set organized by logical type
        common_connectors = {
            # Contrast
            "but", "however", "although", "yet", "nevertheless", "whereas", "on the contrary",
            # Causality
            "because", "since", "therefore", "thus", "hence", "consequently", "as a result",
            # Conditional
            "if", "unless", "provided", "when", "then",
            # Sequence
            "first", "second", "then", "subsequently", "finally", "next", "after",
            # Elaboration
            "specifically", "namely", "including", "for example", "in other words"
        }

        found = []
        text_lower = text.lower()

        # Basic check (can be improved with spaCy dependency parsing if needed)
        for conn in common_connectors:
            # Check for standalone word usage using word boundaries
            pattern = r'\b' + re.escape(conn) + r'\b'
            if re.search(pattern, text_lower):
                found.append(conn)

        return found


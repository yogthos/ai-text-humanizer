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
        # Key phrases (noun phrases, proper nouns) to prevent repetition
        self.key_phrases_history: Deque[str] = collections.deque(maxlen=self.history_window * 2)  # Track more phrases
        # Syntactic fingerprints (ROOT + deps) to prevent structural repetition
        self.syntactic_history: Deque[str] = collections.deque(maxlen=self.history_window)
        
        # Subject tracking for pronoun resolution
        self.subject_history: Deque[str] = collections.deque(maxlen=self.history_window)
        self.entity_registry: Dict[str, int] = {}  # Entity -> Last seen index (global counter)
        self.global_sentence_count = 0

        # Load NLP for connector extraction and phrase extraction (lightweight usage)
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                # Enable NER for proper noun detection, but disable parser for speed
                self.nlp = spacy.load("en_core_web_sm", disable=["parser"])
            except (OSError, IOError):
                # spaCy model not available, will use fallback
                self.nlp = None

    def reset(self):
        """Reset history state (call at start of new document)."""
        self.phrase_history.clear()
        self.connector_history.clear()
        self.structure_history.clear()
        self.key_phrases_history.clear()
        self.syntactic_history.clear()
        self.subject_history.clear()
        self.entity_registry.clear()
        self.global_sentence_count = 0

    def register_usage(
        self,
        text: str,
        connectors: Optional[List[str]] = None,
        structure: Optional[str] = None,
        opener: Optional[str] = None,
        key_phrases: Optional[List[str]] = None
    ):
        """
        Register a generated sentence's style features into the history.

        Args:
            text: Generated text
            connectors: List of connectors used (if None, will extract from text)
            structure: Logical signature (CONTRAST, DEFINITION, etc.)
            opener: Opening phrase (if None, will extract from text)
            key_phrases: List of key phrases to register (if None, will extract from text)
        """
        if not self.enabled:
            return
            
        self.global_sentence_count += 1

        # 1. Register Opener
        if not opener and text:
            opener = self._extract_opener(text)

        if opener:
            self.phrase_history.append(opener.lower().strip())

        # 2. Register Structure
        if structure:
            self.structure_history.append(structure)

        # 3. Register Syntactic Fingerprint (NEW)
        if text:
            fingerprint = self._extract_structural_fingerprint(text)
            if fingerprint:
                self.syntactic_history.append(fingerprint)

        # 4. Register Connectors
        # If connectors aren't provided explicitly, try to extract them
        if connectors is None and text:
            connectors = self._extract_connectors_from_text(text)

        if connectors:
            for c in connectors:
                self.connector_history.append(c.lower().strip())

        # 4. Register Key Phrases (noun phrases, proper nouns)
        if key_phrases is None and text:
            try:
                key_phrases = self._extract_key_phrases(text)
            except Exception:
                # If extraction fails, continue without key phrases
                key_phrases = []

        if key_phrases:
            for phrase in key_phrases:
                if phrase:  # Skip None or empty phrases
                    try:
                        normalized = self._normalize_phrase(phrase)
                        if normalized:  # Only add non-empty normalized phrases
                            self.key_phrases_history.append(normalized)
                    except Exception:
                        # Skip phrases that fail normalization
                        continue
        
        # 5. Register Subject (NEW)
        if text:
            self._extract_and_register_subject(text)

    def _extract_and_register_subject(self, text: str):
        """Extract main subject from text and register it."""
        if not self.nlp or not text:
            return
            
        try:
            doc = self.nlp(text)
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    # Get the full noun chunk for the subject
                    subj_text = token.text.lower()
                    # Try to get the chunk
                    for chunk in doc.noun_chunks:
                        if chunk.root == token:
                            subj_text = chunk.text.lower()
                            break
                    
                    self.subject_history.append(subj_text)
                    self.entity_registry[subj_text] = self.global_sentence_count
                    return # Only register the first main subject
        except Exception:
            pass

    def get_last_subject(self) -> Optional[str]:
        """Return the subject of the most recently registered sentence."""
        if self.subject_history:
            return self.subject_history[-1]
        return None

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

    def get_forbidden_phrases(self) -> List[str]:
        """Return list of recently used key phrases (normalized).

        Returns:
            Unique list of recently used key phrases (normalized for variation matching)
        """
        if not self.enabled:
            return []
        return list(set(self.key_phrases_history))

    def get_forbidden_structures(self) -> List[str]:
        """Return list of recently used syntactic fingerprints.
        
        Returns:
            Unique list of recent structural fingerprints (e.g. "VERB:is+nsubj+attr")
        """
        if not self.enabled:
            return []
        return list(set(self.syntactic_history))

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

    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases (noun phrases, proper nouns) from text using spaCy.
        Handles variations by normalizing phrases.

        Args:
            text: Text to extract phrases from

        Returns:
            List of key phrases (normalized)
        """
        if text is None or not text or not self.nlp:
            # Fallback: extract capitalized phrases and common noun patterns
            return self._extract_key_phrases_fallback(text or "")

        try:
            doc = self.nlp(text)
            phrases = []

            # 1. Extract proper nouns (PROPN) - these are often important concepts
            proper_nouns = []
            for token in doc:
                if token.pos_ == "PROPN":
                    proper_nouns.append(token.text)
                elif token.ent_type_ and token.ent_type_ != "":  # Named entities
                    proper_nouns.append(token.text)

            # 2. Extract noun phrases (compound nouns, noun chunks)
            noun_chunks = []
            for chunk in doc.noun_chunks:
                # Filter out very short chunks (1-2 words) unless they're proper nouns
                if len(chunk) >= 2 or any(t.pos_ == "PROPN" for t in chunk):
                    # Get the root noun and its modifiers
                    chunk_text = chunk.text.strip()
                    if len(chunk_text) > 2:  # Ignore very short chunks
                        noun_chunks.append(chunk_text)

            # 3. Extract important multi-word phrases (proper noun sequences)
            # Look for sequences of proper nouns or capitalized words
            proper_noun_phrases = []
            i = 0
            while i < len(doc):
                if doc[i].pos_ == "PROPN" or (doc[i].is_title and doc[i].pos_ in ["NOUN", "PROPN"]):
                    phrase_words = [doc[i].text]
                    i += 1
                    # Collect consecutive proper nouns/title words
                    while i < len(doc) and (doc[i].pos_ == "PROPN" or
                                          (doc[i].is_title and doc[i].pos_ in ["NOUN", "PROPN"]) or
                                          (doc[i].pos_ == "DET" and i < len(doc) - 1)):
                        if doc[i].pos_ == "DET":
                            phrase_words.append(doc[i].text)
                            i += 1
                            if i < len(doc) and (doc[i].pos_ == "PROPN" or doc[i].is_title):
                                phrase_words.append(doc[i].text)
                                i += 1
                            break
                        else:
                            phrase_words.append(doc[i].text)
                            i += 1
                    if len(phrase_words) >= 2:  # Multi-word proper noun phrases
                        proper_noun_phrases.append(" ".join(phrase_words))
                    continue
                i += 1

            # Combine all phrases
            all_phrases = proper_noun_phrases + noun_chunks
            # Add individual important proper nouns if they're significant
            for pn in proper_nouns:
                if len(pn) > 3:  # Only significant proper nouns
                    all_phrases.append(pn)

            # Normalize and deduplicate
            normalized_phrases = []
            seen = set()
            for phrase in all_phrases:
                normalized = self._normalize_phrase(phrase)
                if normalized and normalized not in seen:
                    normalized_phrases.append(normalized)
                    seen.add(normalized)

            return normalized_phrases

        except Exception as e:
            # Fallback if spaCy processing fails
            return self._extract_key_phrases_fallback(text)

    def _extract_key_phrases_fallback(self, text: str) -> List[str]:
        """
        Fallback method to extract key phrases without spaCy.
        Uses regex to find capitalized phrases and proper noun patterns.

        Args:
            text: Text to extract phrases from

        Returns:
            List of key phrases (normalized)
        """
        if not text:
            return []

        phrases = []

        # Pattern 1: Capitalized multi-word phrases (likely proper nouns or important concepts)
        # Match sequences of capitalized words
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match.split()) >= 2:  # Multi-word phrases
                phrases.append(match)

        # Pattern 2: Quoted phrases (often important concepts)
        quoted_pattern = r'"([^"]+)"'
        quoted = re.findall(quoted_pattern, text)
        phrases.extend(quoted)

        # Normalize and deduplicate
        normalized_phrases = []
        seen = set()
        for phrase in phrases:
            normalized = self._normalize_phrase(phrase)
            if normalized and normalized not in seen:
                normalized_phrases.append(normalized)
                seen.add(normalized)

        return normalized_phrases

    def _extract_structural_fingerprint(self, text: str) -> Optional[str]:
        """Extract a simplified structural fingerprint using spaCy.
        
        Format: "ROOT_POS:ROOT_LEMMA+DEP1+DEP2..."
        Example: "The cat sat." -> "VERB:sit+nsubj+punct"
        
        Args:
            text: Sentence text
            
        Returns:
            Structural fingerprint string or None if spaCy unavailable/parsing fails
        """
        if not self.nlp or not text:
            return None
            
        try:
            doc = self.nlp(text)
            
            # Find root
            roots = [t for t in doc if t.dep_ == "ROOT"]
            if not roots:
                return None
            root = roots[0]
            
            # Get direct children dependency labels (sorted for consistency)
            deps = sorted([t.dep_ for t in root.children])
            
            # Build fingerprint: ROOT_POS:ROOT_LEMMA + child dependencies
            # We include the root lemma to catch repetitive verbs (e.g. "is", "has")
            fingerprint = f"{root.pos_}:{root.lemma_}+" + "+".join(deps)
            
            return fingerprint
        except Exception:
            return None

    def _normalize_phrase(self, phrase: str) -> str:
        """
        Normalize a phrase to handle variations (case, punctuation, articles).

        Args:
            phrase: Phrase to normalize

        Returns:
            Normalized phrase (lowercase, punctuation removed, articles handled)
        """
        if not phrase:
            return ""

        # Convert to lowercase
        normalized = phrase.lower().strip()

        # Remove leading articles (a, an, the) for better matching
        # e.g., "the Dialectical Materialism" -> "dialectical materialism"
        normalized = re.sub(r'^(a|an|the)\s+', '', normalized)

        # Remove punctuation but keep spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)

        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # Filter out very short phrases (less than 3 characters)
        if len(normalized) < 3:
            return ""

        return normalized


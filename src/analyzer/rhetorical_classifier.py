"""Rhetorical Mode Classifier for Template Selection.

This module classifies text into rhetorical modes (NARRATIVE, ARGUMENTATIVE, DESCRIPTIVE)
to enable mode-compatible template selection during paragraph fusion.
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Optional
from src.generator.llm_interface import LLMProvider


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'rhetorical_classifier_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


class RhetoricalClassifier:
    """Classifies text into rhetorical modes using heuristics and LLM fallback."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the rhetorical classifier.

        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.llm_provider = LLMProvider(config_path=config_path)
        self.cache_file = Path(config_path).parent / "rhetorical_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load classification cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self):
        """Save classification cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except IOError:
            pass  # Silently fail if cache can't be written

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _classify_heuristic(self, text: str) -> Optional[str]:
        """Fast heuristic classification (no LLM call).

        Args:
            text: Text to classify.

        Returns:
            Mode string ("NARRATIVE", "ARGUMENTATIVE", "DESCRIPTIVE") or None if ambiguous.
        """
        if not text or not text.strip():
            return "DESCRIPTIVE"  # Default for empty text

        text_lower = text.lower()
        text_stripped = text.strip()

        # ARGUMENTATIVE: Starts with logical connectors or conditionals
        argumentative_starters = [
            "if ", "unless ", "therefore", "thus", "consequently", "because", "since",
            "hence", "accordingly", "as a result", "for this reason", "so ", "it follows"
        ]
        first_words = text_stripped[:50].lower()  # Check first 50 chars
        if any(starter in first_words for starter in argumentative_starters):
            return "ARGUMENTATIVE"

        # ARGUMENTATIVE: Contains logical connectors in body
        argumentative_connectors = [
            "therefore", "thus", "hence", "because", "since", "consequently",
            "accordingly", "as a result", "for this reason", "it follows", "unless"
        ]
        if any(connector in text_lower for connector in argumentative_connectors):
            return "ARGUMENTATIVE"

        # NARRATIVE: Contains time markers + past tense verbs
        time_markers = ["then", "later", "after", "when", "as", "while", "during", "before"]
        has_time_marker = any(marker in text_lower for marker in time_markers)

        # Check for past tense patterns: "I went", "We did", "He said", "They were"
        past_tense_patterns = [
            r'\bi\s+(went|did|said|was|were|saw|felt|thought|knew|came|left|took|gave|made|found|got|had)',
            r'\bwe\s+(went|did|said|were|saw|felt|thought|knew|came|left|took|gave|made|found|got|had)',
            r'\bhe\s+(went|did|said|was|saw|felt|thought|knew|came|left|took|gave|made|found|got|had)',
            r'\bshe\s+(went|did|said|was|saw|felt|thought|knew|came|left|took|gave|made|found|got|had)',
            r'\bthey\s+(went|did|said|were|saw|felt|thought|knew|came|left|took|gave|made|found|got|had)'
        ]
        has_past_tense = any(re.search(pattern, text_lower) for pattern in past_tense_patterns)

        if has_time_marker and has_past_tense:
            return "NARRATIVE"

        # NARRATIVE: First person + action verbs
        if re.search(r'\bi\s+\w+ed\b', text_lower) or re.search(r'\bi\s+\w+ing\b', text_lower):
            if has_time_marker:
                return "NARRATIVE"

        # DESCRIPTIVE: Contains static description patterns
        descriptive_patterns = [
            "is ", "was ", "consists of", "has ", "contains", "includes",
            "is a", "is an", "is the", "was a", "was an", "was the"
        ]
        if any(pattern in text_lower for pattern in descriptive_patterns):
            # But not if it's clearly argumentative
            if not any(connector in text_lower for connector in argumentative_connectors):
                return "DESCRIPTIVE"

        # Ambiguous - return None to trigger LLM fallback
        return None

    def classify_mode(self, text: str) -> str:
        """Classify text into rhetorical mode.

        Uses fast heuristics first, then LLM fallback if ambiguous.
        Results are cached to avoid redundant LLM calls.

        Args:
            text: Text to classify.

        Returns:
            Mode string: "NARRATIVE", "ARGUMENTATIVE", or "DESCRIPTIVE".
        """
        if not text or not text.strip():
            return "DESCRIPTIVE"

        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Try heuristic classification first (fast, no LLM)
        heuristic_result = self._classify_heuristic(text)
        if heuristic_result:
            # Cache and return heuristic result
            self.cache[cache_key] = heuristic_result
            self._save_cache()
            return heuristic_result

        # Heuristic was ambiguous - use LLM fallback
        system_prompt = _load_prompt_template("rhetorical_classifier_system.md")
        user_prompt = f"Classify this text:\n\n{text}"

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=False,
                temperature=0.2,  # Low temperature for consistent classification
                max_tokens=10,
                timeout=15
            )

            # Extract mode from response
            response_upper = response.strip().upper()
            for mode in ["NARRATIVE", "ARGUMENTATIVE", "DESCRIPTIVE"]:
                if mode in response_upper:
                    # Cache and return
                    self.cache[cache_key] = mode
                    self._save_cache()
                    return mode

            # Fallback to DESCRIPTIVE if LLM response is unclear
            mode = "DESCRIPTIVE"
            self.cache[cache_key] = mode
            self._save_cache()
            return mode

        except Exception as e:
            # On error, fallback to heuristic or DESCRIPTIVE
            fallback = self._classify_heuristic(text) or "DESCRIPTIVE"
            self.cache[cache_key] = fallback
            self._save_cache()
            return fallback


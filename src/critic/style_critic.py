"""Style Critic: Validates generated text against style profile statistics.

This module checks that generated text matches the author's statistical
profile (sentence length, punctuation usage) to prevent style "caricatures"
(e.g., overuse of dashes, overly long sentences).
"""

from typing import Dict, Any, Tuple, List, Optional
from src.utils.text_analyzer import TextAnalyzer


class StyleCritic:
    """Validates text against style profile statistics."""

    def __init__(self, style_profile: Optional[Dict[str, Any]] = None, tolerance: float = 0.3):
        """Initialize the style critic.

        Args:
            style_profile: Style profile dictionary containing target metrics.
                Expected keys:
                - structural_dna.avg_words_per_sentence
                - dashes_per_100
                - semicolons_per_100
            tolerance: Allowed deviation from target (default 0.3 = 30%)
        """
        self.style_profile = style_profile or {}
        self.tolerance = tolerance

    def evaluate(self, text: str) -> Tuple[bool, str, List[str]]:
        """Evaluate text against style profile.

        Args:
            text: Text to evaluate

        Returns:
            Tuple of:
            - passed: True if text passes all style checks
            - feedback: Human-readable feedback string
            - violations: List of specific violation messages
        """
        if not self.style_profile:
            # No profile to check against - always pass
            return True, "No style profile provided", []

        if not text or not text.strip():
            return False, "Empty text", ["Text is empty"]

        # Quick scan to get current text statistics
        stats = TextAnalyzer.quick_scan(text)

        violations = []

        # 1. Check Sentence Length
        structural_dna = self.style_profile.get('structural_dna', {})
        target_avg_len = structural_dna.get('avg_words_per_sentence', 25.0)
        current_avg_len = stats.get('avg_words_per_sentence', 0.0)

        if current_avg_len > 0:
            # Fail if average length exceeds target by more than tolerance
            max_allowed = target_avg_len * (1 + self.tolerance)
            if current_avg_len > max_allowed:
                violations.append(
                    f"Sentences too long (Avg: {current_avg_len:.1f} words, "
                    f"Target: {target_avg_len:.1f}, Max allowed: {max_allowed:.1f})"
                )

        # 2. Check Dash Density
        target_dashes = self.style_profile.get('dashes_per_100', 0.0)
        current_dashes = stats.get('dashes_per_100', 0.0)

        # For dashes, we use a stricter upper bound to prevent overuse
        # Allow up to 2x target, but also enforce absolute maximum if target is low
        max_dash_threshold = max(target_dashes * 2.0, 2.0) if target_dashes > 0 else 2.0

        if current_dashes > max_dash_threshold:
            violations.append(
                f"Too many em-dashes ({current_dashes:.1f} per 100 words, "
                f"Target: {target_dashes:.1f}, Max allowed: {max_dash_threshold:.1f})"
            )

        # 3. Check Semicolon Density (optional, less critical)
        target_semicolons = self.style_profile.get('semicolons_per_100', 0.0)
        current_semicolons = stats.get('semicolons_per_100', 0.0)

        # Similar strict upper bound for semicolons
        max_semicolon_threshold = max(target_semicolons * 2.0, 2.0) if target_semicolons > 0 else 2.0

        if current_semicolons > max_semicolon_threshold:
            violations.append(
                f"Too many semicolons ({current_semicolons:.1f} per 100 words, "
                f"Target: {target_semicolons:.1f}, Max allowed: {max_semicolon_threshold:.1f})"
            )

        # Build feedback string
        if violations:
            feedback = f"Style violations detected: {'; '.join(violations)}"
            return False, feedback, violations
        else:
            feedback = (
                f"Style compliant (Length: {current_avg_len:.1f} words avg, "
                f"Dashes: {current_dashes:.1f}/100, Semicolons: {current_semicolons:.1f}/100)"
            )
            return True, feedback, []

    def check_compliance(self, text: str) -> Tuple[bool, List[str]]:
        """Check compliance (backward compatibility alias).

        Args:
            text: Text to check

        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        passed, _, violations = self.evaluate(text)
        return passed, violations


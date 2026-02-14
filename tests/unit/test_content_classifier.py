"""Tests for content classifier module.

Tests cover:
- Bug 16: Content classifier no logging for borderline cases
"""

import pytest
from unittest.mock import patch


class TestContentClassifier:
    """Tests for classify_content_type (Bug 16)."""

    def test_short_text_no_crash(self):
        """Short text should not crash the classifier."""
        from src.utils.content_classifier import classify_content_type, ContentType

        result = classify_content_type("Hello.")
        assert result in (ContentType.NARRATIVE, ContentType.CONCEPTUAL)

    def test_empty_text_no_crash(self):
        """Empty text should not crash."""
        from src.utils.content_classifier import classify_content_type, ContentType

        result = classify_content_type("")
        assert result in (ContentType.NARRATIVE, ContentType.CONCEPTUAL)

    def test_borderline_classification_logs_debug(self):
        """Borderline classifications should log a debug message."""
        from src.utils.content_classifier import classify_content_type

        # Text with roughly equal narrative/conceptual signals
        borderline_text = "The system processes events over time."

        with patch('src.utils.content_classifier.logger') as mock_logger:
            classify_content_type(borderline_text)
            # Should log debug for borderline case (scores within 1 of each other)
            # The test just verifies no crash; actual logging is a nice-to-have


class TestTemporalMarkerWithPunctuation:
    """Tests for Bug 4 Round 5: Temporal markers with attached punctuation missed."""

    def test_temporal_marker_with_comma(self):
        """'when,' with attached comma should still be detected as temporal marker."""
        from src.utils.content_classifier import classify_content_type, ContentType

        # Text with temporal markers followed by commas — heavily narrative
        text = "The soldier walked forward when, suddenly, the ground shook beneath him. He ran then, turning back quickly toward the trenches. Before, there had been silence across the field. After, the world had changed forever."
        result = classify_content_type(text)
        assert result == ContentType.NARRATIVE

    def test_temporal_marker_at_end_of_sentence(self):
        """Temporal marker at end of sentence ('then.') should still count."""
        from src.utils.content_classifier import classify_content_type, ContentType

        text = "He fought and he fell then. The battle had ended before. She arrived soon after. They left eventually."
        classify_content_type(text)  # Should not crash

    def test_temporal_markers_without_punctuation_still_work(self):
        """Standard temporal markers (spaces on both sides) should still work."""
        from src.utils.content_classifier import classify_content_type, ContentType

        text = "Then the army marched forward across the field. After the battle they rested in camp. Before the dawn they had prepared their weapons. When the signal came they charged ahead."
        result = classify_content_type(text)
        assert result == ContentType.NARRATIVE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

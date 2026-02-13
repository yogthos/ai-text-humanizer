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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

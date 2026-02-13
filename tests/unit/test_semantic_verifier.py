"""Tests for semantic verifier module.

Tests cover:
- Bug 6: Singleton ignores thresholds
"""

import pytest
from unittest.mock import patch


class TestSemanticVerifierSingleton:
    """Tests for get_semantic_verifier singleton (Bug 6)."""

    def setup_method(self):
        """Reset singleton before each test."""
        import src.validation.semantic_verifier as sv
        sv._verifier = None

    def test_singleton_with_custom_threshold(self):
        """Singleton should accept and store custom kwargs."""
        from src.validation.semantic_verifier import get_semantic_verifier

        verifier = get_semantic_verifier(entailment_threshold=0.8)
        assert verifier.entailment_threshold == 0.8

    def test_singleton_ignores_subsequent_kwargs(self):
        """Second call should return existing instance, ignoring new kwargs."""
        from src.validation.semantic_verifier import get_semantic_verifier

        v1 = get_semantic_verifier(entailment_threshold=0.8)
        v2 = get_semantic_verifier(entailment_threshold=0.5)
        assert v1 is v2
        assert v2.entailment_threshold == 0.8  # First value wins

    def test_singleton_returns_same_instance(self):
        """Multiple calls should return the same instance."""
        from src.validation.semantic_verifier import get_semantic_verifier

        v1 = get_semantic_verifier()
        v2 = get_semantic_verifier()
        assert v1 is v2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

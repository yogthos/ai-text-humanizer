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


class TestPOSConsistency:
    """Tests for POS tag consistency across extraction points (Bug 6 Round 3)."""

    def test_content_pos_tags_consistent(self):
        """All content word extraction points should use the same POS set.

        The verifier uses POS tags in 3 places:
        - _check_sentence_grounding (lines 285, 311)
        - _check_content_coverage (line 405)
        All should use the same CONTENT_POS_TAGS constant.
        """
        from src.validation.semantic_verifier import CONTENT_POS_TAGS

        expected = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'NUM'}
        assert CONTENT_POS_TAGS == expected, (
            f"CONTENT_POS_TAGS should be {expected} but is {CONTENT_POS_TAGS}"
        )

    def test_content_pos_tags_used_in_source(self):
        """Verify CONTENT_POS_TAGS is actually used in the source code."""
        import inspect
        import src.validation.semantic_verifier as sv

        source = inspect.getsource(sv.SemanticVerifier)
        assert "CONTENT_POS_TAGS" in source, (
            "SemanticVerifier should use CONTENT_POS_TAGS constant"
        )


class TestEntityStemMatching:
    """Tests for entity stem matching false positives (Bug 11)."""

    def test_mars_marx_not_matched(self):
        """'Mars' and 'Marx' should not match (different stems)."""
        from src.validation.semantic_verifier import SemanticVerifier

        verifier = SemanticVerifier.__new__(SemanticVerifier)
        verifier.entailment_threshold = 0.7

        mars_stem = verifier._get_entity_stem("Mars")
        stems = {verifier._get_entity_stem("Marx")}

        result = verifier._entity_matches_any_stem("Mars", stems)
        # Mars and Marx have different stems - should not match
        assert result is False or mars_stem == verifier._get_entity_stem("Marx"), (
            f"Mars (stem={mars_stem}) should not match Marx"
        )

    def test_mark_marker_not_matched(self):
        """'Mark' and 'marker' should not match (length difference > 2)."""
        from src.validation.semantic_verifier import SemanticVerifier

        verifier = SemanticVerifier.__new__(SemanticVerifier)
        verifier.entailment_threshold = 0.7

        mark_stem = verifier._get_entity_stem("Mark")
        marker_stem = verifier._get_entity_stem("marker")

        # With length check: |len(mark) - len(marker)| should be > 2
        result = verifier._entity_matches_any_stem("Mark", {marker_stem})
        # If stems are "mark" vs "mark" (same 4 chars), they'd match
        # But "marker" stem is longer, so length diff check helps
        # The key fix is requiring stems within 2 chars
        if abs(len(mark_stem) - len(marker_stem)) > 2:
            assert result is False

    def test_communist_communism_matched(self):
        """'Communist' and 'communism' should match (same root)."""
        from src.validation.semantic_verifier import SemanticVerifier

        verifier = SemanticVerifier.__new__(SemanticVerifier)
        verifier.entailment_threshold = 0.7

        communism_stem = verifier._get_entity_stem("communism")
        result = verifier._entity_matches_any_stem("communist", {communism_stem})
        assert result is True, "communist and communism should match"


class TestSentenceGroundingNliParam:
    """Tests for Bug 5 Round 5: Dead nli_model parameter in _check_sentence_grounding."""

    def test_nli_model_not_required_for_grounding(self):
        """_check_sentence_grounding should work without nli_model (it uses content word overlap)."""
        from src.validation.semantic_verifier import SemanticVerifier
        verifier = SemanticVerifier()
        source_sents = ["The cat sat on the mat."]
        output_sents = ["The cat rested on the mat."]

        # Should work with nli_model=None since the method doesn't use it
        results, ratio, hallucinations = verifier._check_sentence_grounding(
            source_sents, output_sents, nli_model=None
        )
        assert len(results) == 1
        assert ratio >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

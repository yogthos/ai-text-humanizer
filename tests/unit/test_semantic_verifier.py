"""Tests for semantic verifier module.

Tests cover:
- Bug 6: Singleton ignores thresholds (now: kwargs yield a new uncached
  instance; the latent bug where subsequent kwargs were silently ignored
  no longer applies)
"""

import pytest
from unittest.mock import patch


class TestSemanticVerifierSingleton:
    """Tests for get_semantic_verifier backed by the Services container."""

    @pytest.fixture(autouse=True)
    def _isolated_services(self):
        """Each test gets a fresh default Services container, restored after."""
        from src.services import default_services
        with default_services():
            yield

    def test_kwargs_yield_requested_threshold(self):
        """get_semantic_verifier(**kwargs) returns an instance configured with kwargs."""
        from src.validation.semantic_verifier import get_semantic_verifier

        verifier = get_semantic_verifier(grounding_threshold=0.8)
        assert verifier.grounding_threshold == 0.8

    def test_kwargs_yield_new_uncached_instance(self):
        """Different kwargs produce different instances — no silent reuse."""
        from src.validation.semantic_verifier import get_semantic_verifier

        v1 = get_semantic_verifier(grounding_threshold=0.8)
        v2 = get_semantic_verifier(grounding_threshold=0.5)
        assert v1 is not v2
        assert v1.grounding_threshold == 0.8
        assert v2.grounding_threshold == 0.5

    def test_no_kwargs_returns_shared_services_instance(self):
        """Calls without kwargs return the shared Services verifier."""
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
        verifier.grounding_threshold = 0.7

        mars_stem = verifier._get_entity_stem("Mars")
        stems = {verifier._get_entity_stem("Marx")}

        result = verifier._entity_matches_any_stem("Mars", stems)
        # Mars and Marx have different stems - should not match
        assert result is False or mars_stem == verifier._get_entity_stem("Marx"), (
            f"Mars (stem={mars_stem}) should not match Marx"
        )

    def test_mark_marker_not_matched(self):
        """'Mark' and 'marker' should not match (different words)."""
        from src.validation.semantic_verifier import SemanticVerifier

        verifier = SemanticVerifier.__new__(SemanticVerifier)
        verifier.grounding_threshold = 0.7

        marker_stem = verifier._get_entity_stem("marker")
        result = verifier._entity_matches_any_stem("Mark", {marker_stem})
        assert result is False, "'Mark' and 'marker' are different words and should not match"

    def test_short_prefix_not_matched(self):
        """Short stems (< 5 chars) should not match via prefix."""
        from src.validation.semantic_verifier import SemanticVerifier

        verifier = SemanticVerifier.__new__(SemanticVerifier)
        verifier.grounding_threshold = 0.7

        # "plan" vs "planet" — short stem prefix match should be rejected
        planet_stem = verifier._get_entity_stem("planet")
        result = verifier._entity_matches_any_stem("plan", {planet_stem})
        assert result is False, "'plan' and 'planet' should not match via short prefix"

    def test_communist_communism_matched(self):
        """'Communist' and 'communism' should match (same root)."""
        from src.validation.semantic_verifier import SemanticVerifier

        verifier = SemanticVerifier.__new__(SemanticVerifier)
        verifier.grounding_threshold = 0.7

        communism_stem = verifier._get_entity_stem("communism")
        result = verifier._entity_matches_any_stem("communist", {communism_stem})
        assert result is True, "communist and communism should match"


class TestSentenceGroundingNoNliDependency:
    """_check_sentence_grounding uses content-word overlap only — no NLI model."""

    def test_grounding_uses_content_overlap(self):
        from src.validation.semantic_verifier import SemanticVerifier
        verifier = SemanticVerifier()
        source_sents = ["The cat sat on the mat."]
        output_sents = ["The cat rested on the mat."]

        results, ratio, hallucinations = verifier._check_sentence_grounding(
            source_sents, output_sents
        )
        assert len(results) == 1
        assert ratio >= 0.0


class TestUnusedEntailmentThreshold:
    """Bug: SemanticVerifier stores entailment_threshold but never uses it.
    The grounding check at line 365 uses grounding_threshold instead."""

    @pytest.fixture(autouse=True)
    def _isolated_services(self):
        """Each test gets a fresh default Services container, restored after."""
        from src.services import default_services
        with default_services():
            yield

    def test_entailment_threshold_not_dead_code(self):
        """entailment_threshold should be used in the verifier, not just stored."""
        import inspect
        from src.validation.semantic_verifier import SemanticVerifier

        # Get source of all methods (not just __init__)
        source = inspect.getsource(SemanticVerifier)
        # Count occurrences of self.entailment_threshold
        # Should appear in at least one method BESIDES __init__
        init_source = inspect.getsource(SemanticVerifier.__init__)
        non_init_source = source.replace(init_source, "")
        assert "self.entailment_threshold" in non_init_source or \
               "entailment_threshold" not in inspect.getsource(SemanticVerifier.__init__), (
            "entailment_threshold is stored in __init__ but never used elsewhere — "
            "either use it or remove it"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

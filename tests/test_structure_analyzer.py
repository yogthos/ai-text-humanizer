"""Tests for Structure Analyzer functionality."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from unittest.mock import patch, MagicMock
from src.analysis.structure_analyzer import StructureAnalyzer


class TestStructureAnalyzer:
    """Tests for StructureAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create StructureAnalyzer instance for tests."""
        try:
            return StructureAnalyzer()
        except (ImportError, OSError):
            pytest.skip("spaCy model not available")

    def test_extract_skeleton_tokens_contrast(self, analyzer):
        """Test skeleton extraction from contrast sentence."""
        text = "It is not a dinner party, but an act of violence."
        skeleton = analyzer.extract_skeleton_tokens(text)

        # Should contain: "not", ",", "but", "."
        assert "not" in skeleton
        assert "," in skeleton or "but" in skeleton
        assert "." in skeleton

    def test_extract_skeleton_tokens_list(self, analyzer):
        """Test skeleton extraction from list sentence."""
        text = "The Red Army needs grain, the Red Army needs clothes, and the Red Army needs oil."
        skeleton = analyzer.extract_skeleton_tokens(text)

        # Should contain commas and "and"
        assert "," in skeleton
        assert "and" in skeleton
        assert "." in skeleton

    def test_extract_skeleton_tokens_question(self, analyzer):
        """Test skeleton extraction from question."""
        text = "What is knowledge? It is the reflection of the objective world."
        skeleton = analyzer.extract_skeleton_tokens(text)

        # Should contain "?" and "."
        assert "?" in skeleton
        assert "." in skeleton

    def test_extract_skeleton_tokens_empty(self, analyzer):
        """Test skeleton extraction from empty text."""
        skeleton = analyzer.extract_skeleton_tokens("")
        assert skeleton == []

    def test_calculate_fidelity_perfect_match(self, analyzer):
        """Test fidelity calculation with identical structures."""
        reference = "It is not X, but Y."
        draft = "It is not A, but B."

        score = analyzer.calculate_fidelity(draft, reference)
        assert score == 1.0  # Perfect structural match

    def test_calculate_fidelity_good_match(self, analyzer):
        """Test fidelity calculation with similar structures."""
        reference = "It is not a dinner party, but an act of violence."
        draft = "The phone is not static, but dynamic."

        score = analyzer.calculate_fidelity(draft, reference)
        # Should be high (>0.8) since both have "not", ",", "but", "."
        assert score > 0.8

    def test_calculate_fidelity_poor_match(self, analyzer):
        """Test fidelity calculation with different structures."""
        reference = "It is not X, but Y."
        draft = "The phone is dynamic."

        score = analyzer.calculate_fidelity(draft, reference)
        # Should be low since draft lacks "not", ",", "but"
        assert score < 0.5

    def test_calculate_fidelity_missing_punctuation(self, analyzer):
        """Test fidelity calculation when draft misses punctuation."""
        reference = "It is not X, but Y."
        draft = "It is not X but Y"  # Missing comma and period

        score = analyzer.calculate_fidelity(draft, reference)
        # Should be lower due to missing punctuation
        assert score < 1.0

    def test_calculate_fidelity_empty_reference(self, analyzer):
        """Test fidelity calculation with empty reference."""
        score = analyzer.calculate_fidelity("Some text", "")
        assert score == 1.0  # No structure to match

    def test_calculate_fidelity_empty_draft(self, analyzer):
        """Test fidelity calculation with empty draft."""
        score = analyzer.calculate_fidelity("", "Some reference")
        assert score == 0.0  # Empty draft has no structure

    def test_extract_dynamic_skeleton(self, analyzer):
        """Test dynamic skeleton extraction for prompting."""
        text = "It is not a dinner party, but an act of violence."
        skeleton = analyzer.extract_dynamic_skeleton(text)

        # Should contain structural markers
        assert "not" in skeleton.lower()
        assert "but" in skeleton.lower()
        # Should contain placeholders for content
        assert "..." in skeleton or len(skeleton) > 0

    def test_extract_dynamic_skeleton_empty(self, analyzer):
        """Test dynamic skeleton extraction from empty text."""
        skeleton = analyzer.extract_dynamic_skeleton("")
        assert skeleton == ""

    def test_calculate_fidelity_semicolon_vs_comma(self, analyzer):
        """Test that semicolon vs comma difference is detected."""
        reference = "It is not X, but Y."
        draft = "It is not X; but Y."

        score = analyzer.calculate_fidelity(draft, reference)
        # Should be less than perfect due to punctuation difference
        assert score < 1.0
        # But still reasonably high since structure is similar
        assert score > 0.6

    def test_calculate_fidelity_missing_conjunction(self, analyzer):
        """Test that missing conjunction is detected."""
        reference = "It is not X, but Y."
        draft = "It is not X, Y."  # Missing "but"

        score = analyzer.calculate_fidelity(draft, reference)
        # Should be lower due to missing "but"
        assert score < 0.9


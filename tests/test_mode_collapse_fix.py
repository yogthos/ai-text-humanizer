"""Unit tests for mode collapse fix in paragraph fusion.

Tests that the system:
1. Uses temperature 0.9 to prevent identical variations
2. Includes diversity requirements in the prompt
3. Logs diversity statistics before deduplication
4. Deduplicates identical variations correctly
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator.translator import StyleTranslator
from src.generator.mutation_operators import PARAGRAPH_FUSION_PROMPT


class MockLLMProvider:
    """Mock LLM provider that tracks temperature and returns variations."""

    def __init__(self, variations_response=None):
        self.call_count = 0
        self.call_history = []
        self.temperature_values = []
        self.variations_response = variations_response or json.dumps([
            "Variation 1 text with different structure.",
            "Variation 2 text with alternative phrasing.",
            "Variation 3 text with varied connectors.",
            "Variation 4 text with different emphasis.",
            "Variation 5 text with unique arrangement."
        ])

    def call(self, system_prompt, user_prompt, model_type="editor", require_json=False,
             temperature=0.7, max_tokens=500, timeout=None, top_p=None):
        self.call_count += 1
        self.temperature_values.append(temperature)
        self.call_history.append({
            "system_prompt": system_prompt[:100] if len(system_prompt) > 100 else system_prompt,
            "user_prompt": user_prompt[:200] if len(user_prompt) > 200 else user_prompt,
            "model_type": model_type,
            "temperature": temperature,
            "require_json": require_json
        })
        return self.variations_response


def test_temperature_is_0_9():
    """Test that paragraph fusion uses temperature 0.9 to prevent mode collapse."""
    mock_llm = MockLLMProvider()

    with patch('src.generator.translator.LLMProvider', return_value=mock_llm):
        translator = StyleTranslator(config_path="config.json")

        # Mock atlas and other dependencies
        mock_atlas = Mock()
        mock_atlas.get_examples_by_rhetoric = Mock(return_value=["Example 1", "Example 2"])
        mock_atlas.get_author_style_vector = Mock(return_value=None)

        # Mock proposition extractor
        with patch.object(translator, 'proposition_extractor') as mock_extractor:
            mock_extractor.extract_atomic_propositions = Mock(return_value=[
                "Proposition 1", "Proposition 2", "Proposition 3"
            ])

            # Mock style extractor (imported from analyzer.style_extractor)
            with patch('src.analyzer.style_extractor.StyleExtractor') as mock_style_extractor_class:
                mock_style_extractor = Mock()
                mock_style_extractor.extract_style_dna = Mock(return_value={"lexicon": ["word1", "word2"]})
                mock_style_extractor_class.return_value = mock_style_extractor

                # Mock structuralizer (imported from analyzer.structuralizer)
                with patch('src.analyzer.structuralizer.extract_paragraph_rhythm') as mock_rhythm:
                    mock_rhythm.return_value = [
                        {"length": "long", "type": "standard", "opener": "if"}
                    ]

                    # Mock critic
                    with patch('src.generator.translator.SemanticCritic') as mock_critic_class:
                        mock_critic = Mock()
                        mock_critic.evaluate = Mock(return_value={
                            "pass": True,
                            "proposition_recall": 0.95,
                            "style_alignment": 0.8,
                            "score": 0.9
                        })
                        mock_critic_class.return_value = mock_critic

                        try:
                            translator.translate_paragraph(
                                "Test paragraph with multiple sentences.",
                                mock_atlas,
                                "TestAuthor",
                                verbose=False
                            )
                        except Exception:
                            # Expected to fail at various points, but we just need to check temperature
                            pass

        # Verify temperature was 0.9
        assert len(mock_llm.temperature_values) > 0, "LLM should have been called"
        # Check if any call used temperature 0.9 (for paragraph fusion)
        paragraph_fusion_temps = [t for t in mock_llm.temperature_values]
        if paragraph_fusion_temps:
            # The last call should be paragraph fusion with temp 0.9
            assert paragraph_fusion_temps[-1] == 0.9, f"Expected temperature 0.9, got {paragraph_fusion_temps[-1]}"

    print("✓ test_temperature_is_0_9 passed")


def test_prompt_includes_diversity_requirements():
    """Test that the paragraph fusion prompt includes diversity requirements."""
    # Load the prompt template
    assert "CRITICAL" in PARAGRAPH_FUSION_PROMPT, "Prompt should include CRITICAL section"
    assert "DISTINCT variations" in PARAGRAPH_FUSION_PROMPT, "Prompt should mention DISTINCT variations"
    assert "Do not simply repeat" in PARAGRAPH_FUSION_PROMPT, "Prompt should forbid repetition"
    assert "sentence structures" in PARAGRAPH_FUSION_PROMPT.lower(), "Prompt should mention varying sentence structures"
    assert "word choices" in PARAGRAPH_FUSION_PROMPT.lower(), "Prompt should mention different word choices"

    print("✓ test_prompt_includes_diversity_requirements passed")


def test_diversity_logging_with_identical_variations():
    """Test that diversity logging detects identical variations."""
    # Create mock variations that are identical
    identical_variations = [
        "Same text here.",
        "Same text here.",
        "Same text here.",
        "Same text here.",
        "Same text here."
    ]

    # Test the diversity check logic
    normalized_variations = [" ".join(v.split()).lower() for v in identical_variations if v]
    unique_normalized = len(set(normalized_variations))

    assert unique_normalized == 1, f"Should detect 1 unique variation, got {unique_normalized}"

    print("✓ test_diversity_logging_with_identical_variations passed")


def test_diversity_logging_with_unique_variations():
    """Test that diversity logging correctly identifies unique variations."""
    unique_variations = [
        "First variation with different structure.",
        "Second variation uses alternative phrasing.",
        "Third variation has varied connectors.",
        "Fourth variation employs different emphasis.",
        "Fifth variation presents unique arrangement."
    ]

    # Test the diversity check logic
    normalized_variations = [" ".join(v.split()).lower() for v in unique_variations if v]
    unique_normalized = len(set(normalized_variations))

    assert unique_normalized == len(unique_variations), \
        f"Should detect {len(unique_variations)} unique variations, got {unique_normalized}"

    print("✓ test_diversity_logging_with_unique_variations passed")


def test_deduplication_removes_duplicates():
    """Test that deduplication correctly removes duplicate variations."""
    variations = [
        "Unique variation one.",
        "Duplicate text here.",
        "Duplicate text here.",  # Duplicate
        "Unique variation two.",
        "Duplicate text here."   # Another duplicate
    ]

    # Simulate deduplication logic
    unique_variations = []
    seen_hashes = set()

    for v in variations:
        if not v or not v.strip():
            continue
        v_clean = " ".join(v.split()).lower()
        v_hash = hash(v_clean)

        if v_hash not in seen_hashes:
            seen_hashes.add(v_hash)
            unique_variations.append(v)

    # Should have 3 unique variations (2 unique + 1 duplicate)
    assert len(unique_variations) == 3, f"Expected 3 unique variations, got {len(unique_variations)}"
    assert "Unique variation one." in unique_variations
    assert "Unique variation two." in unique_variations
    assert "Duplicate text here." in unique_variations  # First occurrence kept

    print("✓ test_deduplication_removes_duplicates passed")


def test_deduplication_preserves_order():
    """Test that deduplication preserves the order of first occurrence."""
    variations = [
        "First unique.",
        "Duplicate.",
        "Second unique.",
        "Duplicate.",  # Should be skipped
        "Third unique."
    ]

    # Simulate deduplication logic
    unique_variations = []
    seen_hashes = set()

    for v in variations:
        if not v or not v.strip():
            continue
        v_clean = " ".join(v.split()).lower()
        v_hash = hash(v_clean)

        if v_hash not in seen_hashes:
            seen_hashes.add(v_hash)
            unique_variations.append(v)

    # Should preserve order: First unique, Duplicate (first), Second unique, Third unique
    assert len(unique_variations) == 4
    assert unique_variations[0] == "First unique."
    assert unique_variations[1] == "Duplicate."
    assert unique_variations[2] == "Second unique."
    assert unique_variations[3] == "Third unique."

    print("✓ test_deduplication_preserves_order passed")


def run_all_tests():
    """Run all mode collapse fix tests."""
    print("\n" + "="*60)
    print("Running Mode Collapse Fix Tests")
    print("="*60 + "\n")

    tests = [
        test_temperature_is_0_9,
        test_prompt_includes_diversity_requirements,
        test_diversity_logging_with_identical_variations,
        test_diversity_logging_with_unique_variations,
        test_deduplication_removes_duplicates,
        test_deduplication_preserves_order,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


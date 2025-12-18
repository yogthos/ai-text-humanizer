"""Edge case tests for style DNA extraction.

These tests ensure style extraction handles edge cases gracefully and doesn't
break paragraph fusion when extraction fails.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.style_extractor import StyleExtractor


def test_style_extraction_empty_examples_list():
    """Contract: Empty examples list returns fallback DNA.

    This test ensures style extraction handles empty input gracefully.
    If this test fails, empty input handling is broken.
    """
    extractor = StyleExtractor()

    result = extractor.extract_style_dna([])

    # Should return fallback DNA
    assert isinstance(result, dict), "Should return dictionary"
    assert "lexicon" in result, "Should have lexicon field"
    assert "tone" in result, "Should have tone field"
    assert "structure" in result, "Should have structure field"
    assert result["lexicon"] == [], "Lexicon should be empty list"
    assert result["tone"] == "Neutral", "Tone should be 'Neutral' fallback"
    assert result["structure"] == "Standard sentence structure", "Structure should be fallback"
    print("✓ Contract: Empty examples → fallback DNA returned")


def test_style_extraction_llm_failure_uses_fallback():
    """Contract: LLM extraction failure uses fallback extraction.

    This test ensures graceful handling when LLM extraction fails.
    """
    extractor = StyleExtractor()

    # Mock LLM to raise exception
    original_call = extractor.llm_provider.call
    extractor.llm_provider.call = Mock(side_effect=Exception("LLM failure"))

    examples = [
        "It is a fundamental law of dialectics that all material bodies must undergo the process of internal contradiction.",
        "The historical trajectory of human society demonstrates a continuous struggle between opposing forces."
    ]

    result = extractor.extract_style_dna(examples)

    # Should return fallback DNA (not crash)
    assert isinstance(result, dict), "Should return dictionary even on failure"
    assert "lexicon" in result, "Should have lexicon field"
    assert "tone" in result, "Should have tone field"
    assert "structure" in result, "Should have structure field"

    # Restore original
    extractor.llm_provider.call = original_call
    print("✓ Contract: LLM failure → fallback extraction used")


def test_style_extraction_malformed_json_uses_fallback():
    """Contract: Malformed JSON response uses fallback extraction.

    This test ensures graceful handling when LLM returns invalid JSON.
    """
    extractor = StyleExtractor()

    # Mock LLM to return malformed JSON
    def mock_call(*args, **kwargs):
        return "This is not valid JSON {invalid syntax"

    original_call = extractor.llm_provider.call
    extractor.llm_provider.call = mock_call

    examples = [
        "It is a fundamental law of dialectics that all material bodies must undergo the process of internal contradiction."
    ]

    result = extractor.extract_style_dna(examples)

    # Should return fallback DNA (not crash)
    assert isinstance(result, dict), "Should return dictionary even with malformed JSON"
    assert "lexicon" in result, "Should have lexicon field"
    assert "tone" in result, "Should have tone field"
    assert "structure" in result, "Should have structure field"

    # Restore original
    extractor.llm_provider.call = original_call
    print("✓ Contract: Malformed JSON → fallback extraction used")


def test_style_extraction_missing_fields_uses_defaults():
    """Contract: Missing fields in JSON response use defaults.

    This test ensures missing lexicon/tone/structure fields are handled.
    """
    extractor = StyleExtractor()

    # Mock LLM to return JSON with missing fields
    def mock_call(*args, **kwargs):
        return json.dumps({
            "lexicon": ["test"]  # Missing tone and structure
        })

    original_call = extractor.llm_provider.call
    extractor.llm_provider.call = mock_call

    examples = [
        "It is a fundamental law of dialectics that all material bodies must undergo the process of internal contradiction."
    ]

    result = extractor.extract_style_dna(examples)

    # Should have all fields with defaults for missing ones
    assert isinstance(result, dict), "Should return dictionary"
    assert "lexicon" in result, "Should have lexicon field"
    assert "tone" in result, "Should have tone field (defaulted)"
    assert "structure" in result, "Should have structure field (defaulted)"
    assert result["tone"] == "Neutral", "Missing tone should default to 'Neutral'"
    assert result["structure"] == "Standard sentence structure", "Missing structure should use default"

    # Restore original
    extractor.llm_provider.call = original_call
    print("✓ Contract: Missing fields → defaults applied")


def test_style_extraction_wrong_field_types_handled():
    """Contract: Wrong field types in JSON response are handled.

    This test ensures non-string/non-list fields are converted correctly.
    """
    extractor = StyleExtractor()

    # Mock LLM to return JSON with wrong types
    def mock_call(*args, **kwargs):
        return json.dumps({
            "lexicon": "not a list",  # Should be list
            "tone": 123,  # Should be string
            "structure": ["not", "a", "string"]  # Should be string
        })

    original_call = extractor.llm_provider.call
    extractor.llm_provider.call = mock_call

    examples = [
        "It is a fundamental law of dialectics that all material bodies must undergo the process of internal contradiction."
    ]

    result = extractor.extract_style_dna(examples)

    # Should normalize types
    assert isinstance(result, dict), "Should return dictionary"
    assert isinstance(result["lexicon"], list), "Lexicon should be list (normalized)"
    assert isinstance(result["tone"], str), "Tone should be string (normalized)"
    assert isinstance(result["structure"], str), "Structure should be string (normalized)"

    # Restore original
    extractor.llm_provider.call = original_call
    print("✓ Contract: Wrong field types → normalized correctly")


def test_style_extraction_very_short_examples():
    """Contract: Very short examples still allow extraction.

    This test ensures extraction works even with minimal input.
    """
    extractor = StyleExtractor()

    # Mock LLM to return valid DNA
    def mock_call(*args, **kwargs):
        return json.dumps({
            "lexicon": ["test", "word"],
            "tone": "Authoritative",
            "structure": "Uses complex sentences"
        })

    original_call = extractor.llm_provider.call
    extractor.llm_provider.call = mock_call

    # Very short examples
    examples = [
        "Test.",
        "Short."
    ]

    result = extractor.extract_style_dna(examples)

    # Should still return valid DNA
    assert isinstance(result, dict), "Should return dictionary"
    assert "lexicon" in result, "Should have lexicon field"
    assert len(result["lexicon"]) > 0, "Should extract some lexicon even from short examples"

    # Restore original
    extractor.llm_provider.call = original_call
    print("✓ Contract: Very short examples → extraction still works")


def test_style_extraction_lexicon_limited_to_10():
    """Contract: Lexicon is limited to 10 items maximum.

    This test ensures lexicon doesn't exceed the limit.
    """
    extractor = StyleExtractor()

    # Mock LLM to return many lexicon items
    def mock_call(*args, **kwargs):
        return json.dumps({
            "lexicon": [f"word{i}" for i in range(20)],  # 20 items
            "tone": "Authoritative",
            "structure": "Uses complex sentences"
        })

    original_call = extractor.llm_provider.call
    extractor.llm_provider.call = mock_call

    examples = [
        "It is a fundamental law of dialectics that all material bodies must undergo the process of internal contradiction."
    ]

    result = extractor.extract_style_dna(examples)

    # Lexicon should be limited to 10
    assert len(result["lexicon"]) <= 10, f"Lexicon should be limited to 10, got {len(result['lexicon'])}"

    # Restore original
    extractor.llm_provider.call = original_call
    print("✓ Contract: Lexicon limited to 10 items")


if __name__ == "__main__":
    print("Running Style Extraction Edge Case Tests...\n")

    tests = [
        test_style_extraction_empty_examples_list,
        test_style_extraction_llm_failure_uses_fallback,
        test_style_extraction_malformed_json_uses_fallback,
        test_style_extraction_missing_fields_uses_defaults,
        test_style_extraction_wrong_field_types_handled,
        test_style_extraction_very_short_examples,
        test_style_extraction_lexicon_limited_to_10,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if failed == 0:
        print("\n🎉 All style extraction edge case tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        sys.exit(1)


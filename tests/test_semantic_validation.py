"""Test semantic validation for nonsensical sentences."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validator.critic import _check_semantic_validity, critic_evaluate


def test_incomplete_dependent_clause_end():
    """Test: 'The code, even though it is embedded in every particle and field.' - incomplete"""
    result = _check_semantic_validity("The code, even though it is embedded in every particle and field.")

    print("\nTest: Incomplete dependent clause at end")
    print(f"  Input: 'The code, even though it is embedded in every particle and field.'")
    print(f"  Result: {result}")

    if result is None:
        print("  ❌ FAIL: Should detect incomplete sentence")
        return False

    if result.get("score", 1.0) != 0.0:
        print("  ❌ FAIL: Score should be 0.0")
        return False

    if "incomplete" not in result.get("feedback", "").lower():
        print("  ❌ FAIL: Feedback should mention incomplete sentence")
        return False

    print("  ✓ PASS: Incomplete sentence detected")
    return True


def test_incomplete_dependent_clause_start():
    """Test: 'Even though it is embedded.' - incomplete (no main clause)"""
    result = _check_semantic_validity("Even though it is embedded.")

    print("\nTest: Incomplete dependent clause at start")
    print(f"  Input: 'Even though it is embedded.'")
    print(f"  Result: {result}")

    if result is None:
        print("  ❌ FAIL: Should detect incomplete sentence")
        return False

    if result.get("score", 1.0) != 0.0:
        print("  ❌ FAIL: Score should be 0.0")
        return False

    print("  ✓ PASS: Incomplete sentence detected")
    return True


def test_valid_sentence_with_dependent_clause():
    """Test: 'The code is embedded in every particle and field, even though it is complex.' - valid"""
    result = _check_semantic_validity("The code is embedded in every particle and field, even though it is complex.")

    print("\nTest: Valid sentence with dependent clause")
    print(f"  Input: 'The code is embedded in every particle and field, even though it is complex.'")
    print(f"  Result: {result}")

    if result is not None:
        print("  ❌ FAIL: Should accept valid sentence")
        return False

    print("  ✓ PASS: Valid sentence accepted")
    return True


def test_valid_dependent_clause_start():
    """Test: 'Even though it is complex, the code works.' - valid"""
    result = _check_semantic_validity("Even though it is complex, the code works.")

    print("\nTest: Valid dependent clause at start")
    print(f"  Input: 'Even though it is complex, the code works.'")
    print(f"  Result: {result}")

    if result is not None:
        print("  ❌ FAIL: Should accept valid sentence")
        return False

    print("  ✓ PASS: Valid sentence accepted")
    return True


def test_incomplete_relative_clause():
    """Test: 'The code, which is embedded.' - incomplete (missing main verb)"""
    result = _check_semantic_validity("The code, which is embedded.")

    print("\nTest: Incomplete relative clause")
    print(f"  Input: 'The code, which is embedded.'")
    print(f"  Result: {result}")

    if result is None:
        print("  ❌ FAIL: Should detect incomplete sentence")
        return False

    if result.get("score", 1.0) != 0.0:
        print("  ❌ FAIL: Score should be 0.0")
        return False

    print("  ✓ PASS: Incomplete sentence detected")
    return True


def test_valid_relative_clause():
    """Test: 'The code, which is embedded in every particle, works.' - valid"""
    result = _check_semantic_validity("The code, which is embedded in every particle, works.")

    print("\nTest: Valid relative clause")
    print(f"  Input: 'The code, which is embedded in every particle, works.'")
    print(f"  Result: {result}")

    if result is not None:
        print("  ❌ FAIL: Should accept valid sentence")
        return False

    print("  ✓ PASS: Valid sentence accepted")
    return True


def test_short_incomplete_fragment():
    """Test: 'limits, even though they are only implied by an exterior.' - incomplete fragment"""
    result = _check_semantic_validity("limits, even though they are only implied by an exterior.")

    print("\nTest: Short incomplete fragment")
    print(f"  Input: 'limits, even though they are only implied by an exterior.'")
    print(f"  Result: {result}")

    if result is None:
        print("  ❌ FAIL: Should detect incomplete fragment")
        return False

    if result.get("score", 1.0) != 0.0:
        print("  ❌ FAIL: Score should be 0.0")
        return False

    print("  ✓ PASS: Incomplete fragment detected")
    return True


def test_multiple_sentences_one_incomplete():
    """Test: Multiple sentences where one is incomplete"""
    result = _check_semantic_validity("The code works well. The system, even though it is complex.")

    print("\nTest: Multiple sentences with one incomplete")
    print(f"  Input: 'The code works well. The system, even though it is complex.'")
    print(f"  Result: {result}")

    if result is None:
        print("  ❌ FAIL: Should detect incomplete sentence")
        return False

    print("  ✓ PASS: Incomplete sentence detected")
    return True


def test_real_scenario_from_output():
    """Test: Real scenario from output/small.md"""
    problematic_sentences = [
        "The code, even though it is embedded in every particle and field.",
        "limits, even though they are only implied by an exterior.",
        "The system, even though it is only expressed as complete.",
        "Such an internal framework, even though it eliminates the need for an external container.",
        "Our universe, even though it is only a single iteration of the infinite construct.",
        "bubbles, even though they are only expressed in unique laws of physics.",
        "scale, even though it is only an artifact of our local perspective.",
        "sections, even though they are replicas of the whole.",
    ]

    print("\nTest: Real scenarios from output/small.md")
    all_passed = True

    for sentence in problematic_sentences:
        result = _check_semantic_validity(sentence)
        if result is None:
            print(f"  ❌ FAIL: Should detect incomplete: '{sentence}'")
            all_passed = False
        else:
            print(f"  ✓ PASS: Detected incomplete: '{sentence[:60]}...'")

    return all_passed


def test_semantic_check_in_critic_evaluate():
    """Test that semantic check is integrated into critic_evaluate"""
    config_path = Path("config.json")
    if not config_path.exists():
        print("⚠ Skipping API test: config.json not found")
        return False

    import json
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        provider = config.get("provider", "deepseek")
        if provider == "deepseek":
            deepseek_config = config.get("deepseek", {})
            api_key = deepseek_config.get("api_key")
            if not api_key or api_key == "your-api-key-here":
                print("⚠ Skipping API test: No valid API key in config")
                return False
        elif provider == "ollama":
            ollama_config = config.get("ollama", {})
            if not ollama_config.get("url"):
                print("⚠ Skipping API test: No Ollama URL in config")
                return False
    except Exception as e:
        print(f"⚠ Skipping API test: Error reading config: {e}")
        return False

    # Test that critic_evaluate catches incomplete sentence before LLM call
    incomplete_text = "The code, even though it is embedded in every particle and field."
    structure_match = "These two-dimensional planes cutting through nine-dimensional genetic space give us insight."
    original_text = "The code is embedded in every particle and field."

    try:
        result = critic_evaluate(
            generated_text=incomplete_text,
            structure_match=structure_match,
            situation_match=None,
            original_text=original_text,
            config_path=str(config_path)
        )

        score = result.get("score", 1.0)
        feedback = result.get("feedback", "")
        failure_type = result.get("primary_failure_type", "")

        print(f"\nTest: Semantic check in critic_evaluate")
        print(f"  Input: '{incomplete_text}'")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:100]}...")
        print(f"  Failure type: {failure_type}")

        if score != 0.0:
            print("  ❌ FAIL: Score should be 0.0 for incomplete sentence")
            return False

        if failure_type != "grammar":
            print(f"  ❌ FAIL: Failure type should be 'grammar', got '{failure_type}'")
            return False

        if "incomplete" not in feedback.lower():
            print("  ❌ FAIL: Feedback should mention incomplete sentence")
            return False

        print("  ✓ PASS: Semantic check working in critic_evaluate")
        return True

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Semantic Validation")
    print("=" * 60)

    test1_passed = test_incomplete_dependent_clause_end()
    test2_passed = test_incomplete_dependent_clause_start()
    test3_passed = test_valid_sentence_with_dependent_clause()
    test4_passed = test_valid_dependent_clause_start()
    test5_passed = test_incomplete_relative_clause()
    test6_passed = test_valid_relative_clause()
    test7_passed = test_short_incomplete_fragment()
    test8_passed = test_multiple_sentences_one_incomplete()
    test9_passed = test_real_scenario_from_output()
    test10_passed = test_semantic_check_in_critic_evaluate()

    print("\n" + "=" * 60)
    if all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed,
            test6_passed, test7_passed, test8_passed, test9_passed, test10_passed]):
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)


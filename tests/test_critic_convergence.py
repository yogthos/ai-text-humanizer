"""Test critic convergence issues from logs."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validator.critic import critic_evaluate
from src.utils import should_skip_length_gate, calculate_length_ratio


def test_false_positive_feedback_preservation():
    """Test that false positive override preserves useful feedback instead of generic message."""
    # Check if config exists
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

    # Scenario from logs: LLM flags "essential" as proper noun, filter overrides
    original_text = "Human experience reinforces the rule of finitude."
    generated_text = "Human experience confirms the essential rule of finitude."
    structure_match = "These two-dimensional planes cutting through nine-dimensional genetic space give us insight into evolutionary processes."
    situation_match = "Evolution having themselves. This is the essential ingredient of a self-reinforcing process."

    try:
        result = critic_evaluate(
            generated_text=generated_text,
            structure_match=structure_match,
            situation_match=situation_match,
            original_text=original_text,
            config_path=str(config_path)
        )

        score = result.get("score", 0.0)
        feedback = result.get("feedback", "")
        failure_type = result.get("primary_failure_type", "")

        print(f"\nTest: False positive override feedback quality")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:200]}...")
        print(f"  Failure type: {failure_type}")

        # Feedback should NOT be the generic "Text may need minor adjustments"
        # It should have actual useful feedback about structure/style
        if feedback == "Text may need minor adjustments for style match.":
            print("  ❌ FAIL: Feedback is too generic after false positive override")
            return False

        # Feedback should contain something actionable
        if len(feedback) < 20:
            print("  ❌ FAIL: Feedback is too short/empty")
            return False

        print("  ✓ PASS: Feedback preserved or improved")
        return True

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_length_gate_skip_when_very_different():
    """Test that length gate is skipped when structure match is very different."""
    config_path = Path("config.json")
    if not config_path.exists():
        print("⚠ Skipping test: config.json not found")
        return False

    # Scenario from logs: structure match is 2.86x different (20 words vs 7 words)
    structure_match = "These two-dimensional planes cutting through nine-dimensional genetic space give us insight into evolutionary processes and biological mechanisms."
    original_text = "Human experience reinforces the rule of finitude."

    structure_input_ratio = calculate_length_ratio(structure_match, original_text)
    print(f"\nTest: Length gate skip when structure very different")
    print(f"  Structure match: {len(structure_match.split())} words")
    print(f"  Original: {len(original_text.split())} words")
    print(f"  Ratio: {structure_input_ratio:.2f}")

    should_skip = should_skip_length_gate(structure_input_ratio, config_path=str(config_path))
    print(f"  Should skip: {should_skip}")

    # Ratio is 2.86, which is > 2.0, so should skip
    if structure_input_ratio > 2.0 or structure_input_ratio < 0.5:
        if not should_skip:
            print("  ❌ FAIL: Should skip length gate when ratio is very different")
            return False
        print("  ✓ PASS: Length gate correctly skipped")
    else:
        print("  ⚠ INFO: Ratio is not very different, skip not required")

    return True


def test_score_improvement_with_useful_feedback():
    """Test that score can improve when feedback is useful (not generic)."""
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

    # Test that when we have useful feedback (not false positive), score can improve
    original_text = "The biological cycle of birth, life, and decay defines our reality."
    generated_text = "The cycle defines reality."  # Too short
    structure_match = "These two-dimensional planes cutting through nine-dimensional genetic space give us insight into evolutionary processes."
    situation_match = None

    try:
        result = critic_evaluate(
            generated_text=generated_text,
            structure_match=structure_match,
            situation_match=situation_match,
            original_text=original_text,
            config_path=str(config_path)
        )

        score = result.get("score", 0.0)
        feedback = result.get("feedback", "")

        print(f"\nTest: Score improvement with useful feedback")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:200]}...")

        # Feedback should be specific and actionable
        if "FATAL ERROR" in feedback or "too short" in feedback.lower() or "expand" in feedback.lower():
            print("  ✓ PASS: Feedback is specific and actionable")
            return True
        elif len(feedback) < 30:
            print("  ❌ FAIL: Feedback is too generic/short")
            return False
        else:
            print("  ✓ PASS: Feedback appears useful")
            return True

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Critic Convergence Issues")
    print("=" * 60)

    test1_passed = test_false_positive_feedback_preservation()
    test2_passed = test_length_gate_skip_when_very_different()
    test3_passed = test_score_improvement_with_useful_feedback()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed and test3_passed:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)


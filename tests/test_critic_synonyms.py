"""Test that critic allows synonyms but flags proper noun hallucinations."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validator.critic import critic_evaluate


def test_critic_allows_synonyms():
    """Test that critic allows synonyms like 'essential' for 'important'."""
    # Check if config exists and has valid API key
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
            # Ollama doesn't need API key, just check if URL is configured
            ollama_config = config.get("ollama", {})
            if not ollama_config.get("url"):
                print("⚠ Skipping API test: No Ollama URL in config")
                return False
    except Exception as e:
        print(f"⚠ Skipping API test: Error reading config: {e}")
        return False

    # Test case 1: Synonym should be allowed
    # Original has "important", generated has "essential" (synonym)
    original_text = "Human experience confirms the important rule of finitude."
    generated_text = "Human experience reinforces the essential rule of finitude."
    structure_match = "The biological cycle defines our reality. Every star eventually succumbs to erosion."
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
        pass_flag = result.get("pass", False)
        failure_type = result.get("primary_failure_type", "")

        print(f"\nTest 1: Synonym 'essential' for 'important'")
        print(f"  Score: {score}")
        print(f"  Pass: {pass_flag}")
        print(f"  Feedback: {feedback[:100]}...")
        print(f"  Failure type: {failure_type}")

        # Check that synonym is NOT flagged as hallucination
        if "essential" in feedback.lower() and "does not appear" in feedback.lower():
            print("  ❌ FAIL: Synonym 'essential' was flagged as hallucination")
            return False

        # Score should not be 0.0 for a valid synonym
        if score == 0.0 and failure_type == "meaning":
            print("  ❌ FAIL: Score is 0.0 for valid synonym")
            return False

        print("  ✓ PASS: Synonym allowed")

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test case 2: Another synonym - "necessary" for "required"
    original_text = "The required process must be followed."
    generated_text = "The necessary process must be followed."
    structure_match = "The biological cycle defines our reality."
    situation_match = "Evolution having themselves."

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

        print(f"\nTest 2: Synonym 'necessary' for 'required'")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:100]}...")
        print(f"  Failure type: {failure_type}")

        # Check that synonym is NOT flagged
        if "necessary" in feedback.lower() and "does not appear" in feedback.lower():
            print("  ❌ FAIL: Synonym 'necessary' was flagged as hallucination")
            return False

        if score == 0.0 and failure_type == "meaning":
            print("  ❌ FAIL: Score is 0.0 for valid synonym")
            return False

        print("  ✓ PASS: Synonym allowed")

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_critic_flags_proper_nouns():
    """Test that critic still flags actual proper noun hallucinations."""
    # Check if config exists and has valid API key
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

    # Test case: Proper noun "Schneider" not in original should be flagged
    original_text = "Human experience confirms the important rule of finitude."
    generated_text = "Schneider confirms the important rule of finitude."
    structure_match = "The biological cycle defines our reality."
    situation_match = "Evolution having themselves."

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

        print(f"\nTest 3: Proper noun 'Schneider' not in original")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:150]}...")
        print(f"  Failure type: {failure_type}")

        # Proper noun should be flagged
        if "schneider" in feedback.lower() and ("does not appear" in feedback.lower() or "not present" in feedback.lower()):
            print("  ✓ PASS: Proper noun correctly flagged")
            if score == 0.0:
                print("  ✓ PASS: Score is 0.0 for hallucinated proper noun")
                return True
            else:
                print("  ⚠ WARNING: Proper noun flagged but score is not 0.0")
                return True  # Still pass, just warn
        else:
            print("  ❌ FAIL: Proper noun 'Schneider' was NOT flagged")
            return False

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_critic_real_scenario_from_logs():
    """Test with actual scenario from logs - 'essential' and 'essential ingredient' should NOT be flagged."""
    # Check if config exists and has valid API key
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

    # Real scenario from logs
    original_text = "Human experience reinforces the rule of finitude."
    generated_text = "Human experience confirms the essential rule of finitude."
    structure_match = "These two-dimensional planes cutting through nine-dimensional genetic space give us insight."
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
        pass_flag = result.get("pass", False)

        print(f"\nTest 4: Real scenario - 'essential' (lowercase) should NOT be flagged")
        print(f"  Original: {original_text}")
        print(f"  Generated: {generated_text}")
        print(f"  Score: {score}")
        print(f"  Pass: {pass_flag}")
        print(f"  Feedback: {feedback[:200]}...")
        print(f"  Failure type: {failure_type}")

        # Check that lowercase "essential" is NOT flagged
        feedback_lower = feedback.lower()
        if ("essential" in feedback_lower and
            ("does not appear" in feedback_lower or "not present" in feedback_lower or
             "proper noun" in feedback_lower or "entity" in feedback_lower)):
            print("  ❌ FAIL: Lowercase 'essential' was flagged as proper noun/entity")
            return False

        # Score should not be 0.0 for a valid synonym
        if score == 0.0 and failure_type == "meaning":
            print("  ❌ FAIL: Score is 0.0 for valid lowercase word")
            return False

        print("  ✓ PASS: Lowercase 'essential' not flagged")

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test with "essential ingredient" phrase
    generated_text2 = "This is the essential ingredient of a self-reinforcing process."
    original_text2 = "This is the important component of a self-reinforcing process."

    try:
        result = critic_evaluate(
            generated_text=generated_text2,
            structure_match=structure_match,
            situation_match=situation_match,
            original_text=original_text2,
            config_path=str(config_path)
        )

        score = result.get("score", 0.0)
        feedback = result.get("feedback", "")
        failure_type = result.get("primary_failure_type", "")

        print(f"\nTest 5: Real scenario - 'essential ingredient' (phrase) should NOT be flagged")
        print(f"  Original: {original_text2}")
        print(f"  Generated: {generated_text2}")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:200]}...")
        print(f"  Failure type: {failure_type}")

        # Check that phrase "essential ingredient" is NOT flagged
        feedback_lower = feedback.lower()
        if (("essential ingredient" in feedback_lower or "essential" in feedback_lower) and
            ("does not appear" in feedback_lower or "not present" in feedback_lower or
             "proper noun" in feedback_lower or "entity" in feedback_lower)):
            print("  ❌ FAIL: Phrase 'essential ingredient' was flagged as proper noun/entity")
            return False

        if score == 0.0 and failure_type == "meaning":
            print("  ❌ FAIL: Score is 0.0 for valid phrase")
            return False

        print("  ✓ PASS: Phrase 'essential ingredient' not flagged")

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Critic Synonym Handling")
    print("=" * 60)

    test1_passed = test_critic_allows_synonyms()
    test2_passed = test_critic_flags_proper_nouns()
    test3_passed = test_critic_real_scenario_from_logs()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed and test3_passed:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)


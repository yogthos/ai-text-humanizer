"""Test false positive scenarios from actual debug logs."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validator.critic import critic_evaluate


def test_false_positive_omits_our_reality():
    """Test: 'CRITICAL: Text omits 'our reality' from original' when it's actually present.

    From logs: Sentence 2/8
    Original: "The biological cycle of birth, life, and decay defines our reality."
    Generated likely contains "our reality" but critic flags it as missing.
    """
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

    # Exact scenario from logs
    original_text = "The biological cycle of birth, life, and decay defines our reality."
    generated_text = "The biological cycle—birth, life, decay—defines our reality."
    structure_match = "These two-dimensional planes cutting through nine-dimensional genetic space give us insight into evolutionary processes."
    situation_match = None
    structure_input_ratio = 1.82  # From logs: "Structure match length ratio 1.82 is different"

    try:
        result = critic_evaluate(
            generated_text=generated_text,
            structure_match=structure_match,
            situation_match=situation_match,
            original_text=original_text,
            config_path=str(config_path),
            structure_input_ratio=structure_input_ratio
        )

        score = result.get("score", 0.0)
        feedback = result.get("feedback", "")
        failure_type = result.get("primary_failure_type", "")

        print(f"\nTest: False positive 'omits our reality'")
        print(f"  Original: {original_text}")
        print(f"  Generated: {generated_text}")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:200]}...")
        print(f"  Failure type: {failure_type}")

        # Check if "our reality" is actually in generated text
        if "our reality" in generated_text.lower():
            # It's present, so if critic says it's missing, that's a false positive
            feedback_lower = feedback.lower()
            if "omits" in feedback_lower and "our reality" in feedback_lower:
                print("  ❌ FAIL: Critic falsely claims 'our reality' is omitted when it's present")
                return False

        # Score should not be 0.0 for this (content is preserved)
        if score == 0.0 and failure_type == "meaning":
            print("  ❌ FAIL: Score is 0.0 for false positive content omission")
            return False

        print("  ✓ PASS: No false positive detected")
        return True

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_emdash_grammar_false_positive():
    """Test: 'CRITICAL: Text contains grammatical errors. The em-dash construction creates a fragment'

    From logs: Sentence 2/8, generation attempt 2/5
    Generated text uses em-dashes matching structure reference, but critic flags as grammar error.
    """
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

    # Exact scenario from logs
    original_text = "The biological cycle of birth, life, and decay defines our reality."
    generated_text = "The biological cycle—birth, life, decay—defines our reality."
    structure_match = "These two-dimensional planes—cutting through nine-dimensional genetic space—give us insight."
    situation_match = None
    structure_input_ratio = 1.82

    try:
        result = critic_evaluate(
            generated_text=generated_text,
            structure_match=structure_match,
            situation_match=situation_match,
            original_text=original_text,
            config_path=str(config_path),
            structure_input_ratio=structure_input_ratio
        )

        score = result.get("score", 0.0)
        feedback = result.get("feedback", "")
        failure_type = result.get("primary_failure_type", "")

        print(f"\nTest: Em-dash grammar false positive")
        print(f"  Original: {original_text}")
        print(f"  Generated: {generated_text}")
        print(f"  Structure: {structure_match}")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:200]}...")
        print(f"  Failure type: {failure_type}")

        # Check if structure uses em-dash and generated uses em-dash
        structure_has_emdash = "—" in structure_match
        generated_has_emdash = "—" in generated_text

        if structure_has_emdash and generated_has_emdash:
            # Both use em-dash, so it's valid style
            feedback_lower = feedback.lower()
            if "grammar" in feedback_lower and ("dash" in feedback_lower or "fragment" in feedback_lower or "—" in feedback):
                print("  ❌ FAIL: Em-dash flagged as grammar error when it matches structure")
                return False

        # Score should not be 0.0 for valid style matching
        if score == 0.0 and failure_type == "grammar":
            print("  ❌ FAIL: Score is 0.0 for valid em-dash style matching")
            return False

        print("  ✓ PASS: Em-dash not flagged as grammar error")
        return True

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_colon_punctuation_false_positive():
    """Test: 'CRITICAL: Text omits the colon punctuation structure from the Structural Reference'

    From logs: Sentence 2/8, after structure refresh to "Tony Febbo questioned...:"
    Critic flags missing colon, but colon might not be appropriate for the content.
    """
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

    # Exact scenario from logs
    original_text = "The biological cycle of birth, life, and decay defines our reality."
    generated_text = "The biological cycle of birth, life, and decay defines our reality."
    structure_match = "Tony Febbo questioned this blame-the-leadership syndrome of the pure socialists:"
    situation_match = None
    structure_input_ratio = None  # New structure match

    try:
        result = critic_evaluate(
            generated_text=generated_text,
            structure_match=structure_match,
            situation_match=situation_match,
            original_text=original_text,
            config_path=str(config_path),
            structure_input_ratio=structure_input_ratio
        )

        score = result.get("score", 0.0)
        feedback = result.get("feedback", "")
        failure_type = result.get("primary_failure_type", "")

        print(f"\nTest: Colon punctuation false positive")
        print(f"  Original: {original_text}")
        print(f"  Generated: {generated_text}")
        print(f"  Structure: {structure_match}")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:200]}...")
        print(f"  Failure type: {failure_type}")

        # Check if structure has colon at end (dialogue/question format)
        structure_has_colon = structure_match.strip().endswith(":")

        # If structure has colon but it's a dialogue tag or question format,
        # and generated text is a statement, colon might not be appropriate
        if structure_has_colon:
            feedback_lower = feedback.lower()
            if "colon" in feedback_lower and "omits" in feedback_lower:
                # This might be a false positive if colon is not appropriate for statement format
                # But we need to check if the structure colon is actually a dialogue tag
                if structure_match.strip().startswith(("Tony", "August", "Schneider")):
                    # It's a dialogue tag, colon is not required for statement format
                    print("  ⚠ INFO: Structure has dialogue tag colon, but generated is statement format")
                    # This might be acceptable - colon is for dialogue, not statements
                    if score == 0.0:
                        print("  ❌ FAIL: Score is 0.0 for missing dialogue colon in statement format")
                        return False

        print("  ✓ PASS: Colon punctuation handled appropriately")
        return True

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_false_positive_omits_biological_cycle():
    """Test: 'CRITICAL: Text omits 'biological cycle of birth, life, and decay' from original'

    From logs: Sentence 2/8, after structure refresh
    Generated text likely contains this phrase but critic flags it as missing.
    """
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

    # Exact scenario from logs
    original_text = "The biological cycle of birth, life, and decay defines our reality."
    generated_text = "The biological cycle—birth, life, decay—defines our reality."
    structure_match = "August: They were necessary, but … they are historical experiences."
    situation_match = None
    structure_input_ratio = None

    try:
        result = critic_evaluate(
            generated_text=generated_text,
            structure_match=structure_match,
            situation_match=situation_match,
            original_text=original_text,
            config_path=str(config_path),
            structure_input_ratio=structure_input_ratio
        )

        score = result.get("score", 0.0)
        feedback = result.get("feedback", "")
        failure_type = result.get("primary_failure_type", "")

        print(f"\nTest: False positive 'omits biological cycle'")
        print(f"  Original: {original_text}")
        print(f"  Generated: {generated_text}")
        print(f"  Score: {score}")
        print(f"  Feedback: {feedback[:200]}...")
        print(f"  Failure type: {failure_type}")

        # Check if key concepts are present (with fuzzy matching for word order)
        key_concepts = ["biological cycle", "birth", "life", "decay", "defines", "our reality"]
        generated_lower = generated_text.lower()

        concepts_present = sum(1 for concept in key_concepts if concept in generated_lower)
        concepts_total = len(key_concepts)

        print(f"  Concepts present: {concepts_present}/{concepts_total}")

        if concepts_present >= concepts_total * 0.8:  # 80% of concepts present
            # Most concepts are present, so if critic says they're missing, that's a false positive
            feedback_lower = feedback.lower()
            if "omits" in feedback_lower and "biological cycle" in feedback_lower:
                print("  ❌ FAIL: Critic falsely claims 'biological cycle' is omitted when it's present")
                return False

        # Score should not be 0.0 if most content is preserved
        if score == 0.0 and failure_type == "meaning" and concepts_present >= concepts_total * 0.8:
            print("  ❌ FAIL: Score is 0.0 when 80%+ of content is preserved")
            return False

        print("  ✓ PASS: No false positive detected")
        return True

    except Exception as e:
        print(f"  ❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Critic False Positives from Actual Logs")
    print("=" * 60)

    test1_passed = test_false_positive_omits_our_reality()
    test2_passed = test_emdash_grammar_false_positive()
    test3_passed = test_colon_punctuation_false_positive()
    test4_passed = test_false_positive_omits_biological_cycle()

    print("\n" + "=" * 60)
    if all([test1_passed, test2_passed, test3_passed, test4_passed]):
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)


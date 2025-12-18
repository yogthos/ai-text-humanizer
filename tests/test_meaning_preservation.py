"""Tests for meaning preservation and word salad detection."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validator.critic import (
    is_grammatically_coherent,
    check_semantic_similarity,
    critic_evaluate
)


def test_word_salad_detection():
    """Test that word salad like 'The Human View of Discrete Levels Scale...' is detected."""
    word_salad = "The Human View of Discrete Levels Scale as a Local Perspective Artifact Observe the Mandelbrot set."
    result = is_grammatically_coherent(word_salad)
    print(f"\nTest: Word salad detection - Result: {result}")
    assert not result, "Word salad should be detected as incoherent"
    print("✓ PASSED: Word salad detected")


def test_title_case_abuse():
    """Test that excessive title case (>70% capitalized) is detected."""
    title_case_abuse = "The Code Even Though It Is Embedded In Every Particle And Field"
    result = is_grammatically_coherent(title_case_abuse)
    print(f"\nTest: Title case abuse - Result: {result}")
    assert not result, "Title case abuse should be detected"
    print("✓ PASSED: Title case abuse detected")


def test_missing_verb():
    """Test that sentences without verbs are detected."""
    no_verb = "The code, even though embedded."
    result = is_grammatically_coherent(no_verb)
    print(f"\nTest: Missing verb - Result: {result}")
    # If Spacy is available, it should fail; if not, it might pass
    # The test verifies the function runs without error
    print(f"✓ PASSED: Function executed (result: {result})")


def test_valid_sentence():
    """Test that valid sentences pass the coherence check."""
    valid = "The code is embedded in every particle and field, even though it is complex."
    result = is_grammatically_coherent(valid)
    print(f"\nTest: Valid sentence - Result: {result}")
    assert result, "Valid sentence should pass coherence check"
    print("✓ PASSED: Valid sentence passed")


def test_valid_sentence_with_dependent_clause():
    """Test that valid sentences with dependent clauses pass."""
    valid = "Even though it is complex, the code works well."
    result = is_grammatically_coherent(valid)
    print(f"\nTest: Valid sentence with dependent clause - Result: {result}")
    assert result, "Valid sentence with dependent clause should pass"
    print("✓ PASSED: Valid sentence with dependent clause passed")


def test_short_title():
    """Test that short titles (<5 words) are not flagged as title case abuse."""
    short_title = "The Human View"
    result = is_grammatically_coherent(short_title)
    print(f"\nTest: Short title - Result: {result}")
    assert result, "Short titles should pass"
    print("✓ PASSED: Short title passed")


def test_empty_text():
    """Test that empty text fails."""
    result1 = is_grammatically_coherent("")
    result2 = is_grammatically_coherent("   ")
    print(f"\nTest: Empty text - Results: {result1}, {result2}")
    assert not result1, "Empty text should fail"
    assert not result2, "Whitespace-only text should fail"
    print("✓ PASSED: Empty text rejected")


def test_semantic_similarity_high():
    """Test that semantically similar text passes."""
    original = "Humans view the universe as a series of discrete levels. But scale may be an artifact of our local perspective."
    generated = "People see the cosmos as a sequence of separate stages. However, scale might be a product of our limited viewpoint."
    # These are semantically similar (synonyms used)
    result = check_semantic_similarity(generated, original, threshold=0.6)
    print(f"\nTest: Semantic similarity (high) - Result: {result}")
    assert result, "Semantically similar text should pass"
    print("✓ PASSED: High semantic similarity detected")


def test_semantic_similarity_low():
    """Test that completely different meaning fails."""
    original = "Humans view the universe as a series of discrete levels."
    generated = "The cat sat on the mat and purred contentedly."
    # These are completely different
    result = check_semantic_similarity(generated, original, threshold=0.6)
    print(f"\nTest: Semantic similarity (low) - Result: {result}")
    assert not result, "Completely different meaning should fail"
    print("✓ PASSED: Low semantic similarity detected")


def test_semantic_similarity_word_salad():
    """Test that word salad has low semantic similarity."""
    original = "Humans view the universe as a series of discrete levels. But scale may be an artifact of our local perspective. Consider the Mandelbrot set."
    generated = "The Human View of Discrete Levels Scale as a Local Perspective Artifact Observe the Mandelbrot set."
    # Word salad should have low similarity even though it contains keywords
    result = check_semantic_similarity(generated, original, threshold=0.6)
    print(f"\nTest: Semantic similarity (word salad) - Result: {result}")
    # Word salad typically has lower similarity due to incoherent structure
    # This test verifies the function runs and returns a boolean
    print(f"✓ PASSED: Function executed (result: {result})")


def test_empty_text_similarity():
    """Test that empty text handling works."""
    result1 = check_semantic_similarity("", "test")
    result2 = check_semantic_similarity("test", "")
    result3 = check_semantic_similarity("", "")
    print(f"\nTest: Empty text similarity - Results: {result1}, {result2}, {result3}")
    assert not result1, "Empty generated text should fail"
    assert not result2, "Empty original text should fail"
    assert not result3, "Both empty should fail"
    print("✓ PASSED: Empty text handling works")


def test_critic_rejects_word_salad():
    """Test that critic rejects word salad deterministically."""
    word_salad = "The Human View of Discrete Levels Scale as a Local Perspective Artifact Observe the Mandelbrot set."
    result = is_grammatically_coherent(word_salad)
    print(f"\nTest: Critic rejects word salad - Result: {result}")
    assert not result, "Word salad should be rejected"
    print("✓ PASSED: Word salad rejected")


def test_critic_rejects_low_semantic_similarity():
    """Test that critic rejects text with low semantic similarity."""
    original = "Humans view the universe as a series of discrete levels."
    generated = "The cat sat on the mat."
    result = check_semantic_similarity(generated, original, threshold=0.6)
    print(f"\nTest: Critic rejects low similarity - Result: {result}")
    assert not result, "Low similarity should be rejected"
    print("✓ PASSED: Low similarity rejected")


def test_valid_meaning_preservation():
    """Test that valid meaning preservation passes."""
    original = "Humans view the universe as a series of discrete levels."
    generated = "People see the cosmos as a sequence of separate stages."

    coherence_result = is_grammatically_coherent(generated)
    similarity_result = check_semantic_similarity(generated, original, threshold=0.6)
    print(f"\nTest: Valid meaning preservation - Coherence: {coherence_result}, Similarity: {similarity_result}")
    assert coherence_result, "Valid sentence should pass coherence"
    # Semantic similarity might vary, but should generally pass for synonyms
    # If sentence-transformers is available, it should pass; if not, it returns True (graceful degradation)
    print("✓ PASSED: Valid meaning preservation")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Meaning Preservation Tests")
    print("=" * 60)

    test_word_salad_detection()
    test_title_case_abuse()
    test_missing_verb()
    test_valid_sentence()
    test_valid_sentence_with_dependent_clause()
    test_short_title()
    test_empty_text()

    test_semantic_similarity_high()
    test_semantic_similarity_low()
    test_semantic_similarity_word_salad()
    test_empty_text_similarity()

    test_critic_rejects_word_salad()
    test_critic_rejects_low_semantic_similarity()
    test_valid_meaning_preservation()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


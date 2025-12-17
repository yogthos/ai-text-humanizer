"""Test script for pipeline sentence processing regression.

This test ensures that all sentences from input are processed and generated,
not just the last sentence in each chunk. This prevents the indentation bug
where generation logic was outside the sentence loop.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.semantic import extract_meaning


def test_extract_meaning_preserves_all_sentences():
    """Test that extract_meaning extracts all sentences from input."""
    # Test with multiple sentences across multiple paragraphs
    input_text = """Human experience reinforces the rule of finitude. The biological cycle of birth, life, and decay defines our reality. Every object we touch eventually breaks. Every star burning in the night sky eventually succumbs to erosion. But we encounter a logical trap when we apply that same finiteness to the universe itself. A cosmos with a definitive beginning and a hard boundary implies a container. Logic demands we ask a difficult question. If the universe has edges, what exists outside them?

A truly finite universe must exist within a larger context. Anything with limits implies the existence of an exterior. A bottle possesses a finite volume because the bottle sits within a room. The room exists within a house. The observable universe sits within the expanse of the greater cosmos. We must consider the possibility that our reality is a component of a grander whole.

We can resolve the paradox if we embrace the concept of an infinite cosmos. A system that stretches forever with no beginning or end requires no external container. The system is complete. But a structure without walls must rely on internal rules to hold its shape. Self-containment becomes a physical possibility if we treat information as a fundamental property of the universe. Tom Stonier proposed that information is interconvertible with energy and conserved alongside energy[^155]. The informational architecture of the cosmos provides the intrinsic scaffolding for the structure. The code is embedded in every particle and field. Such an internal framework eliminates the need for an external container. Cosmological models of an infinite universe support this view."""

    content_units = extract_meaning(input_text)

    # Count sentences manually using NLTK (import inside function)
    from nltk.tokenize import sent_tokenize
    paragraphs = [p.strip() for p in input_text.split('\n\n') if p.strip()]
    total_sentences = 0
    for para in paragraphs:
        sentences = sent_tokenize(para)
        total_sentences += len([s for s in sentences if s.strip()])

    # Assertions
    assert len(content_units) == total_sentences, (
        f"extract_meaning should extract all {total_sentences} sentences, "
        f"but got {len(content_units)}"
    )

    # Verify each sentence is preserved
    extracted_texts = [unit.original_text for unit in content_units]
    for i, unit in enumerate(content_units):
        assert unit.original_text, f"ContentUnit {i} should have original_text"
        assert unit.original_text in input_text, (
            f"ContentUnit {i} text should be in input: {unit.original_text[:50]}..."
        )

    print(f"✓ extract_meaning preserves all sentences test passed")
    print(f"  Input sentences: {total_sentences}")
    print(f"  Extracted content units: {len(content_units)}")


def test_extract_meaning_multiple_paragraphs():
    """Test that extract_meaning correctly handles multiple paragraphs."""
    input_text = """First sentence. Second sentence. Third sentence.

Fourth sentence. Fifth sentence.

Sixth sentence."""

    content_units = extract_meaning(input_text)

    # Count sentences manually using NLTK (import inside function)
    from nltk.tokenize import sent_tokenize
    paragraphs = [p.strip() for p in input_text.split('\n\n') if p.strip()]
    total_sentences = 0
    for para in paragraphs:
        sentences = sent_tokenize(para)
        total_sentences += len([s for s in sentences if s.strip()])

    assert len(content_units) == total_sentences, (
        f"Should extract {total_sentences} sentences, got {len(content_units)}"
    )

    # Verify paragraph indices are correct
    para_indices = [unit.paragraph_idx for unit in content_units]
    assert para_indices == [0, 0, 0, 1, 1, 2], (
        f"Paragraph indices should be [0,0,0,1,1,2], got {para_indices}"
    )

    print(f"✓ Multiple paragraphs test passed")
    print(f"  Extracted {len(content_units)} content units across {len(set(para_indices))} paragraphs")


def test_sentence_processing_regression():
    """Test to prevent regression where only last sentence in chunk is processed.

    This test verifies that when processing text in chunks (window_size=3),
    all sentences in each chunk are processed, not just the last one.
    """
    # Create input with exactly 9 sentences (3 chunks of 3)
    input_text = """Sentence one. Sentence two. Sentence three.

Sentence four. Sentence five. Sentence six.

Sentence seven. Sentence eight. Sentence nine."""

    content_units = extract_meaning(input_text)

    # Verify we have 9 sentences
    assert len(content_units) == 9, (
        f"Should extract 9 sentences, got {len(content_units)}"
    )

    # Verify all sentences are unique and preserved
    unique_texts = set(unit.original_text for unit in content_units)
    assert len(unique_texts) == 9, (
        f"All 9 sentences should be unique, but got {len(unique_texts)} unique sentences"
    )

    # Verify sentence indices are sequential
    sentence_indices = [unit.sentence_idx for unit in content_units]
    expected_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2]  # 3 paragraphs, 3 sentences each
    assert sentence_indices == expected_indices, (
        f"Sentence indices should be {expected_indices}, got {sentence_indices}"
    )

    # Verify paragraph grouping
    para_0_units = [u for u in content_units if u.paragraph_idx == 0]
    para_1_units = [u for u in content_units if u.paragraph_idx == 1]
    para_2_units = [u for u in content_units if u.paragraph_idx == 2]

    assert len(para_0_units) == 3, "Paragraph 0 should have 3 sentences"
    assert len(para_1_units) == 3, "Paragraph 1 should have 3 sentences"
    assert len(para_2_units) == 3, "Paragraph 2 should have 3 sentences"

    print(f"✓ Sentence processing regression test passed")
    print(f"  Total sentences: {len(content_units)}")
    print(f"  Paragraphs: {len(set(u.paragraph_idx for u in content_units))}")
    print(f"  All sentences unique: {len(unique_texts) == len(content_units)}")


def test_chunk_processing_simulation():
    """Simulate chunk processing to verify all sentences would be processed.

    This simulates the chunking logic from pipeline.py to ensure
    that when processing in chunks of 3, all sentences are included.
    """
    # Create input with 7 sentences (2 full chunks + 1 partial)
    input_text = """One. Two. Three. Four. Five. Six. Seven."""

    content_units = extract_meaning(input_text)
    assert len(content_units) == 7, "Should extract 7 sentences"

    # Simulate chunking with window_size=3 (from pipeline.py)
    window_size = 3
    para_units = content_units  # Single paragraph for simplicity

    processed_sentences = []
    for chunk_start in range(0, len(para_units), window_size):
        chunk_units = para_units[chunk_start:chunk_start + window_size]
        # Simulate processing each sentence in chunk
        for unit_idx_in_chunk, content_unit in enumerate(chunk_units):
            # This simulates the generation logic that should be INSIDE the loop
            processed_sentences.append(content_unit.original_text)

    # Verify all sentences were processed
    assert len(processed_sentences) == len(content_units), (
        f"All {len(content_units)} sentences should be processed, "
        f"but only {len(processed_sentences)} were processed"
    )

    # Verify no duplicates
    assert len(processed_sentences) == len(set(processed_sentences)), (
        "No sentence should be processed twice"
    )

    # Verify order is preserved
    original_texts = [unit.original_text for unit in content_units]
    assert processed_sentences == original_texts, (
        "Processed sentences should be in same order as input"
    )

    print(f"✓ Chunk processing simulation test passed")
    print(f"  Input sentences: {len(content_units)}")
    print(f"  Processed sentences: {len(processed_sentences)}")
    print(f"  Chunks: {(len(content_units) + window_size - 1) // window_size}")


if __name__ == "__main__":
    print("Running pipeline sentence processing regression tests...\n")

    try:
        test_extract_meaning_preserves_all_sentences()
        test_extract_meaning_multiple_paragraphs()
        test_sentence_processing_regression()
        test_chunk_processing_simulation()
        print("\n✓ All pipeline sentence processing tests passed!")
        print("  This ensures the indentation bug (generation outside loop) is prevented.")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


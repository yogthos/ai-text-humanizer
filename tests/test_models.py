"""Test script to verify that ContentUnit model works correctly."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import ContentUnit


def test_content_unit():
    """Test that ContentUnit can be instantiated and holds values correctly."""
    # Create dummy data
    svo_triples = [("fox", "jump", "dog"), ("cat", "chase", "mouse")]
    entities = ["John", "London"]
    original_text = "The quick brown fox jumps over the dog."
    content_words = ["quick", "brown", "fox", "jumps", "dog"]

    # Instantiate ContentUnit
    content = ContentUnit(
        svo_triples=svo_triples,
        entities=entities,
        original_text=original_text,
        content_words=content_words,
        paragraph_idx=0,
        sentence_idx=0,
        is_first_paragraph=True,
        is_last_paragraph=False,
        is_first_sentence=True,
        is_last_sentence=False,
        total_paragraphs=1,
        paragraph_length=1
    )

    # Assertions
    assert isinstance(content, ContentUnit)
    assert content.svo_triples == svo_triples
    assert isinstance(content.svo_triples, list)
    assert isinstance(content.svo_triples[0], tuple)
    assert len(content.svo_triples[0]) == 3
    assert content.svo_triples[0] == ("fox", "jump", "dog")

    assert content.entities == entities
    assert isinstance(content.entities, list)
    assert content.entities[0] == "John"

    assert content.original_text == original_text
    assert isinstance(content.original_text, str)

    assert content.content_words == content_words
    assert isinstance(content.content_words, list)
    assert len(content.content_words) > 0

    # Test positional metadata
    assert content.paragraph_idx == 0
    assert content.sentence_idx == 0
    assert content.is_first_paragraph is True
    assert content.is_first_sentence is True

    print("✓ ContentUnit test passed")


if __name__ == "__main__":
    print("Running model verification tests...\n")
    test_content_unit()
    print("\n✓ All tests passed!")

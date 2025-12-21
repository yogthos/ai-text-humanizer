"""Text processing utilities for style transfer pipeline."""

import re
from typing import List

def check_zipper_merge(prev_sent: str, new_sent: str) -> bool:
    """
    Returns True if a 'Stitch Glitch' (overlap/repetition) is detected.

    Detects three types of echo:
    - Full Echo: New sentence starts with entire previous sentence
    - Head Echo: Both sentences start with same 2+ words
    - Tail Echo: End of previous sentence matches start of new sentence (word overlap)

    Args:
        prev_sent: Previous sentence (or empty string)
        new_sent: Newly generated sentence

    Returns:
        True if echo/repetition detected, False otherwise
    """
    if not prev_sent or not new_sent:
        return False

    # Normalize to lowercase for comparison
    p_clean = prev_sent.strip().lower()
    n_clean = new_sent.strip().lower()

    if not p_clean or not n_clean:
        return False

    # Strip punctuation from words for better matching
    def strip_punctuation(word: str) -> str:
        """Remove punctuation from word."""
        return re.sub(r'[^\w\s]', '', word)

    # 1. Full Echo (New starts with Prev)
    if n_clean.startswith(p_clean):
        return True

    # 2. Head Echo (Both start with same 2+ words, ignoring punctuation)
    p_words = [strip_punctuation(w) for w in p_clean.split()]
    n_words = [strip_punctuation(w) for w in n_clean.split()]

    # Remove empty strings from punctuation stripping
    p_words = [w for w in p_words if w]
    n_words = [w for w in n_words if w]

    if len(p_words) >= 2 and len(n_words) >= 2:
        # Check if first 2 words match
        if p_words[:2] == n_words[:2]:
            return True
        # Also check first 3 words if both sentences have at least 3 words
        if len(p_words) >= 3 and len(n_words) >= 3:
            if p_words[:3] == n_words[:3]:
                return True

    # 3. Tail Echo (End of Prev matches Start of New)
    # Check if the start of new sentence overlaps with the end of previous sentence
    if len(n_words) >= 2 and len(p_words) >= 2:
        # Get first 2-3 words of new sentence (normalized, no punctuation)
        new_start_words = n_words[:min(3, len(n_words))]
        # Get last 4-6 words of previous sentence (normalized, no punctuation)
        prev_tail_words = p_words[-min(6, len(p_words)):]

        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "was", "are", "were", "it", "this", "that", "these", "those"}

        # Strong signal: First word of new appears as last word of prev (and is not common)
        # This catches cases like "The doors opened." -> "Opened, the room..."
        if len(p_words) >= 1 and n_words[0] == p_words[-1] and n_words[0] not in common_words:
            return True

        # Strong signal: First word of new appears in last 2 words of prev (and is not common)
        if n_words[0] in prev_tail_words[-2:] and n_words[0] not in common_words:
            # Additional check: make sure it's not just a coincidence
            # Require that at least one more word from new_start_words appears in prev_tail_words
            if len(new_start_words) >= 2:
                overlap_count = sum(1 for w in new_start_words if w in prev_tail_words)
                if overlap_count >= 2:
                    return True

        # Check for consecutive word overlap at the END of previous sentence
        # Look for 2+ consecutive words from start of new appearing at the end of prev
        # This catches cases where the end of prev sentence is repeated at start of new
        for i in range(len(new_start_words) - 1):
            bigram = (new_start_words[i], new_start_words[i + 1])
            # Check if this bigram appears at the END of prev_tail_words (last 3 words)
            # This ensures we're detecting tail echo, not just general similarity
            tail_end = prev_tail_words[-3:] if len(prev_tail_words) >= 3 else prev_tail_words
            for j in range(len(tail_end) - 1):
                if (tail_end[j], tail_end[j + 1]) == bigram:
                    return True

    return False


def parse_variants_from_response(response: str, verbose: bool = False) -> List[str]:
    """
    Parse variants from LLM response using robust multi-format regex.

    Handles multiple output formats:
    - VAR: prefix (case insensitive)
    - Numbered lists: 1. or 1)
    - Bullet points: - or *
    - Sentence-like lines (fallback)
    - Entire response (final fallback)

    Args:
        response: LLM response text containing variants
        verbose: Enable verbose output

    Returns:
        List of parsed variant strings
    """
    if not response or not response.strip():
        return []

    variants = []
    lines = response.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1. Check for VAR: prefix (case insensitive) - handle with or without space
        line_upper = line.upper()
        if line_upper.startswith("VAR:"):
            # Strip "VAR:" prefix (4 characters) and any following whitespace
            clean_text = line[4:].lstrip()
            if clean_text:
                variants.append(clean_text)
            continue

        # 2. Check for numbered lists: 1. or 1)
        numbered_match = re.match(r'^(\d+[\.\)])\s+(.*)', line)
        if numbered_match:
            clean_text = numbered_match.group(2).strip()
            if clean_text:
                variants.append(clean_text)
            continue

        # 3. Check for bullet points: - or *
        if line.startswith("- ") or line.startswith("* "):
            clean_text = line[2:].strip()
            if clean_text:
                variants.append(clean_text)
            continue

        # Secondary parsing: Check for sentence-like lines (fallback)
        # Skip chatter lines
        line_lower = line.lower()
        if (line_lower.startswith("here is") or
            line_lower.startswith("generating") or
            line_lower.startswith("variant") or
            line_lower.startswith("option")):
            continue

        # If line looks like a sentence (starts with capital, ends with punctuation)
        # and isn't chatter, treat as variant
        if (len(line) > 10 and  # Reasonable length
            line[0].isupper() and  # Starts with capital
            line[-1] in ".!?\"'"):  # Ends with punctuation
            variants.append(line)

    # Tertiary fallback: If no variants found at all, treat entire response as single variant
    if not variants and response.strip():
        # Strip quotes and clean up
        clean_response = response.strip().strip('"').strip("'")
        if clean_response:
            variants.append(clean_response)

    # Final safety check: Strip any remaining "VAR:" prefixes that might have slipped through
    cleaned_variants = []
    for variant in variants:
        # Remove "VAR:" prefix if present (case insensitive, with or without space)
        variant_upper = variant.upper()
        if variant_upper.startswith("VAR:"):
            variant = variant[4:].lstrip()
        cleaned_variants.append(variant)

    return cleaned_variants


#!/usr/bin/env -S python3
"""Augment Lovecraft training data with conceptual examples and sample-based snowflakes.

Two augmentation modes:

1. conceptual: Extract conceptual/explanatory passages from the Lovecraft corpus,
   neutralize them via RTT, and format with conceptual persona frames. This fixes
   the 98%/2% narrative/conceptual imbalance in existing training data.

2. snowflake-samples: Use data/samples/ chapters (philosophical/scientific content)
   as topic sources for snowflake variations. Takes Lovecraft corpus passages and
   rewrites them to be about the sample topics, teaching the model to apply
   Lovecraft's style to academic/explanatory content.

Usage:
    # Generate conceptual training examples
    python scripts/augment_training_data.py conceptual \
        --corpus data/corpus/curated/lovecraft.txt \
        --author "H.P. Lovecraft" \
        --output data/training/lovecraft/LlamaFactory/conceptual.jsonl

    # Generate snowflake variations from samples
    python scripts/augment_training_data.py snowflake-samples \
        --corpus data/corpus/curated/lovecraft.txt \
        --samples-dir data/samples/ \
        --author "H.P. Lovecraft" \
        --output data/training/lovecraft/LlamaFactory/snowflake_samples.jsonl

    # Merge augmented data into main training file
    python scripts/augment_training_data.py merge \
        --main data/training/lovecraft/LlamaFactory/train_sft.jsonl \
        --extra data/training/lovecraft/LlamaFactory/conceptual.jsonl \
        --extra data/training/lovecraft/LlamaFactory/snowflake_samples.jsonl
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_flat_training import (
    call_deepseek,
    classify_content_type,
    format_training_example,
    neutralize_batch,
    split_into_sentences,
    validate_variation,
    ContentType,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Extract topics from sample chapters
# =============================================================================

def extract_sample_topics(samples_dir: Path, min_words: int = 60, max_words: int = 300) -> List[str]:
    """Extract topic paragraphs from data/samples/ markdown chapters.

    Returns paragraphs suitable for use as snowflake topic descriptions.
    Strips markdown formatting, poem blocks, and headers.
    """
    topics = []
    for md_file in sorted(samples_dir.glob("*.markdown")):
        text = md_file.read_text(encoding="utf-8")

        # Remove poem blocks
        text = re.sub(r'<div class="poem">.*?</div>', "", text, flags=re.DOTALL)
        # Remove footnote references
        text = re.sub(r'\[\^\d+\]', "", text)
        # Remove markdown headers
        text = re.sub(r"^#{1,6}\s+.*$", "", text, flags=re.MULTILINE)
        # Remove horizontal rules
        text = re.sub(r"^---+\s*$", "", text, flags=re.MULTILINE)

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            words = para.split()
            if min_words <= len(words) <= max_words:
                topics.append(para)

    logger.info(f"Extracted {len(topics)} topic paragraphs from {samples_dir}")
    return topics


def create_sample_snowflake(
    lovecraft_passage: str,
    sample_topic: str,
    author: str,
    max_attempts: int = 2,
) -> Optional[str]:
    """Create a snowflake variation using a sample topic paragraph.

    Unlike mundane topic snowflakes (which use short topic labels like
    "making toast"), these use full paragraphs from data/samples/ as the
    topic source, producing Lovecraft-styled versions of philosophical/
    scientific content.
    """
    # Summarize the sample topic to a concise description
    topic_summary = sample_topic[:200].replace("\n", " ")
    if len(sample_topic) > 200:
        topic_summary += "..."

    system = f"""You are a literary style transfer assistant specializing in {author}'s writing style.

Your task: Rewrite the given passage to be about the provided academic/philosophical topic while preserving:
- The EXACT sentence structure (same number of sentences, same clause patterns)
- The author's characteristic rhythm, cadence, and vocabulary
- Similar punctuation patterns (semicolons, dashes, parentheticals)
- The author's distinctive voice and tone

The goal is to demonstrate that {author}'s STYLE transforms even academic content into something distinctive."""

    n_sentences = len(split_into_sentences(lovecraft_passage))
    n_words = len(lovecraft_passage.split())

    prompt = f"""Rewrite this passage by {author} to be about the following topic.

TOPIC:
{sample_topic}

ORIGINAL PASSAGE BY {author.upper()}:
{lovecraft_passage}

Requirements:
1. The new passage must convey the ideas from the TOPIC above
2. Preserve the EXACT sentence structure: {n_sentences} sentences, same clause patterns
3. Use {author}'s distinctive vocabulary and phrasing style
4. Match the word count closely (~{n_words} words)
5. Keep the same punctuation patterns and rhythm
6. The result should sound unmistakably like {author} explaining these concepts

Output only the rewritten passage, nothing else."""

    for attempt in range(max_attempts):
        try:
            varied = call_deepseek(prompt, system, max_retries=2)
            varied = varied.strip('`"\' \n')
            if varied.startswith("```"):
                varied = re.sub(r"^```\w*\n?", "", varied)
                varied = re.sub(r"\n?```$", "", varied)

            is_valid, reason = validate_variation(lovecraft_passage, varied)
            if is_valid:
                return varied
            else:
                logger.debug(f"Sample snowflake rejected: {reason}")
        except Exception as e:
            logger.debug(f"Sample snowflake attempt {attempt + 1} failed: {e}")

    return None


# =============================================================================
# Conceptual passage extraction
# =============================================================================

def extract_conceptual_passages(corpus_path: Path, max_passages: int = 200) -> List[str]:
    """Extract passages from the corpus that are classified as conceptual.

    These are passages that explain mechanisms, describe systems, present
    arguments, or discuss abstract ideas — as opposed to narrative passages
    about events and characters.
    """
    text = corpus_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    conceptual = []
    for para in paragraphs:
        words = len(para.split())
        if words < 100 or words > 500:
            continue
        content_type = classify_content_type(para)
        if content_type == ContentType.CONCEPTUAL:
            conceptual.append(para)

    logger.info(f"Found {len(conceptual)} conceptual passages in corpus (out of {len(paragraphs)} total)")

    if len(conceptual) > max_passages:
        random.shuffle(conceptual)
        conceptual = conceptual[:max_passages]
        logger.info(f"Sampled {max_passages} for processing")

    return conceptual


# =============================================================================
# Generation pipeline
# =============================================================================

def generate_conceptual_entries(
    corpus_path: Path,
    author: str,
    max_passages: int = 200,
    output_format: str = "llama_factory",
) -> List[dict]:
    """Generate training entries from conceptual passages in the corpus.

    Pipeline:
    1. Extract conceptual passages from corpus
    2. Neutralize via RTT (batch)
    3. Format as training pairs with conceptual persona frames
    """
    passages = extract_conceptual_passages(corpus_path, max_passages)
    if not passages:
        logger.warning("No conceptual passages found!")
        return []

    # Batch neutralize
    logger.info(f"Neutralizing {len(passages)} conceptual passages via RTT...")
    neutralized = neutralize_batch(passages, monotone=True,
                                   on_progress=lambda done, total: logger.info(f"RTT: {done}/{total}") if done % 20 == 0 else None)

    entries = []
    skipped = 0
    for passage, neutral in zip(passages, neutralized):
        if neutral is None:
            skipped += 1
            continue

        word_count = len(passage.split())
        entry = format_training_example(
            neutral_text=neutral,
            styled_text=passage,
            author=author,
            word_count=word_count,
            variation_type="original",
            output_format=output_format,
        )
        entries.append(entry)

        # Also generate a robustness variant (heavy perturbation)
        robust_entry = format_training_example(
            neutral_text=neutral,
            styled_text=passage,
            author=author,
            word_count=word_count,
            variation_type="robustness",
            output_format=output_format,
        )
        entries.append(robust_entry)

    logger.info(f"Generated {len(entries)} conceptual entries ({skipped} RTT failures)")
    return entries


def generate_sample_snowflakes(
    corpus_path: Path,
    samples_dir: Path,
    author: str,
    max_pairs: int = 200,
    workers: int = 4,
    output_format: str = "llama_factory",
) -> List[dict]:
    """Generate snowflake training entries using data/samples/ as topic sources.

    Pipeline:
    1. Extract topic paragraphs from samples
    2. Extract Lovecraft passages from corpus (prefer conceptual for better coverage)
    3. Create snowflake variations (Lovecraft structure + sample topic)
    4. Neutralize the snowflake variations via RTT
    5. Format as training pairs
    """
    # Extract topics from samples
    sample_topics = extract_sample_topics(samples_dir)
    if not sample_topics:
        logger.warning("No topics found in samples directory!")
        return []

    # Extract corpus passages (mix of narrative and conceptual)
    text = corpus_path.read_text(encoding="utf-8")
    all_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    eligible = [p for p in all_paragraphs if 100 <= len(p.split()) <= 400]
    random.shuffle(eligible)
    corpus_passages = eligible[:max_pairs]

    logger.info(f"Creating snowflakes: {len(corpus_passages)} corpus passages x {len(sample_topics)} topics")

    # Pair each corpus passage with a random sample topic
    pairs = []
    for passage in corpus_passages:
        topic = random.choice(sample_topics)
        pairs.append((passage, topic))

    # Generate snowflake variations in parallel
    snowflakes = []  # (snowflake_text, original_passage)
    failed = 0
    start_time = time.time()

    def process_pair(pair_idx):
        passage, topic = pairs[pair_idx]
        result = create_sample_snowflake(passage, topic, author)
        return pair_idx, passage, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_pair, i): i for i in range(len(pairs))}

        for future in as_completed(futures):
            try:
                idx, original, snowflake = future.result()
                if snowflake:
                    snowflakes.append((snowflake, original))
                else:
                    failed += 1

                done = len(snowflakes) + failed
                if done % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Snowflakes: {len(snowflakes)}/{len(pairs)} | "
                        f"Failed: {failed} | Rate: {rate:.1f}/s"
                    )
            except Exception as e:
                failed += 1
                logger.debug(f"Snowflake task failed: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Snowflake generation: {len(snowflakes)} created, {failed} failed in {elapsed:.1f}s")

    if not snowflakes:
        return []

    # Batch neutralize the snowflake texts
    snowflake_texts = [s for s, _ in snowflakes]
    logger.info(f"Neutralizing {len(snowflake_texts)} snowflakes via RTT...")
    neutralized = neutralize_batch(snowflake_texts, monotone=True,
                                   on_progress=lambda done, total: logger.info(f"RTT: {done}/{total}") if done % 20 == 0 else None)

    entries = []
    skipped = 0
    for (snowflake_text, _), neutral in zip(snowflakes, neutralized):
        if neutral is None:
            skipped += 1
            continue

        word_count = len(snowflake_text.split())
        entry = format_training_example(
            neutral_text=neutral,
            styled_text=snowflake_text,
            author=author,
            word_count=word_count,
            variation_type="snowflake",
            output_format=output_format,
        )
        entries.append(entry)

    logger.info(f"Generated {len(entries)} snowflake entries ({skipped} RTT failures)")
    return entries


def generate_blended_entries(
    blended_json_path: Path,
    author: str,
    max_entries: int = 300,
    output_format: str = "llama_factory",
) -> List[dict]:
    """Generate training entries from blended secondary-author paragraphs.

    Takes the output of blend_corpuses.py (compatible paragraphs from Sagan,
    Feynman, Mao, etc.) and creates training pairs. These paragraphs are
    conceptual/explanatory in nature and provide diversity beyond Lovecraft's
    primarily narrative corpus.

    The output uses the author's conceptual persona frames so the LoRA learns
    to apply its style to explanatory content.

    Pipeline:
    1. Load blended paragraphs JSON (from blend_corpuses.py output)
    2. Select top entries by similarity score
    3. Neutralize via RTT (batch)
    4. Format as training pairs with conceptual persona frames
    """
    with open(blended_json_path) as f:
        all_entries = json.load(f)

    # Sort by similarity descending, take top N
    all_entries.sort(key=lambda x: -x["similarity"])
    selected = all_entries[:max_entries]

    source_counts = {}
    for e in selected:
        source_counts[e["source"]] = source_counts.get(e["source"], 0) + 1
    logger.info(f"Selected {len(selected)} blended paragraphs: {source_counts}")

    passages = [e["text"] for e in selected]

    # Batch neutralize
    logger.info(f"Neutralizing {len(passages)} blended paragraphs via RTT...")
    neutralized = neutralize_batch(
        passages,
        monotone=True,
        on_progress=lambda done, total: (
            logger.info(f"RTT: {done}/{total}") if done % 20 == 0 else None
        ),
    )

    entries = []
    skipped = 0
    for passage, neutral in zip(passages, neutralized):
        if neutral is None:
            skipped += 1
            continue

        word_count = len(passage.split())

        # Standard entry with conceptual persona frame
        entry = format_training_example(
            neutral_text=neutral,
            styled_text=passage,
            author=author,
            word_count=word_count,
            variation_type="original",
            output_format=output_format,
        )
        entries.append(entry)

        # Robustness variant (heavy perturbation input)
        robust_entry = format_training_example(
            neutral_text=neutral,
            styled_text=passage,
            author=author,
            word_count=word_count,
            variation_type="robustness",
            output_format=output_format,
        )
        entries.append(robust_entry)

    logger.info(f"Generated {len(entries)} blended entries ({skipped} RTT failures)")
    return entries


def merge_training_files(main_path: Path, extra_paths: List[Path]):
    """Merge extra JSONL files into the main training file."""
    # Read existing main file
    with open(main_path) as f:
        existing = [json.loads(line) for line in f]

    logger.info(f"Main file: {len(existing)} entries")

    added = 0
    for extra_path in extra_paths:
        if not extra_path.exists():
            logger.warning(f"File not found: {extra_path}")
            continue
        with open(extra_path) as f:
            new_entries = [json.loads(line) for line in f]
        existing.extend(new_entries)
        added += len(new_entries)
        logger.info(f"  + {len(new_entries)} from {extra_path.name}")

    # Shuffle to mix conceptual entries throughout
    random.shuffle(existing)

    with open(main_path, "w") as f:
        for entry in existing:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Merged: {len(existing)} total entries ({added} new)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Augment training data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # conceptual subcommand
    p_conceptual = subparsers.add_parser("conceptual", help="Generate conceptual training examples")
    p_conceptual.add_argument("--corpus", type=Path, required=True)
    p_conceptual.add_argument("--author", default="H.P. Lovecraft")
    p_conceptual.add_argument("--output", type=Path, required=True)
    p_conceptual.add_argument("--max-passages", type=int, default=200)
    p_conceptual.add_argument("--format", choices=["llama_factory", "mlx"], default="llama_factory")

    # snowflake-samples subcommand
    p_snowflake = subparsers.add_parser("snowflake-samples", help="Generate snowflakes from sample chapters")
    p_snowflake.add_argument("--corpus", type=Path, required=True)
    p_snowflake.add_argument("--samples-dir", type=Path, required=True)
    p_snowflake.add_argument("--author", default="H.P. Lovecraft")
    p_snowflake.add_argument("--output", type=Path, required=True)
    p_snowflake.add_argument("--max-pairs", type=int, default=200)
    p_snowflake.add_argument("--workers", type=int, default=4)
    p_snowflake.add_argument("--format", choices=["llama_factory", "mlx"], default="llama_factory")

    # blended subcommand
    p_blended = subparsers.add_parser("blended", help="Generate entries from blended secondary-author paragraphs")
    p_blended.add_argument("--blended-json", type=Path, required=True,
                           help="Path to blended paragraphs JSON (from blend_corpuses.py)")
    p_blended.add_argument("--author", default="H.P. Lovecraft")
    p_blended.add_argument("--output", type=Path, required=True)
    p_blended.add_argument("--max-entries", type=int, default=300)
    p_blended.add_argument("--format", choices=["llama_factory", "mlx"], default="llama_factory")

    # merge subcommand
    p_merge = subparsers.add_parser("merge", help="Merge extra JSONL into main training file")
    p_merge.add_argument("--main", type=Path, required=True)
    p_merge.add_argument("--extra", type=Path, action="append", required=True)

    args = parser.parse_args()

    if args.command == "conceptual":
        entries = generate_conceptual_entries(
            corpus_path=args.corpus,
            author=args.author,
            max_passages=args.max_passages,
            output_format=args.format,
        )
        with open(args.output, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(entries)} entries to {args.output}")

    elif args.command == "snowflake-samples":
        entries = generate_sample_snowflakes(
            corpus_path=args.corpus,
            samples_dir=args.samples_dir,
            author=args.author,
            max_pairs=args.max_pairs,
            workers=args.workers,
            output_format=args.format,
        )
        with open(args.output, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(entries)} entries to {args.output}")

    elif args.command == "blended":
        entries = generate_blended_entries(
            blended_json_path=args.blended_json,
            author=args.author,
            max_entries=args.max_entries,
            output_format=args.format,
        )
        with open(args.output, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(entries)} entries to {args.output}")

    elif args.command == "merge":
        merge_training_files(args.main, args.extra)


if __name__ == "__main__":
    main()

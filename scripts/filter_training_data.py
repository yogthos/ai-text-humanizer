#!/usr/bin/env python3
"""Filter training data to remove entries with bad input/output word ratios.

Removes entries where the output is significantly longer than the input,
which teaches the model to hallucinate/expand from minimal input.
"""

import argparse
import json
import sys
from pathlib import Path


def word_count(text: str) -> int:
    return len(text.split())


def filter_entries(input_path: Path, output_path: Path, max_ratio: float = 2.0, min_input_words: int = 15):
    """Filter training entries by input/output word ratio and minimum input length."""
    kept = []
    removed = []

    with open(input_path) as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            inp_words = word_count(entry["input"])
            out_words = word_count(entry["output"])
            ratio = out_words / max(inp_words, 1)

            if ratio > max_ratio or inp_words < min_input_words:
                removed.append((i + 1, inp_words, out_words, ratio))
            else:
                kept.append(entry)

    # Write filtered output
    with open(output_path, "w") as f:
        for entry in kept:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return kept, removed


def main():
    parser = argparse.ArgumentParser(description="Filter training data by word ratio")
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSONL file (default: overwrite input)")
    parser.add_argument("--max-ratio", type=float, default=2.0, help="Max output/input word ratio (default: 2.0)")
    parser.add_argument("--min-input-words", type=int, default=15, help="Minimum input word count (default: 15)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without writing")
    args = parser.parse_args()

    output_path = args.output or args.input

    print(f"Filtering {args.input}")
    print(f"  Max ratio: {args.max_ratio}x, Min input words: {args.min_input_words}")

    kept, removed = filter_entries(args.input, output_path if not args.dry_run else "/dev/null",
                                   args.max_ratio, args.min_input_words)

    print(f"\nResults:")
    print(f"  Kept:    {len(kept)}")
    print(f"  Removed: {len(removed)}")

    if removed:
        print(f"\nRemoved entries (worst first):")
        removed.sort(key=lambda x: -x[3])
        for line_num, inp_w, out_w, ratio in removed[:20]:
            print(f"  Line {line_num}: {inp_w} -> {out_w} words ({ratio:.1f}x)")
        if len(removed) > 20:
            print(f"  ... and {len(removed) - 20} more")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")
    else:
        print(f"\nWritten to: {output_path}")


if __name__ == "__main__":
    main()

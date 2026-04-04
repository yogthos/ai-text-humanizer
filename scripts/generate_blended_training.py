#!/usr/bin/env python3
"""Generate training data for a blended author style (e.g., "Howard Russell").

Takes blended paragraphs produced by experiment_blend_sentences.py, neutralizes
them via RTT, and outputs JSONL training data compatible with generate_flat_training.py.

Pipeline:
1. Load blended paragraphs (from JSON output of blending experiment)
2. RTT-neutralize each paragraph to create training inputs
3. Apply persona frames from the blended worldview file
4. Apply perturbation to match training distribution
5. Output JSONL in LLaMA-Factory format

Usage:
    python scripts/generate_blended_training.py \
        --blended data/blended/russell_lovecraft_mood_v4.json \
        --author "Howard Russell" \
        --output data/training/howard_russell \
        --format llama_factory
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional

import requests

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
for _lib in ("httpx", "httpcore", "urllib3", "sentence_transformers",
             "transformers", "huggingface_hub"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# Persona Frames (loaded from worldview file)
# =============================================================================

def load_persona_frames(worldview_path: Path) -> dict:
    """Load persona frames from a worldview file.

    Returns dict with 'narrative' and 'conceptual' lists.
    """
    text = worldview_path.read_text(encoding="utf-8")

    frames = {"narrative": [], "conceptual": []}
    current_section = None

    for line in text.split("\n"):
        line = line.strip()
        if line == "[PERSONA_FRAMES_NARRATIVE]":
            current_section = "narrative"
            continue
        elif line == "[PERSONA_FRAMES_CONCEPTUAL]":
            current_section = "conceptual"
            continue
        elif line == "---":
            continue
        elif line and current_section:
            frames[current_section].append(line)

    logger.info(f"Loaded {len(frames['narrative'])} narrative + "
                f"{len(frames['conceptual'])} conceptual frames")
    return frames


# =============================================================================
# Content Classification (reuse from training pipeline)
# =============================================================================

def classify_content_type(text: str) -> str:
    """Classify text as 'narrative' or 'conceptual'.

    Simple heuristic: narrative has more past-tense verbs, proper nouns,
    temporal markers. Conceptual has more present-tense, abstract nouns,
    logical connectors.
    """
    narrative_markers = [
        r'\b(was|were|had|came|went|saw|found|heard|felt|said|told)\b',
        r'\b(then|suddenly|afterward|meanwhile|finally)\b',
        r'\b(I|he|she|they|we)\s+(was|were|had|did|could|would)\b',
    ]
    conceptual_markers = [
        r'\b(is|are|has|does|can|must|should|would|may|might)\b',
        r'\b(therefore|however|moreover|nevertheless|consequently)\b',
        r'\b(if|whether|unless|although|because|since|while)\b',
        r'\b(concept|principle|theory|argument|proposition|question)\b',
    ]

    narrative_score = sum(
        len(re.findall(p, text, re.IGNORECASE))
        for p in narrative_markers
    )
    conceptual_score = sum(
        len(re.findall(p, text, re.IGNORECASE))
        for p in conceptual_markers
    )

    return "narrative" if narrative_score > conceptual_score else "conceptual"


# =============================================================================
# Constraints (same as generate_flat_training.py)
# =============================================================================

ALWAYS_CONSTRAINTS = [
    "Do not use: 'Moreover', 'Furthermore', 'Therefore', 'Thus', 'Hence', "
    "'In conclusion', 'It is important to note', 'It is worth noting', "
    "'This highlights', 'This underscores', 'In essence', 'Ultimately'.",
    "Do not hedge. Avoid: 'arguably', 'it could be said', 'one might argue', "
    "'perhaps it is', 'it seems that'. State things directly.",
]

FREQUENT_CONSTRAINTS = [
    "Do not start with a topic sentence. Start with a sensory detail, a question, or mid-thought.",
    "Do not use numbered lists or 'Firstly/Secondly/Thirdly' structures.",
]

ROTATING_CONSTRAINTS = [
    "Use fragments. Interrupt yourself with dashes (\u2014).",
    "Let ideas collide without transition words.",
    "Do not explain. Imply.",
    "Use at least one rhetorical question.",
    "Interrupt yourself with a parenthetical thought.",
    "Start the paragraph with a conjunction (But, And, Yet, So).",
    "Be biased. Be opinionated. Do not balance your argument.",
    "Vary sentence lengths dramatically. Follow a long sentence with a short one.",
    "Use concrete nouns instead of abstractions. Not 'the concept' but the thing itself.",
    "End on an image or action, not a summary.",
]


def build_instruction(
    persona_frame: str,
    word_count: int,
    styled_text: str = None,
) -> str:
    """Build a training instruction with persona frame and constraints."""
    instruction = f"{persona_frame}\n\nWrite approximately {word_count} words."

    # Skeleton 50% of the time
    if styled_text and random.random() < 0.50:
        skeleton = extract_simple_skeleton(styled_text)
        if skeleton:
            instruction = f"{instruction}\n\nFollow this structure: {skeleton}"

    # Constraints
    constraints = list(ALWAYS_CONSTRAINTS)
    for c in FREQUENT_CONSTRAINTS:
        if random.random() < 0.70:
            constraints.append(c)
    if random.random() < 0.40:
        constraints.append(random.choice(ROTATING_CONSTRAINTS))

    constraints_text = "\n".join(f"[CONSTRAINT]: {c}" for c in constraints)
    return f"{instruction}\n\n{constraints_text}"


def extract_simple_skeleton(text: str) -> Optional[str]:
    """Extract a simple rhetorical skeleton from text."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 3:
        return None

    moves = []
    for s in sentences[:6]:  # first 6 sentences max
        s_lower = s.lower()
        if any(w in s_lower for w in ["but", "however", "yet", "although"]):
            moves.append("[Counter-argument]")
        elif any(w in s_lower for w in ["because", "since", "for this reason"]):
            moves.append("[Causal reasoning]")
        elif "?" in s:
            moves.append("[Question]")
        elif any(w in s_lower for w in ["i saw", "i heard", "i felt", "there was", "there came"]):
            moves.append("[Observation]")
        elif any(w in s_lower for w in ["consider", "suppose", "imagine", "let us"]):
            moves.append("[Thought experiment]")
        else:
            moves.append("[Claim]")

    # Deduplicate consecutive identical moves
    deduped = [moves[0]]
    for m in moves[1:]:
        if m != deduped[-1]:
            deduped.append(m)

    if len(deduped) < 2:
        return None

    return " \u2192 ".join(deduped)


# =============================================================================
# Perturbation (simplified from generate_flat_training.py)
# =============================================================================

SYNONYMS = {
    "big": ["large", "huge"], "small": ["little", "tiny"],
    "good": ["fine", "decent"], "bad": ["poor", "terrible"],
    "said": ["stated", "remarked"], "think": ["believe", "consider"],
    "show": ["reveal", "demonstrate"], "give": ["provide", "offer"],
    "make": ["create", "produce"], "use": ["employ", "utilize"],
    "see": ["observe", "notice"], "know": ["understand", "realize"],
    "very": ["quite", "extremely"], "really": ["truly", "genuinely"],
}


def perturb_text(text: str, rate: float = 0.08) -> str:
    """Apply random perturbations to text."""
    words = text.split()
    result = []
    droppable = {"the", "a", "an", "very", "really", "just", "quite"}

    for word in words:
        if random.random() > rate:
            result.append(word)
            continue

        word_lower = word.lower().rstrip(".,!?;:")
        choice = random.random()

        if choice < 0.4 and word_lower in SYNONYMS:
            syn = random.choice(SYNONYMS[word_lower])
            if word[0].isupper():
                syn = syn.capitalize()
            result.append(syn + word[len(word_lower):])
        elif choice < 0.7 and word.lower() in droppable:
            pass
        elif len(word) > 3:
            i = random.randint(1, len(word) - 2)
            word = word[:i] + word[i + 1] + word[i] + word[i + 2:]
            result.append(word)
        else:
            result.append(word)

    return " ".join(result)


# =============================================================================
# RTT Neutralization
# =============================================================================

def call_deepseek(prompt: str, system: str = "", max_retries: int = 3) -> str:
    """Call DeepSeek API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def neutralize_text(text: str) -> Optional[str]:
    """Neutralize text via LLM paraphrase (simpler than full RTT for experiments).

    Strips style while preserving all facts and structure.
    """
    system = """You are a text neutralizer. Rewrite the input in plain, neutral English.
Preserve ALL facts, arguments, and logical structure.
Remove all stylistic flourishes, atmospheric language, and distinctive vocabulary.
Use simple, clear sentences. Do not add or remove information.
Output only the neutralized text."""

    prompt = f"Neutralize this text:\n\n{text}"
    return call_deepseek(prompt, system)


# =============================================================================
# Main Pipeline
# =============================================================================

def generate_training_data(
    blended_paragraphs: List[dict],
    author: str,
    persona_frames: dict,
    output_dir: Path,
    output_format: str = "llama_factory",
):
    """Generate training data from blended paragraphs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"

    examples = []
    failed = 0

    for i, para in enumerate(blended_paragraphs):
        # Use the most processed version
        styled_text = para.get("transplanted_text") or para.get("aligned_text") or para["text"]
        word_count = len(styled_text.split())

        logger.info(f"[{i+1}/{len(blended_paragraphs)}] Neutralizing ({word_count} words)...")

        # RTT neutralize
        try:
            neutral = neutralize_text(styled_text)
        except Exception as e:
            logger.warning(f"  Neutralization failed: {e}")
            failed += 1
            continue

        if not neutral:
            failed += 1
            continue

        # Classify content type
        content_type = classify_content_type(styled_text)
        frames = persona_frames.get(content_type, persona_frames["narrative"])

        # Generate training example
        persona_frame = random.choice(frames)
        instruction = build_instruction(persona_frame, word_count, styled_text)
        perturbed = perturb_text(neutral)

        if output_format == "llama_factory":
            example = {
                "instruction": instruction,
                "input": perturbed,
                "output": styled_text,
            }
        else:
            prompt = f"{instruction}\n\n{perturbed}\n###\n"
            example = {
                "text": prompt + styled_text,
                "prompt": prompt,
                "word_count": word_count,
                "variation_type": "blended_original",
            }

        examples.append(example)

        # Also create a snowflake-style variant: same output, info-dropout input
        # Strip adjectives/adverbs from neutral text
        words = neutral.split()
        stripped = []
        skip_words = {
            "great", "small", "large", "old", "new", "good", "bad", "long",
            "short", "dark", "light", "strange", "ancient", "terrible",
            "horrible", "beautiful", "quiet", "loud", "vast", "immense",
            "enormous", "peculiar", "odd", "weird", "deep", "pale", "faint",
        }
        for w in words:
            if w.lower().rstrip(".,!?;:") not in skip_words:
                stripped.append(w)
        stripped_text = " ".join(stripped)

        persona_frame2 = random.choice(frames)
        instruction2 = build_instruction(persona_frame2, word_count, styled_text)
        perturbed2 = perturb_text(stripped_text, rate=0.05)

        if output_format == "llama_factory":
            variant = {
                "instruction": instruction2,
                "input": perturbed2,
                "output": styled_text,
            }
        else:
            prompt2 = f"{instruction2}\n\n{perturbed2}\n###\n"
            variant = {
                "text": prompt2 + styled_text,
                "prompt": prompt2,
                "word_count": word_count,
                "variation_type": "blended_info_dropout",
            }

        examples.append(variant)
        logger.info(f"  OK: 2 examples ({content_type})")

    # Shuffle and write
    random.shuffle(examples)
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"\nDone! {len(examples)} examples written to {train_path}")
    logger.info(f"  ({failed} paragraphs failed neutralization)")

    return examples


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate blended-style training data"
    )
    parser.add_argument("--blended", type=Path, required=True,
                        help="Path to blended paragraphs JSON from experiment_blend_sentences.py")
    parser.add_argument("--author", type=str, default="Howard Russell",
                        help="Blended author name (default: Howard Russell)")
    parser.add_argument("--worldview", type=Path,
                        default=PROJECT_ROOT / "prompts" / "howard_russell_worldview.txt",
                        help="Path to worldview file with persona frames")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for training data")
    parser.add_argument("--format", choices=["llama_factory", "mlx"],
                        default="llama_factory",
                        help="Output format (default: llama_factory)")

    args = parser.parse_args()

    # Load blended paragraphs
    logger.info(f"Loading blended paragraphs from {args.blended}")
    with open(args.blended, "r", encoding="utf-8") as f:
        blended = json.load(f)
    logger.info(f"  {len(blended)} paragraphs loaded")

    # Load persona frames
    logger.info(f"Loading persona frames from {args.worldview}")
    frames = load_persona_frames(args.worldview)

    # Generate training data
    generate_training_data(
        blended_paragraphs=blended,
        author=args.author,
        persona_frames=frames,
        output_dir=args.output,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()

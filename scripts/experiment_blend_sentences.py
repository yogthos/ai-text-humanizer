#!/usr/bin/env python3
"""Experiment: Blend author corpora at sentence level to create chimera paragraphs.

Creates training-ready paragraphs by stitching real sentences from two authors.
The primary author provides the analytical/narrative backbone; the supplementary
author adds flavor sentences at logical seams.

Pipeline:
1. Load both corpora, split into paragraphs → sentences
2. Embed all sentences with sentence-transformers
3. For each primary paragraph, find compatible supplementary sentences
4. Stitch together at logical insertion points
5. Use LLM for minimal subject alignment (pronouns/referents only)
6. Output JSON for manual inspection and downstream training

Usage:
    python scripts/experiment_blend_sentences.py \
        --primary data/corpus/russell/ \
        --primary-author "Bertrand Russell" \
        --supplementary data/corpus/lovecraft.txt \
        --supplementary-author "H.P. Lovecraft" \
        --output data/blended/russell_lovecraft_experiment.json \
        --num-paragraphs 20 \
        --blend-ratio 0.25 \
        --verbose
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import requests

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy library loggers
for _lib in ("httpx", "httpcore", "urllib3", "sentence_transformers",
             "transformers", "huggingface_hub", "pydot"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BlendConfig:
    """Configuration for sentence-level blending experiment."""
    blend_ratio: float = 0.25          # fraction of sentences from supplementary author
    max_insertions: int = 0            # cap on insertions per paragraph (0 = no cap)
    min_similarity: float = 0.15       # minimum cosine similarity for candidate sentences
    min_paragraph_sentences: int = 4   # skip paragraphs with fewer sentences
    max_paragraph_sentences: int = 20  # skip very long paragraphs
    min_sentence_words: int = 8        # skip very short sentences
    max_sentence_words: int = 60       # skip very long sentences
    insertion_modes: List[str] = None  # modes to try: "interleave", "append", "replace"

    def __post_init__(self):
        if self.insertion_modes is None:
            self.insertion_modes = ["interleave", "append", "replace"]


# =============================================================================
# Vocabulary Transplant
# =============================================================================

# Common English adjectives/adverbs to skip (too generic to transplant)
SKIP_WORDS = {
    "good", "bad", "great", "small", "large", "big", "little", "old", "new",
    "long", "high", "low", "young", "first", "last", "next", "few", "many",
    "much", "more", "most", "other", "same", "own", "such", "only", "able",
    "sure", "real", "right", "left", "full", "early", "late", "hard", "far",
    "likely", "certain", "true", "whole", "general", "particular", "possible",
    "necessary", "important", "different", "similar", "available", "common",
    "enough", "second", "third", "several", "various", "further", "quite",
    "rather", "very", "also", "just", "even", "still", "already", "almost",
    "often", "never", "always", "sometimes", "perhaps", "merely", "simply",
    "really", "actually", "probably", "certainly", "indeed", "therefore",
    "however", "especially", "particularly", "completely", "entirely",
    "not", "well", "also", "too", "now", "then", "here", "there",
}


class VocabTransplanter:
    """Replace adjectives/adverbs with author-characteristic vocabulary.

    Extracts distinctive adjectives and adverbs from a source author's corpus,
    groups them by the semantic type of noun they modify, and transplants them
    into blended paragraphs to shift mood and tone.
    """

    def __init__(self, source_text: str, reference_text: str, transplant_ratio: float = 0.5):
        """
        Args:
            source_text: Corpus of the author whose vocabulary we want to inject.
            reference_text: Corpus of the other author (for distinctiveness scoring).
            transplant_ratio: Fraction of eligible adjectives/adverbs to replace.
        """
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
        self.transplant_ratio = transplant_ratio

        logger.info("Building vocabulary profiles...")
        source_adj, source_adv = self._extract_vocab(source_text, "source")
        ref_adj, ref_adv = self._extract_vocab(reference_text, "reference")

        # Score distinctiveness: high in source, low in reference
        self.distinctive_adj = self._score_distinctive(source_adj, ref_adj)
        self.distinctive_adv = self._score_distinctive(source_adv, ref_adv)

        # Group adjectives by what type of noun they typically modify
        self.adj_by_category = self._categorize_adjectives(source_text)

        logger.info(f"  Distinctive adjectives: {len(self.distinctive_adj)}")
        logger.info(f"  Distinctive adverbs: {len(self.distinctive_adv)}")
        logger.info(f"  Top adjectives: {list(self.distinctive_adj.keys())[:15]}")
        logger.info(f"  Top adverbs: {list(self.distinctive_adv.keys())[:15]}")

    def _extract_vocab(self, text: str, label: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Extract adjective and adverb frequencies from corpus."""
        from collections import Counter
        adj_freq = Counter()
        adv_freq = Counter()

        # Process in chunks for speed
        chunk_size = 100000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        logger.info(f"  Processing {label} corpus ({len(chunks)} chunks)...")

        for chunk in self.nlp.pipe(chunks, batch_size=4, n_process=1):
            for token in chunk:
                lemma = token.lemma_.lower()
                if lemma in SKIP_WORDS or len(lemma) < 3 or not lemma.isalpha():
                    continue
                if token.pos_ == "ADJ" and token.dep_ in ("amod", "acomp", "attr", "conj"):
                    adj_freq[lemma] += 1
                elif token.pos_ == "ADV" and token.dep_ in ("advmod", "conj"):
                    adv_freq[lemma] += 1

        return dict(adj_freq), dict(adv_freq)

    def _score_distinctive(
        self, source: Dict[str, int], reference: Dict[str, int], min_count: int = 3
    ) -> Dict[str, float]:
        """Score words by distinctiveness: high in source, low in reference."""
        scores = {}
        for word, count in source.items():
            if count < min_count:
                continue
            ref_count = reference.get(word, 0)
            # Log-ratio score: high when frequent in source, rare in reference
            score = np.log2((count + 1) / (ref_count + 1))
            if score > 0.5:  # at least ~1.4x more frequent in source
                scores[word] = score

        # Sort by score descending
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    def _categorize_adjectives(self, source_text: str) -> Dict[str, List[str]]:
        """Group adjectives by the semantic category of nouns they modify.

        Categories:
        - physical: modifying concrete/physical nouns (room, door, stone, light)
        - abstract: modifying abstract nouns (thought, truth, knowledge, nature)
        - entity: modifying people/creatures (man, thing, creature, being)
        - temporal: modifying time-related nouns (night, age, time, year)
        - sensory: modifying sensory nouns (sound, smell, colour, vision)
        """
        from collections import Counter, defaultdict

        # Track which nouns each adjective modifies
        adj_noun_pairs = defaultdict(Counter)

        # Process a representative sample (first 200k chars)
        sample = source_text[:200000]
        for doc in self.nlp.pipe([sample], batch_size=1, n_process=1):
            for token in doc:
                if token.pos_ == "ADJ" and token.dep_ == "amod":
                    lemma = token.lemma_.lower()
                    head = token.head.lemma_.lower()
                    if lemma not in SKIP_WORDS and len(lemma) >= 3:
                        adj_noun_pairs[lemma][head] += 1

        # Simple noun→category mapping based on common patterns
        physical_nouns = {
            "wall", "door", "stone", "room", "house", "building", "tower",
            "rock", "mountain", "hill", "valley", "sea", "water", "river",
            "light", "shadow", "surface", "floor", "ceiling", "window",
            "structure", "city", "place", "land", "region", "area", "cave",
        }
        abstract_nouns = {
            "thing", "thought", "idea", "truth", "knowledge", "nature",
            "power", "force", "law", "world", "reality", "existence",
            "quality", "sense", "mind", "belief", "theory", "reason",
            "fact", "principle", "relation", "form", "essence", "matter",
        }
        entity_nouns = {
            "man", "men", "creature", "being", "person", "figure", "body",
            "face", "eye", "hand", "voice", "people", "race", "species",
        }
        temporal_nouns = {
            "night", "day", "time", "age", "year", "hour", "moment",
            "century", "aeon", "period", "era", "past", "future",
        }
        sensory_nouns = {
            "sound", "smell", "odour", "colour", "vision", "sight",
            "noise", "silence", "darkness", "glow", "gleam", "flash",
        }

        categories = defaultdict(list)
        categorized = set()

        for adj, noun_counts in adj_noun_pairs.items():
            if adj not in self.distinctive_adj:
                continue

            top_nouns = set(noun_counts.keys())
            # Assign to category based on which noun set has most overlap
            best_cat = "physical"  # default
            best_overlap = 0
            for cat, noun_set in [
                ("physical", physical_nouns),
                ("abstract", abstract_nouns),
                ("entity", entity_nouns),
                ("temporal", temporal_nouns),
                ("sensory", sensory_nouns),
            ]:
                overlap = len(top_nouns & noun_set)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cat = cat

            categories[best_cat].append(adj)
            categorized.add(adj)

        # Add uncategorized distinctive adjectives to "physical" (most common)
        for adj in self.distinctive_adj:
            if adj not in categorized:
                categories["physical"].append(adj)

        return dict(categories)

    def _get_noun_category(self, token) -> str:
        """Determine the semantic category of a noun token."""
        lemma = token.lemma_.lower()

        # Check entity type first
        if token.ent_type_ in ("PERSON", "ORG", "NORP"):
            return "entity"
        if token.ent_type_ in ("DATE", "TIME"):
            return "temporal"

        # Fall back to lemma matching
        physical = {"wall", "door", "stone", "room", "house", "building",
                    "tower", "rock", "mountain", "sea", "water", "light",
                    "shadow", "surface", "floor", "structure", "city", "place",
                    "land", "cave", "street", "road", "sky", "air"}
        abstract = {"thing", "thought", "idea", "truth", "knowledge", "nature",
                    "power", "force", "law", "world", "reality", "existence",
                    "quality", "sense", "mind", "belief", "theory", "reason",
                    "fact", "principle", "way", "kind", "sort", "manner"}
        entity = {"man", "men", "creature", "being", "person", "figure",
                  "body", "face", "eye", "hand", "voice", "people", "race"}
        temporal = {"night", "day", "time", "age", "year", "hour", "moment",
                    "century", "aeon", "period", "era"}
        sensory = {"sound", "smell", "odour", "colour", "vision", "sight",
                   "noise", "silence", "darkness", "glow"}

        for cat, nouns in [("physical", physical), ("abstract", abstract),
                           ("entity", entity), ("temporal", temporal),
                           ("sensory", sensory)]:
            if lemma in nouns:
                return cat
        return "physical"  # default

    def _is_replaceable(self, token) -> bool:
        """Check if an adjective/adverb can be replaced without breaking meaning.

        Uses a block-list approach: everything is replaceable UNLESS it's a
        protected technical term, structural connector, or part of a fixed phrase.
        This is aggressive — the goal is Lovecraftian mood saturation.
        """
        lemma = token.lemma_.lower()
        text_lower = token.text.lower()

        # === BLOCK: too short, abbreviations, or non-alphabetic ===
        if len(lemma) < 4 or not lemma.isalpha():
            return False

        # === BLOCK: technical/classificatory terms ===
        technical = {
            # Scientific classification
            "special", "general", "relative", "absolute",
            "positive", "negative", "electric", "magnetic",
            "continuous", "discrete", "finite", "infinite", "euclidean",
            "synthetic", "analytic", "empirical", "theoretical",
            "mathematical", "physical", "chemical", "biological",
            "atomic", "molecular", "nuclear", "quantum", "classical",
            "mechanical", "static", "dynamic", "linear", "angular",
            "parallel", "perpendicular", "horizontal", "vertical",
            "spatial", "temporal", "structural", "functional",
            "electromagnetic", "gravitational", "convertible", "radiant",
            "discontinuous", "instantaneous", "infinitesimal",
            # Logical/structural
            "logical", "necessary", "sufficient", "possible", "impossible",
            "valid", "invalid", "true", "false", "correct", "incorrect",
            "essential", "fundamental", "basic", "intrinsic", "extrinsic",
            "propositional", "hypothetical", "assertoric", "deductive",
            "inductive", "causal", "relational", "contravariant", "covariant",
            # Latin/philosophical terms (must not be split from phrases)
            "priori", "posteriori", "facto", "hoc", "fortiori",
            # Prefixes/classifiers
            "non", "pre", "post", "semi", "anti", "sub", "super",
            # Measurement/comparison
            "equal", "identical", "opposite", "inverse", "proportional",
            "singular", "plural", "separate", "independent", "dependent",
            "diverse", "compatible", "incompatible", "convertible",
            # Positional
            "internal", "external", "upper", "lower", "inner", "outer",
            "primary", "secondary", "initial", "final", "former", "latter",
            # Domain labels (keep these neutral — Russell's subject matter)
            "political", "social", "economic", "scientific", "religious",
            "pure", "applied", "practical", "intellectual", "rational",
            "philosophical", "metaphysical", "ethical", "moral",
            "mental", "psychological", "historical", "spiritual",
            # Language/nationality (proper-adjective-like)
            "greek", "roman", "latin", "arabic", "hebrew", "chinese",
            "english", "french", "german", "italian", "spanish",
            "european", "american", "british", "christian", "jewish",
            # Grammatical/linguistic terms
            "verbal", "nominal", "adverbial", "conditional", "subjunctive",
            "indicative", "tactual", "visual", "auditory", "olfactory",
            # Science-specific nouns spaCy may tag as ADJ
            "ether", "aether",
        }
        if lemma in technical or text_lower in technical:
            return False

        # === BLOCK: structural/logical/manner adverbs ===
        if token.pos_ == "ADV":
            structural = {
                # Logical connectors
                "secondly", "thirdly", "firstly", "consequently", "accordingly",
                "thus", "hence", "thereby", "therefore", "moreover",
                "furthermore", "nevertheless", "nonetheless", "otherwise",
                "either", "neither", "both", "also", "too", "not",
                # Degree/precision
                "approximately", "exactly", "precisely", "roughly",
                "equally", "independently", "respectively",
                # Manner that carries meaning
                "rapidly", "slowly", "gradually", "immediately", "suddenly",
                "originally", "simultaneously", "previously", "subsequently",
                "merely", "simply", "properly", "strictly", "correctly",
                "literally", "physically", "mentally", "relatively",
            }
            if lemma in structural:
                return False

        # === BLOCK: part of proper noun or named entity ===
        if token.head.ent_type_ in ("PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"):
            return False

        # === BLOCK: head noun is a technical term ===
        head_lemma = token.head.lemma_.lower()
        if head_lemma in technical:
            return False

        # === ALLOW: everything else ===
        # This includes: descriptive, evaluative, sensory, atmospheric,
        # manner adverbs, characterizing adjectives, etc.
        return True

    def transplant(self, text: str) -> Tuple[str, List[Dict]]:
        """Replace a portion of evaluative adjectives/adverbs with source-author vocabulary.

        Only replaces subjective, mood-setting words. Technical terms, classificatory
        adjectives, and structural adverbs are protected.

        Returns (modified_text, list_of_replacements).
        """
        doc = self.nlp(text)
        replacements = []
        tokens = list(doc)

        # Collect eligible replacement positions (only evaluative words)
        eligible = []
        for i, token in enumerate(tokens):
            lemma = token.lemma_.lower()
            if lemma in SKIP_WORDS or len(lemma) < 3:
                continue

            if not self._is_replaceable(token):
                continue

            if token.pos_ == "ADJ" and token.dep_ in ("amod", "acomp", "attr"):
                head_cat = self._get_noun_category(token.head)
                eligible.append((i, token, "adj", head_cat))

        # Select which ones to replace
        n_to_replace = max(1, int(len(eligible) * self.transplant_ratio))
        random.shuffle(eligible)
        to_replace = eligible[:n_to_replace]
        to_replace.sort(key=lambda x: x[0])  # stable order

        # Build replacement text
        result_tokens = [t.text_with_ws for t in tokens]

        # Curated atmospheric Lovecraft vocabulary by noun-category.
        # ONLY true adjectives — no participial forms (-ing, -ed) that break
        # grammar when used as attributive modifiers (e.g., "crumbling with
        # the scent" is wrong; "cyclopean with the scent" is fine because
        # cyclopean is a true adjective).
        atmo_vocab = {
            "physical": [
                "cyclopean", "monolithic", "cavernous", "labyrinthine",
                "windowless", "sunken", "fungoid", "rugose", "pitted",
                "sepulchral", "subterranean", "basaltic", "hewn", "vast",
            ],
            "abstract": [
                "nameless", "unnameable", "blasphemous", "forbidden", "accursed",
                "unspeakable", "unhallowed", "abhorrent", "loathsome", "monstrous",
                "eldritch", "unwholesome", "malign", "noxious", "damnable",
            ],
            "entity": [
                "gaunt", "cadaverous", "pallid", "wizened", "spectral",
                "ghoulish", "furtive", "haggard", "sallow", "wan",
                "lupine", "aquiline", "gaunt", "emaciated", "corpulent",
            ],
            "temporal": [
                "immemorial", "antediluvian", "primordial", "forgotten",
                "primal", "prehistoric", "hoary", "timeless", "ageless",
                "elder", "ancient", "archaic", "primeval", "olden",
            ],
            "sensory": [
                "fetid", "phosphorescent", "luminous", "tenebrous", "gibbous",
                "noisome", "acrid", "sulphurous", "viscous", "gelatinous",
                "pallid", "livid", "iridescent", "opalescent", "crepuscular",
            ],
        }

        for idx, token, word_type, category in to_replace:
            # 70% use atmospheric vocab, 30% keep original — adds vocabulary range
            # so the model doesn't learn that EVERY adjective must be Lovecraftian
            if random.random() > 0.70:
                continue  # keep original adjective

            candidates = atmo_vocab.get(category, atmo_vocab["abstract"])
            replacement = random.choice(candidates)

            # Preserve capitalization
            if token.text[0].isupper():
                replacement = replacement.capitalize()

            # Preserve trailing whitespace
            ws = token.whitespace_
            result_tokens[idx] = replacement + ws

            replacements.append({
                "original": token.text,
                "replacement": replacement,
                "type": word_type,
                "category": category,
                "position": idx,
            })

        return "".join(result_tokens), replacements


@dataclass
class ScoredSentence:
    """A sentence with its embedding and metadata."""
    text: str
    author: str
    source_file: str
    paragraph_index: int
    sentence_index: int
    embedding: np.ndarray = field(repr=False)


@dataclass
class BlendedParagraph:
    """Result of blending: a chimera paragraph with provenance."""
    text: str
    aligned_text: Optional[str]        # after LLM subject alignment
    transplanted_text: Optional[str]   # after vocabulary transplant
    primary_author: str
    supplementary_author: str
    primary_sentences: List[str]
    supplementary_sentences: List[str]
    insertion_mode: str
    insertion_points: List[int]        # indices where supplementary sentences were placed
    similarity_scores: List[float]     # similarity of each inserted sentence to context
    source_paragraph: str              # original primary paragraph
    vocab_replacements: List[Dict] = field(default_factory=list)  # transplant details


# =============================================================================
# Text Processing
# =============================================================================

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex heuristics.

    Handles abbreviations, quoted speech, and ellipses reasonably well.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Simple approach: split on . ! ? followed by space + uppercase letter
    # Then re-join fragments that were split on abbreviations
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)

    abbrevs = {"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
               "vs.", "etc.", "viz.", "al.", "eg.", "ie.", "cf.", "No.",
               "Vol.", "Ch.", "Fig.", "Eq.", "approx.", "dept.", "est."}

    result = []
    buffer = ""
    for fragment in raw:
        if buffer:
            fragment = buffer + " " + fragment
            buffer = ""

        # Check if fragment ends with an abbreviation (not a real sentence end)
        last_word = fragment.rstrip().rsplit(None, 1)[-1] if fragment.strip() else ""
        if last_word in abbrevs:
            buffer = fragment
            continue

        fragment = fragment.strip()
        if fragment and len(fragment.split()) >= 3:
            result.append(fragment)

    # Flush remaining buffer
    if buffer:
        buffer = buffer.strip()
        if buffer and len(buffer.split()) >= 3:
            result.append(buffer)

    return result


def load_corpus_text(path: Path) -> Tuple[str, str]:
    """Load corpus from file or directory. Returns (text, source_name)."""
    if path.is_dir():
        texts = []
        for f in sorted(path.glob("*.txt")):
            texts.append(f.read_text(encoding="utf-8"))
        return "\n\n".join(texts), path.name
    else:
        return path.read_text(encoding="utf-8"), path.stem


def extract_paragraphs(text: str, min_words: int = 40) -> List[str]:
    """Split text into paragraphs, filtering for quality."""
    raw = re.split(r"\n\s*\n", text)
    paragraphs = []
    for p in raw:
        p = p.strip()
        if not p:
            continue
        # Skip headers, chapter markers, dates, etc.
        if len(p.split()) < min_words:
            continue
        if re.match(r"^(Chapter|CHAPTER|Part|PART|\d+\.|[IVXLC]+\.)", p):
            continue
        # Skip paragraphs that are mostly dialogue
        quote_chars = p.count('"') + p.count("'") + p.count("\u201c") + p.count("\u201d")
        if quote_chars > len(p) * 0.05:
            continue
        paragraphs.append(p)
    return paragraphs


# Words that typically signal a new thought/sub-topic when they start a sentence
TRANSITION_WORDS = {
    "But", "Yet", "However", "Nevertheless", "Nonetheless", "Still",
    "Moreover", "Furthermore", "Besides", "Indeed", "Thus", "Hence",
    "Therefore", "Consequently", "Accordingly", "Now", "Then", "Again",
    "Meanwhile", "Afterward", "Later", "Finally", "First", "Second",
    "Third", "Firstly", "Secondly", "Thirdly", "Lastly",
}


def decompose_paragraph(paragraph: str, target_min: int = 60, target_max: int = 100) -> List[str]:
    """Split a long paragraph into shorter sub-paragraphs at natural boundaries.

    Uses greedy sentence accumulation with preference for transition-word breaks.
    A sentence starting with "But", "However", "Thus", "Moreover", etc. signals
    a natural stopping point — if the accumulated window is already in the target
    range, break there. Otherwise keep accumulating until forced to break.

    Returns list of sub-paragraphs in target_min..target_max word range.
    Returns empty list if paragraph is too short to decompose.
    """
    sentences = split_into_sentences(paragraph)
    if len(sentences) < 4:
        return []

    total_words = sum(len(s.split()) for s in sentences)
    if total_words < target_max + target_min:
        return []  # not enough material for multiple sub-paragraphs

    sub_paragraphs = []
    window = []
    window_wc = 0

    for i, sent in enumerate(sentences):
        sw = len(sent.split())
        is_transition = False
        if i > 0:
            first_word = sent.split()[0].rstrip(",.:;!?") if sent.split() else ""
            is_transition = first_word in TRANSITION_WORDS

        # Break decision: if we're at a transition point and window is in target range, break BEFORE adding
        if is_transition and target_min <= window_wc <= target_max:
            sub_paragraphs.append(" ".join(window))
            window = [sent]
            window_wc = sw
            continue

        # Forced break: adding this sentence would exceed target_max and we're already in range
        if window_wc + sw > target_max and window_wc >= target_min:
            sub_paragraphs.append(" ".join(window))
            window = [sent]
            window_wc = sw
            continue

        # Otherwise accumulate
        window.append(sent)
        window_wc += sw

    # Flush remaining window if it meets minimum
    if window_wc >= target_min:
        sub_paragraphs.append(" ".join(window))

    return sub_paragraphs


def augment_with_short_variants(paragraphs: List[str], target_min: int = 60,
                                 target_max: int = 100) -> List[str]:
    """Augment a paragraph list with short variants from long paragraphs.

    Keeps all original paragraphs AND adds decomposed sub-paragraphs for those
    long enough to be usefully split. The sub-paragraphs are real author text
    (sentences lifted verbatim from the original) but at shorter lengths.

    Returns augmented list with originals + short variants.
    """
    augmented = list(paragraphs)  # keep all originals
    n_added = 0

    for para in paragraphs:
        wc = len(para.split())
        if wc < target_max * 1.5:
            continue  # not worth decomposing

        sub_paras = decompose_paragraph(para, target_min=target_min, target_max=target_max)
        augmented.extend(sub_paras)
        n_added += len(sub_paras)

    logger.info(f"  Decomposition added {n_added} short variants from long paragraphs")
    return augmented


def build_sentence_index(
    text: str,
    author: str,
    source_name: str,
    embedding_model,
    config: BlendConfig,
) -> List[ScoredSentence]:
    """Extract and embed all sentences from a corpus."""
    paragraphs = extract_paragraphs(text)
    logger.info(f"  {author}: {len(paragraphs)} paragraphs extracted")

    all_sentences = []
    all_texts = []

    for pi, para in enumerate(paragraphs):
        sentences = split_into_sentences(para)
        for si, sent in enumerate(sentences):
            word_count = len(sent.split())
            if word_count < config.min_sentence_words:
                continue
            if word_count > config.max_sentence_words:
                continue
            all_sentences.append((sent, pi, si))
            all_texts.append(sent)

    logger.info(f"  {author}: {len(all_texts)} sentences after filtering")

    if not all_texts:
        return []

    # Batch embed
    logger.info(f"  {author}: computing embeddings...")
    embeddings = embedding_model.encode(all_texts, show_progress_bar=False, batch_size=64)

    scored = []
    for (sent, pi, si), emb in zip(all_sentences, embeddings):
        scored.append(ScoredSentence(
            text=sent,
            author=author,
            source_file=source_name,
            paragraph_index=pi,
            sentence_index=si,
            embedding=emb,
        ))

    return scored


# =============================================================================
# Similarity & Matching
# =============================================================================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_compatible_sentences(
    primary_sentence: ScoredSentence,
    supplementary_index: List[ScoredSentence],
    supp_embeddings: np.ndarray,
    n: int = 5,
    min_similarity: float = 0.15,
) -> List[Tuple[ScoredSentence, float]]:
    """Find supplementary sentences compatible with a primary sentence.

    Returns top-n candidates sorted by similarity, above threshold.
    """
    # Vectorized cosine similarity
    query = primary_sentence.embedding
    norms = np.linalg.norm(supp_embeddings, axis=1) + 1e-8
    query_norm = np.linalg.norm(query) + 1e-8
    sims = np.dot(supp_embeddings, query) / (norms * query_norm)

    # Get top candidates above threshold
    mask = sims >= min_similarity
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        return []

    # Sort by similarity descending
    sorted_idx = valid_indices[np.argsort(sims[valid_indices])[::-1]][:n]

    return [(supplementary_index[i], float(sims[i])) for i in sorted_idx]


# =============================================================================
# Stitching Strategies
# =============================================================================

def stitch_interleave(
    primary_sentences: List[str],
    compatible_map: Dict[int, List[Tuple[ScoredSentence, float]]],
    blend_ratio: float,
    max_insertions: int = 0,
) -> Tuple[List[str], List[int], List[str], List[float]]:
    """Interleave supplementary sentences after compatible primary sentences.

    Inserts supplementary sentences after the primary sentence they're most
    similar to, creating a natural flow: primary makes a point, supplementary
    adds atmospheric elaboration.
    """
    n_to_insert = max(1, int(len(primary_sentences) * blend_ratio))
    if max_insertions > 0:
        n_to_insert = min(n_to_insert, max_insertions)

    # Score each insertion point by best available similarity
    # Build a list of (position, candidate_list) sorted by best similarity
    scored_positions = []
    for idx, candidates in compatible_map.items():
        if candidates:
            scored_positions.append((idx, candidates))
    scored_positions.sort(key=lambda x: x[1][0][1], reverse=True)

    # Greedily select, deduplicating by sentence text
    selected = []
    used_texts = set()
    for idx, candidates in scored_positions:
        if len(selected) >= n_to_insert:
            break
        for sent, sim in candidates:
            if sent.text not in used_texts:
                selected.append((idx, sent, sim))
                used_texts.add(sent.text)
                break
    selected.sort(key=lambda x: x[0])  # re-sort by position for stable insertion

    # Build stitched paragraph
    result = []
    insertion_points = []
    supp_sentences = []
    sim_scores = []
    offset = 0

    for i, sent in enumerate(primary_sentences):
        result.append(sent)
        # Check if we insert after this sentence
        for idx, supp_sent, sim in selected:
            if idx == i:
                result.append(supp_sent.text)
                insertion_points.append(i + 1 + offset)
                supp_sentences.append(supp_sent.text)
                sim_scores.append(sim)
                offset += 1

    return result, insertion_points, supp_sentences, sim_scores


def stitch_replace(
    primary_sentences: List[str],
    compatible_map: Dict[int, List[Tuple[ScoredSentence, float]]],
    blend_ratio: float,
    max_insertions: int = 0,
) -> Tuple[List[str], List[int], List[str], List[float]]:
    """Replace some primary sentences with supplementary ones.

    Picks primary sentences that have very similar supplementary counterparts,
    swapping them to get the supplementary author's voice on a similar idea.
    """
    n_to_replace = max(1, int(len(primary_sentences) * blend_ratio))
    if max_insertions > 0:
        n_to_replace = min(n_to_replace, max_insertions)

    # Find best replacements, deduplicating by sentence text
    scored_positions = []
    for idx, candidates in compatible_map.items():
        if candidates and candidates[0][1] > 0.3:  # higher threshold for replacement
            scored_positions.append((idx, candidates))
    scored_positions.sort(key=lambda x: x[1][0][1], reverse=True)

    selected = []
    used_texts = set()
    for idx, candidates in scored_positions:
        if len(selected) >= n_to_replace:
            break
        for sent, sim in candidates:
            if sent.text not in used_texts:
                selected.append((idx, sent, sim))
                used_texts.add(sent.text)
                break

    result = []
    insertion_points = []
    supp_sentences = []
    sim_scores = []

    for i, sent in enumerate(primary_sentences):
        replaced = False
        for idx, supp_sent, sim in selected:
            if idx == i:
                result.append(supp_sent.text)
                insertion_points.append(i)
                supp_sentences.append(supp_sent.text)
                sim_scores.append(sim)
                replaced = True
                break
        if not replaced:
            result.append(sent)

    return result, insertion_points, supp_sentences, sim_scores


def stitch_append(
    primary_sentences: List[str],
    compatible_map: Dict[int, List[Tuple[ScoredSentence, float]]],
    blend_ratio: float,
    max_insertions: int = 0,
) -> Tuple[List[str], List[int], List[str], List[float]]:
    """Append a block of supplementary sentences at the end.

    Finds supplementary sentences thematically compatible with the paragraph's
    overall theme and adds them as a concluding flourish.
    """
    n_to_append = max(1, int(len(primary_sentences) * blend_ratio))
    if max_insertions > 0:
        n_to_append = min(n_to_append, max_insertions)

    # Collect all candidates, deduplicate, take best
    all_candidates = []
    seen_texts = set()
    for candidates in compatible_map.values():
        for sent, sim in candidates:
            if sent.text not in seen_texts:
                all_candidates.append((sent, sim))
                seen_texts.add(sent.text)

    all_candidates.sort(key=lambda x: x[1], reverse=True)
    selected = all_candidates[:n_to_append]

    result = list(primary_sentences)
    insertion_points = []
    supp_sentences = []
    sim_scores = []

    for supp_sent, sim in selected:
        insertion_points.append(len(result))
        result.append(supp_sent.text)
        supp_sentences.append(supp_sent.text)
        sim_scores.append(sim)

    return result, insertion_points, supp_sentences, sim_scores


STITCH_STRATEGIES = {
    "interleave": stitch_interleave,
    "replace": stitch_replace,
    "append": stitch_append,
}


# =============================================================================
# LLM Subject Alignment
# =============================================================================

def call_deepseek(prompt: str, system: str = "", max_retries: int = 3) -> str:
    """Call DeepSeek API for subject alignment."""
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
                    "temperature": 0.2,
                    "max_tokens": 2048,
                },
                timeout=90,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def align_subjects(stitched_text: str, primary_author: str, supplementary_author: str) -> str:
    """Use LLM to minimally align subjects/pronouns in stitched paragraph.

    CRITICAL: Only change referents and pronouns to make the paragraph coherent.
    Do NOT rewrite sentences, change vocabulary, or alter sentence structure.
    The authentic author voice lives in the original sentence construction.
    """
    system = """You are a precise text editor. Your ONLY job is to make minor pronoun
and subject-reference adjustments so a paragraph reads as one coherent thought.

Rules:
- ONLY change pronouns, proper nouns, and subject references for consistency
- NEVER rewrite sentence structure
- NEVER change vocabulary, metaphors, or descriptive language
- NEVER add or remove sentences
- NEVER change the register or tone of any sentence
- Keep every sentence's word count within ±3 words of the original
- If the paragraph already reads coherently, return it unchanged"""

    prompt = f"""This paragraph blends sentences from {primary_author} and {supplementary_author}.
Make MINIMAL pronoun/subject adjustments so it reads as one coherent piece.

Paragraph:
{stitched_text}

Return ONLY the adjusted paragraph, nothing else."""

    return call_deepseek(prompt, system)


# =============================================================================
# Main Blending Pipeline
# =============================================================================

def blend_paragraphs(
    primary_sentences_list: List[List[str]],
    primary_paragraphs: List[str],
    supplementary_index: List[ScoredSentence],
    supp_embeddings: np.ndarray,
    primary_author: str,
    supplementary_author: str,
    config: BlendConfig,
    num_paragraphs: int = 20,
    do_alignment: bool = True,
    transplanter: Optional[VocabTransplanter] = None,
    verbose: bool = False,
) -> List[BlendedParagraph]:
    """Run the full blending pipeline."""
    results = []
    used_supp_texts = set()  # avoid reusing supplementary sentences

    # Allocate paragraphs across modes round-robin
    import itertools
    mode_iter = itertools.cycle(config.insertion_modes)

    # Shuffle primary paragraphs for variety
    indices = list(range(len(primary_sentences_list)))
    random.shuffle(indices)

    for idx in indices:
        if len(results) >= num_paragraphs:
            break

        sentences = primary_sentences_list[idx]
        para_text = primary_paragraphs[idx]

        if len(sentences) < config.min_paragraph_sentences:
            continue
        if len(sentences) > config.max_paragraph_sentences:
            continue

        # Build compatibility map: for each primary sentence, find matching supplementary ones
        compatible_map = {}
        _emb_model = _get_embedding_model()
        primary_embs = _emb_model.encode(sentences, show_progress_bar=False)

        for si, (sent, emb) in enumerate(zip(sentences, primary_embs)):
            scored = ScoredSentence(
                text=sent, author=primary_author, source_file="",
                paragraph_index=idx, sentence_index=si,
                embedding=emb,
            )
            candidates = find_compatible_sentences(
                scored, supplementary_index, supp_embeddings,
                n=10, min_similarity=config.min_similarity,
            )
            # Filter out already-used sentences
            candidates = [(s, sim) for s, sim in candidates if s.text not in used_supp_texts]
            compatible_map[si] = candidates

        # Use the next assigned mode, fall back to others if it fails
        target_mode = next(mode_iter)
        modes_to_try = [target_mode] + [m for m in config.insertion_modes if m != target_mode]

        blend_succeeded = False
        for mode in modes_to_try:
            stitch_fn = STITCH_STRATEGIES[mode]
            stitched, ins_points, supp_sents, sim_scores = stitch_fn(
                sentences, compatible_map, config.blend_ratio,
                max_insertions=config.max_insertions,
            )

            if not supp_sents:
                continue  # no compatible sentences found for this mode

            # Skip if average similarity is too low — prevents forced pairings
            # where semantic distance is too vast. Falls through to unblended
            # fallback below so the base author paragraph isn't wasted.
            avg_sim = np.mean(sim_scores)
            if avg_sim < 0.40:
                if verbose:
                    logger.debug(f"  Weak blend: avg similarity {avg_sim:.3f} < 0.40, try next mode")
                continue

            stitched_text = " ".join(stitched)

            # Subject alignment via LLM
            aligned = None
            if do_alignment:
                try:
                    aligned = align_subjects(stitched_text, primary_author, supplementary_author)
                except Exception as e:
                    logger.warning(f"Alignment failed: {e}")

            # Vocabulary transplant
            transplanted = None
            vocab_reps = []
            if transplanter:
                base_text = aligned or stitched_text
                try:
                    transplanted, vocab_reps = transplanter.transplant(base_text)
                except Exception as e:
                    logger.warning(f"Transplant failed: {e}")

            result = BlendedParagraph(
                text=stitched_text,
                aligned_text=aligned,
                transplanted_text=transplanted,
                primary_author=primary_author,
                supplementary_author=supplementary_author,
                primary_sentences=[s for i, s in enumerate(stitched) if i not in set(ins_points)],
                supplementary_sentences=supp_sents,
                insertion_mode=mode,
                insertion_points=ins_points,
                similarity_scores=sim_scores,
                source_paragraph=para_text,
                vocab_replacements=vocab_reps,
            )
            results.append(result)
            blend_succeeded = True

            # Mark supplementary sentences as used
            for s in supp_sents:
                used_supp_texts.add(s)

            if verbose:
                logger.info(
                    f"  [{mode}] Blended paragraph {len(results)}: "
                    f"{len(sentences)} primary + {len(supp_sents)} supplementary sentences, "
                    f"avg similarity {np.mean(sim_scores):.3f}"
                )

            break  # use first successful mode per paragraph

        # Fallback: if no mode produced an acceptable blend, keep the primary
        # paragraph as an unblended entry. The base author text is still valid
        # training material — it contributes vocabulary diversity and range
        # without the forced stylistic collision of a weak blend. The styles
        # have already been deemed compatible at the corpus level; individual
        # paragraph mismatches shouldn't waste real author prose.
        if not blend_succeeded:
            # Apply vocab transplant even to unblended paragraphs so the
            # atmospheric vocabulary appears consistently across training
            transplanted = None
            vocab_reps = []
            if transplanter:
                try:
                    transplanted, vocab_reps = transplanter.transplant(para_text)
                except Exception as e:
                    logger.warning(f"Transplant failed on unblended: {e}")

            result = BlendedParagraph(
                text=para_text,
                aligned_text=None,
                transplanted_text=transplanted,
                primary_author=primary_author,
                supplementary_author=supplementary_author,
                primary_sentences=sentences,
                supplementary_sentences=[],
                insertion_mode="unblended",
                insertion_points=[],
                similarity_scores=[],
                source_paragraph=para_text,
                vocab_replacements=vocab_reps,
            )
            results.append(result)

            if verbose:
                logger.debug(f"  [unblended] Kept primary paragraph {len(results)}: "
                             f"no strong blend available, {len(sentences)} sentences")

    return results


# Singleton embedding model
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# =============================================================================
# Output
# =============================================================================

def save_results(results: List[BlendedParagraph], output_path: Path):
    """Save blended paragraphs as JSON for inspection."""
    output = []
    for r in results:
        entry = {
            "text": r.text,
            "aligned_text": r.aligned_text,
            "transplanted_text": r.transplanted_text,
            "primary_author": r.primary_author,
            "supplementary_author": r.supplementary_author,
            "insertion_mode": r.insertion_mode,
            "insertion_points": r.insertion_points,
            "similarity_scores": r.similarity_scores,
            "primary_sentences": r.primary_sentences,
            "supplementary_sentences": r.supplementary_sentences,
            "source_paragraph": r.source_paragraph,
            "vocab_replacements": r.vocab_replacements,
        }
        output.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(output)} blended paragraphs to {output_path}")


def save_markdown(results: List[BlendedParagraph], output_path: Path):
    """Save blended paragraphs as readable markdown."""
    lines = [
        f"# Blended Style Experiment",
        f"",
        f"**Primary**: {results[0].primary_author} | "
        f"**Supplementary**: {results[0].supplementary_author}",
        f"",
        f"---",
        f"",
    ]

    for i, r in enumerate(results):
        # Use the most processed version available
        final = r.transplanted_text or r.aligned_text or r.text
        avg_sim = np.mean(r.similarity_scores)
        lines.append(f"## Paragraph {i+1} [{r.insertion_mode}] (similarity: {avg_sim:.3f})")
        lines.append("")
        lines.append(final)
        lines.append("")
        lines.append(f"<details><summary>Provenance</summary>")
        lines.append("")
        lines.append(f"**Supplementary sentences inserted:**")
        for s in r.supplementary_sentences:
            lines.append(f"- {s}")
        lines.append("")
        lines.append(f"**Insertion points:** {r.insertion_points}")
        lines.append(f"**Similarity scores:** {[f'{s:.3f}' for s in r.similarity_scores]}")
        if r.vocab_replacements:
            lines.append("")
            lines.append(f"**Vocabulary transplants:**")
            for rep in r.vocab_replacements:
                lines.append(f"- {rep['original']} → {rep['replacement']} "
                             f"({rep['type']}, {rep.get('category', 'n/a')})")
        lines.append("")
        lines.append(f"</details>")
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved markdown to {output_path}")


def print_sample(results: List[BlendedParagraph], n: int = 3):
    """Print a few samples for quick inspection."""
    for i, r in enumerate(results[:n]):
        print(f"\n{'='*70}")
        print(f"SAMPLE {i+1} [{r.insertion_mode}] "
              f"(avg sim: {np.mean(r.similarity_scores):.3f})")
        print(f"{'='*70}")
        print(f"\nPRIMARY ({r.primary_author}) - Original:")
        print(f"  {r.source_paragraph[:200]}...")
        print(f"\nSUPPLEMENTARY ({r.supplementary_author}) - Inserted:")
        for s in r.supplementary_sentences:
            print(f"  >> {s}")
        print(f"\nSTITCHED (raw):")
        print(f"  {r.text[:300]}...")
        if r.aligned_text:
            print(f"\nALIGNED (after LLM):")
            print(f"  {r.aligned_text[:300]}...")
        if r.transplanted_text:
            print(f"\nTRANSPLANTED (after vocab swap):")
            print(f"  {r.transplanted_text[:300]}...")
            if r.vocab_replacements:
                swaps = ", ".join(f"{v['original']}\u2192{v['replacement']}" for v in r.vocab_replacements)
                print(f"  Swaps: {swaps}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment: Blend author sentences into chimera paragraphs"
    )
    parser.add_argument("--primary", type=Path, required=True,
                        help="Path to primary author corpus (file or directory)")
    parser.add_argument("--primary-author", type=str, required=True,
                        help="Primary author name")
    parser.add_argument("--supplementary", type=Path, required=True,
                        help="Path to supplementary author corpus (file or directory)")
    parser.add_argument("--supplementary-author", type=str, required=True,
                        help="Supplementary author name")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path")
    parser.add_argument("--num-paragraphs", type=int, default=20,
                        help="Number of blended paragraphs to generate (default: 20)")
    parser.add_argument("--blend-ratio", type=float, default=0.25,
                        help="Fraction of supplementary sentences (default: 0.25)")
    parser.add_argument("--min-similarity", type=float, default=0.15,
                        help="Minimum cosine similarity for compatibility (default: 0.15)")
    parser.add_argument("--max-insertions", type=int, default=0,
                        help="Max supplementary sentences per paragraph (0 = no cap)")
    parser.add_argument("--min-sentences", type=int, default=4,
                        help="Min sentences per primary paragraph (default: 4)")
    parser.add_argument("--max-sentences", type=int, default=20,
                        help="Max sentences per primary paragraph (default: 20)")
    parser.add_argument("--decompose-long", action="store_true",
                        help="Split long paragraphs at natural boundaries to create "
                             "short variants matching target inference length")
    parser.add_argument("--short-target-min", type=int, default=60,
                        help="Min words for decomposed short variants (default: 60)")
    parser.add_argument("--short-target-max", type=int, default=100,
                        help="Max words for decomposed short variants (default: 100)")
    parser.add_argument("--modes", nargs="+",
                        choices=["interleave", "replace", "append"],
                        default=["interleave", "replace", "append"],
                        help="Insertion modes to try (default: all three)")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Also save readable markdown to this path")
    parser.add_argument("--vocab-transplant", action="store_true",
                        help="Enable vocabulary transplant (replace adj/adv with supplementary author's)")
    parser.add_argument("--transplant-ratio", type=float, default=0.5,
                        help="Fraction of adjectives/adverbs to replace (default: 0.5)")
    parser.add_argument("--skip-alignment", action="store_true",
                        help="Skip LLM subject alignment (faster, for testing)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = BlendConfig(
        blend_ratio=args.blend_ratio,
        max_insertions=args.max_insertions,
        min_similarity=args.min_similarity,
        min_paragraph_sentences=args.min_sentences,
        max_paragraph_sentences=args.max_sentences,
        insertion_modes=args.modes,
    )

    # Load embedding model
    logger.info("Loading embedding model...")
    emb_model = _get_embedding_model()

    # Load and index primary corpus
    logger.info(f"Loading primary corpus: {args.primary}")
    primary_text, primary_source = load_corpus_text(args.primary)
    primary_paragraphs = extract_paragraphs(primary_text)
    logger.info(f"  {len(primary_paragraphs)} paragraphs")

    # Augment with short variants (split long paragraphs at natural boundaries)
    # This creates short training examples matching the target chapter length range
    if args.decompose_long:
        primary_paragraphs = augment_with_short_variants(
            primary_paragraphs,
            target_min=args.short_target_min,
            target_max=args.short_target_max,
        )
        logger.info(f"  {len(primary_paragraphs)} paragraphs after decomposition")

    # Split each paragraph into sentences
    primary_sentences_list = [split_into_sentences(p) for p in primary_paragraphs]

    # Load and index supplementary corpus
    logger.info(f"Loading supplementary corpus: {args.supplementary}")
    supp_text, supp_source = load_corpus_text(args.supplementary)
    supp_index = build_sentence_index(
        supp_text, args.supplementary_author, supp_source, emb_model, config
    )
    logger.info(f"  {len(supp_index)} indexed sentences")

    # Pre-compute supplementary embedding matrix for fast similarity
    supp_embeddings = np.array([s.embedding for s in supp_index])

    # Build vocabulary transplanter if requested
    transplanter = None
    if args.vocab_transplant:
        logger.info("Building vocabulary transplanter...")
        transplanter = VocabTransplanter(
            source_text=supp_text,
            reference_text=primary_text,
            transplant_ratio=args.transplant_ratio,
        )

    # Run blending
    logger.info(f"\nBlending: {args.primary_author} (primary) + "
                f"{args.supplementary_author} (supplementary)")
    logger.info(f"  Blend ratio: {config.blend_ratio}")
    logger.info(f"  Min similarity: {config.min_similarity}")
    logger.info(f"  Modes: {config.insertion_modes}")
    logger.info(f"  Target paragraphs: {args.num_paragraphs}")
    if transplanter:
        logger.info(f"  Vocab transplant: {args.transplant_ratio:.0%} of adj/adv")

    results = blend_paragraphs(
        primary_sentences_list=primary_sentences_list,
        primary_paragraphs=primary_paragraphs,
        supplementary_index=supp_index,
        supp_embeddings=supp_embeddings,
        primary_author=args.primary_author,
        supplementary_author=args.supplementary_author,
        config=config,
        num_paragraphs=args.num_paragraphs,
        do_alignment=not args.skip_alignment,
        transplanter=transplanter,
        verbose=args.verbose,
    )

    # Save and display
    save_results(results, args.output)
    if args.markdown:
        save_markdown(results, args.markdown)
    print_sample(results, n=5)

    # Summary stats
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total blended paragraphs: {len(results)}")
    mode_counts = {}
    for r in results:
        mode_counts[r.insertion_mode] = mode_counts.get(r.insertion_mode, 0) + 1
    for mode, count in mode_counts.items():
        print(f"  {mode}: {count}")
    if results:
        all_sims = [s for r in results for s in r.similarity_scores]
        print(f"Similarity stats: mean={np.mean(all_sims):.3f}, "
              f"min={np.min(all_sims):.3f}, max={np.max(all_sims):.3f}")


if __name__ == "__main__":
    main()

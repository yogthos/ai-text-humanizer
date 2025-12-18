"""Directed mutation operators for gradient-based evolution.

These operators perform specific, narrow tasks based on the type of defect
detected in the current draft, rather than asking the LLM to "fix everything."
"""

from pathlib import Path
from typing import List, Optional
from src.ingestion.blueprint import SemanticBlueprint
from src.atlas.rhetoric import RhetoricalType


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


class MutationOperator:
    """Base class for mutation operators."""

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300
    ) -> str:
        """Generate a mutated version of the draft.

        Args:
            current_draft: Current draft to mutate.
            blueprint: Original semantic blueprint.
            author_name: Target author name.
            style_dna: Style DNA description.
            rhetorical_type: Rhetorical mode.
            llm_provider: LLM provider instance.
            temperature: Temperature for generation.
            max_tokens: Maximum tokens to generate.

        Returns:
            Mutated text.
        """
        raise NotImplementedError


class SemanticInjectionOperator(MutationOperator):
    """Operator for inserting missing keywords (trigger: low recall).

    This operator focuses on adding missing concepts without changing
    the style or rewriting valid parts.
    """

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300,
        missing_keywords: Optional[List[str]] = None
    ) -> str:
        """Insert missing keywords into the draft.

        Args:
            missing_keywords: List of keywords that are missing from the draft.
        """
        if missing_keywords is None:
            # Extract missing keywords by comparing blueprints
            try:
                from src.ingestion.blueprint import BlueprintExtractor
                extractor = BlueprintExtractor()
                draft_blueprint = extractor.extract(current_draft)
                input_keywords = blueprint.core_keywords
                draft_keywords = draft_blueprint.core_keywords
                missing_keywords = list(input_keywords - draft_keywords)[:5]  # Top 5 missing
            except Exception:
                missing_keywords = []

        if not missing_keywords:
            return current_draft  # Nothing to inject

        keywords_text = ", ".join(missing_keywords)

        # Get all required keywords from blueprint (for anchoring)
        all_keywords = sorted(blueprint.core_keywords) if blueprint.core_keywords else []
        required_keywords_text = ", ".join(all_keywords) if all_keywords else "None"

        system_prompt = f"""You are a precision mechanic working on text transformation.
Your task is to INSERT missing keywords into a sentence using Chain-of-Thought reasoning.
You must preserve the existing structure and only add the missing concepts."""

        user_prompt = f"""### TASK: Semantic Injection
**Goal:** Add missing keywords while maintaining perfect grammar.
**Original text:** "{blueprint.original_text}"
**Current draft:** "{current_draft}"
**Missing keywords that MUST be included:** {keywords_text}
**ALL required keywords (do not delete any of these):** {required_keywords_text}

### INSTRUCTIONS
1. **Step 1 (Reasoning):** Analyze the current draft and identify where the missing words logically belong. Consider grammatical structure and natural word order.

2. **Step 2 (Rough Draft):** Write a version that has ALL required keywords (it's okay if it's clunky or awkward). Ensure every keyword from the required list is present.

3. **Step 3 (Polish):** Smooth out Step 2 into natural English while keeping ALL keywords. Fix any grammatical issues introduced in Step 2.

**CRITICAL:**
- Ensure the final output contains **ALL** of these words: {required_keywords_text}
- Do NOT delete existing valid keywords to make room for new ones
- Do NOT change the style or rewrite valid parts unnecessarily

**Output:** Return ONLY the final polished sentence from Step 3."""

        try:
            mutated = llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            from src.generator.llm_interface import clean_generated_text
            mutated = clean_generated_text(mutated)
            mutated = mutated.strip()
            return mutated if mutated else current_draft
        except Exception:
            return current_draft


class GrammarRepairOperator(MutationOperator):
    """Operator for fixing sentence structure (trigger: low fluency).

    This operator focuses on grammar and flow issues like stilted phrasing,
    missing articles, and awkward constructions.
    """

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300
    ) -> str:
        """Fix sentence structure and grammar."""
        system_prompt = f"""You are a professional copyeditor specializing in sentence structure.
Your task is to fix grammatical issues and improve flow WITHOUT changing the meaning or style.
Focus on:
- Converting stilted patterns like 'will, in time, be' to 'eventually is'
- Ensuring Subject-Verb-Object agreement
- Fixing missing articles
- Improving natural word order"""

        # Get all required keywords from blueprint (for anchoring)
        all_keywords = sorted(blueprint.core_keywords) if blueprint.core_keywords else []
        required_keywords_text = ", ".join(all_keywords) if all_keywords else "None"

        user_prompt = f"""### TASK: Grammar Repair
**Goal:** Fix sentence structure while preserving ALL meaning and keywords.
**Original text:** "{blueprint.original_text}"
**Current draft:** "{current_draft}"
**ALL required keywords (must keep all of these):** {required_keywords_text}

### INSTRUCTIONS
1. **Step 1 (Reasoning):** Identify the grammatical issues in the current draft. What makes it awkward or incorrect?

2. **Step 2 (Rough Draft):** Write a grammatically correct version that includes ALL required keywords. It's okay if it's not perfectly polished yet.

3. **Step 3 (Polish):** Refine Step 2 into natural, flowing English while ensuring ALL keywords remain present.

**CRITICAL:**
- Keep ALL keywords: {required_keywords_text}
- Do NOT remove any keywords while fixing grammar
- Preserve the exact meaning from the original text

**Output:** Return ONLY the final polished sentence from Step 3."""

        try:
            mutated = llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            from src.generator.llm_interface import clean_generated_text
            mutated = clean_generated_text(mutated)
            mutated = mutated.strip()
            return mutated if mutated else current_draft
        except Exception:
            return current_draft


class StylePolishOperator(MutationOperator):
    """Operator for enhancing style (trigger: high recall+fluency, low style).

    This operator focuses on matching the target author's voice while
    preserving all nouns and verbs exactly.
    """

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300,
        style_lexicon: Optional[List[str]] = None,
        style_structure: Optional[str] = None
    ) -> str:
        """Enhance style while preserving meaning.

        Args:
            style_lexicon: Optional list of style words from extracted DNA.
            style_structure: Optional structural rule from extracted DNA.
        """
        system_prompt = f"""You are a ghostwriter specializing in the style of {author_name}.

Style characteristics: {style_dna}

Your task is to rewrite the sentence in this style while preserving ALL nouns and verbs exactly.
You may change:
- Adjectives and adverbs
- Sentence structure
- Word order
- Phrasing

You must NOT change:
- Core nouns
- Core verbs
- Meaning"""

        # Get all required keywords from blueprint (for anchoring)
        all_keywords = sorted(blueprint.core_keywords) if blueprint.core_keywords else []
        required_keywords_text = ", ".join(all_keywords) if all_keywords else "None"

        # Use extracted style DNA if provided
        style_instruction = ""
        if style_lexicon:
            lexicon_text = ", ".join(style_lexicon[:10])  # Limit to 10
            style_instruction = f"\n**Style Lexicon (integrate these words):** {lexicon_text}"
            if style_structure:
                style_instruction += f"\n**Style Structure:** {style_structure}"

        user_prompt = f"""### TASK: Style Polish
**Goal:** Enhance style while preserving ALL meaning and keywords.
**Original text:** "{blueprint.original_text}"
**Current draft:** "{current_draft}"
**Rhetorical type:** {rhetorical_type.value}
**ALL required keywords (must keep all of these):** {required_keywords_text}{style_instruction}

### INSTRUCTIONS
1. **Step 1 (Reasoning):** Analyze how the current draft differs from the target style. What stylistic elements need enhancement?

2. **Step 2 (Rough Draft):** Write a version in the target style that includes ALL required keywords. Focus on style transformation first, even if it's not perfectly polished.

3. **Step 3 (Polish):** Refine Step 2 into natural, flowing prose in the target style while ensuring ALL keywords remain present.

**CRITICAL:**
- Keep ALL keywords: {required_keywords_text}
- Preserve ALL nouns and verbs exactly
- Do NOT remove any keywords while enhancing style
- Maintain the exact meaning from the original text
{f"- **Vocabulary:** You MUST attempt to integrate words from this list: {', '.join(style_lexicon[:10])}" if style_lexicon else ""}
{f"- **Structure:** Follow this structural rule: {style_structure}" if style_structure else ""}

**Output:** Return ONLY the final polished sentence from Step 3."""

        try:
            mutated = llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            from src.generator.llm_interface import clean_generated_text
            mutated = clean_generated_text(mutated)
            mutated = mutated.strip()
            return mutated if mutated else current_draft
        except Exception:
            return current_draft


class DynamicStyleOperator(MutationOperator):
    """Operator for dynamic style enhancement using RAG-extracted style DNA.

    This operator uses style lexicon and structure extracted from ChromaDB examples
    to perform style transfer without hardcoding.
    """

    def generate(
        self,
        current_draft: str,
        blueprint: SemanticBlueprint,
        author_name: str,
        style_dna: str,
        rhetorical_type: RhetoricalType,
        llm_provider,
        temperature: float = 0.6,
        max_tokens: int = 300,
        style_lexicon: Optional[List[str]] = None,
        style_structure: Optional[str] = None,
        style_tone: Optional[str] = None
    ) -> str:
        """Enhance style using dynamically extracted style DNA.

        Args:
            style_lexicon: List of style words extracted from examples.
            style_structure: Structural rule extracted from examples.
            style_tone: Tone adjective extracted from examples.
        """
        if not style_lexicon:
            # Fallback to regular style polish if no lexicon
            operator = StylePolishOperator()
            return operator.generate(
                current_draft, blueprint, author_name, style_dna,
                rhetorical_type, llm_provider, temperature, max_tokens
            )

        # Get all required keywords from blueprint
        all_keywords = sorted(blueprint.core_keywords) if blueprint.core_keywords else []
        required_keywords_text = ", ".join(all_keywords) if all_keywords else "None"

        lexicon_text = ", ".join(style_lexicon[:10])  # Limit to 10

        system_prompt = f"""You are a ghostwriter specializing in the style of {author_name}.

Style Tone: {style_tone or 'Authoritative'}
Style Structure: {style_structure or 'Standard sentence structure'}

Your task is to rewrite the sentence to match this author's distinctive voice using their signature vocabulary and structural patterns."""

        user_prompt = f"""### TASK: Dynamic Style Enhancement
**Goal:** Rewrite text to match target voice using extracted style characteristics.
**Original text:** "{blueprint.original_text}"
**Current draft:** "{current_draft}"
**Rhetorical type:** {rhetorical_type.value}
**ALL required keywords (must keep all of these):** {required_keywords_text}

**Extracted Style DNA:**
- **Lexicon (integrate these words):** {lexicon_text}
- **Tone:** {style_tone or 'Authoritative'}
- **Structure:** {style_structure or 'Standard sentence structure'}

### INSTRUCTIONS
1. **Step 1 (Reasoning):** Analyze the current draft. How can you integrate the style lexicon words naturally? What structural changes are needed?

2. **Step 2 (Rough Draft):** Write a version that:
   - Includes ALL required keywords
   - Attempts to integrate words from the lexicon: {lexicon_text}
   - Follows the structural rule: {style_structure or 'Standard sentence structure'}
   It's okay if it's clunky at this stage.

3. **Step 3 (Polish):** Refine Step 2 into natural, flowing prose that:
   - Maintains ALL required keywords
   - Naturally incorporates style lexicon words
   - Follows the structural pattern
   - Preserves the exact meaning

**CRITICAL:**
- Keep ALL keywords: {required_keywords_text}
- Do NOT lose original meaning
- You MAY add 'connective tissue' words to fit the style
- Integrate lexicon words naturally, don't force them

**Output:** Return ONLY the final polished sentence from Step 3."""

        try:
            mutated = llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            from src.generator.llm_interface import clean_generated_text
            mutated = clean_generated_text(mutated)
            mutated = mutated.strip()
            return mutated if mutated else current_draft
        except Exception:
            return current_draft


# Operator constants for easy reference
OP_SEMANTIC_INJECTION = "semantic_injection"
OP_GRAMMAR_REPAIR = "grammar_repair"
OP_STYLE_POLISH = "style_polish"
OP_DYNAMIC_STYLE = "dynamic_style"


def get_operator(operator_type: str) -> MutationOperator:
    """Get a mutation operator by type.

    Args:
        operator_type: One of OP_SEMANTIC_INJECTION, OP_GRAMMAR_REPAIR, OP_STYLE_POLISH, OP_DYNAMIC_STYLE.

    Returns:
        MutationOperator instance.
    """
    operators = {
        OP_SEMANTIC_INJECTION: SemanticInjectionOperator(),
        OP_GRAMMAR_REPAIR: GrammarRepairOperator(),
        OP_STYLE_POLISH: StylePolishOperator(),
        OP_DYNAMIC_STYLE: DynamicStyleOperator()
    }
    return operators.get(operator_type, GrammarRepairOperator())  # Default fallback


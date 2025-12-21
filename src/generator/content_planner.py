"""Content Planner for distributing neutral content into sentence slots.

This module handles the "Fact Distribution" step of the Assembly Line architecture,
mapping neutral content facts into specific sentence slots based on structure map.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from src.generator.llm_provider import LLMProvider
from src.utils.nlp_manager import NLPManager


class ContentPlanner:
    """Distributes neutral content facts into sentence slots for assembly-line construction."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the content planner.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.llm_provider = LLMProvider(config_path=config_path)
        # Lazy-load spaCy model for semantic drift detection
        self._nlp = None

    def plan_content(self, neutral_text: str, structure_map: List[Dict], author_name: str, source_text: Optional[str] = None) -> List[str]:
        """Distribute neutral content facts into sentence slots.

        Args:
            neutral_text: Neutral summary text containing facts to distribute
            structure_map: List of slot dicts with target_len and type
            author_name: Author name (for context, not used in planning)
            source_text: Optional original source text for semantic drift detection

        Returns:
            List of content strings, one per slot, aligned with structure_map
        """
        if not neutral_text or not neutral_text.strip():
            return [""] * len(structure_map)

        if not structure_map:
            return []

        # Build slot descriptions
        slot_descriptions = []
        for i, slot in enumerate(structure_map, 1):
            target_len = slot.get('target_len', 20)
            slot_type = slot.get('type', 'moderate')
            slot_descriptions.append(f"Slot {i}: {target_len} words ({slot_type})")

        slot_descriptions_text = "\n".join(slot_descriptions)

        # Load prompt template
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        try:
            template_path = prompts_dir / "content_planner_user.md"
            if template_path.exists():
                template = template_path.read_text().strip()
            else:
                # Fallback template
                template = self._get_fallback_template()
        except Exception:
            template = self._get_fallback_template()

        # Format prompt (handle both template formats)
        try:
            user_prompt = template.format(
                neutral_text=neutral_text,
                slot_descriptions=slot_descriptions_text,
                num_slots=len(structure_map)
            )
        except KeyError:
            # Template might have different placeholders, use fallback
            user_prompt = f"""# Task: Content Distribution

## Neutral Content:
{neutral_text}

## Structure Slots:
{slot_descriptions_text}

## Instructions:
Distribute the content above into {len(structure_map)} slots. Each slot has a target word count.
Output only the content for each slot, one per line. Do not include slot numbers or labels.
"""

        # Call LLM
        try:
            system_prompt = (
                "You are a content planner. Distribute facts into sentence slots based on target word counts.\n"
                "CRITICAL: You are a structural analyst, not a creative writer. "
                "Your content plan must map STRICTLY to the events present in the Source Text. "
                "Do NOT invent sequel events, future timelines, or abstract theoretical conclusions "
                "unless they are explicitly in the source."
            )
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=False,
                temperature=0.3,  # Low temperature for factual distribution
                max_tokens=1000
            )

            # Parse response: one content string per line
            content_slots = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove slot numbers/labels if present
                if ':' in line:
                    # Try to extract content after colon
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        line = parts[1].strip()
                # Preserve EMPTY markers (case-insensitive check)
                if line.upper() == "EMPTY":
                    content_slots.append("EMPTY")
                elif line:
                    content_slots.append(line)

            # Ensure we have the right number of slots
            while len(content_slots) < len(structure_map):
                content_slots.append("")
            if len(content_slots) > len(structure_map):
                content_slots = content_slots[:len(structure_map)]

            # Apply semantic drift guard if source_text is provided
            if source_text:
                content_slots = self._prune_hallucinated_slots(content_slots, source_text)

            return content_slots

        except Exception as e:
            print(f"Warning: Content planning failed: {e}")
            # Fallback: distribute evenly
            return self._fallback_distribution(neutral_text, len(structure_map))

    def _get_fallback_template(self) -> str:
        """Get fallback template if file doesn't exist."""
        return """# Task: Content Distribution

## Neutral Content:
{neutral_text}

## Structure Slots:
{slot_descriptions}

## Instructions:
Distribute the content above into {num_slots} slots. Each slot has a target word count.
Output only the content for each slot, one per line. Do not include slot numbers or labels.
"""

    def _fallback_distribution(self, neutral_text: str, num_slots: int) -> List[str]:
        """Fallback: distribute content evenly across slots."""
        if num_slots == 0:
            return []

        # Simple split by sentences
        sentences = neutral_text.split('. ')
        if len(sentences) < num_slots:
            # Not enough sentences, pad with empty
            result = sentences + [''] * (num_slots - len(sentences))
        else:
            # Distribute sentences across slots
            per_slot = len(sentences) // num_slots
            result = []
            for i in range(num_slots):
                start = i * per_slot
                end = start + per_slot if i < num_slots - 1 else len(sentences)
                slot_content = '. '.join(sentences[start:end])
                if slot_content and not slot_content.endswith('.'):
                    slot_content += '.'
                result.append(slot_content)

        return result[:num_slots]

    def _get_nlp(self):
        """Get or load spaCy model for semantic similarity checking.

        Returns:
            spaCy language model, or None if unavailable
        """
        if self._nlp is None:
            try:
                self._nlp = NLPManager.get_nlp()
            except Exception:
                # If spaCy not available, return None (check will be skipped)
                self._nlp = False
        return self._nlp if self._nlp is not False else None

    def _prune_hallucinated_slots(self, plan: List[str], source_text: str) -> List[str]:
        """
        Removes final slots that have drifted semantically from the source text.
        Uses spaCy similarity to detect 'filler' topics.

        Args:
            plan: List of content strings, one per slot
            source_text: Original source text to compare against

        Returns:
            Pruned plan with hallucinated final slots removed
        """
        if not plan or not source_text:
            return plan

        # Get spaCy model
        nlp = self._get_nlp()
        if not nlp:
            # If spaCy not available, return original plan
            return plan

        # Process source once
        try:
            source_doc = nlp(source_text)
        except Exception:
            # If processing fails, return original plan
            return plan

        # Check the final slot (where hallucinations usually happen)
        last_slot_content = plan[-1] if plan else ""

        # Skip if slot is empty or marked as EMPTY
        if not last_slot_content or last_slot_content.strip().upper() == "EMPTY":
            return plan

        # Process last slot content
        try:
            topic_doc = nlp(last_slot_content)
        except Exception:
            # If processing fails, return original plan
            return plan

        # Calculate similarity
        # Note: spaCy similarity is 0.0-1.0.
        # Generic filler usually scores < 0.3 against specific narratives.
        try:
            similarity = source_doc.similarity(topic_doc)
        except Exception:
            # If similarity calculation fails, return original plan
            return plan

        # Threshold: 0.25 is a conservative cutoff for "completely unrelated"
        if similarity < 0.25:
            print(f"Warning: Pruning hallucinated slot: '{last_slot_content[:50]}...' (Similarity: {similarity:.2f})")
            return plan[:-1]  # Drop the last slot

        return plan


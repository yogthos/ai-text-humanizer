"""Semantic Fracturer for Dynamic Graph Fracturing.

Groups propositions into logical clusters that match the target style's density,
preserving causal chains and logical relationships.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generator.llm_provider import LLMProvider


class SemanticFracturer:
    """Groups propositions into logical clusters for sentence generation."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize the Semantic Fracturer.

        Args:
            llm_provider: LLM provider instance for grouping propositions.
        """
        self.llm_provider = llm_provider

    def _strip_markdown_code_blocks(self, text: str) -> str:
        """Strip markdown code blocks from text.

        Args:
            text: Text that may contain markdown code blocks.

        Returns:
            Text with code blocks removed.
        """
        # Remove ```json ... ``` blocks
        text = re.sub(r'```json\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
        # Remove ``` ... ``` blocks (generic)
        text = re.sub(r'```\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
        # Remove any remaining ``` markers
        text = text.replace('```', '')
        return text.strip()

    def fracture(
        self,
        propositions: List[str],
        target_density: int = 4,
        max_density: int = 6
    ) -> List[List[int]]:
        """Group propositions into logical clusters.

        Args:
            propositions: List of proposition strings.
            target_density: Target number of propositions per group.
            max_density: Maximum number of propositions per group (hard cap).

        Returns:
            List of lists of indices, where each inner list represents a cluster.
            Example: [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
        """
        if not propositions:
            return []

        if len(propositions) <= target_density:
            # If we have fewer propositions than target, return single group
            return [list(range(len(propositions)))]

        # Try LLM-based fracturing
        try:
            groups = self._fracture_with_llm(propositions, target_density, max_density)
            if groups and self._validate_groups(groups, len(propositions)):
                return groups
        except Exception as e:
            print(f"Warning: LLM fracturing failed: {e}, using fallback")

        # Fallback: simple fixed-size chunking
        return self._fallback_chunking(propositions, target_density)

    def _fracture_with_llm(
        self,
        propositions: List[str],
        target_density: int,
        max_density: int
    ) -> Optional[List[List[int]]]:
        """Use LLM to group propositions into logical clusters.

        Args:
            propositions: List of proposition strings.
            target_density: Target number of propositions per group.
            max_density: Maximum number of propositions per group.

        Returns:
            List of lists of indices, or None if LLM call fails.
        """
        # Create indexed list for LLM
        indexed_propositions = [
            f"{i}: {prop}" for i, prop in enumerate(propositions)
        ]

        system_prompt = (
            "You are a Logic Architect. Your task is to group atomic facts "
            "into logical clusters that form coherent sentences. "
            "Preserve causal chains and logical relationships."
        )

        user_prompt = f"""Propositions:
{chr(10).join(indexed_propositions)}

Target Density: ~{target_density} propositions per group.
Max Density: {max_density} propositions per group.

Task: Group these propositions into logical clusters that form coherent sentences.
- Keep causal chains together (e.g., if proposition 1 causes proposition 2, keep them in the same group).
- Keep related facts together (e.g., conditions and consequences).
- Each group should have roughly {target_density} propositions (aim for {target_density}, but can vary from 2 to {max_density}).
- Do NOT break logical dependencies.

Output: JSON list of lists of indices (e.g., [[0, 1, 2], [3, 4], [5, 6, 7, 8]]).
Each inner list represents one sentence cluster."""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=500
            )

            # Strip markdown code blocks
            response = self._strip_markdown_code_blocks(response)

            # Parse JSON
            groups = json.loads(response)

            # Validate format
            if not isinstance(groups, list):
                return None

            # Ensure all inner items are lists
            for group in groups:
                if not isinstance(group, list):
                    return None

            return groups

        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Failed to parse LLM response for fracturing: {e}")
            return None

    def _validate_groups(self, groups: List[List[int]], total_propositions: int) -> bool:
        """Validate that groups cover all propositions exactly once.

        Args:
            groups: List of lists of indices.
            total_propositions: Total number of propositions.

        Returns:
            True if groups are valid, False otherwise.
        """
        if not groups:
            return False

        # Collect all indices
        all_indices = []
        for group in groups:
            if not isinstance(group, list):
                return False
            for idx in group:
                if not isinstance(idx, int):
                    return False
                if idx < 0 or idx >= total_propositions:
                    return False
                all_indices.append(idx)

        # Check that all indices are present exactly once
        if len(all_indices) != total_propositions:
            return False

        if set(all_indices) != set(range(total_propositions)):
            return False

        return True

    def _fallback_chunking(
        self,
        propositions: List[str],
        target_density: int
    ) -> List[List[int]]:
        """Fallback to simple fixed-size chunking.

        Args:
            propositions: List of proposition strings.
            target_density: Target number of propositions per chunk.

        Returns:
            List of lists of indices.
        """
        groups = []
        for i in range(0, len(propositions), target_density):
            group = list(range(i, min(i + target_density, len(propositions))))
            groups.append(group)
        return groups


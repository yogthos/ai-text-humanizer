"""Semantic analyzer for extracting atomic propositions from text.

This module provides functionality to decompose text into atomic propositions,
stripping away style and connectors to extract core factual statements.
"""

import json
import re
from pathlib import Path
from typing import List, Optional
from src.generator.llm_provider import LLMProvider


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'semantic_analyzer_propositions.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


class PropositionExtractor:
    """Extracts atomic propositions from text using LLM-based decomposition."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the proposition extractor.

        Args:
            config_path: Path to configuration file for LLM provider.
        """
        self.llm_provider = LLMProvider(config_path=config_path)

    def extract_atomic_propositions(self, text: str) -> List[str]:
        """Extract atomic propositions from text.

        Args:
            text: Input text to analyze (can be paragraph or multiple sentences).

        Returns:
            List of atomic proposition strings (standalone factual statements).
        """
        if not text or not text.strip():
            return []

        # Load and format the prompt
        prompt_template = _load_prompt_template("semantic_analyzer_propositions.md")
        prompt = prompt_template.format(text=text.strip())

        try:
            # Call LLM with JSON mode
            response = self.llm_provider.call(
                system_prompt="You are a semantic analyzer that extracts atomic propositions from text. Output only valid JSON arrays.",
                user_prompt=prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3  # Low temperature for consistent extraction
            )

            # Parse JSON response
            propositions = self._parse_json_response(response)

            # Clean and validate propositions
            cleaned = [p.strip() for p in propositions if p and p.strip()]

            return cleaned if cleaned else [text.strip()]  # Fallback to original if extraction fails

        except Exception as e:
            # Fallback: if LLM extraction fails, try simple sentence splitting
            return self._fallback_extraction(text)

    def _parse_json_response(self, response: str) -> List[str]:
        """Parse JSON response from LLM.

        Args:
            response: LLM response string (may contain JSON or markdown).

        Returns:
            List of proposition strings.
        """
        # Try to extract JSON array from response
        # Handle cases where LLM wraps JSON in markdown code blocks
        json_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if json_match:
            try:
                propositions = json.loads(json_match.group(0))
                if isinstance(propositions, list):
                    return [str(p) for p in propositions]
            except json.JSONDecodeError:
                pass

        # Try parsing entire response as JSON
        try:
            propositions = json.loads(response.strip())
            if isinstance(propositions, list):
                return [str(p) for p in propositions]
        except json.JSONDecodeError:
            pass

        # If all parsing fails, return empty list
        return []

    def _fallback_extraction(self, text: str) -> List[str]:
        """Fallback extraction using simple sentence splitting.

        Args:
            text: Input text.

        Returns:
            List of sentences (as atomic propositions).
        """
        # Simple sentence splitting as fallback
        sentences = re.split(r'[.!?]+\s+', text)
        cleaned = [s.strip() for s in sentences if s.strip()]
        return cleaned if cleaned else [text.strip()]


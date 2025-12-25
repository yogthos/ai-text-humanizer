"""Structure Analyzer for extracting logical chains from text.

Analyzes text chunks to identify the sequence of logical operations
(e.g., DEFINITION|CONTRAST|ATTRIBUTION) for structural indexing.
"""

import json
import re
from typing import Optional

from src.generator.llm_provider import LLMProvider


class StructureAnalyzer:
    """Analyzes text to extract logical structure chains."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize the Structure Analyzer.

        Args:
            llm_provider: LLM provider instance for analysis.
        """
        self.llm_provider = llm_provider

        # Valid logical tags
        self.valid_tags = {
            'DEFINITION', 'CONTRAST', 'CAUSALITY', 'SEQUENCE',
            'PURPOSE', 'ATTRIBUTION', 'ORIGIN', 'COMPOSITION', 'ELABORATION'
        }

    def analyze_structure(self, text: str) -> str:
        """
        Analyzes text and returns a 'Logic Chain' string (e.g., 'DEFINITION|CONTRAST').

        Args:
            text: Input text chunk to analyze (typically 1-3 sentences).

        Returns:
            Pipe-separated string of logical tags representing the flow.
            Example: "DEFINITION|CONTRAST|ATTRIBUTION"
        """
        if not text or not text.strip():
            return "DEFINITION"  # Default fallback

        system_prompt = (
            "You are a Logical Structure Analyzer. Your task is to identify "
            "the sequence of logical operations in a text chunk. Focus on the "
            "rhetorical flow, not the specific content."
        )

        user_prompt = f"""Analyze the rhetorical structure of the following text.
Identify the logical steps taken in the sentences.

Valid Tags:
- DEFINITION (Defining what something is, "X is Y")
- CONTRAST (Saying what it is not, or opposing ideas, "Not X, but Y")
- CAUSALITY (Explaining why/because, "Because X, Y")
- SEQUENCE (Temporal order, 'then', 'next', "First X, then Y")
- PURPOSE (Explaining the goal/use, "X in order to Y", "X is for Y")
- ATTRIBUTION (Citing a source or author, "X by Y", "Y said X")
- ORIGIN (Source/location, "X from Y", "X comes from Y")
- COMPOSITION (Parts/wholes, "X consists of Y", "X includes Y")
- ELABORATION (Giving details or lists, "X, specifically Y", "X, Y, and Z")

TEXT: "{text}"

Return ONLY a pipe-separated list of tags representing the logical flow.
- Use 1-3 tags maximum (most texts have 1-2 logical steps)
- Order matters: tags should reflect the sequence of logical operations
- Example: "DEFINITION|CONTRAST"
- Example: "ATTRIBUTION|PURPOSE"
- Example: "CONTRAST"
- If the text is primarily a definition, return "DEFINITION"
- If the text contrasts ideas, return "CONTRAST" or "DEFINITION|CONTRAST"

Output format: Return only the pipe-separated tags, nothing else."""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=False,
                temperature=0.3,
                max_tokens=100
            )

            # Clean and validate the response
            logic_signature = self._clean_tags(response)
            return logic_signature

        except Exception as e:
            # Fallback to default
            print(f"Warning: Structure analysis failed: {e}, using default 'DEFINITION'")
            return "DEFINITION"

    def _clean_tags(self, response: str) -> str:
        """Clean and validate the tag string from LLM response.

        Args:
            response: Raw LLM response string.

        Returns:
            Cleaned pipe-separated tag string.
        """
        if not response:
            return "DEFINITION"

        # Strip whitespace and newlines
        cleaned = response.strip()

        # Remove markdown code blocks if present
        cleaned = re.sub(r'```[a-z]*\n?', '', cleaned)
        cleaned = cleaned.replace('```', '')

        # Remove quotes if present
        cleaned = cleaned.strip('"\'')
        cleaned = cleaned.strip()

        # Extract pipe-separated tags
        # Handle various formats: "DEFINITION|CONTRAST", "DEFINITION | CONTRAST", etc.
        tags = [tag.strip().upper() for tag in cleaned.split('|')]

        # Validate tags
        valid_tags = []
        for tag in tags:
            # Remove any trailing punctuation or extra text
            tag_clean = re.sub(r'[^A-Z].*$', '', tag)
            if tag_clean in self.valid_tags:
                valid_tags.append(tag_clean)
            elif tag_clean:
                # Try to match partial tags (e.g., "DEF" -> "DEFINITION")
                matched = None
                for valid_tag in self.valid_tags:
                    if valid_tag.startswith(tag_clean) or tag_clean in valid_tag:
                        matched = valid_tag
                        break
                if matched:
                    valid_tags.append(matched)

        # If no valid tags found, return default
        if not valid_tags:
            return "DEFINITION"

        # Limit to 3 tags max
        valid_tags = valid_tags[:3]

        return '|'.join(valid_tags)


"""Input Logic Mapper.

Converts unstructured propositions into structured logical dependency graphs
using rhetorical topology analysis.
"""

import json
import re
from typing import List, Optional, Dict, Any

from src.generator.llm_provider import LLMProvider


class InputLogicMapper:
    """Maps propositions to logical dependency graphs using LLM analysis."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize the Input Logic Mapper.

        Args:
            llm_provider: LLM provider instance for graph generation.
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

    def map_propositions(self, propositions: List[str]) -> Optional[Dict[str, Any]]:
        """Map propositions to a logical dependency graph.

        Args:
            propositions: List of atomic facts (e.g., ['The phone needs power', 'Without it, it dies']).

        Returns:
            Dictionary with 'mermaid', 'description', 'node_map', and 'node_count',
            or None if mapping fails.
        """
        if not propositions:
            raise ValueError("Propositions list cannot be empty")

        system_prompt = (
            "You are a Rhetorical Topologist. Analyze the logical flow. "
            "When describing the graph, use structural terms like 'contrast', 'causality', "
            "'concession', 'enumeration', and 'definition' to match the style index."
        )

        # Format propositions for the prompt
        propositions_text = "\n".join([f"{i}. {prop}" for i, prop in enumerate(propositions)])

        user_prompt = f"""Propositions:
{propositions_text}

Task:
1. **CRITICAL: De-duplicate facts.** If the text repeats an idea (e.g., 'He made it' and 'It was created by him'), extract it ONLY ONCE. Remove redundant propositions before creating the graph.
2. Create a Mermaid graph using IDs P0, P1, P2... corresponding to the list index (after deduplication).
3. Label edges with logic (cause, contrast, support).
4. Write a 1-sentence description of the flow using rhetorical topology terms (causal, contrastive, conditional, enumeration, concession, definition).
5. Analyze the propositions. What is the primary rhetorical intent? Choose one: `DEFINITION`, `ARGUMENT`, `NARRATIVE`, `INTERROGATIVE`, `IMPERATIVE`.
   - `DEFINITION`: Explaining what something is (e.g., "The phone is a tool.").
   - `ARGUMENT`: Persuading or debating (e.g., "Therefore, we must reject...").
   - `NARRATIVE`: Telling a sequence of events (e.g., "At that time, the army moved...").
   - `INTERROGATIVE`: Asking rhetorical questions.
   - `IMPERATIVE`: Giving commands/directives.

Output JSON:
{{
  "mermaid": "graph LR; P0 --cause--> P1",
  "description": "A causal chain...",
  "intent": "DEFINITION",
  "node_map": {{ "P0": "text of prop 0", "P1": "text of prop 1" }}
}}"""

        try:
            # Call LLM with require_json=True
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=500
            )

            # Strip markdown code blocks before parsing
            response = self._strip_markdown_code_blocks(response)

            # Parse JSON
            result = json.loads(response)

            # Validate required fields
            required_fields = ['mermaid', 'description', 'node_map']
            if not all(field in result for field in required_fields):
                print(f"Warning: Missing required fields in LLM response. Expected: {required_fields}")
                return None

            # Validate intent if present, or set default
            valid_intents = ['DEFINITION', 'ARGUMENT', 'NARRATIVE', 'INTERROGATIVE', 'IMPERATIVE']
            if 'intent' not in result:
                # Try to infer from description if intent not provided
                description_lower = result.get('description', '').lower()
                if 'definition' in description_lower or 'define' in description_lower:
                    result['intent'] = 'DEFINITION'
                elif 'narrative' in description_lower or 'sequence' in description_lower or 'event' in description_lower:
                    result['intent'] = 'NARRATIVE'
                elif 'question' in description_lower or 'interrogative' in description_lower:
                    result['intent'] = 'INTERROGATIVE'
                elif 'command' in description_lower or 'imperative' in description_lower or 'directive' in description_lower:
                    result['intent'] = 'IMPERATIVE'
                else:
                    result['intent'] = 'ARGUMENT'  # Default fallback
            elif result.get('intent') not in valid_intents:
                print(f"Warning: Invalid intent '{result.get('intent')}', defaulting to 'ARGUMENT'")
                result['intent'] = 'ARGUMENT'

            # Validate types
            if not isinstance(result['mermaid'], str):
                print("Warning: Mermaid field is not a string")
                return None

            if not isinstance(result['description'], str):
                print("Warning: Description field is not a string")
                return None

            if not isinstance(result['node_map'], dict):
                print("Warning: Node map field is not a dictionary")
                return None

            # Validate node_map keys match P0, P1, P2... pattern
            node_keys = list(result['node_map'].keys())
            expected_keys = [f"P{i}" for i in range(len(propositions))]
            if set(node_keys) != set(expected_keys):
                print(f"Warning: Node map keys don't match expected pattern. Got: {node_keys}, Expected: {expected_keys}")
                # Try to normalize - this is lenient but logs a warning
                normalized_map = {}
                for i, prop in enumerate(propositions):
                    key = f"P{i}"
                    if key in result['node_map']:
                        normalized_map[key] = result['node_map'][key]
                    else:
                        # Use the proposition text if key not found
                        normalized_map[key] = prop
                result['node_map'] = normalized_map

            # Add node_count derived from node_map length
            result['node_count'] = len(result['node_map'])

            return result

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            print(f"Warning: LLM call failed: {e}")
            return None

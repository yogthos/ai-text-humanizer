"""Semantic fidelity validation using DeepSeek.

Replaces the multi-step post-processing pipeline (semantic verification,
repair loop, grammar correction, repetition reduction) with a single
DeepSeek call that validates semantic equivalence and fixes only genuine
errors while preserving the restyled text's voice and structure.
"""

import json
from dataclasses import dataclass, field
from typing import List

from ..utils.logging import get_logger
from ..utils.prompts import load_prompt

logger = get_logger(__name__)


@dataclass
class FidelityResult:
    """Result of semantic fidelity validation."""
    original: str
    corrected: str
    changes: List[dict] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return len(self.changes) > 0


def validate_semantic_fidelity(
    original: str,
    restyled: str,
    critic_provider,
) -> FidelityResult:
    """Validate and minimally correct restyled text for semantic fidelity.

    Uses the critic LLM to compare the restyled text against the original,
    fixing only genuine semantic errors (missing facts, reversed meaning,
    broken grammar) while preserving the restyled voice and structure.

    Args:
        original: The original source text (ground truth for meaning).
        restyled: The restyled text to validate.
        critic_provider: LLM provider for the validation call.

    Returns:
        FidelityResult with the corrected text and list of changes made.
    """
    system_prompt = load_prompt("semantic_fidelity")
    user_prompt = f"ORIGINAL:\n{original}\n\nRESTYLED:\n{restyled}"

    try:
        response = critic_provider.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=max(1024, len(restyled.split()) * 4),
            require_json=True,
        )

        result = json.loads(response)
        changes = result.get("changes", [])
        corrected = result.get("result", restyled)

        if not isinstance(corrected, str) or not corrected.strip():
            corrected = restyled
            logger.warning("Semantic fidelity returned empty/null result, keeping original restyled text")

        if changes and isinstance(changes, list):
            for change in changes:
                if isinstance(change, dict):
                    logger.info(f"Semantic fix: {change.get('issue', '?')}")

        return FidelityResult(
            original=original,
            corrected=corrected,
            changes=changes,
        )

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse fidelity response: {e}")
        return FidelityResult(original=original, corrected=restyled)
    except Exception as e:
        logger.warning(f"Semantic fidelity check failed: {e}")
        return FidelityResult(original=original, corrected=restyled)

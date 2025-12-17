"""Adversarial Critic for style transfer quality control.

This module provides a critic LLM that evaluates generated text against
a reference style paragraph to detect style mismatches and "AI slop".
"""

import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory.

    Args:
        template_name: Name of the template file (e.g., 'critic_system.md')

    Returns:
        Template content as string.
    """
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    template_path = prompts_dir / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return template_path.read_text().strip()


def _load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from config.json."""
    with open(config_path, 'r') as f:
        return json.load(f)


def _call_deepseek_api(system_prompt: str, user_prompt: str, api_key: str, api_url: str, model: str) -> str:
    """Call DeepSeek API for critic evaluation."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"}
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", "")


def critic_evaluate(
    generated_text: str,
    structure_match: str,
    situation_match: Optional[str] = None,
    original_text: Optional[str] = None,
    config_path: str = "config.json"
) -> Dict[str, any]:
    """Evaluate generated text against dual RAG references.

    Uses an LLM to compare the generated text with structure_match (for rhythm)
    and situation_match (for vocabulary) to check for style mismatches.

    Args:
        generated_text: The generated text to evaluate.
        structure_match: Reference paragraph for rhythm/structure evaluation.
        situation_match: Optional reference paragraph for vocabulary evaluation.
        original_text: Original input text (for checking reference/quotation preservation).
        config_path: Path to configuration file.

    Returns:
        Dictionary with:
        - "pass": bool - Whether the generated text passes style check
        - "feedback": str - Specific feedback on what to improve
        - "score": float - Style match score (0-1)
    """
    config = _load_config(config_path)
    provider = config.get("provider", "deepseek")

    if provider == "deepseek":
        deepseek_config = config.get("deepseek", {})
        api_key = deepseek_config.get("api_key")
        api_url = deepseek_config.get("api_url")
        model = deepseek_config.get("critic_model", deepseek_config.get("editor_model", "deepseek-chat"))

        if not api_key or not api_url:
            raise ValueError("DeepSeek API key or URL not found in config")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Load system prompt from template
    system_prompt = _load_prompt_template("critic_system.md")

    # Build sections for user prompt
    structure_section = f"""STRUCTURAL REFERENCE (for rhythm/structure):
"{structure_match}"
"""

    if situation_match:
        situation_section = f"""
SITUATIONAL REFERENCE (for vocabulary):
"{situation_match}"
"""
    else:
        situation_section = """
SITUATIONAL REFERENCE: Not provided (no similar topic found in corpus).
"""

    if original_text:
        original_section = f"""
ORIGINAL TEXT (for reference preservation check):
"{original_text}"
"""
        preservation_checks = """
- CRITICAL: All [^number] style citation references from Original Text must be preserved exactly
- CRITICAL: All direct quotations (text in quotes) from Original Text must be preserved exactly"""
        preservation_instruction = """

CRITICAL: Check that all [^number] citations and direct quotations from Original Text are preserved exactly in Generated Text. If any are missing or modified, this is a critical failure."
"""
    else:
        original_section = ""
        preservation_checks = ""
        preservation_instruction = ""

    # Load and format user prompt template
    template = _load_prompt_template("critic_user.md")
    user_prompt = template.format(
        structure_section=structure_section,
        situation_section=situation_section,
        original_section=original_section,
        generated_text=generated_text,
        preservation_checks=preservation_checks,
        preservation_instruction=preservation_instruction
    )

    # Note: preservation_instruction already includes the critical failure message if original_text is provided

    try:
        # Call API
        response_text = _call_deepseek_api(system_prompt, user_prompt, api_key, api_url, model)

        # Parse JSON response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback: create result from text
                result = {
                    "pass": False,
                    "feedback": "Could not parse critic response. Please retry.",
                    "score": 0.5
                }

        # Ensure required fields
        if "pass" not in result:
            result["pass"] = result.get("score", 0.5) >= 0.75
        if "feedback" not in result:
            result["feedback"] = "No specific feedback provided."
        if "score" not in result:
            result["score"] = 0.5

        # Ensure types
        result["pass"] = bool(result["pass"])
        result["score"] = float(result["score"])
        result["feedback"] = str(result["feedback"])

        return result

    except Exception as e:
        # Fallback on error
        return {
            "pass": True,  # Don't block generation on critic failure
            "feedback": f"Critic evaluation failed: {str(e)}",
            "score": 0.5
        }


class ConvergenceError(Exception):
    """Raised when critic cannot converge to minimum score threshold."""
    pass


def _extract_issues_from_feedback(feedback: str) -> List[str]:
    """Extract individual issues/action items from feedback text.

    Args:
        feedback: Feedback string that may contain numbered items or sentences.

    Returns:
        List of extracted issues/action items.
    """
    issues = []

    # Try to extract numbered items (e.g., "1. Make sentences shorter")
    numbered_pattern = r'\d+[\.\)]\s*([^\.]+(?:\.[^\.]+)*)'
    matches = re.findall(numbered_pattern, feedback)
    if matches:
        issues.extend([m.strip() for m in matches])
        return issues

    # If no numbered items, try to split by sentences and extract action-oriented ones
    sentences = re.split(r'[\.!?]\s+', feedback)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Look for action-oriented sentences (contain verbs like "make", "use", "match", "fix")
        action_verbs = ['make', 'use', 'match', 'fix', 'change', 'adjust', 'improve', 'reduce', 'increase']
        if any(verb in sentence.lower() for verb in action_verbs):
            issues.append(sentence)

    # If still no issues, return the whole feedback as a single issue
    if not issues and feedback.strip():
        issues.append(feedback.strip())

    return issues


def _group_similar_issues(issues: List[str]) -> Dict[str, List[str]]:
    """Group similar issues together.

    Args:
        issues: List of issue strings.

    Returns:
        Dictionary mapping canonical issue to list of similar variations.
    """
    issue_groups: Dict[str, List[str]] = {}

    for issue in issues:
        issue_lower = issue.lower()
        matched = False

        # Check if this issue is similar to any existing group
        for canonical, variations in issue_groups.items():
            canonical_lower = canonical.lower()

            # Simple similarity check: shared keywords
            issue_words = set(issue_lower.split())
            canonical_words = set(canonical_lower.split())

            # If they share significant words, group them
            common_words = issue_words & canonical_words
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'in', 'on', 'at', 'for', 'with'}
            common_words -= stop_words

            if len(common_words) >= 2 or (len(common_words) == 1 and len(issue_words) <= 3):
                issue_groups[canonical].append(issue)
                matched = True
                break

        if not matched:
            # Create new group with this issue as canonical
            issue_groups[issue] = [issue]

    return issue_groups


def _consolidate_feedback(feedback_history: List[str]) -> str:
    """Consolidate multiple feedback attempts into clear action items.

    Extracts key issues, groups similar ones, and formats as actionable steps.

    Args:
        feedback_history: List of feedback strings from previous attempts.

    Returns:
        Consolidated feedback as numbered action items.
    """
    if not feedback_history:
        return ""

    # Extract all issues from all feedback
    all_issues = []
    for feedback in feedback_history:
        issues = _extract_issues_from_feedback(feedback)
        all_issues.extend(issues)

    if not all_issues:
        return ""

    # Group similar issues and count frequency
    issue_groups = _group_similar_issues(all_issues)

    # Sort by frequency (most mentioned first)
    sorted_issues = sorted(issue_groups.items(), key=lambda x: len(x[1]), reverse=True)

    # Format as action items
    action_items = []
    for idx, (canonical_issue, variations) in enumerate(sorted_issues, 1):
        # Use the canonical issue (first one found)
        action_items.append(f"{idx}. {canonical_issue}")

    return "ACTION ITEMS TO FIX:\n" + "\n".join(action_items)


def generate_with_critic(
    generate_fn,
    content_unit,
    structure_match: str,
    situation_match: Optional[str] = None,
    config_path: str = "config.json",
    max_retries: int = 3,
    min_score: float = 0.75
) -> Tuple[str, Dict[str, any]]:
    """Generate text with adversarial critic loop.

    Generates text, evaluates it with critic, and retries with feedback
    if the style doesn't match well enough.

    Args:
        generate_fn: Function to generate text (takes content_unit, structure_match, situation_match, hint).
        content_unit: ContentUnit to generate from.
        structure_match: Reference paragraph for rhythm/structure (required).
        situation_match: Optional reference paragraph for vocabulary.
        config_path: Path to configuration file.
        max_retries: Maximum number of retry attempts (default: 3).
        min_score: Minimum critic score to accept (default: 0.75).

    Returns:
        Tuple of:
        - generated_text: Best generated text
        - critic_result: Final critic evaluation result
    """
    if not structure_match:
        # No structure match, cannot proceed
        generated = content_unit.original_text
        return generated, {"pass": False, "feedback": "No structure match provided", "score": 0.0}

    best_text = None
    best_score = 0.0
    best_result = None
    feedback_history: List[str] = []

    for attempt in range(max_retries):
        # Build consolidated hint from all previous attempts
        hint = None
        if feedback_history:
            consolidated = _consolidate_feedback(feedback_history)
            if consolidated:
                hint = f"CRITICAL FEEDBACK FROM ALL PREVIOUS ATTEMPTS:\n{consolidated}\n\nPlease address ALL of these action items in your rewrite."

        # Generate with hint from previous attempts
        generated = generate_fn(content_unit, structure_match, situation_match, config_path, hint=hint)

        # Evaluate with critic
        critic_result = critic_evaluate(
            generated,
            structure_match,
            situation_match,
            original_text=content_unit.original_text,
            config_path=config_path
        )
        score = critic_result.get("score", 0.0)

        # Track best result
        if score > best_score:
            best_text = generated
            best_score = score
            best_result = critic_result

        # Check if we should accept
        if critic_result.get("pass", False) and score >= min_score:
            return generated, critic_result

        # Accumulate feedback for next attempt
        feedback = critic_result.get("feedback", "")
        if feedback:
            feedback_history.append(feedback)

    # Enforce minimum score - raise exception if not met
    if best_score < min_score:
        consolidated_feedback = _consolidate_feedback(feedback_history) if feedback_history else "No specific feedback available"
        raise ConvergenceError(
            f"Failed to converge to minimum score {min_score} after {max_retries} attempts. "
            f"Best score: {best_score:.3f}. Issues: {consolidated_feedback}"
        )

    # Return best result if it meets threshold
    return best_text or generated, best_result or {"pass": False, "feedback": "Max retries reached", "score": best_score}


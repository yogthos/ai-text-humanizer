Task 1: Verify factual recall.
Compare these two sentences and determine if the generated text preserves the core meaning of the original.

Original: "{original_text}"

Generated: "{generated_text}"

Does the generated text preserve the core meaning of the original? Pay attention to:
- Logical relationships: Does it add conditions (e.g., "only when", "if... then") that weren't in the original?
- Meaning shifts: Does it change a universal statement into a conditional, or vice versa?
- Contradictions: Does it imply something that contradicts the original meaning?

{global_context_section}

{intent_task}

Respond with JSON:
{{
    "meaning_preserved": true/false,
    "confidence": 0.0-1.0,
    {intent_score_field}
    "explanation": "brief reason (mention if conditional relationship, logic mismatch, meaning shift, or intent mismatch detected)"
}}


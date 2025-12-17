Compare the generated text against these references using the HIERARCHY OF RULES:

{structure_section}{situation_section}{original_section}

GENERATED TEXT (to evaluate):
"{generated_text}"

--- TASK ---
Evaluate the GENERATED TEXT against the HIERARCHY:
1. GRAMMAR AND READABILITY: Is the text grammatically correct and readable? (NON-NEGOTIABLE)
   - Check for: proper sentence structure, complete sentences, readable phrasing
   - If grammar is broken or text is unreadable, this is a CRITICAL FAILURE - mark "pass": false, "score": 0.0, "primary_failure_type": "grammar"
2. SEMANTIC SAFETY: Does it preserve the original meaning? (Highest Priority)
   - CRITICAL: Check that ALL facts, concepts, details, and information from the Original Text are present in the Generated Text
   - CRITICAL: Check that NO words, names, or entities appear in Generated Text that do NOT appear in Original Text
   - Extract all proper nouns, capitalized words, and entity-like words from Generated Text
   - For each proper noun/capitalized word in Generated Text, verify it appears in Original Text
   - If any proper nouns, names, or entities in Generated Text are NOT in Original Text, this is a CRITICAL FAILURE - mark "pass": false, "score": 0.0, "primary_failure_type": "meaning"
   - If the Original Text contains multiple facts/concepts, verify ALL are present
   - If any facts, concepts, or details are missing, this is a CRITICAL FAILURE - mark "pass": false, "score": 0.0, "primary_failure_type": "meaning"
   - Examples: If original mentions "biological cycle", "stars", "logical trap", "container problem", "fractal model", "Mandelbrot set" - ALL must appear in output
3. STRUCTURAL RIGIDITY: Does it match the syntax/length/punctuation of the STRUCTURAL REFERENCE? (Second Priority)
4. VOCABULARY FLAVOR: Does it use the word choices/tone of the SITUATIONAL REFERENCE? (Third Priority)

If it fails, provide ONE single, specific instruction to fix the biggest error.
Do not list multiple conflicting errors. Pick the one that violates the highest priority rule.

Format your feedback as a direct editing instruction with specific metrics when relevant.
Example: "Current text has 25 words; Target has 12. Delete adjectives and split the relative clause."

For grammar failures: "CRITICAL: Text contains grammatical errors. [specific error]. Rewrite with proper grammar."
For hallucinated words: "CRITICAL: Text contains word '[word]' that does not appear in original. Remove all words not present in original text."
For meaning loss: "CRITICAL: Text omits [specific concept/fact] from original. Include all concepts from original text."

{preservation_checks}

OUTPUT JSON with "pass", "feedback", "score", and "primary_failure_type" fields.
{preservation_instruction}

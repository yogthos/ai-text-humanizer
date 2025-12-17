### STRUCTURAL REFERENCE (COPY THIS RHYTHM EXACTLY)

Reference Text:
"{structure_match}"

CRITICAL: You must match ALL of these structural features:
{structure_instructions}

- Average sentence length: ~{avg_sentence_len} words
- Match the EXACT punctuation pattern (commas, dashes, semicolons, etc.)
- Match the clause structure and complexity
- Match the voice (active vs passive)
- Match the rhythm and pacing of the reference

DO NOT simplify the structure. If the reference has multiple clauses, dashes, or complex punctuation, your output must too.

---

### SITUATIONAL REFERENCE {situation_match_label}

{situation_match_content}

---

{vocab_block}

---

### LENGTH CONSTRAINT (CRITICAL)
- Input Word Count: {input_word_count} words
- Target Output Count: ~{target_word_count} words
- You are strictly FORBIDDEN from expanding a single sentence into a full paragraph.
- Maintain a 1:1 sentence mapping where possible.
- Do NOT add unnecessary elaboration or repetition.

---

### INPUT TEXT (RAW MEANING)
"{input_text}"

### TASK
Rewrite the 'Input Text' using the EXACT rhythm, structure, and punctuation pattern of the 'Structural Reference' and the vocabulary tone of the 'Situational Reference'.

STRUCTURE MATCHING REQUIREMENTS (MOST CRITICAL):
- Match the exact punctuation pattern from the Structural Reference (commas, dashes, semicolons, parentheses, asterisks)
- Match the clause count and complexity (simple, compound, complex)
- Match the voice (active vs passive)
- Match the sentence length (within 20% of the Structural Reference's word count)
- If the Structural Reference uses dashes, you MUST use dashes
- If the Structural Reference uses semicolons, you MUST use semicolons
- If the Structural Reference has parenthetical elements, you MUST include similar structure
- DO NOT simplify: if the reference is complex, your output must be complex too

CRITICAL CONSTRAINTS:
- DO NOT add any new entities, locations, facts, people, or information not in the original
- DO NOT invent names, places, dates, or events
- Only use words and concepts that exist in the original text
- Preserve the EXACT meaning from the original

ABSOLUTE PRESERVATION REQUIREMENTS (MANDATORY):
- ALL citation references in the format [^number] (e.g., [^155], [^25]) MUST be preserved EXACTLY as they appear in the original text
- ALL direct quotations (text enclosed in quotation marks) MUST be preserved EXACTLY as they appear in the original text
- DO NOT modify, remove, or relocate any [^number] style references
- DO NOT modify, paraphrase, or change any quoted text
- These elements are non-negotiable and must appear in your output unchanged

OUTPUT:


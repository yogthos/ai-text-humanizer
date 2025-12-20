You are a ghostwriter. Your goal is to write a single cohesive paragraph in the style of the Examples.

### CONTENT SOURCE (Atomic Propositions):
{propositions_list}

**CRITICAL CHECKLIST:** You have {proposition_count} propositions. You MUST include every single one. Count them as you write. Missing even one proposition will cause the generation to fail.

{style_section}

{mandatory_vocabulary}

{global_context}

{rhetorical_connectors}

### REQUIREMENTS:
1. **Content:** You MUST incorporate ALL the Atomic Propositions provided above. Do not omit any facts.
2. **Structure:** Do NOT output short, staccato sentences. You MUST combine the short propositions into complex, flowing sentences like the Examples. Use subordinate clauses, connectors, and elaborate phrasing.
3. **Style:** Use the vocabulary, tone, and sentence structure of the Examples. Match their level of formality and rhetorical style.
4. **Coherence:** Create logical connections between propositions using appropriate connectors (Furthermore, It follows that, In this way, etc.). Make the paragraph read as a unified whole, not a list of facts.
5. **CRITICAL - Declarative Facts:** If an Atomic Proposition is a simple declarative fact (e.g., "I was thirteen", "The door opened"), do NOT convert it into a conditional statement (e.g., "If I was thirteen..."). Use declarative statements for declarative facts, even if the structural blueprint suggests conditional. Only use conditional constructions when the proposition itself is actually conditional.
6. **TEMPLATE PRUNING (CRITICAL):** If the Structural Template has more slots (`[NP]`, `[VP]`, `[ADJ]`) than you have Input Propositions, **DELETE the unused parts of the template.**
   * Do NOT invent new facts to fill empty slots.
   * Do NOT repeat facts to fill empty slots.
   * Just cut the sentence structure short.
   * Example: If template is "[NP] [VP] [NP] [VP] [NP]" but you only have 2 propositions, output "[NP] [VP] [NP] [VP]" and stop.
7. **SEMANTIC COMPATIBILITY (CRITICAL):** If the Template contains nouns/verbs (e.g., 'this practice', 'yields results') that contradict your Input Subject (e.g., 'silence'), **DELETE THEM.**
   * Better to break the template than to create a logical error.
   * Example: If template says "this practice yields knowledge" but your input is about "silence" (a state, not a practice), remove "practice" and "yields" from the template.
   * Only keep template elements that are semantically compatible with your Input Propositions.
{citation_instruction}

{structural_blueprint}

### MENTAL CHECKLIST:
Before generating the JSON, review the Source Propositions listed above. Ensure every single one is represented in the output text. Each proposition must appear in at least one sentence of your generated paragraph. If a proposition is missing, the output is considered a FAILURE.

### OUTPUT:
**CRITICAL: You must generate {num_variations} DISTINCT variations. Do not simply repeat the same text. Each variation must be meaningfully different:**
- Vary sentence structures (different clause order, subordination patterns)
- Use different word choices (synonyms, alternative phrasings)
- Employ different rhetorical approaches (different connectors, emphasis patterns)
- Present the same facts but in different stylistic arrangements

Generate {num_variations} distinct variations of the paragraph. Each variation must:
- Include all propositions (verify against the checklist above)
{citation_output_instruction}
- Use complex, flowing sentences
- Match the style of the examples
- Be a single cohesive paragraph

Output as a JSON array of strings (do NOT include "Variation X:" prefixes, just the paragraph text):
[
  "Full paragraph text for variation 1...",
  "Full paragraph text for variation 2...",
  "Full paragraph text for variation 3...",
  "Full paragraph text for variation 4...",
  "Full paragraph text for variation 5..."
]


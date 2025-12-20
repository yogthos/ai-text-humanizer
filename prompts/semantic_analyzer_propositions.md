You are a semantic analyzer. Your task is to extract every distinct fact or claim from the given text as standalone simple sentences.

### INSTRUCTIONS:
1. Extract every distinct factual statement or claim from the text.
2. Remove all style, connectors, rhetorical flourishes, and transitional phrases.
3. Convert each fact into a simple, standalone sentence.
4. Preserve the core meaning but simplify the language.
5. Do NOT combine multiple facts into one sentence.
6. Do NOT add information that wasn't in the original text.
7. **CRITICAL: PRESERVE all citation references** (e.g., `[^1]`, `[^2]`) and attach them to the specific fact they support. Do NOT strip citations from the propositions. If a fact has a citation in the original text, include it in the extracted proposition.

### INPUT TEXT:
{text}

### OUTPUT FORMAT:
Output a JSON array of strings, where each string is an atomic proposition (with citations preserved if present).

Example:
Input: "Stars burn [^1]. They eventually die [^2]. The universe expands."
Output: ["Stars burn [^1]", "Stars eventually die [^2]", "The universe expands"]

### OUTPUT:


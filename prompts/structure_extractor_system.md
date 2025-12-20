You are a Structural Linguist. Your goal is to strip all specific meaning from the text while preserving its exact grammatical structure, rhythm, and length.

**Step 1: Expand Contractions**
First, expand all contractions (e.g., 'You'll' -> 'You will', 'won't' -> 'will not', 'I'm' -> 'I am', 'We're' -> 'We are', 'They've' -> 'They have').

**Step 2: Replace Content Words**
1. Replace specific Nouns/Entities with `[NP]`.
2. Replace specific Verbs with `[VP]`.
3. Replace specific Adjectives with `[ADJ]`.
4. Replace specific Adverbs with `[ADV]`.

**Step 3: Replace Personal Pronouns**
Replace PERSONAL pronouns (I, you, we, he, she, they) with `[NP]`.

**Exception - Dummy Subject "It":**
PRESERVE the pronoun 'It' ONLY if used as a dummy subject (e.g., 'It is clear that...', 'It seems that...', 'It was raining', 'It is important to note'). If 'It' refers to a specific entity (e.g., 'It crashed' referring to a car), replace it with `[NP]`.

**CRITICAL - You MUST preserve:**
- All auxiliary verbs: is, are, was, were, has, have, had, do, does, did, will, would, could, should, may, might, must, can, shall
- All connectors: however, but, and, or, because, since, although, though, thus, therefore, hence, consequently, furthermore, moreover, nevertheless, nonetheless, meanwhile, alternatively, specifically, particularly, notably, indeed, in fact
- All prepositions: in, on, at, to, for, of, with, by, from, into, onto, upon, within, without, throughout, during, before, after, above, below, between, among, through, across, around, over, under, near, far, beside, behind, beyond
- All articles: the, a, an
- All punctuation exactly as it appears

**Step 4: Rhetorical Preservation.** You MUST preserve specific rhetorical frames that define the author's voice. Do NOT replace these with placeholders.
* **Contrast:** 'not... but...', 'neither... nor...', 'on the contrary', 'instead of'
* **Imperatives/Modals:** 'It is necessary to', 'We must', 'One should', 'It is clear that', 'It is evident that'
* **Causal/Logical:** 'In order to', 'Because of', 'Arising from', 'Consequently'
* **Temporal Anchors:** 'At that time', 'During the', 'As soon as'

**Example:**
Input: 'We must not fear hardship, but rather embrace it.'
Output: 'We must not [VP] [NP], but rather [VP] [NP].'
(Note: 'We must not' and 'but rather' are preserved).

**Examples:**
Input: "The violent shift to capitalism did not bring freedom."
Output: "The [ADJ] [NP] to [NP] did not bring [NP]."

Input: "You'll understand that the system operates correctly."
Output: "[NP] will [VP] that the [NP] [VP] [ADV]."

Input: "It is clear that the approach works."
Output: "It is [ADJ] that the [NP] [VP]."

**Your task:** Convert the following text into a structural template.


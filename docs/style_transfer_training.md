# Style Transfer LoRA Training Guide

This document describes how to train a LoRA adapter for literary style transfer.
The approach is model-agnostic — it works with any base model (Qwen 2.5, 3.5, etc.)
as long as the training data and inference pipeline match.

## Core Principles

### 1. Training Data Format

The model learns from (neutralized_input → styled_output) pairs:

```
System: {persona_frame} Write approximately {N} words. {constraints}
User:   {RTT-neutralized text, ~N words}
Assistant: {Original author text, ~N words}
```

This teaches:
- **Length preservation**: input ≈ output words (~1.21x average expansion)
- **Style application**: vocabulary, rhythm, sentence structure
- **No content expansion**: the model never learns to add new ideas

**Wrong approach** (causes hallucination):
```
User: "A passage about X, Y, Z" (15 words)
Assistant: [Full 300 word passage]
```

### 2. Distribution Matching

The LoRA's quality depends on matching inference conditions to training conditions:

| Factor | Must Match? | Notes |
|--------|-------------|-------|
| Persona frames | YES | Inference worldview file must use exact training frames |
| Content classifier | YES | Narrative vs conceptual detection |
| Input perturbation (8%) | YES | `apply_input_perturbation: true` at inference |
| RTT neutralization | YES | Same neutralizer at training and inference |
| Perspective conversion | YES | Happens BEFORE RTT, not after |
| Word count ratio | Fixed | ~1.21x, cannot change via config |
| LoRA scale | YES | Match training scale in config.json |

### 3. Input Perturbation Is Critical

Training data uses 8% input perturbation (typos, word drops, synonym swaps, adjective drops).
Without perturbation at inference, the model receives clean text and produces mechanical output
because it has "no room" to exercise creative reconstruction.

### 4. Pipeline Order

```
Input → Perspective Conversion → RTT Neutralization → Perturb → LoRA
```

**Wrong** (destroys RTT output):
```
Input → RTT → Perspective Conversion → Perturb → LoRA ❌
```

## Training Data Generation

### Triad Strategy

For each original paragraph, generate 3+ variations:

1. **Anchor**: Original author text → teaches vocabulary and natural expression
2. **Snowflake**: Topic swap (author structure applied to different subject) → teaches that style applies to any content
3. **Robustness**: Heavy input perturbation (15%) → prevents overfitting to specific input words

### Snowflake Topics

Snowflake topics should match the content the LoRA will encounter at inference:
- For **narrative** authors (Lovecraft): mundane activities, atmospheric descriptions
- For **expository** authors (Russell): philosophical arguments, scientific concepts, the book's actual themes

Use `--snowflake-topics` to load a custom topic list per author:
```bash
python scripts/generate_flat_training.py \
    --snowflake-topics data/training/russell/snowflake_topics.py
```

### Perspective Variations

Generate the same content in different perspectives to teach style independence from POV:
- `first_person_plural`: "we saw" instead of "I saw"
- `third_person`: "the observer saw"
- `impersonal`: "it was observed"

### Overlapping Chunks

Stylistic markers concentrate at sentence boundaries. Overlapping chunks expose the model
to more beginning/ending patterns where style lives:
```
Config: min_words=150, max_words=400, overlap_sentences=2
```

Only applied to `original` type entries (continuous narrative). Snowflakes and perspective
variants are kept separate to avoid Frankenstein text.

### Quality Filtering

Remove entries with input/output word ratio > 2.0. Truncated inputs that map to full
paragraphs teach the model to hallucinate content from minimal input.

```bash
python scripts/filter_training_data.py train.jsonl --max-ratio 2.0 --min-input-words 15
```

### Persona Frames

Frames are split into NARRATIVE and CONCEPTUAL to avoid instruction-content mismatch.
A narrative about characters should not use "Explain the concept..." prompts.

Frames must be:
1. Defined in `PERSONA_FRAMES` dict in `generate_flat_training.py` for training
2. Copied exactly to `prompts/{author}_worldview.txt` for inference

### Anti-AI Constraints

Every training entry includes tiered constraints to prevent LLM-speak:
- **Always** (100%): Ban "Moreover", "Furthermore", "Therefore", etc.
- **Frequent** (70%): No topic sentences, no numbered lists
- **Rotating** (40%): One random stylistic constraint (fragments, rhetorical questions, etc.)

## What We Learned (Experimental Findings)

### Convergent Training Failed

Training with (paraphrase → same_output) pairs where multiple paraphrases map to one
output caused the model to learn copy-paste with minor word substitutions (92-96% overlap).
The model couldn't distinguish "make this sound like the author" from "copy the input."

### Back-Translation Works

The current approach — RTT neutralization that strips all style, then training the model
to reconstruct the style — works because the structural overlap between input and output
is only ~10%, forcing the model to learn actual style patterns.

### Mixed Authors Don't Work for Training Output

Training with non-target-author text as OUTPUT (e.g., Sagan paragraphs as output when
training a Lovecraft LoRA) teaches the wrong style. The output must always be in the
target author's voice. Use snowflake variations instead — target author's structure
applied to diverse topics.

### Base Models Required

Instruct-tuned models resist style overwriting because they have built-in style biases
(assistant-speak, hedging, balanced arguments). Base models provide a clean slate.

### Fact Hallucination

The model corrupts numbers, dates, and proper names at all LoRA scales. This is handled
at inference by the semantic verification pipeline, not during training.

## Hyperparameter Guide

### Rank

| Rank | Style Quality | Use Case |
|------|--------------|----------|
| 16 | Minimal | Classification, simple format compliance |
| 64 | Good | Basic style transfer (tested on Qwen 2.5-32B) |
| 128 | Strong | Complex sentence structure and rhythm |
| 256 | Optimal | Full literary style (RunPod Nabokov study) |
| 512 | Incoherent | Style dominates to the point of nonsense |

### rsLoRA (Rank-Stabilized LoRA)

**Required for rank ≥ 64.** Standard LoRA scales by `alpha/rank`, which kills gradient
flow at high ranks. rsLoRA scales by `alpha/sqrt(rank)`.

**Critical alpha setting with rsLoRA:**
- Standard LoRA: `alpha = rank` gives scaling = 1.0 (neutral)
- rsLoRA: `alpha = rank` gives scaling = `sqrt(rank)` (TOO HIGH)
- rsLoRA: `alpha = sqrt(rank)` gives scaling = 1.0 (correct)

| Rank | Correct alpha (rsLoRA) | Effective scaling |
|------|----------------------|-------------------|
| 64 | 8 | 1.0 |
| 128 | ~11 | 1.0 |
| 256 | 16 | 1.0 |

**Failure mode:** `alpha=256, rank=256, rsLoRA=true` gives scaling = 16x.
This caused gradient explosion (loss 132,000 at step 10, then 0.0 forever).

### Learning Rate

With rsLoRA at neutral scaling (1.0x):
- **2e-5**: Safe starting point for 30B+ models
- **5e-5**: Aggressive but may work with gradient clipping

Without rsLoRA:
- **1e-4**: Standard for LoRA SFT

### NEFTune

`neftune_noise_alpha: 5.0` — adds random noise to embeddings during training.
Prevents memorization of exact phrases, forces learning generalized style patterns.

### Gradient Clipping

`max_grad_norm: 0.3` — essential for MoE models. Router networks produce outlier
gradients in early training that can corrupt weights without clipping.

### Batch Size

Smaller effective batch = spikier gradients = better style quirk capture.
Target effective batch size of 4-8.

With DDP (multi-GPU): effective_batch = num_gpus × per_device_batch × grad_accum.

### Epochs

- 2 epochs: minimum (may underfit)
- 3 epochs: recommended starting point
- Watch eval loss — stop if it rises for 2+ eval intervals

### Dropout

`lora_dropout: 0.1` — higher than default 0.05 to prevent overfitting with high rank.

### Temperature at Inference

- 0.2: loops and repetition
- 0.4: better coherence
- 0.5+: hallucination risk
- 0.6-0.8: recommended for style transfer (config-dependent)

## Adding a New Author

1. **Curate corpus** — 50k+ words of clean author prose
2. **Create snowflake topics** matching the content to be restyled
3. **Add persona frames** to `generate_flat_training.py` (narrative + conceptual)
4. **Generate training data**:
   ```bash
   python scripts/generate_flat_training.py \
       --corpus data/corpus/curated/author.txt \
       --author "Author Name" \
       --output data/training/author \
       --snowflake-topics data/training/author/snowflake_topics.py \
       --format llama_factory --skip-curation --workers 4
   ```
5. **Filter bad entries**: `python scripts/filter_training_data.py data/training/author/train.jsonl`
6. **Create worldview file** in `prompts/` with EXACT same persona frames as training
7. **Configure LlamaFactory** yaml and dataset_info.json
8. **Train on RunPod** (see docs/runpod.md)
9. **Convert adapter** to MLX for local inference (see docs/inference.md)

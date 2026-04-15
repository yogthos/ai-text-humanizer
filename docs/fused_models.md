# Fused Models

A fused model is a standalone model with LoRA weights merged directly into the base
model. This removes the need for separate adapter files at inference time and can
improve load performance.

The `scripts/fuse_model.py` script supports two fusion backends:

- **MLX** — Apple Silicon only. Works with quantized models, no OOM risk.
- **PEFT** — Cross-platform (PyTorch). Requires enough RAM to load the full model.

## Prerequisites

### MLX fusion

```bash
pip install mlx mlx-lm
```

Requires an MLX-format base model and an MLX-format adapter directory containing
`adapters.safetensors` and `adapter_config.json`.

If your adapter is a raw PEFT checkpoint (HF format with `adapter_model.safetensors`),
convert it first:

```bash
python scripts/convert_peft_to_mlx.py \
    --input checkpoints/checkpoint-10200 \
    --output lora_adapters/my-adapter-mlx \
    --mlx-model models/Qwen2.5-32B-Base-8bit-MLX
```

### PEFT fusion

```bash
pip install torch transformers peft accelerate
```

Requires a HuggingFace base model and a PEFT checkpoint directory containing
`adapter_config.json` and `adapter_model.safetensors`.

## MLX Fusion

Recommended on Apple Silicon. Loads the quantized MLX model + adapter into memory,
fuses them, and saves the result. No PyTorch needed.

```bash
python scripts/fuse_model.py \
    --model models/Qwen2.5-32B-Base-8bit-MLX \
    --checkpoint lora_adapters/howard_russell_checkpoint_10200 \
    --output models/Qwen2.5-32B-howard-lovecraft-10200 \
    --mlx
```

### Re-quantize after fusion

Fusion dequantizes the weights during merging. To re-quantize the output:

```bash
python scripts/fuse_model.py \
    --model models/Qwen2.5-32B-Base-8bit-MLX \
    --checkpoint lora_adapters/howard_russell_checkpoint_10200 \
    --output models/Qwen2.5-32B-howard-lovecraft-10200 \
    --mlx --qbits 8
```

`--qbits` accepts 2, 3, 4, 6, or 8. Use `--group-size 64` (default) to control
quantization granularity.

### Override the LoRA scale at fuse time

The adapter's `adapter_config.json` records the `scale` used during training
(typically 2.0). `--scale` overrides that multiplier when computing the merged
delta — useful for dialling the adapter's influence up or down without
retraining:

```bash
python scripts/fuse_model.py \
    --model models/Qwen2.5-32B-Base-8bit-MLX \
    --checkpoint lora_adapters/howard_russell_checkpoint_10200 \
    --output models/Qwen2.5-32B-howard-lovecraft-10200-weak \
    --mlx --scale 1.0
```

Rule of thumb: `<1.0` weakens the style, `>1.0` amplifies it but may degrade
coherence. MLX-only; not supported by the PEFT path.

## PEFT Fusion

Cross-platform approach using HuggingFace Transformers and PEFT. Loads the full
base model into CPU memory, merges the adapter, and saves.

```bash
python scripts/fuse_model.py \
    --model models/Qwen2.5-32B \
    --checkpoint checkpoints/checkpoint-10200 \
    --output models/Qwen2.5-32B-howard-lovecraft-10200-hf
```

### Convert PEFT output to MLX

If you fuse with PEFT on a Linux/CUDA machine and later want to run inference
on Apple Silicon:

```bash
python scripts/fuse_model.py \
    --model models/Qwen2.5-32B-howard-lovecraft-10200-hf \
    --output models/Qwen2.5-32B-howard-lovecraft-10200-MLX \
    --convert-mlx-only
```

Or do both steps in one command:

```bash
python scripts/fuse_model.py \
    --model models/Qwen2.5-32B \
    --checkpoint checkpoints/checkpoint-10200 \
    --output models/Qwen2.5-32B-howard-lovecraft-10200-hf \
    --convert-mlx
```

This creates the HF fused model and an adjacent `*-MLX` directory.

## Configuration

Add the fused model to `config.json` with `use_adapter: false`:

```json
{
  "generation": {
    "use_adapter": false,
    "models": {
      "models/Qwen2.5-32B-howard-lovecraft-10200": {
        "enabled": true,
        "author": "Howard Russell",
        "temperature": 0.7,
        "top_p": 0.92,
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "max_tokens": 2048,
        "worldview": "howard_russell_worldview.txt"
      }
    }
  }
}
```

### Per-model settings

| Field | Default | Description |
|---|---|---|
| `enabled` | `true` | Skip this model when `false` |
| `author` | `""` | Author name (avoids needing `--author` on CLI) |
| `temperature` | `0.6` | Generation temperature |
| `top_p` | `0.92` | Nucleus sampling threshold |
| `min_p` | `0.05` | Minimum probability filter |
| `repetition_penalty` | `1.15` | Token repetition penalty |
| `max_tokens` | `512` | Max tokens per generation |
| `worldview` | `""` | Persona worldview file in `prompts/` |
| `perspective` | `null` | Override output perspective |
| `verify_entailment` | `null` | Override semantic verification |
| `expand_for_texture` | `null` | Override texture expansion |
| `merge_paragraphs` | `null` | Min words per merged paragraph block |

## Running style transfer

With config (`use_adapter: false` and `models` configured):

```bash
python restyle.py input.md -o output.md
```

With CLI `--model` flag (overrides config):

```bash
python restyle.py input.md -o output.md \
    --model models/Qwen2.5-32B-howard-lovecraft-10200 \
    --author "Howard Russell"
```

## Summary of paths

```
Raw PEFT checkpoint          MLX adapter
checkpoints/checkpoint-10200  lora_adapters/...-10200
         |                            |
         | convert_peft_to_mlx.py     |
         v                            v
         +-------> MLX fusion (--mlx) -----+
                                           |
                                           v
                                    Fused MLX model
                              models/Qwen2.5-32B-...-10200

Raw PEFT checkpoint
checkpoints/checkpoint-10200
         |
         v  PEFT fusion (default)
    Fused HF model
    models/...-hf
         |
         v  --convert-mlx-only
    Fused MLX model
    models/...-MLX
```

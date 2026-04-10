# Qwen 3.5 MoE LoRA Training

Training notes specific to Qwen 3.5 MoE models (35B-A3B).

## Model

- **Qwen/Qwen3.5-35B-A3B-Base**: 35B total params, 3B active per token
- Architecture: `qwen3_5_moe` — Gated DeltaNet hybrid attention + Sparse MoE
  - 40 decoder layers: 30 DeltaNet (linear attention) + 10 full attention (3:1 pattern)
  - 256 routed experts + 1 shared expert per layer, 9 active (8 routed + 1 shared)

## Current Config (Russell, rank 256)

```yaml
model_name_or_path: Qwen/Qwen3.5-35B-A3B-Base
template: qwen3_5_nothink
lora_rank: 256
lora_alpha: 16              # sqrt(256) — gives 1.0x effective scaling with rsLoRA
lora_dropout: 0.1
use_rslora: true
neftune_noise_alpha: 5.0
lora_target: all            # Attention + DeltaNet + shared expert + router
cutoff_len: 2048
per_device_train_batch_size: 1
gradient_accumulation_steps: 4   # With 2x GPU DDP: effective batch = 8
learning_rate: 2.0e-5
num_train_epochs: 3.0
warmup_ratio: 0.10
max_grad_norm: 0.3
optim: paged_adamw_8bit
flash_attn: sdpa
bf16: true
gradient_checkpointing: true
```

## MoE-Specific Considerations

### No QLoRA

QLoRA (4-bit quantization during training) breaks MoE fused expert `nn.Parameter` tensors.
Always use bf16 LoRA.

### Flash Attention

Use `sdpa`, not `fa2`. Flash Attention 2 causes CUDA errors with Qwen 3.5.

### Gradient Clipping

`max_grad_norm: 0.3` — MoE router networks produce outlier gradients in early training.
Without clipping, these spikes corrupt weights.

### LoRA Target Modules

`lora_target: all` applies LoRA to all `nn.Linear` modules:
- **Full attention** (layers 3,7,11,...,39): q_proj, k_proj, v_proj, o_proj
- **DeltaNet linear attention** (layers 0,1,2,4,5,6,...): in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
- **Shared expert**: gate_proj, up_proj, down_proj
- **Router**: shared_expert_gate

Routed expert `nn.Parameter` tensors are NOT targeted (PEFT can't target fused parameters
without `target_parameters`). Research shows attention + shared experts is sufficient.

### DDP on Multi-GPU

DDP replicates the full 70GB model on each GPU — per-GPU memory is the same as single GPU.
2x GPU gives 2x throughput (parallel batches), not 2x memory.

Effective batch with DDP: `num_gpus × per_device_batch × grad_accum`

To actually split the model across GPUs, you'd need DeepSpeed ZeRO Stage 3 or FSDP,
which are more complex to configure with MoE models.

## rsLoRA Alpha Calculation

**This is the #1 source of training failures.** Getting alpha wrong causes immediate
gradient explosion.

With rsLoRA, effective scaling = `alpha / sqrt(rank)`:

| alpha | rank | Effective scaling | Result |
|-------|------|-------------------|--------|
| 256 | 256 | 16.0x | GRADIENT EXPLOSION |
| 128 | 128 | ~11.3x | GRADIENT EXPLOSION |
| 16 | 256 | 1.0x | Correct |
| 11 | 128 | ~1.0x | Correct |
| 8 | 64 | 1.0x | Correct |

**Rule: set alpha = sqrt(rank) for neutral 1.0x scaling.**

## Template

LlamaFactory template: `qwen3_5_nothink`

- Requires LlamaFactory 0.9.5.dev0+ (install from git, not PyPI)
- Requires transformers == 5.2.0 (first version with Qwen 3.5 support)
- The `nothink` variant suppresses `<think>` reasoning tokens

At inference, the base model's tokenizer injects `<think>` tags in the generation prompt.
The code in `lora_generator.py` overrides the chat template to prevent this.

## PEFT → MLX Adapter Conversion

Key prefix mismatch between PEFT and MLX:
- PEFT: `base_model.model.model.language_model.layers.N.`
- MLX: `language_model.model.layers.N.`

The conversion script auto-detects and fixes this when `--mlx-model` is provided:
```bash
python scripts/convert_peft_to_mlx.py \
    --input lora_adapters/russell_peft \
    --output lora_adapters/russell_mlx \
    --mlx-model models/Qwen3.5-35B-A3B-Base-6bit-MLX
```

## Stop Tokens at Inference

Qwen 3.5 base model only has `<|endoftext|>` (248044) as EOS, but chat format uses
`<|im_end|>` (248046) to end assistant turns. Without adding `<|im_end|>` to the stop
token set, generation continues past the response into fake multi-turn conversation.

This is handled in `lora_generator.py`.

## Config Evolution

| Parameter | Round 1 (failed) | Round 2 (current) | Why Changed |
|-----------|-----------------|-------------------|-------------|
| lora_rank | 16 | 256 | 16 = "minimal stylistic influence" |
| lora_alpha | 32 | 16 | rsLoRA needs alpha=sqrt(rank) |
| use_rslora | no | yes | Enables high rank |
| neftune_noise_alpha | — | 5.0 | Prevents memorization |
| lora_dropout | 0.05 | 0.1 | More regularization |
| learning_rate | 1e-4 | 2e-5 | Safe at 1.0x rsLoRA scaling |
| max_grad_norm | — | 0.3 | MoE router stability |
| warmup_ratio | 0.05 | 0.10 | Optimizer needs time for MoE |
| optim | adamw_torch | paged_adamw_8bit | Memory savings |
| cutoff_len | 2048 | 2048 | 4096 OOMs even on H100 80GB |
| GPU | 1x A100 80GB | 2x H100 80GB | Rank 256 needs headroom |

### Failed Run: alpha=256 with rsLoRA

`lora_alpha=256, lora_rank=256, use_rslora=true` gave effective scaling of 16x.
Combined with lr=5e-5, the effective learning rate was 8e-4.
Result: loss spiked to 132,000 at step 10, then collapsed to 0.0 (NaN weights).

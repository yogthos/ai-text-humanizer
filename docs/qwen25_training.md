# Qwen 2.5 LoRA Training

Training notes specific to Qwen 2.5 dense models (14B, 32B).

## Model

- **Qwen2.5-32B-Base-4bit-MLX** for local MLX training/inference
- **Qwen/Qwen2.5-32B** for RunPod/CUDA training
- Architecture: `qwen2` (standard transformer, no MoE)

## Config

```yaml
model_name_or_path: Qwen/Qwen2.5-32B
template: qwen
lora_rank: 64
lora_alpha: 256
lora_dropout: 0.05
lora_target: all
quantization_bit: 4          # QLoRA works fine on dense models
flash_attn: sdpa
optim: paged_adamw_32bit
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
num_train_epochs: 3
max_steps: 600               # ~1 epoch, stop early if val loss plateaus
```

## Key Differences from Qwen 3.5

| Feature | Qwen 2.5 | Qwen 3.5 MoE |
|---------|----------|---------------|
| Architecture | Dense transformer | Gated DeltaNet + MoE |
| QLoRA | Works | Breaks MoE experts |
| Template | `qwen` | `qwen3_5_nothink` |
| Rank needed | 64 sufficient | 128-256 needed |
| rsLoRA | Optional | Required at high rank |
| Gradient clipping | Optional | Required (MoE routers) |
| Adapter key prefix | `model.layers.N.` | `language_model.model.layers.N.` |

## MLX Config (Local Training)

```yaml
model: "./models/Qwen2.5-32B-Base-4bit-MLX"
fine_tune_type: lora
batch_size: 1
grad_accumulation: 4
iters: 2100
learning_rate: 1e-5
num_layers: -1
lora_parameters:
  rank: 64
  scale: 2.0
  dropout: 0.1
  keys:
    - "self_attn.q_proj"
    - "self_attn.k_proj"
    - "self_attn.v_proj"
    - "self_attn.o_proj"
    - "mlp.gate_proj"
    - "mlp.up_proj"
    - "mlp.down_proj"
```

## Known Working Results

Lovecraft adapter trained on Qwen2.5-32B with rank 64 produced detectable style transfer.
This was the baseline that confirmed the training pipeline works before moving to Qwen 3.5.

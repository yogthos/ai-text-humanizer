# RunPod LoRA Training Setup

## Requirements

**Recommended:** 2x H100 80GB SXM for rank 256 training (~$3.29/hr each).
**Minimum:** 1x A100 80GB SXM for rank 128, cutoff 2048 (~$1.64/hr).

| Config | GPU | Rank | Cutoff | Est. VRAM |
|--------|-----|------|--------|-----------|
| Full quality | 2x H100 80GB | 256 | 4096 | ~80 GB / 160 avail |
| Good quality | 1x A100 80GB | 128 | 2048 | ~75 GB / 80 avail |
| Budget | 1x A100 80GB | 64 | 2048 | ~74 GB / 80 avail |

- **Container disk**: 20GB is fine (packages only)
- **Volume disk**: 150GB+ (for model weights + checkpoints)
- **Template**: RunPod PyTorch 2.x (CUDA 12.x)

## Pod Setup

```bash
# tmux so you can disconnect
apt update && apt install -y tmux
tmux new -s train

# attach later
tmux attach -t train

# CRITICAL: Point HF and datasets cache to workspace (root overlay is only 20GB)
export HF_HOME=/workspace/huggingface_cache
export HF_DATASETS_CACHE=/workspace/huggingface_cache/datasets
mkdir -p $HF_DATASETS_CACHE

# Install LlamaFactory from git (0.9.5.dev0+ has qwen3_5 templates)
pip install "llamafactory[torch] @ git+https://github.com/hiyouga/LLaMA-Factory.git"

# needed for paged_adamw_8bit
pip install bitsandbytes

# Install transformers 5.2.0 (first version with Qwen 3.5 support,
# also the max version LlamaFactory 0.9.5.dev0 accepts)
pip install transformers==5.2.0 bitsandbytes
```

## Upload Training Data

Clone the repo directly on the pod:
```bash
cd /workspace
git clone -b qwen-35 <your-repo-url> revenant
```

Then set up the training directory:
```bash
# For Russell:
mkdir -p /workspace/russell_training/data
cp revenant/data/training/russell/LlamaFactory/qwen35_35b_lora.yaml /workspace/russell_training/
cp revenant/data/training/russell/LlamaFactory/dataset_info.json /workspace/russell_training/data/
cp revenant/data/training/russell/train.jsonl /workspace/russell_training/data/

# For Lovecraft:
mkdir -p /workspace/lovecraft_training/data
cp revenant/data/training/lovecraft/LlamaFactory/qwen35_35b_lora.yaml /workspace/lovecraft_training/
cp revenant/data/training/lovecraft/LlamaFactory/dataset_info.json /workspace/lovecraft_training/data/
cp revenant/data/training/lovecraft/LlamaFactory/train_sft.jsonl /workspace/lovecraft_training/data/
```

## Start Training

```bash
cd /workspace/russell_training
llamafactory-cli train qwen35_35b_lora.yaml
```

Model (`Qwen/Qwen3.5-35B-A3B-Base`) auto-downloads from HuggingFace on first run (~70GB).

## Monitor Training

```bash
# In another terminal / tmux pane
tail -f /workspace/russell_training/saves/Qwen3.5-35B-A3B/lora/russell/trainer_log.jsonl
```

Watch val loss — if it increases for 2+ eval intervals (every 200 steps), training is overfitting.

### Expected Loss Curve (Lovecraft round 1: rank 16, 9,371 entries, A100)

| Step | Epoch | Train Loss | Eval Loss |
|------|-------|-----------|-----------|
| 100  | 0.47  | 1.077     | 1.079     |
| 200  | 0.95  | 0.994     | 1.020     |
| 300  | 1.42  | 1.003     | 0.993     |
| 400  | 1.89  | 0.944     | 0.985     |
| 424  | 2.00  | 0.947     | 0.985     |

Note: Round 1 with rank 16 produced a LoRA with minimal style effect. Round 2 uses
rank 256 with rsLoRA, NEFTune, and lower learning rate — expect different loss dynamics.

## Upload Adapter to HuggingFace

```bash
export HF_TOKEN=your_token_here

python -c "
import os
from huggingface_hub import HfApi, login
login(token=os.environ['HF_TOKEN'])
api = HfApi()
api.create_repo('yogthos/russell-qwen35-35b-a3b-lora', private=True, exist_ok=True)
api.upload_folder(
    folder_path='saves/Qwen3.5-35B-A3B/lora/russell/',
    repo_id='yogthos/russell-qwen35-35b-a3b-lora',
)
print('Done!')
"
```

## Convert to MLX for Local Inference (M1 Max 64GB)

### 1. Quantize Base Model

6-bit (~26GB) is recommended — fits comfortably with plenty of headroom.
8-bit (~35GB) works but may OOM on long documents. Do NOT use bf16 (~70GB, won't fit).

Requires up-to-date mlx-lm (`pip install --upgrade mlx-lm`) for Qwen 3.5 MoE support.

```bash
# 6-bit (recommended for 64GB)
venv/bin/python -m mlx_lm convert \
    --hf-path Qwen/Qwen3.5-35B-A3B-Base \
    --mlx-path models/Qwen3.5-35B-A3B-Base-6bit-MLX \
    -q --q-bits 6

# 8-bit (better quality, tighter on memory)
venv/bin/python -m mlx_lm convert \
    --hf-path Qwen/Qwen3.5-35B-A3B-Base \
    --mlx-path models/Qwen3.5-35B-A3B-Base-8bit-MLX \
    -q --q-bits 8
```

Downloads ~70GB bf16 weights on first run, quantizes locally. Subsequent conversions reuse cached weights.

### 2. Convert PEFT Adapter to MLX Format

The LoRA adapter from LlamaFactory is in PEFT (PyTorch) format. MLX needs its own format.
The conversion script:
- Auto-detects all LoRA target modules from the weight names
- Detects and fixes layer key prefix mismatches between PEFT and MLX
  (e.g. PEFT uses `model.language_model.layers.` but MLX uses `language_model.model.layers.`)
- Writes the local quantized model path in adapter configs (avoiding bf16 OOM)

**IMPORTANT:** Always pass `--mlx-model` to point at your local quantized model. Without it,
the adapter configs will reference the HuggingFace repo ID, causing MLX to download and load
the full bf16 model (~66GB) at inference — instant OOM on 64GB machines.

```bash
venv/bin/python scripts/convert_peft_to_mlx.py \
    --input lora_adapters/russell_Qwen3.5-35B-A3B \
    --output lora_adapters/russell_35b_6b_mlx \
    --mlx-model models/Qwen3.5-35B-A3B-Base-6bit-MLX
```

This will:
1. Convert weight keys from PEFT format to MLX format
2. Fix the `model.language_model.layers.` → `language_model.model.layers.` prefix
3. Set `metadata.json:base_model` and `adapter_config.json:model` to the local 6-bit path

### 3. Test Inference

```bash
venv/bin/python restyle.py input.md -o output.md \
    --adapter lora_adapters/russell_35b_6b_mlx \
    --author "Bertrand Russell" --verbose
```

### Memory Budget at Inference

| Component | 6-bit | 8-bit |
|-----------|-------|-------|
| Base model | ~26 GB | ~35 GB |
| LoRA adapter | ~180 MB (rank 256) | ~180 MB |
| KV cache | ~2-4 GB | ~2-4 GB |
| OS + overhead | ~4-6 GB | ~4-6 GB |
| **Total** | **~33 GB** | **~43 GB** |
| **Headroom** | **~31 GB** | **~21 GB** |

## Training Config Reference

Config file: `data/training/russell/LlamaFactory/qwen35_35b_lora.yaml`

| Parameter | Value | Notes |
|-----------|-------|-------|
| model | Qwen/Qwen3.5-35B-A3B-Base | Base, not Instruct |
| template | qwen3_5_nothink | No thinking tokens |
| lora_rank | 256 | Optimal for literary style (RunPod Nabokov study) |
| lora_alpha | 256 | 1:1 ratio with rsLoRA |
| use_rslora | true | Critical — prevents gradient collapse at high ranks |
| neftune_noise_alpha | 5.0 | Prevents memorization, forces generalized style |
| lora_dropout | 0.1 | Prevents overfitting with high rank |
| lora_target | all | Attention + DeltaNet + shared expert + router |
| quantization | none (bf16) | QLoRA breaks MoE expert layers |
| flash_attn | sdpa | FA2 causes CUDA errors with Qwen 3.5 |
| optim | paged_adamw_8bit | 8-bit optimizer saves ~30% memory |
| learning_rate | 5e-5 | Lower than standard — rsLoRA amplifies updates ~16x |
| epochs | 3 | Previous run (2 epochs) hadn't converged |
| effective batch | 16 (2 × 8) | Small batch = spikier gradients, better for style |
| cutoff_len | 4096 | Captures full argumentative structures |

### Config Evolution

| Parameter | Round 1 (failed) | Round 2 | Why |
|-----------|-----------------|---------|-----|
| lora_rank | 16 | 256 | 16 = "minimal stylistic influence" |
| lora_alpha | 32 | 256 | 1:1 ratio with rsLoRA |
| use_rslora | no | yes | Enables high rank without gradient collapse |
| neftune_noise_alpha | none | 5.0 | Prevents memorization |
| lora_dropout | 0.05 | 0.1 | More regularization for higher rank |
| learning_rate | 1e-4 | 5e-5 | rsLoRA amplifies updates, lower LR stabilizes |
| optim | adamw_torch | paged_adamw_8bit | Memory savings for larger config |
| cutoff_len | 2048 | 4096 | Longer context for argumentative prose |
| epochs | 2 | 3 | Loss was still declining |
| GPU | 1x A100 80GB | 2x H100 80GB | Needed for rank 256 + cutoff 4096 |

## Troubleshooting

- **"No space left on device"**: Set `export HF_HOME=/workspace/huggingface_cache` before training. The root overlay is only 20GB by default.
- **"Disk quota exceeded"**: Same cause — HF downloads going to root. Ensure `HF_HOME` is set.
- **I/O error during preprocessing**: Set `export HF_DATASETS_CACHE=/workspace/huggingface_cache/datasets` and reduce `preprocessing_num_workers` to 1.
- **Template `qwen3_5` not found**: Need LlamaFactory from git (`0.9.5.dev0+`), not PyPI release (`0.9.4`). Template is `qwen3_5_nothink`.
- **transformers version errors**: Must be exactly `5.2.0` — first version with Qwen 3.5, max version LlamaFactory 0.9.5.dev0 accepts. If the pod has a pre-installed dev version (5.5.0.dev0), force reinstall: `pip install transformers==5.2.0`.
- **`AutoModelForVision2Seq` import error**: transformers version too new (5.5.0.dev0 renamed this class). Install 5.2.0.
- **`HybridCache` import error**: peft/transformers version mismatch. Run `pip install --upgrade peft accelerate`.
- **`bitsandbytes` not found**: `pip install bitsandbytes` — needed for `paged_adamw_8bit`.
- **CUDA OOM during training**: Reduce `cutoff_len` to 2048, then `gradient_accumulation_steps` to 4, then `lora_rank` to 128 or 64. Each step frees memory.
- **QLoRA (4-bit training)**: Do NOT use — breaks MoE fused expert nn.Parameter tensors. Use bf16 LoRA.
- **flash_attn**: Use `sdpa`, not `fa2` — Flash Attention 2 causes CUDA errors with Qwen 3.5.
- **`huggingface-cli` not found**: Use `python -c "from huggingface_hub import ..."` directly instead.
- **OOM at inference despite quantized model**: Check `metadata.json` and `adapter_config.json` in the adapter directory. If they point to `Qwen/Qwen3.5-35B-A3B-Base` (HF repo), MLX loads the full bf16 model (~66GB). Re-run `convert_peft_to_mlx.py` with `--mlx-model` pointing to the local quantized model.
- **`Model type qwen3_5_moe not supported`**: mlx-lm is too old. Run `pip install --upgrade mlx-lm`.
- **Model generates thinking tokens / analysis instead of restyling**: The Qwen 3.5 base model tokenizer injects `<think>` tags by default. The code in `lora_generator.py` uses a nothink chat template override to prevent this. If you see `<think>` in output, check that the nothink template override is in place.
- **Adapter weight key mismatch (LoRA not activating)**: PEFT uses `model.language_model.layers.N.` but MLX expects `language_model.model.layers.N.`. The `--mlx-model` flag on `convert_peft_to_mlx.py` auto-detects and fixes this. Verify with: `python -c "from safetensors.numpy import load_file; print(list(load_file('adapter/adapters.safetensors').keys())[:3])"`

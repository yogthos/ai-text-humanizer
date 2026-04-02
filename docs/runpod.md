# RunPod LoRA Training Setup

## Requirements

- **GPU**: A100 80GB SXM (bf16 LoRA needs ~74GB VRAM)
- **Container disk**: 50GB+ (for packages — default 20GB is too small)
- **Volume disk**: 150GB+ (for model weights + checkpoints)
- **Template**: RunPod PyTorch 2.x (CUDA 12.x)

## Pod Setup

```bash
# tmux so you can disconnect
apt update && apt install -y tmux
tmux new -s train

# CRITICAL: Point HF cache to workspace (root overlay is only 20-50GB, model is ~70GB)
export HF_HOME=/workspace/huggingface_cache
mkdir -p $HF_HOME

# Install LlamaFactory from git (0.9.5.dev0+ has qwen3_5 templates)
pip install "llamafactory[torch] @ git+https://github.com/hiyouga/LLaMA-Factory.git"

# Install transformers 5.2.0 (first version with Qwen 3.5 support,
# also the max version LlamaFactory 0.9.5.dev0 accepts)
pip install transformers==5.2.0
```

## Upload Training Data

From local machine:
```bash
scp data/training/lovecraft/LlamaFactory/train_sft.jsonl \
    data/training/lovecraft/LlamaFactory/dataset_info.json \
    data/training/lovecraft/LlamaFactory/qwen35_35b_lora.yaml \
    runpod:/workspace/lovecraft_training/
```

Or use `runpodctl`:
```bash
# Local
runpodctl send data/training/lovecraft/LlamaFactory/

# On pod
runpodctl receive <code>
```

## Start Training

```bash
cd /workspace/lovecraft_training

# LlamaFactory expects dataset_info.json in data/ subdirectory
mkdir -p data
cp dataset_info.json data/
cp train_sft.jsonl data/

# Launch
llamafactory-cli train qwen35_35b_lora.yaml
```

Model (`Qwen/Qwen3.5-35B-A3B-Base`) auto-downloads from HuggingFace on first run (~70GB).

## tmux Controls

- **Detach**: `Ctrl+B` then `D`
- **Reattach**: `tmux attach -t train`
- **List sessions**: `tmux ls`

## Monitor Training

```bash
# In another terminal / tmux pane
tail -f /workspace/lovecraft_training/saves/Qwen3.5-35B-A3B/lora/lovecraft/trainer_log.jsonl
```

Watch val loss — if it increases for 2+ eval intervals (every 100 steps), training is overfitting.

### Expected Loss Curve (from April 2026 run, 9,371 entries)

| Step | Epoch | Train Loss | Eval Loss |
|------|-------|-----------|-----------|
| 100  | 0.47  | 1.077     | 1.079     |
| 200  | 0.95  | 0.994     | 1.020     |
| 300  | 1.42  | 1.003     | 0.993     |
| 400  | 1.89  | 0.944     | 0.985     |
| 424  | 2.00  | 0.947     | 0.985     |

Total training time was ~32 hours on A100 80GB SXM (~$53-80 at RunPod rates).

## Upload Adapter to HuggingFace

```bash
export HF_TOKEN=your_token_here

python -c "
import os
from huggingface_hub import HfApi, login
login(token=os.environ['HF_TOKEN'])
api = HfApi()
api.create_repo('yogthos/lovecraft-qwen35-35b-a3b-lora', private=True, exist_ok=True)
api.upload_folder(
    folder_path='saves/Qwen3.5-35B-A3B/lora/lovecraft/',
    repo_id='yogthos/lovecraft-qwen35-35b-a3b-lora',
)
print('Done!')
"
```

## Convert to MLX for Local Inference (M1 Max 64GB)

### 1. Quantize Base Model

8-bit is the best quality that fits comfortably in 64GB (~35GB model + KV cache + OS).
6-bit (~26GB) also works with more headroom. Do NOT use bf16 (~70GB, won't fit).

```bash
venv/bin/python -m mlx_lm.convert \
    --hf-path Qwen/Qwen3.5-35B-A3B-Base \
    --mlx-path models/Qwen3.5-35B-A3B-Base-8bit-MLX \
    -q --q-bits 8
```

Downloads ~70GB bf16 weights, quantizes to ~35GB. Takes a while on first run.

### 2. Convert PEFT Adapter to MLX Format

The LoRA adapter from LlamaFactory is in PEFT (PyTorch) format. MLX needs its own format.
The conversion script auto-detects all LoRA target modules from the weight names.

```bash
venv/bin/python scripts/convert_peft_to_mlx.py \
    --input lora_adapters/lovecraft_Qwen3.5-35B-A3B \
    --output lora_adapters/lovecraft_35b_mlx
```

### 3. Test Inference

```bash
venv/bin/python restyle.py test_input.md -o test_output.md \
    --adapter lora_adapters/lovecraft_35b_mlx \
    --author "H.P. Lovecraft" --verbose
```

### Memory Budget at Inference (8-bit)

| Component | Size |
|-----------|------|
| Base model (8-bit) | ~35 GB |
| LoRA adapter | ~90 MB |
| KV cache | ~2-4 GB |
| OS + overhead | ~4-6 GB |
| **Total** | **~43 GB** |
| **Headroom** | **~21 GB** |

## Training Config Reference

Config file: `data/training/lovecraft/LlamaFactory/qwen35_35b_lora.yaml`

| Parameter | Value | Notes |
|-----------|-------|-------|
| model | Qwen/Qwen3.5-35B-A3B-Base | Base, not Instruct |
| template | qwen3_5_nothink | No thinking tokens |
| lora_rank | 16 | MoE experts are 512-dim, rank 64 is overkill |
| lora_alpha | 32 | 2:1 ratio with rank |
| lora_target | all | Attention + DeltaNet + shared expert + router |
| quantization | none (bf16) | QLoRA breaks MoE expert layers |
| flash_attn | sdpa | FA2 causes CUDA errors with Qwen 3.5 |
| optim | adamw_torch | paged_adamw needs bitsandbytes |
| epochs | 2 | MoE overfits faster than dense |
| effective batch | 16 (1 × 16) | ~586 steps per epoch |

## Troubleshooting

- **"No space left on device"**: Set `export HF_HOME=/workspace/huggingface_cache` before training. The root overlay is only 20GB by default. Resize container disk to 50GB+ or redirect HF cache to `/workspace/`.
- **"Disk quota exceeded"**: Same cause — HF downloads going to root. Ensure `HF_HOME` is set.
- **Template `qwen3_5` not found**: Need LlamaFactory from git (`0.9.5.dev0+`), not PyPI release (`0.9.4`). Template is `qwen3_5_nothink`.
- **transformers version errors**: Must be exactly `5.2.0` — first version with Qwen 3.5, max version LlamaFactory 0.9.5.dev0 accepts. If the pod has a pre-installed dev version (5.5.0.dev0), force reinstall: `pip install transformers==5.2.0`.
- **`AutoModelForVision2Seq` import error**: transformers version too new (5.5.0.dev0 renamed this class). Install 5.2.0.
- **`HybridCache` import error**: peft/transformers version mismatch. Run `pip install --upgrade peft accelerate`.
- **QLoRA (4-bit training)**: Do NOT use — breaks MoE fused expert nn.Parameter tensors. Use bf16 LoRA.
- **flash_attn**: Use `sdpa`, not `fa2` — Flash Attention 2 causes CUDA errors with Qwen 3.5.
- **`huggingface-cli` not found**: Use `python -c "from huggingface_hub import ..."` directly instead.

# RunPod Setup

Operational guide for running LoRA training on RunPod.
For training concepts and hyperparameter rationale, see `style_transfer_training.md`.
For model-specific configs, see `qwen35_training.md`.

## Pod Selection

| Config | GPU | Use Case |
|--------|-----|----------|
| **2x H100 80GB SXM** | Rank 256, fastest | ~$3.29/hr each |
| 1x A100 80GB SXM | Rank 128 or lower | ~$1.64/hr |

- **Container disk**: 20GB default is fine
- **Volume disk**: 150GB+ (model weights + checkpoints)
- **Template**: RunPod PyTorch 2.x (CUDA 12.x)

## Setup

```bash
# tmux so you can disconnect
apt update && apt install -y tmux
tmux new -s train

# attach later: tmux attach -t train

# CRITICAL: Point caches to workspace (root overlay is only 20GB)
export HF_HOME=/workspace/huggingface_cache
export HF_DATASETS_CACHE=/workspace/huggingface_cache/datasets
mkdir -p $HF_DATASETS_CACHE

# Install
pip install "llamafactory[torch] @ git+https://github.com/hiyouga/LLaMA-Factory.git"
pip install transformers==5.2.0 bitsandbytes

# Clone repo
cd /workspace
git clone -b qwen-35 <your-repo-url> revenant
```

## Prepare Training Directory

```bash
# Russell
mkdir -p /workspace/russell_training/data
cp revenant/data/training/russell/LlamaFactory/qwen35_35b_lora.yaml /workspace/russell_training/
cp revenant/data/training/russell/LlamaFactory/dataset_info.json /workspace/russell_training/data/
cp revenant/data/training/russell/train.jsonl /workspace/russell_training/data/
```

## Train

```bash
cd /workspace/russell_training
llamafactory-cli train qwen35_35b_lora.yaml
```

Model auto-downloads from HuggingFace on first run (~70GB).

## Monitor

```bash
tail -f saves/Qwen3.5-35B-A3B/lora/russell/trainer_log.jsonl
```

First 10-20 steps: loss should be in the 1-3 range and declining. If loss spikes above
1000 or drops to 0.0, the config has a problem (see qwen35_training.md for diagnosis).

## Upload Adapter

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

## Troubleshooting

**Installation:**
- **`qwen3_5` template not found**: Need LlamaFactory from git (0.9.5.dev0+), not PyPI (0.9.4)
- **transformers version errors**: Must be exactly 5.2.0
- **`bitsandbytes` not found**: `pip install bitsandbytes`

**Disk:**
- **"No space left on device"**: Set `HF_HOME` and `HF_DATASETS_CACHE` to `/workspace/`
- **I/O error during preprocessing**: Set `HF_DATASETS_CACHE`, reduce `preprocessing_num_workers` to 1

**Training:**
- **CUDA OOM**: Reduce cutoff_len → grad_accum → rank (in that order)
- **Loss spike then 0.0**: rsLoRA alpha too high — see qwen35_training.md
- **DDP replicates model**: Per-GPU memory = single GPU. Multi-GPU gives throughput, not more memory per card

**Upload:**
- **`huggingface-cli` not found**: Use Python `HfApi` directly (see upload section above)

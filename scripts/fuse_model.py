#!/usr/bin/env python3
"""Fuse a LoRA checkpoint with the base model.

Merges PEFT LoRA weights into the base model using transformers/PEFT,
producing a standalone fused model. Optionally converts the result to MLX format.

Usage:
    # Fuse PEFT checkpoint into base model
    python scripts/fuse_model.py \\
        --model Qwen/Qwen2.5-32B \\
        --checkpoint checkpoints/checkpoint-10200 \\
        --output models/style-transfer-fused

    # Also convert to MLX format
    python scripts/fuse_model.py \\
        --model Qwen/Qwen2.5-32B \\
        --checkpoint checkpoints/checkpoint-10200 \\
        --output models/style-transfer-fused \\
        --convert-mlx

    # Convert a previously fused HF model to MLX
    python scripts/fuse_model.py \\
        --model models/style-transfer-fused \\
        --output models/style-transfer-mlx \\
        --convert-mlx-only
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def fuse_peft(
    model_name_or_path: str, checkpoint_path: Path, output_path: Path
) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    print(f"Loading PEFT adapter from {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path))

    print("Merging adapter weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving fused model to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    with open(checkpoint_path / "adapter_config.json") as f:
        adapter_cfg = json.load(f)

    metadata = {
        "base_model": model_name_or_path,
        "checkpoint": str(checkpoint_path),
        "lora_rank": adapter_cfg.get("r"),
        "lora_alpha": adapter_cfg.get("lora_alpha"),
    }
    with open(output_path / "fuse_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def convert_to_mlx(model_path: Path, output_path: Path) -> None:
    try:
        from mlx_lm.utils import convert
    except ImportError:
        print(
            "Error: mlx-lm is required for MLX conversion. Install with: pip install mlx-lm",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nConverting to MLX format: {model_path} -> {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading fused model for MLX conversion...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    weights = dict(tree_flatten(hf_model.state_dict()))
    del hf_model

    from mlx_lm.utils import save_weights

    save_weights(str(output_path), weights)
    tokenizer.save_pretrained(str(output_path))

    config_path = model_path / "config.json"
    if config_path.exists():
        shutil.copy2(config_path, output_path / "config.json")

    for name in [
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]:
        src = model_path / name
        if src.exists():
            shutil.copy2(src, output_path / name)

    if (model_path / "fuse_metadata.json").exists():
        with open(model_path / "fuse_metadata.json") as f:
            metadata = json.load(f)
        metadata["mlx_converted"] = True
        with open(output_path / "fuse_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA checkpoint with base model")
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="HuggingFace model name or local path to base model",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=None,
        help="Path to PEFT checkpoint directory to fuse",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for the fused model",
    )
    parser.add_argument(
        "--convert-mlx",
        action="store_true",
        help="Convert the fused model to MLX format after fusing",
    )
    parser.add_argument(
        "--convert-mlx-only",
        action="store_true",
        help="Skip fusion; just convert an existing HF model to MLX",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.convert_mlx_only:
        convert_to_mlx(Path(args.model), output_path)
        print(f"\nDone! MLX model saved to {output_path}")
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required when not using --convert-mlx-only")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    fuse_peft(args.model, checkpoint_path, output_path)
    print(f"\nFused model saved to {output_path}")

    if args.convert_mlx:
        mlx_output = output_path.parent / (output_path.name + "-MLX")
        convert_to_mlx(output_path, mlx_output)
        print(f"\nMLX model saved to {mlx_output}")

    print(f"\nUsage:")
    print(f"  python restyle.py input.txt -o output.txt --model {output_path}")


if __name__ == "__main__":
    main()

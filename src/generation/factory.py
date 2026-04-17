"""Factory for creating style generators based on backend configuration.

This module provides a unified interface for creating style generators
that can use either MLX (Apple Silicon) or PyTorch (CUDA/CPU) backends.
"""

import sys
from typing import Optional, List

from ..utils.logging import get_logger
from .base_generator import GenerationConfig

logger = get_logger(__name__)


def detect_best_backend() -> str:
    """Detect the best available backend for the current system.

    Returns:
        "mlx" on Apple Silicon with MLX installed, "pytorch" otherwise.

    Raises:
        RuntimeError: If no backend is available.
    """
    # Check MLX first (preferred on Apple Silicon)
    if sys.platform == "darwin":
        try:
            import mlx  # noqa: F401

            logger.debug("MLX backend available")
            return "mlx"
        except ImportError:
            logger.debug("MLX not available on macOS")

    # Check PyTorch
    try:
        import torch  # noqa: F401

        logger.debug("PyTorch backend available")
        return "pytorch"
    except ImportError:
        pass

    raise RuntimeError(
        "No backend available. Install one of:\n"
        "  - MLX (Apple Silicon): pip install mlx mlx-lm\n"
        "  - PyTorch: pip install torch transformers peft"
    )


def _set_fiction_markers(generator, model_path: Optional[str]) -> None:
    """Load fiction markers from adapter or fused-model config and set on generator.

    The same `model_path` may refer to either a LoRA adapter (looked up in
    `lora_adapters`) or a standalone fused model (looked up in `models`).
    We try the adapter config first, then fall back to the fused-model config.
    """
    if not model_path:
        return
    # Narrow catch: get_*_config already log+default on config errors, so the
    # only exceptions we expect here are ImportError (module broken) or
    # AttributeError (non-standard generator). Letting anything else surface
    # keeps real bugs from being silently swallowed.
    try:
        from ..config import get_adapter_config, get_fused_model_config

        adapter_config = get_adapter_config(model_path)
        if adapter_config.fiction_markers:
            generator.fiction_markers = adapter_config.fiction_markers
            return
        fused_config = get_fused_model_config(model_path)
        if fused_config.fiction_markers:
            generator.fiction_markers = fused_config.fiction_markers
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not load fiction markers: {e}")


def create_style_generator(
    adapter_path: Optional[str] = None,
    base_model: Optional[str] = None,
    config: Optional[GenerationConfig] = None,
    checkpoint: Optional[str] = None,
    adapters: Optional[List] = None,
    backend: str = "auto",
    device: str = "auto",
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    fused_models: Optional[List[str]] = None,
):
    """Create a style generator using the specified backend.

    This factory function creates the appropriate generator based on the
    backend parameter, automatically detecting the best backend if "auto".

    Args:
        adapter_path: Path to adapter directory (local or HuggingFace repo).
        base_model: Base model name/path. Defaults vary by backend.
        config: Generation configuration.
        checkpoint: Specific checkpoint to use (e.g., "checkpoint-600").
        adapters: List of AdapterSpec for multi-adapter blending (MLX only).
        backend: Backend to use ("auto", "mlx", "pytorch").
        device: Device for PyTorch ("auto", "cuda", "cpu", "mps").
        load_in_4bit: Use 4-bit quantization (PyTorch with CUDA only).
        load_in_8bit: Use 8-bit quantization (PyTorch with CUDA only).
        fused_models: List of fused model paths to use directly (no adapter).

    Returns:
        Style generator instance (LoRAStyleGenerator or PyTorchStyleGenerator).

    Raises:
        RuntimeError: If the requested backend is not available.
        ValueError: If an unknown backend is specified.
    """
    if backend == "auto":
        backend = detect_best_backend()

    logger.info(f"Using {backend} backend")

    if backend == "mlx":
        from .lora_generator import LoRAStyleGenerator, AdapterSpec

        if fused_models:
            generator = LoRAStyleGenerator(
                adapter_path=None,
                base_model=fused_models[0],
                config=config,
                adapters=None,
            )
            _set_fiction_markers(generator, fused_models[0])
            return generator

        # Convert adapters if provided as list of dicts or strings
        adapter_specs = None
        if adapters:
            adapter_specs = []
            for a in adapters:
                if isinstance(a, AdapterSpec):
                    adapter_specs.append(a)
                elif isinstance(a, str):
                    adapter_specs.append(AdapterSpec.parse(a))
                elif isinstance(a, dict):
                    adapter_specs.append(
                        AdapterSpec(
                            path=a.get("path", ""),
                            scale=a.get("scale", 1.0),
                            checkpoint=a.get("checkpoint"),
                        )
                    )

        generator = LoRAStyleGenerator(
            adapter_path=adapter_path,
            base_model=base_model or "mlx-community/Qwen3-8B-Base-bf16",
            config=config,
            checkpoint=checkpoint,
            adapters=adapter_specs,
        )
        # Load fiction markers from adapter config
        _set_fiction_markers(generator, adapter_path)
        return generator

    elif backend == "pytorch":
        from .pytorch_generator import PyTorchStyleGenerator

        if fused_models:
            generator = PyTorchStyleGenerator(
                adapter_path=None,
                base_model=fused_models[0],
                config=config,
                device=device,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            _set_fiction_markers(generator, fused_models[0])
            return generator

        generator = PyTorchStyleGenerator(
            adapter_path=adapter_path,
            base_model=base_model or "Qwen/Qwen2.5-32B-Instruct",
            config=config,
            checkpoint=checkpoint,
            device=device,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        _set_fiction_markers(generator, adapter_path)
        return generator

    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: 'auto', 'mlx', 'pytorch'"
        )


def list_available_backends() -> List[str]:
    """List all available backends on this system.

    Returns:
        List of available backend names.
    """
    available = []

    # Check MLX
    if sys.platform == "darwin":
        try:
            import mlx  # noqa: F401

            available.append("mlx")
        except ImportError:
            pass

    # Check PyTorch
    try:
        import torch  # noqa: F401

        available.append("pytorch")
    except ImportError:
        pass

    return available

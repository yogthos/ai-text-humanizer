"""Configuration management for the style transfer pipeline."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

from .utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMProviderConfig:
    """Configuration for a specific LLM provider."""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120


@dataclass
class LLMProviderRoles:
    """Configuration for role-based LLM provider assignment.

    Allows using different providers for different tasks:
    - writer: Fast local model for style generation (e.g., MLX with LoRA)
    - critic: Smarter API model for validation and repair (e.g., DeepSeek)
    - rtt: Provider for Round-Trip Translation neutralization (e.g., DeepSeek)
    """
    writer: str = "mlx"  # Provider for generation
    critic: str = "deepseek"  # Provider for critique/repair
    rtt: str = "deepseek"  # Provider for RTT neutralization


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProviderRoles = field(default_factory=LLMProviderRoles)
    providers: Dict[str, LLMProviderConfig] = field(default_factory=dict)
    max_retries: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0

    def get_provider_config(self, provider_name: str) -> LLMProviderConfig:
        """Get configuration for a specific provider."""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
        return self.providers[provider_name]

    def get_writer_provider(self) -> str:
        """Get the provider name for generation/writing tasks."""
        return self.provider.writer

    def get_critic_provider(self) -> str:
        """Get the provider name for critique/repair tasks."""
        return self.provider.critic


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    # Length control settings
    max_expansion_ratio: float = 2.5  # Max output/input word ratio before warning
    target_expansion_ratio: float = 1.5  # Target for LoRA generation (1.5 = 50% expansion for flourish)
    expand_for_texture: bool = False  # Add stronger expansion prompt to encourage elaboration/flourishes

    # LoRA adapter settings (path -> config mapping)
    lora_adapters: Dict[str, "LoRAAdapterConfig"] = field(default_factory=dict)

    # Neutralization settings
    skip_neutralization: bool = False  # If True, skip RTT and use original text as input

    # Document handling
    use_document_context: bool = False  # Extract document-level context (disabled: extracted but never used)
    pass_headings_unchanged: bool = True  # Don't transform headings
    min_paragraph_words: int = 10  # Skip paragraphs shorter than this

    # RAG settings
    use_structural_rag: bool = True  # Enable structural RAG for rhythm/syntax guidance
    use_structural_grafting: bool = True  # Enable structural grafting for argument skeletons
    rag_sample_size: int = 300  # Number of corpus chunks to sample for rhythm pattern analysis

    # Persona settings
    use_persona: bool = True  # Enable persona-based prompting
    apply_input_perturbation: bool = True  # Apply 8% noise to match training distribution


@dataclass
class LoRAAdapterConfig:
    """Configuration for a specific LoRA adapter.

    These settings control the balance between style strength and coherence.
    Each adapter in lora_adapters can have its own settings.
    """
    enabled: bool = True  # Whether this adapter is active
    scale: float = 1.0  # Adapter influence (0.0=base only, 1.0=full, >1.0=amplified)
    temperature: float = 0.6  # Higher = more creative, lower = more coherent
    top_p: float = 0.92  # Nucleus sampling threshold
    min_p: float = 0.05  # Minimum probability filter
    repetition_penalty: float = 1.15  # Penalty for repeating tokens
    max_tokens: int = 512  # Maximum tokens to generate
    worldview: str = ""  # Author worldview prompt file
    fiction_markers: List[str] = field(default_factory=list)  # Fiction markers to filter from output
    checkpoint: Optional[str] = None  # Specific checkpoint to use
    # Backend configuration
    backend: str = "auto"  # "auto", "mlx", or "pytorch"
    device: str = "auto"  # Device for PyTorch: "auto", "cuda", "cpu", "mps"
    load_in_4bit: bool = True  # Use 4-bit quantization (PyTorch with CUDA only)
    load_in_8bit: bool = False  # Use 8-bit quantization (PyTorch with CUDA only)
    hf_adapter_path: Optional[str] = None  # HuggingFace adapter path (if different from local)
    expand_for_texture: Optional[bool] = None  # Per-adapter override for texture expansion (None = use global)
    perspective: Optional[str] = None  # Per-adapter perspective override (None = use global)
    verify_entailment: Optional[bool] = None  # Per-adapter verification override (None = use global/CLI)
    merge_paragraphs: Optional[int] = None  # Merge short paragraphs to reach this minimum word count before LoRA


@dataclass
@dataclass
class StyleConfig:
    """Configuration for style transfer settings."""
    perspective: str = "preserve"  # preserve, first_person_singular, first_person_plural, third_person, author_voice_third_person

    def __post_init__(self):
        """Auto-correct invalid perspective on init."""
        if not self.validate_perspective():
            logger.warning(
                f"Invalid perspective '{self.perspective}', using 'preserve'"
            )
            self.perspective = "preserve"

    def validate_perspective(self) -> bool:
        """Check if perspective setting is valid."""
        valid_perspectives = {
            "preserve",
            "first_person_singular",
            "first_person_plural",
            "third_person",
            "author_voice_third_person",  # Writes AS the author using third person (not about the author)
        }
        return self.perspective in valid_perspectives

    @staticmethod
    def get_perspective_instruction(perspective: str, author: str) -> str:
        """Get the instruction text for a given perspective."""
        instructions = {
            "preserve": "Maintain the same perspective (first/third person) as the source text.",
            "first_person_singular": "Write in first person singular (I, me, my).",
            "first_person_plural": "Write in first person plural (we, us, our).",
            "third_person": "Write in third person (he, she, they, it).",
            "author_voice_third_person": f"Write AS {author} would write, using third person perspective. Channel {author}'s voice and style while referring to subjects in third person.",
        }
        return instructions.get(perspective, "")


@dataclass
class Config:
    """Main configuration container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    log_level: str = "INFO"
    log_json: bool = False


def _resolve_env_vars(value: Any) -> Any:
    """Resolve environment variables in string values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        resolved = os.environ.get(env_var, "")
        if not resolved:
            logger.warning(f"Environment variable {env_var} not set")
        return resolved
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


def _parse_llm_provider_config(data: Dict) -> LLMProviderConfig:
    """Parse LLM provider configuration."""
    return LLMProviderConfig(
        api_key=_resolve_env_vars(data.get("api_key", "")),
        base_url=_resolve_env_vars(data.get("base_url", "")),
        model=_resolve_env_vars(data.get("model", "")),
        max_tokens=data.get("max_tokens", 4096),
        temperature=data.get("temperature", 0.7),
        timeout=data.get("timeout", 120),
    )


_KNOWN_ADAPTER_FIELDS = {
    "enabled", "scale", "temperature", "top_p", "min_p", "repetition_penalty",
    "max_tokens", "worldview", "fiction_markers", "checkpoint", "backend", "device",
    "load_in_4bit", "load_in_8bit", "hf_adapter_path", "expand_for_texture",
    "perspective", "verify_entailment", "merge_paragraphs",
}


def _parse_lora_adapter_config(data: Dict) -> LoRAAdapterConfig:
    """Parse LoRA adapter configuration from dict."""
    unknown_fields = set(data.keys()) - _KNOWN_ADAPTER_FIELDS
    if unknown_fields:
        logger.warning(f"Unknown adapter config fields (ignored): {', '.join(sorted(unknown_fields))}")

    return LoRAAdapterConfig(
        enabled=data.get("enabled", True),
        scale=data.get("scale", 1.0),
        temperature=data.get("temperature", 0.6),
        top_p=data.get("top_p", 0.92),
        min_p=data.get("min_p", 0.05),
        repetition_penalty=data.get("repetition_penalty", 1.15),
        max_tokens=data.get("max_tokens", 512),
        worldview=data.get("worldview", ""),
        fiction_markers=data.get("fiction_markers", []),
        checkpoint=data.get("checkpoint"),
        # Backend configuration
        backend=data.get("backend", "auto"),
        device=data.get("device", "auto"),
        load_in_4bit=data.get("load_in_4bit", True),
        load_in_8bit=data.get("load_in_8bit", False),
        hf_adapter_path=data.get("hf_adapter_path"),
        expand_for_texture=data.get("expand_for_texture"),
        perspective=data.get("perspective"),
        verify_entailment=data.get("verify_entailment"),
        merge_paragraphs=data.get("merge_paragraphs"),
    )


def _parse_lora_adapters(data: Dict) -> Dict[str, LoRAAdapterConfig]:
    """Parse lora_adapters section into typed configs.

    Handles both old format (path -> scale) and new format (path -> config dict).
    """
    result = {}
    for path, value in data.items():
        if isinstance(value, dict):
            # New format: full config dict
            result[path] = _parse_lora_adapter_config(value)
        else:
            # Old format: just a scale number
            result[path] = LoRAAdapterConfig(scale=float(value))
    return result


def _parse_llm_config(data: Dict) -> LLMConfig:
    """Parse LLM configuration section."""
    providers = {}
    for name, provider_data in data.get("providers", {}).items():
        providers[name] = _parse_llm_provider_config(provider_data)

    retry_config = data.get("retry", {})
    provider_data = data.get("provider", {})

    provider_roles = LLMProviderRoles(
        writer=provider_data.get("writer", "mlx"),
        critic=provider_data.get("critic", "deepseek"),
        rtt=provider_data.get("rtt", "deepseek"),
    )

    return LLMConfig(
        provider=provider_roles,
        providers=providers,
        max_retries=retry_config.get("max_attempts", 5),
        base_delay=retry_config.get("base_delay", 2.0),
        max_delay=retry_config.get("max_delay", 60.0),
    )


# Module-level config cache
_config_cache: dict = {}


def load_config(config_path: str = "config.json") -> Config:
    """Load configuration from a JSON file.

    Uses a cache to avoid reloading the same config file multiple times.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Parsed configuration object.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    # Return cached config if available
    if config_path in _config_cache:
        return _config_cache[config_path]

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please copy config.json.sample to config.json and configure it."
        )

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")

    # Parse each section
    config = Config()

    if "llm" in data:
        config.llm = _parse_llm_config(data["llm"])

    if "generation" in data:
        gen = data["generation"]
        config.generation = GenerationConfig(
            max_expansion_ratio=gen.get("max_expansion_ratio", 2.5),
            target_expansion_ratio=gen.get("target_expansion_ratio", 1.5),
            expand_for_texture=gen.get("expand_for_texture", False),
            lora_adapters=_parse_lora_adapters(gen.get("lora_adapters", {})),
            skip_neutralization=gen.get("skip_neutralization", False),
            use_document_context=gen.get("use_document_context", False),
            pass_headings_unchanged=gen.get("pass_headings_unchanged", True),
            min_paragraph_words=gen.get("min_paragraph_words", 10),
            use_structural_rag=gen.get("use_structural_rag", True),
            use_structural_grafting=gen.get("use_structural_grafting", True),
            rag_sample_size=gen.get("rag_sample_size", 300),
            use_persona=gen.get("use_persona", True),
            apply_input_perturbation=gen.get("apply_input_perturbation", True),
        )

    if "style" in data:
        style_data = data["style"]
        config.style = StyleConfig(
            perspective=style_data.get("perspective", "preserve"),
        )
        # Note: validation is handled by StyleConfig.__post_init__

    # Validate worldview files exist in prompts/ directory
    prompts_dir = Path(__file__).parent.parent / "prompts"
    for adapter_path_key, adapter_cfg in config.generation.lora_adapters.items():
        if adapter_cfg.worldview:
            worldview_path = prompts_dir / adapter_cfg.worldview
            if not worldview_path.exists():
                logger.warning(
                    f"Worldview file '{adapter_cfg.worldview}' for adapter '{adapter_path_key}' "
                    f"not found in {prompts_dir}. Persona prompts will use fallback frames."
                )

    config.log_level = data.get("log_level", "INFO")
    config.log_json = data.get("log_json", False)

    # Cache the config
    _config_cache[config_path] = config

    logger.info(f"Loaded configuration from {config_path}")
    return config


def create_default_config() -> Dict:
    """Create a default configuration dictionary."""
    return {
        "llm": {
            "provider": {
                "writer": "mlx",
                "critic": "deepseek"
            },
            "providers": {
                "deepseek": {
                    "api_key": "${DEEPSEEK_API_KEY}",
                    "base_url": "https://api.deepseek.com",
                    "model": "deepseek-chat",
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "timeout": 120
                },
                "mlx": {
                    "model": "mlx-community/Qwen3-8B-4bit",
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                "ollama": {
                    "base_url": "http://localhost:11434",
                    "model": "llama3",
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
            },
            "retry": {
                "max_attempts": 5,
                "base_delay": 2,
                "max_delay": 60
            }
        },
        "generation": {},
        "log_level": "INFO"
    }


def get_adapter_config(adapter_path: Optional[str] = None) -> LoRAAdapterConfig:
    """Get LoRA adapter config for a specific adapter path.

    Args:
        adapter_path: Path to the adapter directory. If None, returns defaults.

    Returns:
        LoRAAdapterConfig for the adapter, or defaults if not found.
    """
    if not adapter_path:
        return LoRAAdapterConfig()

    try:
        config = load_config()
        adapters = config.generation.lora_adapters

        # Try exact match first
        if adapter_path in adapters:
            return adapters[adapter_path]

        # Try matching by adapter directory name
        from pathlib import Path
        adapter_name = Path(adapter_path).name
        for path, adapter_config in adapters.items():
            if Path(path).name == adapter_name:
                return adapter_config

    except Exception as e:
        logger.debug(f"Could not load adapter config: {e}")

    return LoRAAdapterConfig()

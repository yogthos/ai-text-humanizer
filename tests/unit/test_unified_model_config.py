"""Tests for C4: unified ModelConfig dataclass.

LoRAAdapterConfig and FusedModelConfig had near-identical fields with only
two distinct concerns: LoRA-specific options (scale, backend, quantization)
and the fused-only `author` field. Collapse them into one ModelConfig and
keep the old names as aliases so external call sites don't break.
"""

import json
import os
import tempfile


class TestModelConfigExists:
    """ModelConfig should be importable and carry all fields."""

    def test_model_config_importable(self):
        from src.config import ModelConfig

        cfg = ModelConfig()
        assert cfg is not None

    def test_model_config_has_lora_fields(self):
        """LoRA-specific fields from LoRAAdapterConfig should live on ModelConfig."""
        from src.config import ModelConfig

        cfg = ModelConfig()
        assert hasattr(cfg, "scale")
        assert hasattr(cfg, "checkpoint")
        assert hasattr(cfg, "backend")
        assert hasattr(cfg, "device")
        assert hasattr(cfg, "load_in_4bit")
        assert hasattr(cfg, "load_in_8bit")
        assert hasattr(cfg, "hf_adapter_path")

    def test_model_config_has_fused_fields(self):
        """Fused-specific fields from FusedModelConfig should live on ModelConfig."""
        from src.config import ModelConfig

        cfg = ModelConfig()
        assert hasattr(cfg, "author")

    def test_model_config_has_shared_fields(self):
        """All shared fields should exist on ModelConfig."""
        from src.config import ModelConfig

        cfg = ModelConfig()
        for name in (
            "enabled",
            "temperature",
            "top_p",
            "min_p",
            "repetition_penalty",
            "max_tokens",
            "worldview",
            "fiction_markers",
            "expand_for_texture",
            "perspective",
            "verify_entailment",
            "merge_paragraphs",
        ):
            assert hasattr(cfg, name), f"ModelConfig missing field: {name}"


class TestBackwardsCompatAliases:
    """LoRAAdapterConfig and FusedModelConfig must still be importable as aliases."""

    def test_lora_adapter_config_is_model_config(self):
        from src.config import LoRAAdapterConfig, ModelConfig

        assert LoRAAdapterConfig is ModelConfig

    def test_fused_model_config_is_model_config(self):
        from src.config import FusedModelConfig, ModelConfig

        assert FusedModelConfig is ModelConfig


class TestGetConfigReturnTypes:
    """get_adapter_config and get_fused_model_config should return ModelConfig."""

    def test_get_adapter_config_returns_model_config(self):
        from src.config import ModelConfig, get_adapter_config

        cfg = get_adapter_config(None)
        assert isinstance(cfg, ModelConfig)

    def test_get_fused_model_config_returns_model_config(self):
        from src.config import ModelConfig, get_fused_model_config

        cfg = get_fused_model_config(None)
        assert isinstance(cfg, ModelConfig)


class TestUnifiedParsing:
    """Unified parser should parse both lora_adapters and models entries."""

    def test_parse_lora_adapter_entry_with_all_fields(self):
        from src.config import _config_cache, get_adapter_config, load_config

        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "generation": {
                "lora_adapters": {
                    "lora/custom": {
                        "scale": 1.7,
                        "temperature": 0.55,
                        "worldview": "custom.txt",
                        "fiction_markers": ["m1"],
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            try:
                _config_cache.clear()
                cfg = load_config(f.name)
                _config_cache["config.json"] = cfg

                adapter = get_adapter_config("lora/custom")
                assert adapter.scale == 1.7
                assert adapter.temperature == 0.55
                assert adapter.worldview == "custom.txt"
                assert adapter.fiction_markers == ["m1"]
            finally:
                _config_cache.clear()
                os.unlink(f.name)

    def test_parse_fused_model_entry_with_author(self):
        from src.config import _config_cache, get_fused_model_config, load_config

        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "generation": {
                "use_adapter": False,
                "models": {
                    "models/baz": {
                        "author": "Baz Author",
                        "temperature": 0.3,
                        "worldview": "baz.txt",
                    }
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            f.flush()
            try:
                _config_cache.clear()
                cfg = load_config(f.name)
                _config_cache["config.json"] = cfg

                model = get_fused_model_config("models/baz")
                assert model.author == "Baz Author"
                assert model.temperature == 0.3
                assert model.worldview == "baz.txt"
            finally:
                _config_cache.clear()
                os.unlink(f.name)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

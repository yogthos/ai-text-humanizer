"""Unit tests for configuration loading."""

import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.config import (
    load_config,
    create_default_config,
    Config,
    LLMConfig,
    LLMProviderRoles,
)


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_minimal_config(self):
        """Test loading a minimal valid config."""
        minimal_config = {
            "llm": {
                "provider": {
                    "writer": "mlx",
                    "critic": "deepseek"
                },
                "providers": {
                    "deepseek": {
                        "api_key": "test-key",
                        "model": "deepseek-chat"
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.llm.provider.writer == "mlx"
                assert config.llm.provider.critic == "deepseek"
                assert config.llm.providers["deepseek"].api_key == "test-key"
            finally:
                os.unlink(f.name)

    def test_load_full_config(self):
        """Test loading a complete config with all sections."""
        full_config = create_default_config()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(full_config, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.llm.provider.writer == "mlx"
                assert config.llm.provider.critic == "deepseek"
                assert config.generation.max_repair_attempts == 3  # explicitly set to 3 in config
            finally:
                os.unlink(f.name)

    def test_missing_config_file_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config("nonexistent_config.json")
        assert "Configuration file not found" in str(exc_info.value)

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            f.flush()

            try:
                with pytest.raises(ValueError) as exc_info:
                    load_config(f.name)
                assert "Invalid JSON" in str(exc_info.value)
            finally:
                os.unlink(f.name)


class TestEnvironmentVariables:
    """Test environment variable resolution."""

    def test_resolves_env_var(self):
        """Test that ${VAR} syntax resolves environment variables."""
        os.environ["TEST_API_KEY"] = "secret-from-env"

        config_data = {
            "llm": {
                "provider": {
                    "writer": "mlx",
                    "critic": "deepseek"
                },
                "providers": {
                    "deepseek": {
                        "api_key": "${TEST_API_KEY}",
                        "model": "test"
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.llm.providers["deepseek"].api_key == "secret-from-env"
            finally:
                os.unlink(f.name)
                del os.environ["TEST_API_KEY"]

    def test_missing_env_var_returns_empty(self):
        """Test that missing env var returns empty string with warning."""
        # Ensure var doesn't exist
        os.environ.pop("NONEXISTENT_VAR", None)

        config_data = {
            "llm": {
                "provider": {
                    "writer": "mlx",
                    "critic": "deepseek"
                },
                "providers": {
                    "deepseek": {
                        "api_key": "${NONEXISTENT_VAR}",
                        "model": "test"
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.llm.providers["deepseek"].api_key == ""
            finally:
                os.unlink(f.name)


class TestDefaultConfig:
    """Test default configuration creation."""

    def test_create_default_config(self):
        """Test creating default config dictionary."""
        config = create_default_config()

        assert "llm" in config
        assert "generation" in config

        # Provider is now a dict with writer/critic
        assert config["llm"]["provider"]["writer"] == "mlx"
        assert config["llm"]["provider"]["critic"] == "deepseek"
        assert "deepseek" in config["llm"]["providers"]
        assert "ollama" in config["llm"]["providers"]
        assert "mlx" in config["llm"]["providers"]

    def test_default_values(self):
        """Test that Config has sensible defaults."""
        config = Config()

        assert config.llm.max_retries == 5
        assert config.generation.max_repair_attempts == 5  # default is 5
        assert config.validation.entailment_threshold == 0.7
        assert config.validation.max_hallucinations_before_reject == 2


class TestLLMConfig:
    """Test LLM configuration."""

    def test_get_provider_config(self):
        """Test getting provider-specific config."""
        config_data = {
            "llm": {
                "provider": {
                    "writer": "mlx",
                    "critic": "deepseek"
                },
                "providers": {
                    "deepseek": {
                        "api_key": "key1",
                        "model": "model1"
                    },
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model": "model2"
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)

                deepseek_config = config.llm.get_provider_config("deepseek")
                assert deepseek_config.api_key == "key1"

                ollama_config = config.llm.get_provider_config("ollama")
                assert ollama_config.base_url == "http://localhost:11434"

                # Get writer and critic providers
                assert config.llm.get_writer_provider() == "mlx"
                assert config.llm.get_critic_provider() == "deepseek"
            finally:
                os.unlink(f.name)

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        llm_config = LLMConfig(
            provider=LLMProviderRoles(writer="test", critic="test"),
            providers={}
        )

        with pytest.raises(ValueError) as exc_info:
            llm_config.get_provider_config("unknown")
        assert "Unknown LLM provider" in str(exc_info.value)


class TestDefaultValueMatches:
    """Tests for default value consistency (Bug 15)."""

    def test_generation_config_max_repair_attempts(self):
        """GenerationConfig max_repair_attempts should match config.json default."""
        from src.config import GenerationConfig
        assert GenerationConfig().max_repair_attempts == 5

    def test_generation_config_rag_sample_size(self):
        """GenerationConfig rag_sample_size should match config.json default."""
        from src.config import GenerationConfig
        assert GenerationConfig().rag_sample_size == 300

    def test_transfer_config_sentence_length_variance(self):
        """TransferConfig sentence_length_variance should match config.json default."""
        from src.generation.transfer import TransferConfig
        assert TransferConfig().sentence_length_variance == 0.4


class TestWorldviewFileValidation:
    """Tests for worldview file existence check on load (Bug 20)."""

    def test_nonexistent_worldview_logs_warning(self):
        """Config with nonexistent worldview file should log a warning."""
        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "generation": {
                "lora_adapters": {
                    "test_adapter": {
                        "scale": 2.0,
                        "worldview": "nonexistent_worldview_xyz.txt",
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                with patch('src.config.logger') as mock_logger:
                    config = load_config(f.name)
                    # Should have logged a warning about missing worldview file
                    warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
                    assert any("nonexistent_worldview_xyz.txt" in str(c) for c in warning_calls), \
                        f"Expected warning about missing worldview file, got: {warning_calls}"
            finally:
                os.unlink(f.name)


class TestDualEntailmentThreshold:
    """Tests for entailment_threshold sync between generation and validation (Bug 8)."""

    def test_validation_threshold_propagates(self):
        """When only set in validation section, generation should match."""
        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "validation": {
                "entailment_threshold": 0.85,
                "max_hallucinations_before_reject": 3,
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                # Both should be 0.85 since validation is the only source
                assert config.validation.entailment_threshold == 0.85
                assert config.generation.entailment_threshold == 0.85
            finally:
                os.unlink(f.name)

    def test_generation_threshold_takes_precedence(self):
        """When set in both, generation value should win."""
        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "generation": {
                "entailment_threshold": 0.6,
            },
            "validation": {
                "entailment_threshold": 0.85,
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                # generation explicitly set should win
                assert config.generation.entailment_threshold == 0.6
            finally:
                os.unlink(f.name)


class TestEnvVarResolution:
    """Tests for environment variable resolution in config (Bug 13)."""

    def test_base_url_env_var_resolved(self):
        """base_url should resolve ${ENV_VAR} syntax."""
        os.environ["TEST_BASE_URL"] = "http://test.example.com"

        config_data = {
            "llm": {
                "provider": {"writer": "mlx", "critic": "test_provider"},
                "providers": {
                    "test_provider": {
                        "api_key": "key",
                        "base_url": "${TEST_BASE_URL}",
                        "model": "test-model",
                    }
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.llm.providers["test_provider"].base_url == "http://test.example.com"
            finally:
                os.unlink(f.name)
                del os.environ["TEST_BASE_URL"]

    def test_model_env_var_resolved(self):
        """model should resolve ${ENV_VAR} syntax."""
        os.environ["TEST_MODEL"] = "gpt-4"

        config_data = {
            "llm": {
                "provider": {"writer": "mlx", "critic": "test_provider"},
                "providers": {
                    "test_provider": {
                        "api_key": "key",
                        "model": "${TEST_MODEL}",
                    }
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                config = load_config(f.name)
                assert config.llm.providers["test_provider"].model == "gpt-4"
            finally:
                os.unlink(f.name)
                del os.environ["TEST_MODEL"]


class TestUnknownConfigFields:
    """Tests for unknown config field warnings (Bug 14)."""

    def test_unknown_adapter_field_logs_warning(self):
        """Unknown fields in adapter config should trigger a warning."""
        config_data = {
            "llm": {"provider": {"writer": "mlx", "critic": "deepseek"}, "providers": {}},
            "generation": {
                "lora_adapters": {
                    "test_adapter": {
                        "scale": 2.0,
                        "typo_field": "oops",
                        "another_bad_key": 42,
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            f.flush()

            try:
                with patch('src.config.logger') as mock_logger:
                    config = load_config(f.name)
                    # Just verify config loaded without crashing
                    assert "test_adapter" in config.generation.lora_adapters
                    # Check warning was logged
                    warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
                    assert any("typo_field" in str(c) for c in warning_calls) or \
                           any("another_bad_key" in str(c) for c in warning_calls), \
                           f"Expected warning about unknown fields, got: {warning_calls}"
            finally:
                os.unlink(f.name)

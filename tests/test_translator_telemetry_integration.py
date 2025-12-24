"""Tests for telemetry integration in StyleTranslator."""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.translator import StyleTranslator
from tests.mocks.mock_llm_provider import MockLLMProvider


class TestTranslatorTelemetryIntegration:
    """Test telemetry integration in StyleTranslator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        self.log_path = Path(self.temp_dir) / "logs" / "graph_debug.md"

        # Create config with telemetry enabled and required LLM provider config
        config = {
            "provider": "deepseek",
            "deepseek": {
                "api_key": "test_key",
                "api_url": "https://api.deepseek.com/v1/chat/completions",
                "editor_model": "deepseek-chat",
                "critic_model": "deepseek-chat"
            },
            "graph_pipeline": {
                "telemetry": {
                    "enabled": True,
                    "log_path": str(self.log_path)
                }
            },
            "generation": {},
            "translator": {},
            "llm_provider": {}
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

    def test_telemetry_initialized_when_enabled(self):
        """Test that telemetry is initialized when enabled in config."""
        translator = StyleTranslator(config_path=str(self.config_path))

        assert translator.telemetry is not None
        assert hasattr(translator.telemetry, 'log_match')

    def test_telemetry_not_initialized_when_disabled(self):
        """Test that telemetry is None when disabled in config."""
        config = {
            "provider": "deepseek",
            "deepseek": {
                "api_key": "test_key",
                "api_url": "https://api.deepseek.com/v1/chat/completions",
                "editor_model": "deepseek-chat",
                "critic_model": "deepseek-chat"
            },
            "graph_pipeline": {
                "telemetry": {
                    "enabled": False
                }
            },
            "generation": {},
            "translator": {},
            "llm_provider": {}
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

        translator = StyleTranslator(config_path=str(self.config_path))

        assert translator.telemetry is None

    def test_telemetry_defaults_to_enabled(self):
        """Test that telemetry defaults to enabled when config missing."""
        config = {
            "provider": "deepseek",
            "deepseek": {
                "api_key": "test_key",
                "api_url": "https://api.deepseek.com/v1/chat/completions",
                "editor_model": "deepseek-chat",
                "critic_model": "deepseek-chat"
            },
            "graph_pipeline": {},
            "generation": {},
            "translator": {},
            "llm_provider": {}
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

        translator = StyleTranslator(config_path=str(self.config_path))

        # Should default to enabled
        assert translator.telemetry is not None

    def test_telemetry_called_after_successful_generation(self):
        """Test that telemetry.log_match is called after successful graph generation."""
        translator = StyleTranslator(config_path=str(self.config_path))
        translator.llm_provider = MockLLMProvider()

        # Mock input_mapper and graph_matcher
        translator.input_mapper = MagicMock()
        translator.graph_matcher = MagicMock()

        # Mock successful graph pipeline
        propositions = ["Fact 1", "Fact 2"]
        translator.llm_provider.call = Mock(side_effect=[
            json.dumps(propositions),  # Proposition extraction
            "Generated sentence."  # Graph generation
        ])

        input_graph = {
            'mermaid': 'graph LR; P0 --> P1',
            'description': 'A chain',
            'node_map': {'P0': 'Fact 1', 'P1': 'Fact 2'},
            'node_count': 2
        }
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {'ROOT': 'P0', 'CLAIM': 'P1'},
            'style_metadata': {},
            'distance': 0.5
        }

        translator.input_mapper.map_propositions.return_value = input_graph
        translator.graph_matcher.get_best_match.return_value = blueprint

        # Call translate_paragraph_propositions
        result = translator.translate_paragraph_propositions(
            "Some text",
            "Mao",
            document_context={'current_index': 5, 'total_paragraphs': 10},
            verbose=False
        )

        # Verify telemetry was called
        assert translator.telemetry is not None
        # Check that log file was written
        assert self.log_path.exists()

        # Read log to verify content
        with open(self.log_path, 'r') as f:
            content = f.read()
            assert "## Paragraph 5" in content
            assert "graph LR; P0 --> P1" in content
            assert "graph LR; ROOT --> CLAIM" in content

    def test_telemetry_not_called_when_disabled(self):
        """Test that telemetry is not called when disabled."""
        config = {
            "provider": "deepseek",
            "deepseek": {
                "api_key": "test_key",
                "api_url": "https://api.deepseek.com/v1/chat/completions",
                "editor_model": "deepseek-chat",
                "critic_model": "deepseek-chat"
            },
            "graph_pipeline": {
                "telemetry": {
                    "enabled": False
                }
            },
            "generation": {},
            "translator": {},
            "llm_provider": {}
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

        translator = StyleTranslator(config_path=str(self.config_path))
        translator.llm_provider = MockLLMProvider()
        translator.input_mapper = MagicMock()
        translator.graph_matcher = MagicMock()

        # Mock successful graph pipeline
        propositions = ["Fact 1", "Fact 2"]
        translator.llm_provider.call = Mock(side_effect=[
            json.dumps(propositions),
            "Generated sentence."
        ])

        input_graph = {
            'mermaid': 'graph LR; P0 --> P1',
            'description': 'A chain',
            'node_map': {'P0': 'Fact 1', 'P1': 'Fact 2'},
            'node_count': 2
        }
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {'ROOT': 'P0', 'CLAIM': 'P1'},
            'style_metadata': {},
            'distance': 0.5
        }

        translator.input_mapper.map_propositions.return_value = input_graph
        translator.graph_matcher.get_best_match.return_value = blueprint

        translator.translate_paragraph_propositions(
            "Some text",
            "Mao",
            verbose=False
        )

        # Telemetry should be None
        assert translator.telemetry is None

    def test_telemetry_handles_exception_gracefully(self):
        """Test that telemetry exceptions don't break generation."""
        translator = StyleTranslator(config_path=str(self.config_path))
        translator.llm_provider = MockLLMProvider()
        translator.input_mapper = MagicMock()
        translator.graph_matcher = MagicMock()

        # Mock telemetry to raise exception
        translator.telemetry.log_match = Mock(side_effect=Exception("Telemetry error"))

        # Mock successful graph pipeline
        propositions = ["Fact 1", "Fact 2"]
        translator.llm_provider.call = Mock(side_effect=[
            json.dumps(propositions),
            "Generated sentence."
        ])

        input_graph = {
            'mermaid': 'graph LR; P0 --> P1',
            'description': 'A chain',
            'node_map': {'P0': 'Fact 1', 'P1': 'Fact 2'},
            'node_count': 2
        }
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {'ROOT': 'P0', 'CLAIM': 'P1'},
            'style_metadata': {},
            'distance': 0.5
        }

        translator.input_mapper.map_propositions.return_value = input_graph
        translator.graph_matcher.get_best_match.return_value = blueprint

        # Should not raise exception
        result = translator.translate_paragraph_propositions(
            "Some text",
            "Mao",
            verbose=True
        )

        # Should still generate text
        assert isinstance(result, tuple)
        text, _, _ = result
        assert len(text) > 0

    def test_telemetry_uses_correct_paragraph_index(self):
        """Test that telemetry uses correct paragraph index from document_context."""
        translator = StyleTranslator(config_path=str(self.config_path))
        translator.llm_provider = MockLLMProvider()
        translator.input_mapper = MagicMock()
        translator.graph_matcher = MagicMock()

        # Mock successful graph pipeline - need 3 calls for proposition extraction (one per index)
        # and 3 calls for graph generation
        propositions = ["Fact 1"]
        translator.llm_provider.call = Mock(side_effect=[
            json.dumps(propositions),  # Index 0: proposition extraction
            "Generated sentence 0.",   # Index 0: graph generation
            json.dumps(propositions),  # Index 5: proposition extraction
            "Generated sentence 5.",   # Index 5: graph generation
            json.dumps(propositions),  # Index 10: proposition extraction
            "Generated sentence 10."   # Index 10: graph generation
        ])

        input_graph = {
            'mermaid': 'graph LR; P0',
            'description': 'Single node',
            'node_map': {'P0': 'Fact 1'},
            'node_count': 1
        }
        blueprint = {
            'style_mermaid': 'graph LR; ROOT',
            'node_mapping': {'ROOT': 'P0'},
            'style_metadata': {},
            'distance': 0.3
        }

        translator.input_mapper.map_propositions.return_value = input_graph
        translator.graph_matcher.get_best_match.return_value = blueprint

        # Test with different paragraph indices
        for idx in [0, 5, 10]:
            translator.translate_paragraph_propositions(
                "Some text",
                "Mao",
                document_context={'current_index': idx, 'total_paragraphs': 20},
                verbose=False
            )

        # Check log file
        with open(self.log_path, 'r') as f:
            content = f.read()

        # Should have entries for all indices
        assert "## Paragraph 0" in content
        assert "## Paragraph 5" in content
        assert "## Paragraph 10" in content

    def test_telemetry_not_called_when_no_blueprint(self):
        """Test that telemetry is not called when blueprint is None."""
        translator = StyleTranslator(config_path=str(self.config_path))
        translator.llm_provider = MockLLMProvider()
        translator.input_mapper = MagicMock()
        translator.graph_matcher = MagicMock()

        # Mock graph_matcher returning None
        propositions = ["Fact 1", "Fact 2"]
        translator.llm_provider.call = Mock(side_effect=[
            json.dumps(propositions),
            "Fallback text."
        ])

        input_graph = {
            'mermaid': 'graph LR; P0 --> P1',
            'description': 'A chain',
            'node_map': {'P0': 'Fact 1', 'P1': 'Fact 2'},
            'node_count': 2
        }

        translator.input_mapper.map_propositions.return_value = input_graph
        translator.graph_matcher.get_best_match.return_value = None

        # Mock telemetry to track calls
        log_match_spy = Mock()
        translator.telemetry.log_match = log_match_spy

        translator.translate_paragraph_propositions(
            "Some text",
            "Mao",
            verbose=False
        )

        # Telemetry should not be called when blueprint is None
        log_match_spy.assert_not_called()


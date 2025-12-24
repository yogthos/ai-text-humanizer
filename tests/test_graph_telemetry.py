"""Comprehensive unit tests for GraphLogger telemetry."""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.graph_telemetry import GraphLogger


class TestGraphLogger:
    """Test suite for GraphLogger."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        self.log_path = Path(self.temp_dir) / "logs" / "graph_debug.md"

        # Create config with telemetry enabled
        config = {
            "graph_pipeline": {
                "telemetry": {
                    "enabled": True,
                    "log_path": str(self.log_path)
                }
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

    def test_init_creates_log_directory(self):
        """Test that initialization creates logs directory."""
        logger = GraphLogger(config_path=str(self.config_path))

        assert self.log_path.parent.exists()
        assert self.log_path.parent.is_dir()

    def test_init_creates_log_file_with_header(self):
        """Test that initialization creates log file with header."""
        logger = GraphLogger(config_path=str(self.config_path))

        assert self.log_path.exists()

        with open(self.log_path, 'r') as f:
            content = f.read()
            assert "# Graph Pipeline Debug Log" in content
            assert "This file contains graph matching decisions" in content

    def test_init_does_not_overwrite_existing_log(self):
        """Test that initialization doesn't overwrite existing log."""
        # Create existing log with content
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            f.write("Existing content\n")

        logger = GraphLogger(config_path=str(self.config_path))

        # Should still have existing content
        with open(self.log_path, 'r') as f:
            content = f.read()
            assert "Existing content" in content

    def test_init_uses_default_path_when_not_in_config(self):
        """Test that default log path is used when not in config."""
        config = {"graph_pipeline": {}}
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

        logger = GraphLogger(config_path=str(self.config_path))

        # Should use default path (logs/graph_debug.md relative to project root)
        assert logger.log_path.name == "graph_debug.md"

    def test_log_match_writes_correct_format(self):
        """Test that log_match writes correct Markdown format."""
        logger = GraphLogger(config_path=str(self.config_path))

        input_graph = {
            'mermaid': 'graph LR; P0 --> P1',
            'description': 'A causal chain',
            'node_count': 2,
            'node_map': {'P0': 'Fact 1', 'P1': 'Fact 2'}
        }

        style_match = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {'ROOT': 'P0', 'CLAIM': 'P1'},
            'distance': 0.3456,
            'style_metadata': {'node_count': 2}
        }

        final_text = "Generated text here."

        logger.log_match(
            paragraph_index=0,
            input_graph=input_graph,
            style_match=style_match,
            final_text=final_text
        )

        # Read log file
        with open(self.log_path, 'r') as f:
            content = f.read()

        # Check format
        assert "## Paragraph 0" in content
        assert "### Input Graph" in content
        assert "```mermaid" in content
        assert "graph LR; P0 --> P1" in content
        assert "### Matched Style Graph" in content
        assert "graph LR; ROOT --> CLAIM" in content
        assert "**Distance:** 0.3456" in content
        assert "**Node Count:** Input=2, Style=2" in content
        assert "### Node Mapping" in content
        assert "```json" in content
        assert '"ROOT": "P0"' in content
        assert "### Generated Text" in content
        assert "> Generated text here." in content
        assert "---" in content

    def test_log_match_handles_missing_fields(self):
        """Test that log_match handles missing optional fields."""
        logger = GraphLogger(config_path=str(self.config_path))

        input_graph = {
            'mermaid': 'graph LR; P0 --> P1',
            'node_count': 2
        }

        style_match = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {}
        }

        logger.log_match(
            paragraph_index=1,
            input_graph=input_graph,
            style_match=style_match,
            final_text="Text"
        )

        # Should not raise exception
        assert self.log_path.exists()

    def test_log_match_handles_non_numeric_distance(self):
        """Test that log_match handles non-numeric distance values."""
        logger = GraphLogger(config_path=str(self.config_path))

        input_graph = {'mermaid': 'graph LR; P0 --> P1', 'node_count': 2}
        style_match = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {},
            'distance': 'N/A'
        }

        logger.log_match(
            paragraph_index=0,
            input_graph=input_graph,
            style_match=style_match,
            final_text="Text"
        )

        with open(self.log_path, 'r') as f:
            content = f.read()
            assert "**Distance:** N/A" in content

    def test_log_match_handles_missing_style_metadata(self):
        """Test that log_match handles missing style_metadata."""
        logger = GraphLogger(config_path=str(self.config_path))

        input_graph = {'mermaid': 'graph LR; P0 --> P1', 'node_count': 2}
        style_match = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {}
        }

        logger.log_match(
            paragraph_index=0,
            input_graph=input_graph,
            style_match=style_match,
            final_text="Text"
        )

        with open(self.log_path, 'r') as content:
            content_str = content.read()
            assert "Style=N/A" in content_str

    def test_log_match_appends_multiple_entries(self):
        """Test that multiple log_match calls append entries."""
        logger = GraphLogger(config_path=str(self.config_path))

        input_graph = {'mermaid': 'graph LR; P0 --> P1', 'node_count': 2}
        style_match = {'style_mermaid': 'graph LR; ROOT --> CLAIM', 'node_mapping': {}}

        # Log multiple entries
        logger.log_match(0, input_graph, style_match, "Text 1")
        logger.log_match(1, input_graph, style_match, "Text 2")
        logger.log_match(2, input_graph, style_match, "Text 3")

        with open(self.log_path, 'r') as f:
            content = f.read()

        assert content.count("## Paragraph") == 3
        assert "## Paragraph 0" in content
        assert "## Paragraph 1" in content
        assert "## Paragraph 2" in content
        assert "> Text 1" in content
        assert "> Text 2" in content
        assert "> Text 3" in content

    def test_log_match_handles_file_write_error(self):
        """Test that log_match handles file write errors gracefully."""
        logger = GraphLogger(config_path=str(self.config_path))

        # Make log path a directory to cause write error
        self.log_path.unlink(missing_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)

        input_graph = {'mermaid': 'graph LR; P0 --> P1', 'node_count': 2}
        style_match = {'style_mermaid': 'graph LR; ROOT --> CLAIM', 'node_mapping': {}}

        # Should not raise exception
        logger.log_match(0, input_graph, style_match, "Text")

        # Should print warning (we can't easily test print, but no exception is good)

    def test_log_match_json_serialization(self):
        """Test that node_mapping is properly JSON serialized."""
        logger = GraphLogger(config_path=str(self.config_path))

        input_graph = {'mermaid': 'graph LR; P0 --> P1', 'node_count': 2}
        style_match = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {
                'ROOT': 'P0, P1',
                'CLAIM': 'P2',
                'EXTRA': 'UNUSED'
            }
        }

        logger.log_match(0, input_graph, style_match, "Text")

        with open(self.log_path, 'r') as f:
            content = f.read()

        # Check JSON is properly formatted
        json_start = content.find("```json")
        json_end = content.find("```", json_start + 7)
        json_content = content[json_start + 7:json_end].strip()

        # Should be valid JSON
        parsed = json.loads(json_content)
        assert parsed['ROOT'] == 'P0, P1'
        assert parsed['CLAIM'] == 'P2'
        assert parsed['EXTRA'] == 'UNUSED'

    def test_log_match_with_complex_graphs(self):
        """Test logging with complex multi-node graphs."""
        logger = GraphLogger(config_path=str(self.config_path))

        input_graph = {
            'mermaid': 'graph LR; P0 --> P1; P1 --> P2; P2 --> P3',
            'description': 'A complex causal chain',
            'node_count': 4,
            'node_map': {
                'P0': 'First fact',
                'P1': 'Second fact',
                'P2': 'Third fact',
                'P3': 'Conclusion'
            }
        }

        style_match = {
            'style_mermaid': 'graph LR; ROOT --> A; A --> B; B --> C; C --> D',
            'node_mapping': {
                'ROOT': 'P0',
                'A': 'P1',
                'B': 'P2',
                'C': 'P3',
                'D': 'UNUSED'
            },
            'distance': 0.1234,
            'style_metadata': {'node_count': 5}
        }

        logger.log_match(0, input_graph, style_match, "Complex generated text.")

        with open(self.log_path, 'r') as f:
            content = f.read()

        # Verify all components are present
        assert "graph LR; P0 --> P1; P1 --> P2; P2 --> P3" in content
        assert "graph LR; ROOT --> A; A --> B; B --> C; C --> D" in content
        assert "**Node Count:** Input=4, Style=5" in content
        assert '"D": "UNUSED"' in content


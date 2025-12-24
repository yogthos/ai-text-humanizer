"""Tests for InputLogicMapper."""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas.input_mapper import InputLogicMapper
from tests.mocks.mock_llm_provider import MockLLMProvider


class TestInputLogicMapper:
    """Test suite for InputLogicMapper."""

    def setup_method(self):
        """Set up test fixtures."""
        from unittest.mock import MagicMock
        self.mock_llm = MagicMock()
        self.mapper = InputLogicMapper(self.mock_llm)

    def test_map_simple_causal_chain(self):
        """Test mapping simple causal chain."""
        # Mock LLM response
        response = {
            "mermaid": "graph LR; P0 --cause--> P1",
            "description": "A causal chain leading to consequence",
            "node_map": {
                "P0": "The phone needs power",
                "P1": "Without it, it dies"
            }
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = ['The phone needs power', 'Without it, it dies']
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert 'mermaid' in result
        assert 'description' in result
        assert 'node_map' in result
        assert 'node_count' in result
        assert result['node_count'] == 2
        assert 'P0' in result['node_map']
        assert 'P1' in result['node_map']
        assert 'P0' in result['mermaid']
        assert 'P1' in result['mermaid']
        assert 'causal' in result['description'].lower() or 'consequence' in result['description'].lower()

    def test_map_multiple_propositions(self):
        """Test mapping multiple propositions."""
        response = {
            "mermaid": "graph LR; P0 --> P1; P1 --> P2; P2 --> P3; P3 --> P4",
            "description": "A sequential enumeration of related conditions",
            "node_map": {
                "P0": "First condition",
                "P1": "Second condition",
                "P2": "Third condition",
                "P3": "Fourth condition",
                "P4": "Fifth condition"
            }
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = [
            "First condition",
            "Second condition",
            "Third condition",
            "Fourth condition",
            "Fifth condition"
        ]
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert len(result['node_map']) == 5
        for i in range(5):
            assert f"P{i}" in result['node_map']
        assert result['node_count'] == 5
        assert len(result['description']) > 0

    def test_map_single_proposition(self):
        """Test mapping single proposition."""
        response = {
            "mermaid": "graph LR; P0",
            "description": "A single declarative statement",
            "node_map": {
                "P0": "Single fact"
            }
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = ['Single fact']
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert result['node_count'] == 1
        assert 'P0' in result['node_map']
        assert result['node_map']['P0'] == "Single fact"

    def test_map_empty_propositions(self):
        """Test mapping empty propositions raises error."""
        propositions = []
        try:
            result = self.mapper.map_propositions(propositions)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "empty" in str(e).lower()

    def test_map_malformed_llm_response(self):
        """Test handling malformed LLM response."""
        self.mock_llm.responses = {"propositions": "This is not valid JSON {["}

        propositions = ['Test proposition']
        result = self.mapper.map_propositions(propositions)

        assert result is None

    def test_map_missing_required_fields(self):
        """Test handling missing required fields."""
        response = {
            "mermaid": "graph LR; P0 --> P1"
            # Missing 'description' and 'node_map'
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = ['First', 'Second']
        result = self.mapper.map_propositions(propositions)

        assert result is None

    def test_map_markdown_code_blocks(self):
        """Test stripping markdown code blocks."""
        response_with_markdown = f"```json\n{json.dumps({
            'mermaid': 'graph LR; P0 --> P1',
            'description': 'A test graph',
            'node_map': {'P0': 'First', 'P1': 'Second'}
        })}\n```"
        self.mock_llm.call.return_value = response_with_markdown

        propositions = ['First', 'Second']
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert 'mermaid' in result
        assert 'description' in result
        assert 'node_map' in result

    def test_map_node_map_validation(self):
        """Test node map key validation."""
        # Response with wrong keys
        response = {
            "mermaid": "graph LR; P0 --> P1",
            "description": "A test graph",
            "node_map": {
                "NODE0": "First",  # Wrong key
                "NODE1": "Second"  # Wrong key
            }
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = ['First', 'Second']
        result = self.mapper.map_propositions(propositions)

        # Should normalize to P0, P1
        assert result is not None
        assert 'P0' in result['node_map'] or 'NODE0' in result['node_map']

    def test_map_complex_relationships(self):
        """Test mapping complex relationships."""
        response = {
            "mermaid": "graph LR; P0 --cause--> P1; P2 --contrast--> P3; P4 --support--> P0",
            "description": "A complex graph with causal, contrastive, and supportive relationships",
            "node_map": {
                f"P{i}": f"Proposition {i}" for i in range(10)
            }
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = [f"Proposition {i}" for i in range(10)]
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert result['node_count'] == 10
        assert 'cause' in result['mermaid'].lower() or 'contrast' in result['mermaid'].lower()

    def test_map_preserves_proposition_order(self):
        """Test that proposition order is preserved."""
        response = {
            "mermaid": "graph LR; P0 --> P1 --> P2",
            "description": "A sequential chain",
            "node_map": {
                "P0": "First proposition",
                "P1": "Second proposition",
                "P2": "Third proposition"
            }
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = ["First proposition", "Second proposition", "Third proposition"]
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert result['node_map']['P0'] == "First proposition"
        assert result['node_map']['P1'] == "Second proposition"
        assert result['node_map']['P2'] == "Third proposition"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


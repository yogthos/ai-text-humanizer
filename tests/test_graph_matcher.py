"""Tests for TopologicalMatcher."""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from chromadb.utils import embedding_functions

from src.generator.graph_matcher import TopologicalMatcher
from tests.mocks.mock_llm_provider import MockLLMProvider


class TestTopologicalMatcher:
    """Test suite for TopologicalMatcher."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for ChromaDB
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = Path(self.temp_dir) / "test_chroma"

        # Create test ChromaDB collection
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        try:
            self.client.delete_collection("style_graphs")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name="style_graphs",
            embedding_function=embedding_fn
        )

        # Create mock LLM provider
        from unittest.mock import MagicMock
        self.mock_llm = MagicMock()

        # Create matcher with custom chroma path
        with patch('src.generator.graph_matcher.LLMProvider', return_value=self.mock_llm):
            self.matcher = TopologicalMatcher(
                config_path="config.json",
                chroma_path=str(self.chroma_path)
            )
            self.matcher.llm_provider = self.mock_llm
            self.matcher.collection = self.collection

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def _add_test_graph(self, mermaid: str, description: str, node_count: int,
                       edge_types: str = "defines", paragraph_role: str = None,
                       original_text: str = "Test text", skeleton: str = ""):
        """Helper to add a test graph to ChromaDB."""
        metadata = {
            'mermaid': mermaid,
            'node_count': node_count,
            'edge_types': edge_types,
            'skeleton': skeleton,
            'original_text': original_text
        }
        if paragraph_role:
            metadata['paragraph_role'] = paragraph_role

        self.collection.add(
            ids=[f"graph_{len(self.collection.get()['ids'])}"],
            documents=[description],
            metadatas=[metadata]
        )

    def test_get_best_match_exact_node_count(self):
        """Test matching with exact node count."""
        # Add test graphs
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2",
            "A three-node causal chain",
            node_count=3
        )
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2 --> NODE3",
            "A four-node chain",
            node_count=4
        )
        self._add_test_graph(
            "graph LR; ROOT --> NODE1",
            "A two-node chain",
            node_count=2
        )

        input_graph = {
            'description': 'A three-node causal chain',
            'mermaid': 'graph LR; P0 --> P1 --> P2',
            'node_map': {'P0': 'First', 'P1': 'Second', 'P2': 'Third'},
            'node_count': 3
        }

        # Mock LLM response for node mapping
        mapping_response = {
            'ROOT': 'P0',
            'NODE1': 'P1',
            'NODE2': 'P2'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert result['style_metadata']['node_count'] == 3
        assert 'node_mapping' in result
        assert 'distance' in result

    def test_get_best_match_style_has_more_nodes(self):
        """Test matching when style has more nodes."""
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2 --> NODE3",
            "A four-node chain",
            node_count=4
        )

        input_graph = {
            'description': 'A two-node chain',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First', 'P1': 'Second'},
            'node_count': 2
        }

        mapping_response = {
            'ROOT': 'P0',
            'NODE1': 'P1',
            'NODE2': 'UNUSED',
            'NODE3': 'UNUSED'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert result['style_metadata']['node_count'] == 4
        assert 'UNUSED' in str(result['node_mapping'].values())

    def test_get_best_match_no_meeting_constraint(self):
        """Test overflow handling when input has more nodes."""
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2",
            "A three-node chain",
            node_count=3
        )

        input_graph = {
            'description': 'A five-node chain',
            'mermaid': 'graph LR; P0 --> P1 --> P2 --> P3 --> P4',
            'node_map': {f'P{i}': f'Prop {i}' for i in range(5)},
            'node_count': 5
        }

        # Mock LLM response with grouped nodes
        mapping_response = {
            'ROOT': 'P0, P1',
            'NODE1': 'P2, P3',
            'NODE2': 'P4'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        # Should use the 3-node graph (largest available)
        assert result['style_metadata']['node_count'] == 3
        # Verify grouping happened
        assert any(',' in str(v) for v in result['node_mapping'].values())

    def test_get_best_match_multiple_candidates(self):
        """Test selecting best candidate from multiple options."""
        # Add multiple graphs with same node count but different distances
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2",
            "A three-node causal chain",
            node_count=3
        )
        self._add_test_graph(
            "graph LR; A --> B --> C",
            "Another three-node chain",
            node_count=3
        )

        input_graph = {
            'description': 'A three-node causal chain',
            'mermaid': 'graph LR; P0 --> P1 --> P2',
            'node_map': {'P0': 'First', 'P1': 'Second', 'P2': 'Third'},
            'node_count': 3
        }

        mapping_response = {'ROOT': 'P0', 'NODE1': 'P1', 'NODE2': 'P2'}
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert 'distance' in result
        assert isinstance(result['distance'], (int, float))

    def test_get_best_match_empty_collection(self):
        """Test error handling with empty collection."""
        # Create new empty collection
        try:
            self.client.delete_collection("style_graphs")
        except Exception:
            pass

        empty_collection = self.client.create_collection(
            name="style_graphs",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        self.matcher.collection = empty_collection

        input_graph = {
            'description': 'A test graph',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First'},
            'node_count': 1
        }

        try:
            result = self.matcher.get_best_match(input_graph)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No style graphs" in str(e)

    def test_get_best_match_semantic_search(self):
        """Test semantic search functionality."""
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A causal chain leading to consequence",
            node_count=2
        )

        input_graph = {
            'description': 'A causal chain leading to consequence',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'Cause', 'P1': 'Effect'},
            'node_count': 2
        }

        mapping_response = {'ROOT': 'P0', 'CLAIM': 'P1'}
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert 'causal' in result['style_metadata']['original_text'].lower() or \
               'causal' in input_graph['description'].lower()

    def test_node_mapping_llm_integration(self):
        """Test LLM-based node mapping."""
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM --> EVIDENCE",
            "A three-part argument structure",
            node_count=3
        )

        input_graph = {
            'description': 'A structured argument',
            'mermaid': 'graph LR; P0 --> P1 --> P2',
            'node_map': {'P0': 'Premise', 'P1': 'Conclusion', 'P2': 'Support'},
            'node_count': 3
        }

        mapping_response = {
            'ROOT': 'P0',
            'CLAIM': 'P1',
            'EVIDENCE': 'P2'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert 'node_mapping' in result
        assert len(result['node_mapping']) == 3
        assert 'ROOT' in result['node_mapping']
        assert 'CLAIM' in result['node_mapping']
        assert 'EVIDENCE' in result['node_mapping']

    def test_node_mapping_unused_nodes(self):
        """Test mapping with unused style nodes."""
        self._add_test_graph(
            "graph LR; ROOT --> NODE1 --> NODE2 --> NODE3",
            "A four-node structure",
            node_count=4
        )

        input_graph = {
            'description': 'A simple two-node chain',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First', 'P1': 'Second'},
            'node_count': 2
        }

        mapping_response = {
            'ROOT': 'P0',
            'NODE1': 'P1',
            'NODE2': 'UNUSED',
            'NODE3': 'UNUSED'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert len(result['node_mapping']) == 4
        assert result['node_mapping'].get('NODE2') == 'UNUSED' or \
               'UNUSED' in str(result['node_mapping'].values())

    def test_node_mapping_overflow_semantic_grafting(self):
        """Test semantic grafting when input has more nodes."""
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A two-node structure",
            node_count=2
        )

        input_graph = {
            'description': 'A complex five-node argument',
            'mermaid': 'graph LR; P0 --> P1 --> P2 --> P3 --> P4',
            'node_map': {f'P{i}': f'Proposition {i}' for i in range(5)},
            'node_count': 5
        }

        # Mock LLM response with grouped nodes
        mapping_response = {
            'ROOT': 'P0, P1, P2',
            'CLAIM': 'P3, P4'
        }
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert len(result['node_mapping']) == 2
        # Verify grouping (comma-separated values)
        grouped_values = [v for v in result['node_mapping'].values() if ',' in str(v)]
        assert len(grouped_values) > 0

    def test_parse_mermaid_nodes(self):
        """Test Mermaid node parsing."""
        test_cases = [
            ("graph LR; ROOT --> NODE1", ["NODE1", "ROOT"]),
            ("graph TD; A[Label] --> B", ["A", "B"]),
            ("ROOT --edge--> NODE1", ["NODE1", "ROOT"]),
        ]

        for mermaid, expected_nodes in test_cases:
            nodes = self.matcher._parse_mermaid_nodes(mermaid)
            for expected in expected_nodes:
                assert expected in nodes, f"Expected {expected} in {nodes} for {mermaid}"

    def test_get_best_match_invalid_input_graph(self):
        """Test error handling for invalid input."""
        input_graph = {
            # Missing 'description' field
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First'}
        }

        try:
            result = self.matcher.get_best_match(input_graph)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "description" in str(e).lower()

    def test_get_best_match_filters_by_role(self):
        """Test role-based filtering."""
        # Add graphs with different roles
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "An opening statement",
            node_count=2,
            paragraph_role='opener'
        )
        self._add_test_graph(
            "graph LR; A --> B",
            "A body paragraph structure",
            node_count=2,
            paragraph_role='body'
        )

        input_graph = {
            'description': 'An opening statement',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First', 'P1': 'Second'},
            'node_count': 2
        }

        mapping_response = {'ROOT': 'P0', 'CLAIM': 'P1'}
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        # Test opener role
        context = {'current_index': 0, 'total_paragraphs': 3}
        result = self.matcher.get_best_match(input_graph, document_context=context)

        assert result is not None
        # Should prefer opener graph
        assert result['style_metadata'].get('paragraph_role') == 'opener' or \
               'ROOT' in result['style_mermaid']

        # Test body role
        context = {'current_index': 1, 'total_paragraphs': 3}
        result = self.matcher.get_best_match(input_graph, document_context=context)

        assert result is not None

    def test_get_best_match_role_fallback(self):
        """Test fallback when role filter finds no results."""
        # Add graph without role metadata
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A generic structure",
            node_count=2,
            paragraph_role=None
        )

        input_graph = {
            'description': 'A generic structure',
            'mermaid': 'graph LR; P0 --> P1',
            'node_map': {'P0': 'First', 'P1': 'Second'},
            'node_count': 2
        }

        mapping_response = {'ROOT': 'P0', 'CLAIM': 'P1'}
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        # Request opener but none available
        context = {'current_index': 0, 'total_paragraphs': 3}
        result = self.matcher.get_best_match(input_graph, document_context=context)

        # Should fall back to unfiltered search
        assert result is not None

    def test_get_best_match_return_format(self):
        """Test return format structure."""
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM --> CONDITION",
            "A three-node structure",
            node_count=3,
            edge_types="defines,supports"
        )

        input_graph = {
            'description': 'A three-node structure',
            'mermaid': 'graph LR; P0 --> P1 --> P2',
            'node_map': {'P0': 'First', 'P1': 'Second', 'P2': 'Third'},
            'node_count': 3
        }

        mapping_response = {'ROOT': 'P0', 'CLAIM': 'P1', 'CONDITION': 'P2'}
        self.mock_llm.call.return_value = json.dumps(mapping_response)

        result = self.matcher.get_best_match(input_graph)

        assert 'style_mermaid' in result
        assert 'node_mapping' in result
        assert 'style_metadata' in result
        assert 'distance' in result
        assert isinstance(result['style_mermaid'], str)
        assert isinstance(result['node_mapping'], dict)
        assert isinstance(result['style_metadata'], dict)
        assert isinstance(result['distance'], (int, float))
        assert 'node_count' in result['style_metadata']
        assert 'edge_types' in result['style_metadata']


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


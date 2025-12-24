"""Integration tests for the graph pipeline."""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from chromadb.utils import embedding_functions

from src.atlas.input_mapper import InputLogicMapper
from src.generator.graph_matcher import TopologicalMatcher
from tests.mocks.mock_llm_provider import MockLLMProvider


class TestGraphPipelineIntegration:
    """Integration tests for the complete graph pipeline."""

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

        # Create mapper
        self.mapper = InputLogicMapper(self.mock_llm)

        # Create matcher
        with patch('src.generator.graph_matcher.LLMProvider', return_value=self.mock_llm):
            self.matcher = TopologicalMatcher(
                config_path="config.json",
                chroma_path=str(self.chroma_path)
            )
            self.matcher.llm_provider = self.mock_llm
            self.matcher.collection = self.collection

    def teardown_method(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def _add_test_graph(self, mermaid: str, description: str, node_count: int,
                       edge_types: str = "defines", paragraph_role: str = None,
                       skeleton: str = ""):
        """Helper to add a test graph to ChromaDB."""
        metadata = {
            'mermaid': mermaid,
            'node_count': node_count,
            'edge_types': edge_types,
            'skeleton': skeleton,
            'original_text': 'Test original text'
        }
        if paragraph_role:
            metadata['paragraph_role'] = paragraph_role

        self.collection.add(
            ids=[f"graph_{len(self.collection.get()['ids'])}"],
            documents=[description],
            metadatas=[metadata]
        )

    def test_end_to_end_propositions_to_mapping(self):
        """Test complete pipeline from propositions to mapping."""
        # Add a style graph
        self._add_test_graph(
            "graph LR; ROOT --defines--> CLAIM --supports--> EVIDENCE",
            "A definition followed by supporting evidence",
            node_count=3
        )

        # Step 1: Map propositions to input graph
        propositions = [
            "The concept requires definition",
            "It must be clearly stated",
            "Evidence supports the definition"
        ]

        # Mock LLM response for mapping propositions
        mapper_response = {
            "mermaid": "graph LR; P0 --cause--> P1; P1 --support--> P2",
            "description": "A definition followed by supporting evidence",
            "node_map": {
                "P0": "The concept requires definition",
                "P1": "It must be clearly stated",
                "P2": "Evidence supports the definition"
            }
        }
        self.mock_llm.call.return_value = json.dumps(mapper_response)

        input_graph = self.mapper.map_propositions(propositions)

        assert input_graph is not None
        assert input_graph['node_count'] == 3

        # Step 2: Match to style graph
        matcher_response = {
            'ROOT': 'P0',
            'CLAIM': 'P1',
            'EVIDENCE': 'P2'
        }
        self.mock_llm.call.return_value = json.dumps(matcher_response)

        result = self.matcher.get_best_match(input_graph)

        assert result is not None
        assert 'style_mermaid' in result
        assert 'node_mapping' in result
        assert len(result['node_mapping']) == 3

        # Verify all propositions are accounted for
        all_mapped = []
        for mapping_value in result['node_mapping'].values():
            if mapping_value != 'UNUSED':
                if ',' in str(mapping_value):
                    all_mapped.extend([p.strip() for p in str(mapping_value).split(',')])
                else:
                    all_mapped.append(str(mapping_value))

        # Check that all input nodes (P0, P1, P2) are in the mapping
        assert 'P0' in all_mapped or 'P0' in str(result['node_mapping'].values())
        assert 'P1' in all_mapped or 'P1' in str(result['node_mapping'].values())
        assert 'P2' in all_mapped or 'P2' in str(result['node_mapping'].values())

    def test_pipeline_with_real_chromadb(self):
        """Test pipeline with actual ChromaDB (if available)."""
        # This test uses the actual ChromaDB setup
        # Add multiple style graphs
        self._add_test_graph(
            "graph LR; ROOT --> CLAIM",
            "A causal chain leading to consequence",
            node_count=2,
            paragraph_role='opener'
        )
        self._add_test_graph(
            "graph LR; A --> B --> C",
            "A sequential enumeration",
            node_count=3,
            paragraph_role='body'
        )

        propositions = ["First point", "Second point", "Third point"]

        # Mock mapper response
        mapper_response = {
            "mermaid": "graph LR; P0 --> P1 --> P2",
            "description": "A sequential enumeration",
            "node_map": {
                "P0": "First point",
                "P1": "Second point",
                "P2": "Third point"
            }
        }
        self.mock_llm.call.return_value = json.dumps(mapper_response)

        input_graph = self.mapper.map_propositions(propositions)

        assert input_graph is not None

        # Test with context (should match body graph)
        context = {'current_index': 1, 'total_paragraphs': 3}

        matcher_response = {
            'A': 'P0',
            'B': 'P1',
            'C': 'P2'
        }
        self.mock_llm.call.return_value = json.dumps(matcher_response)

        result = self.matcher.get_best_match(input_graph, document_context=context)

        assert result is not None
        # Should prefer body graph due to context
        assert result['style_metadata'].get('paragraph_role') == 'body' or \
               result['style_metadata']['node_count'] == 3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


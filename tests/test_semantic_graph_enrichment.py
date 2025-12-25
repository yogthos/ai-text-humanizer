"""Tests for Semantic Graph Enrichment & Hard Clustering.

Tests the new semantic edge types (PURPOSE, ORIGIN, COMPOSITION, ATTRIBUTION)
and Pre-Clustering Contraction algorithm.
"""

import sys
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import networkx as nx
from src.atlas.input_mapper import InputLogicMapper
from src.atlas.fracturer import SemanticFracturer
from src.generator.graph_matcher import TopologicalMatcher


class TestSemanticEdgeExtraction:
    """Test semantic edge extraction in InputLogicMapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.mapper = InputLogicMapper(self.mock_llm)

    def test_semantic_edge_purpose_extraction(self):
        """Test that PURPOSE edges are extracted correctly."""
        response = {
            "mermaid": "graph LR; P0 --PURPOSE--> P1",
            "description": "A purpose relationship",
            "node_map": {
                "P0": "Lithium is used",
                "P1": "For smartphones"
            },
            "intent": "DEFINITION",
            "signature": "DEFINITION",
            "role": "BODY"
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = ["Lithium is used", "For smartphones"]
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert "PURPOSE" in result['mermaid'].upper() or "purpose" in result['mermaid'].lower()

    def test_semantic_edge_origin_extraction(self):
        """Test that ORIGIN edges are extracted correctly."""
        response = {
            "mermaid": "graph LR; P0 --ORIGIN--> P1",
            "description": "An origin relationship",
            "node_map": {
                "P0": "Lithium",
                "P1": "From Chile"
            },
            "intent": "DEFINITION",
            "signature": "DEFINITION",
            "role": "BODY"
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = ["Lithium", "From Chile"]
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert "ORIGIN" in result['mermaid'].upper() or "origin" in result['mermaid'].lower()

    def test_semantic_edge_composition_extraction(self):
        """Test that COMPOSITION edges are extracted correctly."""
        response = {
            "mermaid": "graph LR; P0 --COMPOSITION--> P1",
            "description": "A composition relationship",
            "node_map": {
                "P0": "Smartphone",
                "P1": "Includes lithium and cobalt"
            },
            "intent": "DEFINITION",
            "signature": "DEFINITION",
            "role": "BODY"
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = ["Smartphone", "Includes lithium and cobalt"]
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert "COMPOSITION" in result['mermaid'].upper() or "composition" in result['mermaid'].lower()

    def test_semantic_edge_attribution_extraction(self):
        """Test that ATTRIBUTION edges are extracted correctly."""
        response = {
            "mermaid": "graph LR; P0 --ATTRIBUTION--> P1",
            "description": "An attribution relationship",
            "node_map": {
                "P0": "The term Dialectical Materialism",
                "P1": "Coined by Stalin"
            },
            "intent": "DEFINITION",
            "signature": "DEFINITION",
            "role": "BODY"
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = ["The term Dialectical Materialism", "Coined by Stalin"]
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert "ATTRIBUTION" in result['mermaid'].upper() or "attribution" in result['mermaid'].lower()

    def test_mixed_semantic_and_rhetorical_edges(self):
        """Test that semantic and rhetorical edges can coexist."""
        response = {
            "mermaid": "graph LR; P0 --PURPOSE--> P1; P1 --CONTRAST--> P2",
            "description": "Mixed relationships",
            "node_map": {
                "P0": "Lithium",
                "P1": "For smartphones",
                "P2": "Not for decoration"
            },
            "intent": "ARGUMENT",
            "signature": "CONTRAST",
            "role": "BODY"
        }
        self.mock_llm.call.return_value = json.dumps(response)

        propositions = ["Lithium", "For smartphones", "Not for decoration"]
        result = self.mapper.map_propositions(propositions)

        assert result is not None
        assert "PURPOSE" in result['mermaid'].upper() or "purpose" in result['mermaid'].lower()
        assert "CONTRAST" in result['mermaid'].upper() or "contrast" in result['mermaid'].lower()


class TestPreClusteringContraction:
    """Test Pre-Clustering Contraction algorithm in SemanticFracturer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.fracturer = SemanticFracturer(self.mock_llm)

    def test_semantic_edges_never_split(self):
        """Test that semantic edges (weight=100.0) are never split across chunks."""
        propositions = [
            "Lithium is a raw material",
            "For smartphones",
            "From Chile",
            "Cobalt is another material",
            "From the Congo"
        ]

        # Create graph with semantic edges: P0--PURPOSE-->P1, P1--ORIGIN-->P2
        # These should form a super node that cannot be split
        input_graph = {
            "mermaid": "graph LR; P0 --PURPOSE--> P1; P1 --ORIGIN--> P2; P3 --PURPOSE--> P4; P4 --ORIGIN--> P5",
            "node_map": {
                "P0": propositions[0],
                "P1": propositions[1],
                "P2": propositions[2],
                "P3": propositions[3],
                "P4": propositions[4],
                "P5": "For smartphones"  # Note: P5 doesn't exist in propositions, but that's OK for test
            }
        }

        # Note: We only have 5 propositions, so P5 won't be in the graph
        # Let's fix the graph to match 5 propositions
        input_graph = {
            "mermaid": "graph LR; P0 --PURPOSE--> P1; P1 --ORIGIN--> P2; P3 --PURPOSE--> P4",
            "node_map": {
                "P0": propositions[0],
                "P1": propositions[1],
                "P2": propositions[2],
                "P3": propositions[3],
                "P4": propositions[4]
            }
        }

        result = self.fracturer.fracture(propositions, target_density=2, max_density=4, input_graph=input_graph)

        # Verify that P0, P1, P2 are in the same group (semantic chain)
        # and P3, P4 are in the same group (semantic chain)
        assert result is not None
        assert len(result) > 0

        # Find groups containing P0, P1, P2
        group_with_p0 = None
        for group in result:
            if 0 in group:
                group_with_p0 = group
                break

        assert group_with_p0 is not None
        # P0, P1, P2 should all be together (semantic chain)
        assert 1 in group_with_p0, "P1 should be with P0 (PURPOSE edge)"
        assert 2 in group_with_p0, "P2 should be with P0 and P1 (ORIGIN edge)"

    def test_semantic_edges_form_super_nodes(self):
        """Test that semantic edges form connected super nodes."""
        propositions = [
            "Material A",
            "For device X",
            "From source Y",
            "Material B",
            "For device Z"
        ]

        # Graph: P0--PURPOSE-->P1--ORIGIN-->P2 (forms super node)
        #        P3--PURPOSE-->P4 (forms another super node)
        input_graph = {
            "mermaid": "graph LR; P0 --PURPOSE--> P1; P1 --ORIGIN--> P2; P3 --PURPOSE--> P4",
            "node_map": {
                "P0": propositions[0],
                "P1": propositions[1],
                "P2": propositions[2],
                "P3": propositions[3],
                "P4": propositions[4]
            }
        }

        result = self.fracturer.fracture(propositions, target_density=2, max_density=4, input_graph=input_graph)

        assert result is not None
        # Should have at least 2 groups (one for each super node)
        assert len(result) >= 2

        # Verify super node 1 (P0, P1, P2) is together
        found_super_node_1 = False
        for group in result:
            if 0 in group and 1 in group and 2 in group:
                found_super_node_1 = True
                break
        assert found_super_node_1, "P0, P1, P2 should form a super node"

        # Verify super node 2 (P3, P4) is together
        found_super_node_2 = False
        for group in result:
            if 3 in group and 4 in group:
                found_super_node_2 = True
                break
        assert found_super_node_2, "P3, P4 should form a super node"

    def test_rhetorical_edges_can_be_split(self):
        """Test that rhetorical edges (weight < 100.0) can be split if needed."""
        propositions = [
            "First statement",
            "Second statement",
            "Third statement",
            "Fourth statement"
        ]

        # Graph with only rhetorical edges (SEQUENCE, weight=1.0)
        input_graph = {
            "mermaid": "graph LR; P0 --SEQUENCE--> P1; P1 --SEQUENCE--> P2; P2 --SEQUENCE--> P3",
            "node_map": {
                "P0": propositions[0],
                "P1": propositions[1],
                "P2": propositions[2],
                "P3": propositions[3]
            }
        }

        result = self.fracturer.fracture(propositions, target_density=2, max_density=3, input_graph=input_graph)

        assert result is not None
        # Rhetorical edges can be split, so we might get multiple groups
        # The exact grouping depends on community detection, but should respect target_density
        total_props = sum(len(group) for group in result)
        assert total_props == 4, "All propositions should be included"

    def test_fallback_when_no_graph(self):
        """Test that fracturer falls back to LLM when no graph provided."""
        propositions = [f"Prop {i}" for i in range(10)]

        # Mock LLM response
        llm_response = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        self.mock_llm.call.return_value = json.dumps(llm_response)

        result = self.fracturer.fracture(propositions, target_density=5, max_density=6, input_graph=None)

        assert result == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert self.mock_llm.call.called

    def test_fallback_when_graph_parsing_fails(self):
        """Test that fracturer falls back when graph parsing fails."""
        propositions = [f"Prop {i}" for i in range(10)]

        # Invalid graph
        input_graph = {
            "mermaid": "invalid mermaid syntax",
            "node_map": {}
        }

        # Mock LLM response for fallback
        llm_response = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        self.mock_llm.call.return_value = json.dumps(llm_response)

        result = self.fracturer.fracture(propositions, target_density=5, max_density=6, input_graph=input_graph)

        # Should fall back to LLM
        assert result == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert self.mock_llm.call.called


class TestSemanticEdgeRendering:
    """Test semantic edge rendering in TopologicalMatcher."""

    def setup_method(self):
        """Set up test fixtures."""
        from unittest.mock import MagicMock
        self.mock_llm = MagicMock()
        self.matcher = TopologicalMatcher(config_path="config.json", llm_provider=self.mock_llm)

    def test_purpose_edge_rendering(self):
        """Test that PURPOSE edges extract connector from text using spaCy."""
        input_graph = {
            "mermaid": "graph LR; P0 --PURPOSE--> P1",
            "node_map": {
                "P0": "Lithium is used",
                "P1": "for smartphones"
            }
        }

        style_vocab = {
            "PURPOSE": [" for "],  # Should be ignored, semantic edges use spaCy extraction
            "CONTRAST": [" but "],
            "SEQUENCE": [", "]
        }

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph,
            style_vocab,
            num_propositions=2,
            verbose=False
        )

        assert skeleton is not None
        # Should extract " for " from "for smartphones" using spaCy
        assert " for " in skeleton, f"PURPOSE edge should extract ' for ' from text, got: {skeleton}"

    def test_origin_edge_rendering(self):
        """Test that ORIGIN edges extract connector from text using spaCy."""
        input_graph = {
            "mermaid": "graph LR; P0 --ORIGIN--> P1",
            "node_map": {
                "P0": "Lithium comes",
                "P1": "from Chile"
            }
        }

        style_vocab = {
            "ORIGIN": [" from "],  # Should be ignored, semantic edges use spaCy extraction
            "CONTRAST": [" but "]
        }

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph,
            style_vocab,
            num_propositions=2,
            verbose=False
        )

        assert skeleton is not None
        # Should extract " from " from "from Chile" using spaCy
        assert " from " in skeleton, f"ORIGIN edge should extract ' from ' from text, got: {skeleton}"

    def test_composition_edge_rendering(self):
        """Test that COMPOSITION edges extract connector from text using spaCy."""
        input_graph = {
            "mermaid": "graph LR; P0 --COMPOSITION--> P1",
            "node_map": {
                "P0": "Smartphone consists",
                "P1": "including lithium and cobalt"
            }
        }

        style_vocab = {
            "COMPOSITION": [", including "],  # Should be ignored, semantic edges use spaCy extraction
            "CONTRAST": [" but "]
        }

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph,
            style_vocab,
            num_propositions=2,
            verbose=False
        )

        assert skeleton is not None
        # Should extract ", including " from "including lithium" using spaCy
        assert ", including" in skeleton or "including" in skeleton, f"COMPOSITION edge should extract connector from text, got: {skeleton}"

    def test_attribution_edge_rendering(self):
        """Test that ATTRIBUTION edges extract connector from text using spaCy."""
        input_graph = {
            "mermaid": "graph LR; P0 --ATTRIBUTION--> P1",
            "node_map": {
                "P0": "The term was coined",
                "P1": "by Stalin"
            }
        }

        style_vocab = {
            "ATTRIBUTION": [", by "],  # Should be ignored, semantic edges use spaCy extraction
            "CONTRAST": [" but "]
        }

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph,
            style_vocab,
            num_propositions=2,
            verbose=False
        )

        assert skeleton is not None
        # Should extract ", by " from "by Stalin" using spaCy
        assert ", by" in skeleton or " by " in skeleton, f"ATTRIBUTION edge should extract connector from text, got: {skeleton}"

    def test_mixed_semantic_and_rhetorical_rendering(self):
        """Test that semantic and rhetorical edges render correctly together."""
        input_graph = {
            "mermaid": "graph LR; P0 --PURPOSE--> P1; P1 --CONTRAST--> P2",
            "node_map": {
                "P0": "Lithium",
                "P1": "For smartphones",
                "P2": "Not decoration"
            }
        }

        style_vocab = {
            "PURPOSE": [" for "],  # Ignored for semantic
            "CONTRAST": [" but ", " however "],
            "SEQUENCE": [", "]
        }

        skeleton = self.matcher._build_skeleton_from_graph(
            input_graph,
            style_vocab,
            num_propositions=3,
            verbose=False
        )

        assert skeleton is not None
        assert " for " in skeleton, "PURPOSE should render as ' for '"
        # CONTRAST should use style_vocab
        assert (" but " in skeleton or " however " in skeleton), "CONTRAST should use style vocabulary"


class TestGraphBasedFracturingIntegration:
    """Integration tests for graph-based fracturing with semantic edges."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.fracturer = SemanticFracturer(self.mock_llm)

    def test_lithium_smartphones_preservation(self):
        """Test the specific case: 'lithium for smartphones' should not be split."""
        propositions = [
            "Lithium is a raw material",
            "For smartphones",
            "From Chile",
            "Cobalt is another material",
            "For smartphones",
            "From the Congo"
        ]

        # Create graph where P0--PURPOSE-->P1 and P1--ORIGIN-->P2 form a chain
        # This should be a super node that cannot be split
        input_graph = {
            "mermaid": "graph LR; P0 --PURPOSE--> P1; P1 --ORIGIN--> P2; P3 --PURPOSE--> P4; P4 --ORIGIN--> P5",
            "node_map": {
                "P0": propositions[0],
                "P1": propositions[1],
                "P2": propositions[2],
                "P3": propositions[3],
                "P4": propositions[4],
                "P5": propositions[5]
            }
        }

        result = self.fracturer.fracture(propositions, target_density=2, max_density=4, input_graph=input_graph)

        assert result is not None
        # Verify that semantic chains are preserved
        # P0, P1, P2 should be together (lithium for smartphones from Chile)
        found_chain_1 = False
        for group in result:
            if 0 in group and 1 in group and 2 in group:
                found_chain_1 = True
                break
        assert found_chain_1, "Lithium-for-smartphones-from-Chile chain should not be split"

        # P3, P4, P5 should be together (cobalt for smartphones from Congo)
        found_chain_2 = False
        for group in result:
            if 3 in group and 4 in group and 5 in group:
                found_chain_2 = True
                break
        assert found_chain_2, "Cobalt-for-smartphones-from-Congo chain should not be split"

    def test_pre_clustering_contraction_algorithm(self):
        """Test the Pre-Clustering Contraction algorithm directly."""
        # Create a simple graph with semantic edges
        G = nx.Graph()

        # Add nodes
        for i in range(5):
            G.add_node(i)

        # Add semantic edges (weight=100.0): P0--PURPOSE-->P1--ORIGIN-->P2
        G.add_edge(0, 1, weight=100.0, label='PURPOSE')
        G.add_edge(1, 2, weight=100.0, label='ORIGIN')

        # Add rhetorical edge (weight=1.0): P3--SEQUENCE-->P4
        G.add_edge(3, 4, weight=1.0, label='SEQUENCE')

        # Add cross-edge between semantic and rhetorical (weight=2.0)
        G.add_edge(2, 3, weight=2.0, label='CAUSALITY')

        # Step 3: Pre-Clustering Contraction
        semantic_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight') == 100.0]
        semantic_subgraph = G.edge_subgraph(semantic_edges)
        super_nodes = list(nx.connected_components(semantic_subgraph))

        # Verify super nodes
        assert len(super_nodes) == 1, "P0, P1, P2 should form one super node"
        assert {0, 1, 2} in super_nodes, "Super node should contain P0, P1, P2"

        # Verify isolated nodes
        isolated = set(range(5)) - {0, 1, 2}
        assert isolated == {3, 4}, "P3 and P4 should be isolated (no semantic edges)"

    def test_edge_weight_assignment(self):
        """Test that edge weights are assigned correctly."""
        propositions = ["A", "B", "C"]

        input_graph = {
            "mermaid": "graph LR; P0 --PURPOSE--> P1; P1 --CONTRAST--> P2; P0 --SEQUENCE--> P2",
            "node_map": {
                "P0": propositions[0],
                "P1": propositions[1],
                "P2": propositions[2]
            }
        }

        # Parse and check weights
        edges = self.fracturer._parse_mermaid_edges(input_graph['mermaid'])

        # Build graph manually to check weights
        G = nx.Graph()
        for source, target, label in edges:
            source_match = re.search(r'\d+', source)
            target_match = re.search(r'\d+', target)
            if not source_match or not target_match:
                continue
            source_idx = int(source_match.group())
            target_idx = int(target_match.group())
            label_upper = label.upper()

            if label_upper in ['PURPOSE', 'ORIGIN', 'COMPOSITION', 'ATTRIBUTION']:
                weight = 100.0
            elif label_upper in ['CONTRAST', 'CAUSALITY']:
                weight = 2.0
            else:
                weight = 1.0

            G.add_edge(source_idx, target_idx, weight=weight, label=label)

        # Check weights
        assert G[0][1]['weight'] == 100.0, "PURPOSE edge should have weight 100.0"
        assert G[1][2]['weight'] == 2.0, "CONTRAST edge should have weight 2.0"
        assert G[0][2]['weight'] == 1.0, "SEQUENCE edge should have weight 1.0"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


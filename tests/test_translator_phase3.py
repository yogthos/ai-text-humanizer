"""Comprehensive unit tests for Phase 3: Graph Renderer implementation."""

import sys
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.translator import StyleTranslator
from tests.mocks.mock_llm_provider import MockLLMProvider


class TestExtractPropositions:
    """Tests for _extract_propositions_from_text method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.translator = StyleTranslator(config_path="config.json")
        self.translator.llm_provider = self.mock_llm

    def test_extract_propositions_success(self):
        """Test successful proposition extraction."""
        # Mock LLM response
        propositions = [
            "The phone needs power",
            "Without power, it dies",
            "Batteries provide power"
        ]
        self.mock_llm.responses["Break the following text"] = json.dumps(propositions)

        text = "The phone needs power. Without power, it dies. Batteries provide power."
        result = self.translator._extract_propositions_from_text(text)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(p, str) for p in result)
        assert "The phone needs power" in result

    def test_extract_propositions_empty_text(self):
        """Test extraction with empty text."""
        result = self.translator._extract_propositions_from_text("")
        assert result == []

        result = self.translator._extract_propositions_from_text("   ")
        assert result == []

    def test_extract_propositions_with_markdown_code_blocks(self):
        """Test extraction when LLM returns markdown code blocks."""
        propositions = ["Fact 1", "Fact 2"]
        wrapped_response = f"```json\n{json.dumps(propositions)}\n```"
        # Mock the call method directly
        self.mock_llm.call = Mock(return_value=wrapped_response)

        result = self.translator._extract_propositions_from_text("Some text")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_extract_propositions_json_decode_error(self):
        """Test handling of JSON decode errors."""
        self.mock_llm.call = Mock(return_value="Not valid JSON {")

        result = self.translator._extract_propositions_from_text("Some text")
        assert result == []

    def test_extract_propositions_non_list_response(self):
        """Test handling when LLM returns non-list."""
        self.mock_llm.call = Mock(return_value=json.dumps({"error": "not a list"}))

        result = self.translator._extract_propositions_from_text("Some text")
        assert result == []

    def test_extract_propositions_filters_empty_strings(self):
        """Test that empty strings are filtered out."""
        propositions = ["Fact 1", "", "   ", "Fact 2"]
        self.mock_llm.call = Mock(return_value=json.dumps(propositions))

        result = self.translator._extract_propositions_from_text("Some text")
        assert len(result) == 2
        assert "Fact 1" in result
        assert "Fact 2" in result


class TestGenerateFromGraph:
    """Tests for _generate_from_graph method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.translator = StyleTranslator(config_path="config.json")
        self.translator.llm_provider = self.mock_llm

    def test_generate_from_graph_success(self):
        """Test successful graph-based generation."""
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --cause--> CLAIM',
            'node_mapping': {
                'ROOT': 'P0',
                'CLAIM': 'P1'
            },
            'style_metadata': {'skeleton': ''}
        }
        input_node_map = {
            'P0': 'The phone needs power',
            'P1': 'Without it, it dies'
        }

        self.mock_llm.responses["Blueprint Graph"] = "The phone needs power, and without it, it dies."

        result = self.translator._generate_from_graph(
            blueprint,
            input_node_map,
            "Mao",
            verbose=False
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_from_graph_with_comma_separated_nodes(self):
        """Test generation with multiple input nodes mapped to one style node."""
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {
                'ROOT': 'P0, P1',
                'CLAIM': 'P2'
            },
            'style_metadata': {'skeleton': ''}
        }
        input_node_map = {
            'P0': 'First fact',
            'P1': 'Second fact',
            'P2': 'Conclusion'
        }

        self.mock_llm.responses["Blueprint Graph"] = "Generated text."

        result = self.translator._generate_from_graph(
            blueprint,
            input_node_map,
            "Mao",
            verbose=False
        )

        assert isinstance(result, str)
        # Verify that the LLM was called with resolved content
        assert self.mock_llm.call_count > 0

    def test_generate_from_graph_with_unused_nodes(self):
        """Test generation when some style nodes are marked UNUSED."""
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM --> EXTRA',
            'node_mapping': {
                'ROOT': 'P0',
                'CLAIM': 'P1',
                'EXTRA': 'UNUSED'
            },
            'style_metadata': {'skeleton': ''}
        }
        input_node_map = {
            'P0': 'First fact',
            'P1': 'Second fact'
        }

        self.mock_llm.responses["Blueprint Graph"] = "Generated text."

        result = self.translator._generate_from_graph(
            blueprint,
            input_node_map,
            "Mao",
            verbose=False
        )

        assert isinstance(result, str)
        # UNUSED nodes should not appear in resolved content

    def test_generate_from_graph_missing_node_in_map(self):
        """Test handling when input node is missing from input_node_map."""
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {
                'ROOT': 'P0',
                'CLAIM': 'P999'  # Missing from input_node_map
            },
            'style_metadata': {'skeleton': ''}
        }
        input_node_map = {
            'P0': 'First fact'
        }

        self.mock_llm.responses["Blueprint Graph"] = "Generated text."

        result = self.translator._generate_from_graph(
            blueprint,
            input_node_map,
            "Mao",
            verbose=True
        )

        # Should still generate, just with missing content
        assert isinstance(result, str)

    def test_generate_from_graph_adds_punctuation(self):
        """Test that generated text gets proper punctuation."""
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {'ROOT': 'P0', 'CLAIM': 'P1'},
            'style_metadata': {'skeleton': ''}
        }
        input_node_map = {'P0': 'First', 'P1': 'Second'}

        self.mock_llm.responses["Blueprint Graph"] = "Text without punctuation"

        result = self.translator._generate_from_graph(
            blueprint,
            input_node_map,
            "Mao",
            verbose=False
        )

        assert result.endswith('.')

    def test_generate_from_graph_empty_response(self):
        """Test handling of empty LLM response."""
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {'ROOT': 'P0', 'CLAIM': 'P1'},
            'style_metadata': {'skeleton': ''}
        }
        input_node_map = {'P0': 'First', 'P1': 'Second'}

        # Mock to return empty string
        from unittest.mock import Mock
        mock_llm_empty = Mock()
        mock_llm_empty.call.return_value = ""
        self.translator.llm_provider = mock_llm_empty

        result = self.translator._generate_from_graph(
            blueprint,
            input_node_map,
            "Mao",
            verbose=False
        )

        assert result == ""


class TestTranslateParagraphPropositions:
    """Tests for translate_paragraph_propositions method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.translator = StyleTranslator(config_path="config.json")
        self.translator.llm_provider = self.mock_llm

        # Mock input_mapper and graph_matcher
        self.translator.input_mapper = MagicMock()
        self.translator.graph_matcher = MagicMock()

    def test_translate_paragraph_propositions_success(self):
        """Test successful paragraph translation."""
        # Mock proposition extraction
        propositions = ["Fact 1", "Fact 2", "Fact 3", "Fact 4"]
        # Use side_effect to return different responses for different calls
        self.mock_llm.call = Mock(side_effect=[
            json.dumps(propositions),  # First call: proposition extraction
            "Generated sentence from graph."  # Second call: graph generation
        ])

        # Mock input graph mapping
        input_graph = {
            'mermaid': 'graph LR; P0 --> P1',
            'description': 'A causal chain',
            'node_map': {
                'P0': 'Fact 1',
                'P1': 'Fact 2'
            },
            'node_count': 2
        }
        self.translator.input_mapper.map_propositions.return_value = input_graph

        # Mock style graph matching
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {'ROOT': 'P0', 'CLAIM': 'P1'},
            'style_metadata': {'skeleton': ''},
            'distance': 0.5
        }
        self.translator.graph_matcher.get_best_match.return_value = blueprint

        result = self.translator.translate_paragraph_propositions(
            "Some paragraph text",
            "Mao",
            document_context={'current_index': 0, 'total_paragraphs': 5},
            verbose=False
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        text, arch_id, compliance_score = result
        assert isinstance(text, str)
        assert len(text) > 0
        assert arch_id == 0
        assert compliance_score == 1.0

    def test_translate_paragraph_propositions_empty_input(self):
        """Test with empty paragraph."""
        result = self.translator.translate_paragraph_propositions(
            "",
            "Mao",
            verbose=False
        )

        # Empty input should return tuple with empty string
        assert isinstance(result, tuple)
        text, arch_id, compliance_score = result
        assert text == ""
        assert arch_id == 0
        assert compliance_score == 1.0

    def test_translate_paragraph_propositions_no_propositions(self):
        """Test when proposition extraction returns empty."""
        # Mock proposition extraction returning empty, then fallback generation
        self.mock_llm.call = Mock(side_effect=[
            json.dumps([]),  # First call: empty propositions
            "Fallback generated text."  # Second call: fallback generation
        ])

        result = self.translator.translate_paragraph_propositions(
            "Some text",
            "Mao",
            verbose=False
        )

        assert isinstance(result, tuple)
        text, _, _ = result
        assert len(text) > 0

    def test_translate_paragraph_propositions_chunking(self):
        """Test that propositions are chunked correctly."""
        # Create 10 propositions to test chunking (chunk_size = 4)
        propositions = [f"Fact {i}" for i in range(10)]
        # Mock: first call extracts propositions, then 3 calls for graph generation (3 chunks)
        self.mock_llm.call = Mock(side_effect=[
            json.dumps(propositions),  # Proposition extraction
            "Generated sentence 1.",  # Chunk 1 generation
            "Generated sentence 2.",  # Chunk 2 generation
            "Generated sentence 3."   # Chunk 3 generation
        ])

        # Mock input graph and blueprint
        input_graph = {
            'mermaid': 'graph LR; P0 --> P1',
            'description': 'A chain',
            'node_map': {'P0': 'Fact 0', 'P1': 'Fact 1'},
            'node_count': 2
        }
        blueprint = {
            'style_mermaid': 'graph LR; ROOT --> CLAIM',
            'node_mapping': {'ROOT': 'P0', 'CLAIM': 'P1'},
            'style_metadata': {'skeleton': ''},
            'distance': 0.5
        }

        self.translator.input_mapper.map_propositions.return_value = input_graph
        self.translator.graph_matcher.get_best_match.return_value = blueprint
        self.mock_llm.responses["Blueprint Graph"] = "Generated sentence."

        result = self.translator.translate_paragraph_propositions(
            "Some text with many facts",
            "Mao",
            verbose=False
        )

        # Should process multiple chunks (10 facts / 4 per chunk = 3 chunks)
        assert self.translator.input_mapper.map_propositions.call_count == 3

    def test_translate_paragraph_propositions_graph_mapping_fails(self):
        """Test fallback when graph mapping fails."""
        propositions = ["Fact 1", "Fact 2"]
        self.mock_llm.call = Mock(side_effect=[
            json.dumps(propositions),  # Proposition extraction
            "Fallback text."  # Fallback generation
        ])

        # Mock mapping failure
        self.translator.input_mapper.map_propositions.return_value = None

        # Mock fallback generation
        self.mock_llm.responses["Write a sentence combining"] = "Fallback text."

        result = self.translator.translate_paragraph_propositions(
            "Some text",
            "Mao",
            verbose=False
        )

        assert isinstance(result, tuple)
        text, _, _ = result
        assert len(text) > 0

    def test_translate_paragraph_propositions_no_style_match(self):
        """Test fallback when no style graph match found."""
        propositions = ["Fact 1", "Fact 2"]
        self.mock_llm.call = Mock(side_effect=[
            json.dumps(propositions),  # Proposition extraction
            "Fallback text."  # Fallback generation
        ])

        input_graph = {
            'mermaid': 'graph LR; P0 --> P1',
            'description': 'A chain',
            'node_map': {'P0': 'Fact 1', 'P1': 'Fact 2'},
            'node_count': 2
        }
        self.translator.input_mapper.map_propositions.return_value = input_graph

        # Mock no match
        self.translator.graph_matcher.get_best_match.return_value = None

        # Mock fallback
        self.mock_llm.responses["Write a sentence combining"] = "Fallback text."

        result = self.translator.translate_paragraph_propositions(
            "Some text",
            "Mao",
            verbose=False
        )

        assert isinstance(result, tuple)
        text, _, _ = result
        assert len(text) > 0

    def test_translate_paragraph_propositions_generation_fails(self):
        """Test fallback when graph generation returns empty."""
        propositions = ["Fact 1", "Fact 2"]
        self.mock_llm.call = Mock(side_effect=[
            json.dumps(propositions),  # Proposition extraction
            "",  # Empty graph generation
            "Fallback text."  # Fallback generation
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
            'style_metadata': {'skeleton': ''},
            'distance': 0.5
        }

        self.translator.input_mapper.map_propositions.return_value = input_graph
        self.translator.graph_matcher.get_best_match.return_value = blueprint

        # Mock empty generation
        self.mock_llm.responses["Blueprint Graph"] = ""

        # Mock fallback
        self.mock_llm.responses["Write a sentence combining"] = "Fallback text."

        result = self.translator.translate_paragraph_propositions(
            "Some text",
            "Mao",
            verbose=False
        )

        assert isinstance(result, tuple)
        text, _, _ = result
        assert len(text) > 0

    def test_translate_paragraph_propositions_exception_handling(self):
        """Test exception handling in chunk processing."""
        propositions = ["Fact 1", "Fact 2"]
        self.mock_llm.call = Mock(side_effect=[
            json.dumps(propositions),  # Proposition extraction
            "Fallback text."  # Fallback generation
        ])

        # Mock exception in mapping
        self.translator.input_mapper.map_propositions.side_effect = Exception("Test error")

        # Mock fallback
        self.mock_llm.responses["Write a sentence combining"] = "Fallback text."

        result = self.translator.translate_paragraph_propositions(
            "Some text",
            "Mao",
            verbose=False
        )

        assert isinstance(result, tuple)
        text, _, _ = result
        assert len(text) > 0

    def test_translate_paragraph_propositions_document_context_passed(self):
        """Test that document context is passed to graph matcher."""
        propositions = ["Fact 1", "Fact 2"]
        self.mock_llm.call = Mock(side_effect=[
            json.dumps(propositions),  # Proposition extraction
            "Generated text."  # Graph generation
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
            'style_metadata': {'skeleton': ''},
            'distance': 0.5
        }

        self.translator.input_mapper.map_propositions.return_value = input_graph
        self.translator.graph_matcher.get_best_match.return_value = blueprint
        self.mock_llm.responses["Blueprint Graph"] = "Generated text."

        document_context = {'current_index': 2, 'total_paragraphs': 5}

        self.translator.translate_paragraph_propositions(
            "Some text",
            "Mao",
            document_context=document_context,
            verbose=False
        )

        # Verify document_context was passed
        self.translator.graph_matcher.get_best_match.assert_called()
        call_args = self.translator.graph_matcher.get_best_match.call_args
        assert call_args[0][1] == document_context


class TestFallbackMethods:
    """Tests for fallback generation methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.translator = StyleTranslator(config_path="config.json")
        self.translator.llm_provider = self.mock_llm

    def test_fallback_simple_generation_chunk(self):
        """Test fallback generation for a chunk."""
        chunk = ["Fact 1", "Fact 2"]
        self.mock_llm.responses["Write a sentence combining"] = "Combined sentence."

        result = self.translator._fallback_simple_generation_chunk(
            chunk,
            "Mao",
            verbose=False
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert result.endswith('.')

    def test_fallback_simple_generation_chunk_empty(self):
        """Test fallback with empty chunk."""
        result = self.translator._fallback_simple_generation_chunk(
            [],
            "Mao",
            verbose=False
        )

        assert result == ""

    def test_fallback_simple_generation_paragraph(self):
        """Test fallback generation for entire paragraph."""
        paragraph = "Some paragraph text."
        self.mock_llm.call = Mock(return_value="Generated paragraph.")

        result = self.translator._fallback_simple_generation(
            paragraph,
            "Mao",
            verbose=False
        )

        # Now returns tuple
        assert isinstance(result, tuple)
        text, arch_id, compliance_score = result
        assert isinstance(text, str)
        assert len(text) > 0
        assert text.endswith('.')
        assert arch_id == 0
        assert compliance_score == 1.0

    def test_fallback_simple_generation_exception(self):
        """Test fallback handles exceptions gracefully."""
        paragraph = "Some text"
        # Create a mock that raises exception
        from unittest.mock import Mock
        mock_llm_with_error = Mock()
        mock_llm_with_error.call.side_effect = Exception("LLM error")
        self.translator.llm_provider = mock_llm_with_error

        result = self.translator._fallback_simple_generation(
            paragraph,
            "Mao",
            verbose=True
        )

        # Now returns tuple
        assert isinstance(result, tuple)
        text, arch_id, compliance_score = result
        assert text == ""
        assert arch_id == 0
        assert compliance_score == 1.0


class TestIntegration:
    """Integration tests for the full graph pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.translator = StyleTranslator(config_path="config.json")
        self.translator.llm_provider = self.mock_llm

        # Use real input_mapper and graph_matcher but with mocked LLM
        from src.atlas.input_mapper import InputLogicMapper
        from src.generator.graph_matcher import TopologicalMatcher

        self.translator.input_mapper = InputLogicMapper(self.mock_llm)
        self.translator.graph_matcher = TopologicalMatcher(config_path="config.json")
        self.translator.graph_matcher.llm_provider = self.mock_llm

    def test_full_pipeline_success(self):
        """Test the complete pipeline from text to generated paragraph."""
        # Mock proposition extraction
        propositions = ["The phone needs power", "Without power it dies"]
        # Mock multiple LLM calls: extraction, input mapping, node mapping, generation
        self.mock_llm.call = Mock(side_effect=[
            json.dumps(propositions),  # Proposition extraction
            json.dumps({  # Input graph mapping
                "mermaid": "graph LR; P0 --cause--> P1",
                "description": "A causal chain leading to consequence",
                "node_map": {
                    "P0": "The phone needs power",
                    "P1": "Without power it dies"
                },
                "node_count": 2
            }),
            json.dumps({"ROOT": "P0", "CLAIM": "P1"}),  # Node mapping
            "The phone needs power, and without it, it dies."  # Graph generation
        ])

        # Mock input graph mapping
        input_graph_response = {
            "mermaid": "graph LR; P0 --cause--> P1",
            "description": "A causal chain leading to consequence",
            "node_map": {
                "P0": "The phone needs power",
                "P1": "Without power it dies"
            },
            "node_count": 2
        }
        self.mock_llm.responses["Propositions:"] = json.dumps(input_graph_response)

        # Mock style graph matching (would need ChromaDB, so we'll mock the collection)
        with patch.object(self.translator.graph_matcher, 'collection') as mock_collection:
            # Mock ChromaDB query result
            mock_collection.query.return_value = {
                'ids': [['test_id']],
                'distances': [[0.3]],
                'metadatas': [[{
                    'mermaid': 'graph LR; ROOT --cause--> CLAIM',
                    'node_count': 2,
                    'edge_types': 'cause',
                    'original_text': 'Original style text'
                }]],
                'documents': [['A causal chain']]
            }

            # Mock node mapping
            node_mapping_response = {
                "ROOT": "P0",
                "CLAIM": "P1"
            }
            self.mock_llm.responses["Input Graph:"] = json.dumps(node_mapping_response)

            # Mock graph generation
            self.mock_llm.responses["Blueprint Graph"] = "The phone needs power, and without it, it dies."

            result = self.translator.translate_paragraph_propositions(
                "The phone needs power. Without power it dies.",
                "Mao",
                document_context={'current_index': 0, 'total_paragraphs': 3},
                verbose=False
            )

            assert isinstance(result, tuple)
            text, arch_id, compliance_score = result
            assert isinstance(text, str)
            assert len(text) > 0


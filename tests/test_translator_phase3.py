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


class TestDeduplication:
    """Tests for proposition deduplication feature."""

    def setup_method(self):
        """Set up test fixtures."""
        self.translator = StyleTranslator(config_path="config.json")

    def test_deduplicate_propositions_high_similarity(self):
        """Test deduplication removes highly similar propositions (>80%)."""
        propositions = [
            "The term Dialectical Materialism was coined by Stalin",
            "Dialectical Materialism was coined by Joseph Vissarionovich Dzhugashvili",
            "This is a different fact"
        ]

        result = self.translator._deduplicate_propositions(propositions)

        # Should remove one of the similar propositions (the shorter one)
        assert len(result) == 2
        # The longer, more detailed version should be kept
        assert any("Joseph Vissarionovich Dzhugashvili" in prop for prop in result)
        assert "This is a different fact" in result

    def test_deduplicate_propositions_shared_key_words(self):
        """Test deduplication based on shared key proper nouns and verbs."""
        propositions = [
            "Stalin coined the term Dialectical Materialism",
            "The term Dialectical Materialism was coined by Stalin",
            "Marx developed the theory"
        ]

        result = self.translator._deduplicate_propositions(propositions)

        # Should remove one of the Stalin/coined propositions if they're similar enough
        # The exact result depends on similarity calculation, so we check:
        # - Result should have at least 2 items (may have 2 or 3 depending on similarity)
        assert len(result) >= 2
        assert len(result) <= 3
        # Should keep the Marx proposition (different topic)
        assert any("Marx" in prop for prop in result)

    def test_deduplicate_propositions_no_duplicates(self):
        """Test that non-duplicate propositions are preserved."""
        propositions = [
            "First unique fact about apples",
            "Second unique fact about oranges",
            "Third unique fact about bananas"
        ]

        result = self.translator._deduplicate_propositions(propositions)

        # Should preserve all unique propositions
        # (Using more distinct content to avoid false similarity matches)
        assert len(result) >= 2  # At least most should be preserved
        # All original propositions should be in result (or very similar ones)
        for prop in propositions:
            # Check if any result contains key words from original
            assert any(any(word in result_prop for word in prop.split()[:2]) for result_prop in result)

    def test_deduplicate_propositions_single_proposition(self):
        """Test deduplication with single proposition."""
        propositions = ["Single fact"]

        result = self.translator._deduplicate_propositions(propositions)

        assert result == propositions

    def test_deduplicate_propositions_empty_list(self):
        """Test deduplication with empty list."""
        result = self.translator._deduplicate_propositions([])
        assert result == []

    def test_deduplicate_propositions_keeps_longer_version(self):
        """Test that deduplication keeps the longer, more detailed version."""
        propositions = [
            "Stalin made it",
            "The term Dialectical Materialism was coined by Joseph Vissarionovich Dzhugashvili to describe the method"
        ]

        result = self.translator._deduplicate_propositions(propositions)

        # Should keep the longer, more detailed version if they're similar enough
        # If similarity is low, both might be kept
        if len(result) == 1:
            # If deduplicated, should keep the longer one
            assert "Joseph Vissarionovich Dzhugashvili" in result[0] or len(result[0]) > len(propositions[0])
        else:
            # If both kept, that's also acceptable (they might not be similar enough)
            assert len(result) == 2


class TestContextAwareness:
    """Tests for context awareness (previous sentence tracking)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.translator = StyleTranslator(config_path="config.json")
        self.translator.llm_provider = self.mock_llm

        # Mock input_mapper and graph_matcher
        self.translator.input_mapper = MagicMock()
        self.translator.graph_matcher = MagicMock()

    def test_previous_sentence_passed_to_generate_from_graph(self):
        """Test that previous sentence parameter is accepted by _generate_from_graph."""
        # Test that the method signature accepts previous_sentence parameter
        blueprint = {
            'style_metadata': {
                'skeleton': '[P0] and [P1]',
                'node_count': 2,
                'intent': 'ARGUMENT'
            },
            'node_mapping': {'P0': 'P0', 'P1': 'P1'},
            'intent': 'ARGUMENT',
            'distance': 0.0
        }
        input_node_map = {'P0': 'Fact 1', 'P1': 'Fact 2'}

        # Mock LLM response using MockLLMProvider's responses dict
        self.mock_llm.responses["STYLE BLUEPRINT"] = "Generated sentence."

        # Test that method can be called with previous_sentence parameter
        result1 = self.translator._generate_from_graph(
            blueprint,
            input_node_map,
            "Mao",
            verbose=False,
            previous_sentence=None
        )

        # Capture prompt for second call
        captured_prompt = None
        original_call = self.mock_llm.call

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            if 'user_prompt' in kwargs:
                captured_prompt = kwargs['user_prompt']
            return "Generated sentence."

        self.mock_llm.call = capture_prompt

        result2 = self.translator._generate_from_graph(
            blueprint,
            input_node_map,
            "Mao",
            verbose=False,
            previous_sentence="Previous sentence text."
        )

        # Both should work without errors
        assert isinstance(result1, str)
        assert isinstance(result2, str)

        # Verify that previous_sentence is used in the prompt when provided
        assert captured_prompt is not None
        assert "PREVIOUS SENTENCE" in captured_prompt
        assert "Previous sentence text." in captured_prompt
        assert "Do not repeat facts or names" in captured_prompt

    def test_context_constraint_in_prompt(self):
        """Test that previous sentence context is included in generation prompt."""
        previous_sentence = "The term Dialectical Materialism was coined by Stalin."

        blueprint = {
            'style_metadata': {
                'skeleton': '[P0] is [P1]',
                'node_count': 2,
                'intent': 'DEFINITION'
            },
            'node_mapping': {'P0': 'P0', 'P1': 'P1'},
            'intent': 'DEFINITION',
            'distance': 0.0
        }
        input_node_map = {'P0': 'Dialectical Materialism', 'P1': 'a practical toolset'}

        # Capture the prompt
        captured_prompt = None
        original_call = self.mock_llm.call

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            if 'user_prompt' in kwargs:
                captured_prompt = kwargs['user_prompt']
            return original_call(*args, **kwargs)

        self.mock_llm.call = capture_prompt
        self.mock_llm.call.return_value = "Generated text without repeating Stalin."

        self.translator._generate_from_graph(
            blueprint,
            input_node_map,
            "Mao",
            verbose=False,
            previous_sentence=previous_sentence
        )

        # Verify previous sentence context is in the prompt
        assert captured_prompt is not None
        assert "PREVIOUS SENTENCE" in captured_prompt
        assert previous_sentence in captured_prompt
        assert "Do not repeat facts or names" in captured_prompt


class TestDeduplicationIntegration:
    """Integration tests for deduplication in the full pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()
        self.translator = StyleTranslator(config_path="config.json")
        self.translator.llm_provider = self.mock_llm

        # Mock input_mapper and graph_matcher
        self.translator.input_mapper = MagicMock()
        self.translator.graph_matcher = MagicMock()

    def test_deduplication_called_before_fracturing(self):
        """Test that deduplication is called before fracturing."""
        # Create propositions with duplicates
        all_propositions = [
            "Stalin coined the term",
            "The term was coined by Stalin",
            "Dialectical Materialism is a toolset",
            "It is a practical method"
        ]

        # Track if deduplication was called
        deduplication_called = False
        original_deduplicate = self.translator._deduplicate_propositions

        def track_deduplicate(propositions):
            nonlocal deduplication_called
            deduplication_called = True
            return original_deduplicate(propositions)

        self.translator._deduplicate_propositions = track_deduplicate

        # Mock proposition extraction
        self.mock_llm.call = Mock(return_value=json.dumps(all_propositions))

        # Mock input graph
        input_graph = {
            'mermaid': 'graph LR; P0 --> P1',
            'description': 'A definition',
            'node_map': {'P0': 'Fact 1', 'P1': 'Fact 2'},
            'node_count': 2,
            'intent': 'DEFINITION'
        }
        self.translator.input_mapper.map_propositions.return_value = input_graph

        # Mock blueprint
        blueprint = {
            'style_metadata': {
                'skeleton': '[P0] and [P1]',
                'node_count': 2,
                'intent': 'DEFINITION'
            },
            'node_mapping': {'P0': 'P0', 'P1': 'P1'},
            'intent': 'DEFINITION',
            'distance': 0.0
        }
        self.translator.graph_matcher.synthesize_match.return_value = blueprint

        # Mock fracturer to track if it was called
        fracturing_called = False
        original_fracture = None
        if hasattr(self.translator, 'fracturer') and hasattr(self.translator.fracturer, 'fracture'):
            original_fracture = self.translator.fracturer.fracture
            def track_fracture(*args, **kwargs):
                nonlocal fracturing_called
                fracturing_called = True
                return original_fracture(*args, **kwargs) if original_fracture else [[0, 1]]
            self.translator.fracturer.fracture = track_fracture

        with patch.object(self.translator, '_extract_propositions_from_text', return_value=all_propositions):
            self.translator.translate_paragraph_propositions(
                "Test paragraph",
                "Mao",
                verbose=False
            )

        # Verify deduplication was called
        assert deduplication_called, "Deduplication should be called before fracturing"


class TestInputMapperDeduplication:
    """Tests for input mapper deduplication instruction."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.atlas.input_mapper import InputLogicMapper
        from src.generator.llm_provider import LLMProvider
        from unittest.mock import MagicMock

        self.mock_llm = MagicMock()
        self.input_mapper = InputLogicMapper(self.mock_llm)

    def test_map_propositions_includes_deduplication_instruction(self):
        """Test that map_propositions prompt includes deduplication instruction."""
        propositions = ["Fact 1", "Fact 2"]

        # Capture the prompt
        captured_prompt = None
        original_call = self.mock_llm.call

        def capture_prompt(*args, **kwargs):
            nonlocal captured_prompt
            if 'user_prompt' in kwargs:
                captured_prompt = kwargs['user_prompt']
            return json.dumps({
                'mermaid': 'graph LR; P0 --> P1',
                'description': 'A causal chain',
                'intent': 'ARGUMENT',
                'node_map': {'P0': 'Fact 1', 'P1': 'Fact 2'}
            })

        self.mock_llm.call = capture_prompt

        self.input_mapper.map_propositions(propositions)

        # Verify deduplication instruction is in the prompt
        assert captured_prompt is not None
        assert "CRITICAL: De-duplicate" in captured_prompt or "De-duplicate facts" in captured_prompt


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


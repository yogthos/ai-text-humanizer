
import pytest
from unittest.mock import MagicMock, patch
from src.generator.graph_matcher import TopologicalMatcher
from src.planning.sentence_plan import SentenceNode
from src.generator.translator import StyleTranslator

class TestBrokenSentenceRegression:
    """Tests for regression: broken sentences (fragments) and fallback logic."""

    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    @pytest.fixture
    def matcher(self, mock_llm):
        # Create matcher with mocked LLM
        matcher = TopologicalMatcher(config_path="config.json", llm_provider=mock_llm)
        # Mock ChromaDB collection to avoid actual DB calls
        matcher.collection = MagicMock()
        matcher.collection.query.return_value = {'ids': [], 'distances': [], 'metadatas': []}
        return matcher

    def test_synthesize_rejects_broken_endings(self, matcher, mock_llm):
        """Test that synthesize_sentence_node rejects sentences ending in connectors."""
        
        # Setup a node
        node = SentenceNode(
            id="S1",
            propositions=["The concept is complex."],
            role="Definition",
            transition_type="None",
            target_length=15
        )

        # Mock LLM to return a broken sentence ending with "to"
        # The code strips markdown blocks, so we can return plain text or markdown
        broken_output = "The concept is complex compared to" 
        mock_llm.call.return_value = broken_output

        # Execute
        result = matcher.synthesize_sentence_node(
            node, 
            previous_text=None, 
            style_profile={}, 
            style_tracker=None, 
            default_perspective="Third Person", 
            author_name="TestAuthor"
        )

        # Assert: Should be empty string (rejection)
        assert result == "", "Matcher should reject sentences ending with 'to'"

        # Test another broken ending
        mock_llm.call.return_value = "It depends on the usage of"
        result = matcher.synthesize_sentence_node(
            node, previous_text=None, style_profile={}, style_tracker=None, 
            default_perspective="Third Person", author_name="TestAuthor"
        )
        assert result == "", "Matcher should reject sentences ending with 'of'"

    def test_synthesize_enforces_punctuation(self, matcher, mock_llm):
        """Test that synthesize_sentence_node adds punctuation if missing."""
        
        node = SentenceNode(
            id="S1",
            propositions=["The concept is complex."],
            role="Definition",
            transition_type="None",
            target_length=15
        )

        # Mock LLM to return a valid sentence BUT missing period
        valid_but_no_punct = "The concept is inherently complex and multifaceted"
        mock_llm.call.return_value = valid_but_no_punct

        # Execute
        result = matcher.synthesize_sentence_node(
            node, previous_text=None, style_profile={}, style_tracker=None, 
            default_perspective="Third Person", author_name="TestAuthor"
        )

        # Assert: Should have period added
        assert result == valid_but_no_punct + ".", "Matcher should add missing terminal punctuation"

    def test_synthesize_rejects_short_garbage(self, matcher, mock_llm):
        """Test that synthesize_sentence_node rejects extremely short output."""
        
        node = SentenceNode(
            id="S1",
            propositions=["The concept is complex."],
            role="Definition",
            transition_type="None",
            target_length=15
        )

        # Mock LLM to return garbage
        short_garbage = "No."
        mock_llm.call.return_value = short_garbage

        # Execute
        result = matcher.synthesize_sentence_node(
            node, previous_text=None, style_profile={}, style_tracker=None, 
            default_perspective="Third Person", author_name="TestAuthor"
        )

        # Assert: Should be empty string (rejection due to length < 10)
        assert result == "", "Matcher should reject extremely short sentences"

@patch("src.generator.translator.StyleTranslator._extract_propositions_from_text")
@patch("src.generator.translator.StyleTranslator._deduplicate_propositions")
@patch("src.generator.translator.InputLogicMapper")
@patch("src.generator.translator.GraphPlanner")
@patch("src.generator.translator.TopologicalMatcher")
class TestTranslatorFallback:
    """Integration tests for Translator fallback logic."""

    def test_fallback_when_generation_fails(
        self, 
        MockMatcher, 
        MockPlanner, 
        MockMapper, 
        MockDedup, 
        MockExtract
    ):
        """Test that translator falls back to propositions when synthesis returns empty."""
        
        # Setup Translator
        translator = StyleTranslator(config_path="config.json")
        
        # Setup Mocks
        # 1. Extraction returns valid props
        propositions = ["Prop 1 is true", "Prop 2 is false"]
        MockExtract.return_value = propositions
        MockDedup.return_value = propositions
        
        # 2. Mapper returns None (to trigger fallback? No, we want to test loop fallback)
        # Let's say Mapper works, Planner works
        mock_mapper_instance = MockMapper.return_value
        mock_mapper_instance.map_propositions.return_value = {"mermaid": "graph LR", "intent": "ARGUMENT"}
        
        mock_planner_instance = MockPlanner.return_value
        
        # Create a mock plan with 2 nodes
        node1 = SentenceNode(id="S1", propositions=[propositions[0]], role="Body", transition_type="None", target_length=10)
        node2 = SentenceNode(id="S2", propositions=[propositions[1]], role="Body", transition_type="None", target_length=10)
        
        mock_plan = MagicMock()
        mock_plan.nodes = [node1, node2]
        mock_planner_instance.create_plan.return_value = mock_plan
        
        # 3. Matcher (GraphMatcher) - THIS IS KEY
        # We assume translator.graph_matcher is replaced by MockMatcher instance
        # Actually translator creates its own matcher in __init__, so we need to mock THAT instance
        # But we are patching the class, so we need to ensure the instance used is the mock
        
        # NOTE: StyleTranslator.__init__ instantiates TopologicalMatcher. 
        # Since we patch the class 'src.generator.translator.TopologicalMatcher', 
        # the translator.graph_matcher will be an instance of our MockMatcher.
        
        mock_matcher_instance = translator.graph_matcher
        
        # Configure synthesize_sentence_node to FAIL (return empty) for both nodes
        # This simulates the "broken output" rejection we tested above
        mock_matcher_instance.synthesize_sentence_node.return_value = "" 
        
        # Execute
        result_text, _, _ = translator.translate_paragraph_propositions(
            paragraph="Input paragraph text",
            author_name="TestAuthor",
            verbose=True
        )
        
        # Assert
        # Since synthesis failed (returned empty string), the loop should have fallen back 
        # to using the propositions text: "Prop 1 is true" and "Prop 2 is false"
        # The result should contain these strings
        
        assert propositions[0] in result_text, "Should fall back to proposition 1 text"
        assert propositions[1] in result_text, "Should fall back to proposition 2 text"
        
        # Ensure it's not just empty
        assert len(result_text) > 0

    def test_fallback_when_validation_fails(
        self, 
        MockMatcher, 
        MockPlanner, 
        MockMapper, 
        MockDedup, 
        MockExtract
    ):
        """Test fallback when synthesis succeeds but semantic score is low (repair fails)."""
        
        # Setup Translator
        translator = StyleTranslator(config_path="config.json")
        
        # Setup Mocks
        propositions = ["Prop 1 is critical"]
        MockExtract.return_value = propositions
        MockDedup.return_value = propositions
        
        # Plan with 1 node
        node1 = SentenceNode(id="S1", propositions=[propositions[0]], role="Body", transition_type="None", target_length=10)
        mock_plan = MagicMock()
        mock_plan.nodes = [node1]
        
        mock_planner_instance = MockPlanner.return_value
        mock_planner_instance.create_plan.return_value = mock_plan
        
        # Matcher generates a "hallucination" (something unrelated)
        mock_matcher_instance = translator.graph_matcher
        mock_matcher_instance.synthesize_sentence_node.return_value = "The sky is blue." 
        
        # Mock _evaluate_variant to return LOW score (fail)
        # We need to patch the internal method on the translator instance or mock the critic
        with patch.object(translator, '_evaluate_variant') as mock_eval:
            mock_eval.return_value = {'score': 0.1} # Low score
            
            # Mock _repair_sentence_semantic to also fail (return same bad text)
            with patch.object(translator, '_repair_sentence_semantic') as mock_repair:
                mock_repair.return_value = "The sky is blue." # Repair failed
                
                # Execute
                result_text, _, _ = translator.translate_paragraph_propositions(
                    paragraph="Input text",
                    author_name="TestAuthor",
                    verbose=True
                )
                
                # Assert
                # Should fall back to proposition because "The sky is blue" scored 0.1
                assert propositions[0] in result_text, "Should fall back to proposition when semantic score is low"
                assert "The sky is blue" not in result_text, "Should discard the hallucination"


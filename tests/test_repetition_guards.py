
import unittest
from unittest.mock import MagicMock
from src.planning.graph_planner import GraphPlanner
from src.generator.graph_matcher import TopologicalMatcher
from src.planning.sentence_plan import SentenceNode

class TestRepetitionGuards(unittest.TestCase):
    
    def test_parallel_clustering(self):
        """Test that GraphPlanner clusters parallel propositions."""
        planner = GraphPlanner()
        
        # Scenario: P0 and P1 both CONTRAST with P2
        propositions = ["P0", "P1", "P2"]
        input_graph = {
            "mermaid": "graph LR; P0 --CONTRAST--> P2; P1 --CONTRAST--> P2",
            "node_map": {0: "P0", 1: "P1", 2: "P2"}
        }
        
        # Force parameters that would normally cause splitting
        planner.avg_words_per_sentence = 5 
        planner.burstiness = 0.0 
        
        plan = planner.create_plan(propositions, input_graph)
        
        # Check if P0 and P1 are clustered
        p0_node = next((n for n in plan.nodes if 0 in n.global_indices), None)
        p1_node = next((n for n in plan.nodes if 1 in n.global_indices), None)
        
        self.assertIsNotNone(p0_node)
        self.assertIsNotNone(p1_node)
        self.assertEqual(p0_node, p1_node, "P0 and P1 should be in the same cluster due to parallel structure")

    def test_suffix_repetition_guard(self):
        """Test that synthesize_sentence_node rejects suffix repetition."""
        mock_llm = MagicMock()
        matcher = TopologicalMatcher(config_path="config.json", llm_provider=mock_llm)
        matcher.collection = MagicMock()
        matcher.collection.query.return_value = {'ids': [], 'distances': [], 'metadatas': []}
        
        previous_text = "The concept creates a contradiction between knowledge and material reality."
        
        # Mock LLM generating the exact same suffix
        # Suffix (last 5 words): "between knowledge and material reality."
        repetitive_output = "This creates a contradiction between knowledge and material reality."
        mock_llm.call.return_value = repetitive_output
        
        node = SentenceNode(
            id="S1",
            propositions=["Some content"],
            role="Body",
            transition_type="None",
            target_length=15
        )
        
        result = matcher.synthesize_sentence_node(
            node, 
            previous_text=previous_text,
            style_profile={}, 
            style_tracker=None, 
            default_perspective="Third Person", 
            author_name="TestAuthor",
            verbose=True
        )
        
        # Should be rejected (empty string)
        self.assertEqual(result, "", "Should reject repetitive suffix")
        
        # Mock LLM generating valid output (different suffix)
        valid_output = "This creates a contradiction that resolves in synthesis."
        mock_llm.call.return_value = valid_output
        
        result = matcher.synthesize_sentence_node(
            node, 
            previous_text=previous_text,
            style_profile={}, 
            style_tracker=None, 
            default_perspective="Third Person", 
            author_name="TestAuthor",
            verbose=True
        )
        
        self.assertEqual(result, valid_output, "Should accept different suffix")

if __name__ == "__main__":
    unittest.main()

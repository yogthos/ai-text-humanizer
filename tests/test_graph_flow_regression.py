
import unittest
from src.planning.graph_planner import GraphPlanner

class TestGraphFlowRegression(unittest.TestCase):
    
    def test_stable_lexicographical_sort(self):
        """
        Ensures that independent nodes preserve their original index order.
        This prevents disjointed paragraphs where unrelated facts are shuffled.
        """
        planner = GraphPlanner()
        # 5 propositions with NO logical dependencies
        propositions = ["P0", "P1", "P2", "P3", "P4"]
        edges = [] # No edges
        
        sorted_indices = planner._get_topological_sort(edges, len(propositions))
        
        # Should be exactly [0, 1, 2, 3, 4]
        self.assertEqual(sorted_indices, [0, 1, 2, 3, 4], 
                         "Independent nodes should preserve original narrative order")

    def test_topological_constraint_preservation(self):
        """
        Ensures that logical dependencies (arrows) correctly force order,
        while independent nodes still respect lexicographical order.
        """
        planner = GraphPlanner()
        # Input order: 0, 1, 2, 3
        # Logic: P3 must come before P1 (e.g. Attribution -> Concept)
        propositions = ["P0", "P1", "P2", "P3"]
        edges = [(3, 1, "DEPENDENCY")]
        
        sorted_indices = planner._get_topological_sort(edges, len(propositions))
        
        # Expected: 
        # 0 comes first (independent, lowest index)
        # 2 comes second (independent, lowest remaining index)
        # 3 comes before 1 (forced by edge)
        # Result: [0, 2, 3, 1]
        
        self.assertEqual(sorted_indices, [0, 2, 3, 1], 
                         "Topological sort should respect edges but prefer index order for ties")

    def test_clustering_narrative_flow(self):
        """
        Ensures that clustering respects the stable sort order and doesn't 
        accidentally pull future independent facts into an earlier sentence 
        if they aren't parallel.
        """
        planner = GraphPlanner()
        propositions = ["P0", "P1", "P2", "P3"]
        # P0 -> P1 (Sequence)
        # P2 -> P3 (Sequence)
        # No connection between (0,1) and (2,3)
        input_graph = {
            "mermaid": "graph LR; P0 --> P1; P2 --> P3",
            "node_map": {"P0": "P0", "P1": "P1", "P2": "P2", "P3": "P3"}
        }
        
        planner.avg_words_per_sentence = 20 # Large enough for 2 props
        planner.burstiness = 0.0 # Force even 2-2 split
        
        plan = planner.create_plan(propositions, input_graph)
        
        # Expected Nodes:
        # S0: [0, 1]
        # S1: [2, 3]
        
        self.assertEqual(len(plan.nodes), 2)
        self.assertEqual(plan.nodes[0].global_indices, [0, 1])
        self.assertEqual(plan.nodes[1].global_indices, [2, 3])

if __name__ == "__main__":
    unittest.main()

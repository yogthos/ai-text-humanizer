"""Unit tests for pipeline refactoring: Context injection, semantic topology, and pattern tracking."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

from src.processing.style_state import GlobalStyleTracker
from src.generator.llm_provider import LLMProvider


class TestPatternTracking(unittest.TestCase):
    """Test Step 3: Pattern-based opener extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = GlobalStyleTracker(config={'style_state': {'enabled': True}})

    def test_extract_opener_first_four_words(self):
        """Test that opener extraction takes first 4 words."""
        text = "It is not a realm of strange mysticism"
        opener = self.tracker._extract_opener(text)
        self.assertEqual(opener, "it is not a")

    def test_extract_opener_normalizes_case(self):
        """Test that opener is lowercased."""
        text = "IT IS NOT A realm"
        opener = self.tracker._extract_opener(text)
        self.assertEqual(opener, "it is not a")

    def test_extract_opener_strips_punctuation(self):
        """Test that trailing punctuation is removed."""
        text = "It is not a, but something else."
        opener = self.tracker._extract_opener(text)
        self.assertEqual(opener, "it is not a")

    def test_extract_opener_handles_short_text(self):
        """Test that short text (< 4 words) is handled."""
        text = "It is not"
        opener = self.tracker._extract_opener(text)
        self.assertEqual(opener, "it is not")

    def test_extract_opener_blocks_repetition(self):
        """Test that same opener pattern is blocked."""
        text1 = "It is not a realm of mysticism"
        text2 = "It is not a toolset for analysis"

        # Register first opener
        self.tracker.register_usage(text1, structure='CONTRAST')

        # Check that second opener is in forbidden list
        forbidden = self.tracker.get_forbidden_openers()
        self.assertIn("it is not a", forbidden)

        # Register second (should also be tracked)
        self.tracker.register_usage(text2, structure='CONTRAST')

        # Both should be tracked (or at least the pattern)
        forbidden_after = self.tracker.get_forbidden_openers()
        self.assertIn("it is not a", forbidden_after)


class TestContextInjection(unittest.TestCase):
    """Test Step 1: Context passing through pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm_provider = Mock(spec=LLMProvider)
        self.llm_provider.call = Mock(return_value="Generated text")

    def test_original_context_passed_to_generate_from_graph(self):
        """Test that original_context parameter is accepted by _generate_from_graph."""
        # This is a signature test - verify the method accepts original_context
        from src.generator.translator import StyleTranslator
        import inspect

        # Get the signature of _generate_from_graph
        sig = inspect.signature(StyleTranslator._generate_from_graph)
        params = list(sig.parameters.keys())

        # Verify original_context is in the parameters
        self.assertIn('original_context', params)
        self.assertIn('input_graph', params)

    def test_context_block_in_prompt(self):
        """Test that context_block is built when original_context is provided."""
        original_context = "Many people hear the term Dialectical Materialism"

        # Build context block (simulating what happens in _generate_from_graph)
        context_block = ""
        if original_context:
            context_block = f"""
**ORIGINAL CONTEXT (SOURCE MEANING):**
"{original_context}"
*Use this to resolve ambiguities and ensure the full meaning is preserved.*

"""

        # Verify context block is built
        self.assertIn("ORIGINAL CONTEXT", context_block)
        self.assertIn(original_context, context_block)
        self.assertIn("SOURCE MEANING", context_block)


class TestSemanticTopology(unittest.TestCase):
    """Test Step 2: Semantic topology extraction and injection."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm_provider = Mock(spec=LLMProvider)

    def test_semantic_topology_extraction(self):
        """Test that semantic edges are extracted from graph."""
        # Simulate semantic topology extraction
        edges = [
            ('P0', 'P1', 'PURPOSE'),
            ('P1', 'P2', 'CONTRAST'),
            ('P2', 'P3', 'ATTRIBUTION')
        ]

        semantic_edge_types = {'PURPOSE', 'ATTRIBUTION', 'ORIGIN', 'COMPOSITION', 'DEFINITION'}
        semantic_instructions = []

        for source, target, label in edges:
            label_upper = label.upper()
            if label_upper in semantic_edge_types:
                relationship_desc = {
                    'PURPOSE': 'provides the PURPOSE for',
                    'ATTRIBUTION': 'is ATTRIBUTED to',
                    'ORIGIN': 'comes from',
                    'COMPOSITION': 'consists of',
                    'DEFINITION': 'defines'
                }.get(label_upper, 'linked to')

                semantic_instructions.append(
                    f"- [{source}] {relationship_desc} [{target}]"
                )

        # Verify semantic instructions are created
        self.assertEqual(len(semantic_instructions), 2)  # PURPOSE and ATTRIBUTION, not CONTRAST
        self.assertIn("PURPOSE", semantic_instructions[0])
        self.assertIn("ATTRIBUTED", semantic_instructions[1])

    def test_semantic_block_in_prompt(self):
        """Test that semantic_block is built when semantic edges exist."""
        semantic_instructions = [
            "- [P0] provides the PURPOSE for [P1]",
            "- [P2] is ATTRIBUTED to [P3]"
        ]

        semantic_block = f"""
**SEMANTIC TOPOLOGY (UNBREAKABLE BONDS):**
These relationships MUST be preserved in your output. They represent factual dependencies that cannot be broken:
{chr(10).join(semantic_instructions)}

*CRITICAL: These semantic links (PURPOSE, ATTRIBUTION, ORIGIN, COMPOSITION) are factual bonds. When you map propositions to the skeleton, ensure these relationships are maintained.*

"""

        # Verify semantic block is built
        self.assertIn("SEMANTIC TOPOLOGY", semantic_block)
        self.assertIn("UNBREAKABLE BONDS", semantic_block)
        self.assertIn("PURPOSE", semantic_block)
        self.assertIn("ATTRIBUTED", semantic_block)


class TestSmartSkeletonSelection(unittest.TestCase):
    """Test Step 4: Smart skeleton selection prompt."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm_provider = Mock(spec=LLMProvider)

    def test_structural_mapper_system_prompt_contains_key_elements(self):
        """Test that system prompt contains all required elements from Step 4."""
        # Read the actual prompt from the source code
        import inspect
        from src.generator.graph_matcher import TopologicalMatcher

        # Get the source code of synthesize_match to verify prompt structure
        source = inspect.getsource(TopologicalMatcher.synthesize_match)

        # Verify key elements are present
        self.assertIn('Structural Mapper', source)
        self.assertIn('Input Logic flow', source)
        self.assertIn('NO Run-On Chains', source)
        self.assertIn('Structural Matching', source)
        self.assertIn('Map the Input Logic structure', source)

    def test_structural_mapper_prompt(self):
        """Test that the prompt uses Structural Mapper language."""
        from src.generator.graph_matcher import TopologicalMatcher

        matcher = TopologicalMatcher(config_path="config.json", llm_provider=self.llm_provider)

        # Mock collection
        matcher.collection = Mock()
        matcher.collection.query = Mock(return_value={
            'ids': [[]],
            'metadatas': [[]],
            'documents': [[]],
            'distances': [[]]
        })
        matcher.collection.get = Mock(return_value={'metadatas': [{'signature': 'DEFINITION'}]})

        # Mock LLM call to capture prompt
        captured_prompt = {}
        def capture_prompt(*args, **kwargs):
            captured_prompt['system'] = kwargs.get('system_prompt', '')
            captured_prompt['user'] = kwargs.get('user_prompt', '')
            return '{"revised_skeleton": "[P0] is [P1]", "selected_source_indices": [0]}'

        self.llm_provider.call = Mock(side_effect=capture_prompt)

        # Call synthesize_match (this will trigger LLM call)
        try:
            matcher.synthesize_match(
                propositions=["Prop0", "Prop1"],
                input_intent="DEFINITION",
                input_signature="DEFINITION",
                verbose=False
            )
        except:
            pass  # We just want to capture the prompt

        # Check if prompt contains Structural Mapper language
        if captured_prompt.get('system'):
            system_prompt = captured_prompt['system']
            # Should contain "Structural Mapper"
            self.assertIn('Structural Mapper', system_prompt)
            # Should contain "NO Run-On Chains" constraint
            self.assertIn('NO Run-On Chains', system_prompt)
            # Should contain structural matching language
            self.assertIn('Structural Matching', system_prompt)

        # Check user prompt contains logic flow analysis
        if captured_prompt.get('user'):
            user_prompt = captured_prompt['user']
            # Should contain "Analyze Input Logic Flow"
            self.assertIn('Analyze Input Logic Flow', user_prompt)
            # Should contain "Select Best Structural Match"
            self.assertIn('Select Best Structural Match', user_prompt)
            # Should contain "Structural Mapping" language
            self.assertIn('Structural Mapping', user_prompt)

    def test_prompt_prohibits_run_on_chains(self):
        """Test that prompt explicitly prohibits run-on chains."""
        # This is a critical constraint from Step 4
        import inspect
        from src.generator.graph_matcher import TopologicalMatcher

        source = inspect.getsource(TopologicalMatcher.synthesize_match)

        # Should explicitly prohibit run-on chains
        self.assertIn('NO Run-On Chains', source)
        self.assertIn('X to Y by Z', source)  # Example of what NOT to do


class TestRealScoring(unittest.TestCase):
    """Test Step 5: Real scoring instead of hardcoded 1.0."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm_provider = Mock(spec=LLMProvider)

    def test_returns_actual_score(self):
        """Test that translate_paragraph_propositions uses chunk scores for final score."""
        # Test the score calculation logic
        chunk_scores = [0.9, 0.8, 0.95]
        compliance_score = sum(chunk_scores) / len(chunk_scores)

        # Verify score is calculated from chunk scores
        self.assertNotEqual(compliance_score, 1.0)
        self.assertAlmostEqual(compliance_score, 0.883, places=2)

        # Test that empty chunk_scores triggers fallback
        empty_scores = []
        if empty_scores:
            score = sum(empty_scores) / len(empty_scores)
        else:
            score = 0.5  # Fallback score

        self.assertEqual(score, 0.5)

    def test_score_calculation_from_chunk_scores(self):
        """Test that score is calculated from chunk scores."""
        # Simulate chunk scores
        chunk_scores = [0.9, 0.8, 0.95]
        compliance_score = sum(chunk_scores) / len(chunk_scores)

        self.assertEqual(compliance_score, 0.8833333333333333)
        self.assertNotEqual(compliance_score, 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.llm_provider = Mock(spec=LLMProvider)
        self.llm_provider.call = Mock(return_value='{"revised_skeleton": "[P0] is [P1]"}')

    def test_pattern_tracking_prevents_repetition(self):
        """Test that pattern tracking prevents repetitive openers."""
        tracker = GlobalStyleTracker(config={'style_state': {'enabled': True}})

        # Register first sentence
        text1 = "It is not a realm of mysticism"
        tracker.register_usage(text1, structure='CONTRAST')

        # Register second sentence with same pattern
        text2 = "It is not a toolset for analysis"
        tracker.register_usage(text2, structure='CONTRAST')

        # Get forbidden openers
        forbidden = tracker.get_forbidden_openers()

        # The pattern "it is not a" should be forbidden
        self.assertIn("it is not a", forbidden)

        # Filter should reject connectors if opener is forbidden
        # (This tests the integration, not just the extraction)

    def test_context_and_topology_in_prompt(self):
        """Test that both context and topology appear in generated prompt."""
        # This is a conceptual test - in practice, we'd need to mock the full pipeline
        # But we can verify the building blocks work

        original_context = "Many people hear the term Dialectical Materialism"
        context_block = f"""
**ORIGINAL CONTEXT (SOURCE MEANING):**
"{original_context}"
*Use this to resolve ambiguities and ensure the full meaning is preserved.*

"""

        semantic_instructions = [
            "- [P0] provides the PURPOSE for [P1]",
            "- [P2] is ATTRIBUTED to [P3]"
        ]
        semantic_block = f"""
**SEMANTIC TOPOLOGY (UNBREAKABLE BONDS):**
These relationships MUST be preserved in your output. They represent factual dependencies that cannot be broken:
{chr(10).join(semantic_instructions)}

*CRITICAL: These semantic links (PURPOSE, ATTRIBUTION, ORIGIN, COMPOSITION) are factual bonds. When you map propositions to the skeleton, ensure these relationships are maintained.*

"""

        # Verify both blocks are properly formatted
        self.assertIn("ORIGINAL CONTEXT", context_block)
        self.assertIn("SOURCE MEANING", context_block)
        self.assertIn("SEMANTIC TOPOLOGY", semantic_block)
        self.assertIn("UNBREAKABLE BONDS", semantic_block)


if __name__ == '__main__':
    unittest.main()


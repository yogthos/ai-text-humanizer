"""Tests for SemanticFracturer."""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.atlas.fracturer import SemanticFracturer


class TestSemanticFracturer:
    """Test suite for SemanticFracturer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock()
        self.fracturer = SemanticFracturer(self.mock_llm)

    def test_fracture_basic_split(self):
        """Test correct splitting of 10 props into logical groups."""
        # Create 10 propositions
        propositions = [f"Proposition {i}" for i in range(10)]

        # Mock LLM response: split into 2 groups of 5
        llm_response = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        self.mock_llm.call.return_value = json.dumps(llm_response)

        # Test with target_density=5
        result = self.fracturer.fracture(propositions, target_density=5, max_density=6)

        # Verify result
        assert result == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert self.mock_llm.call.called

    def test_fracture_small_input(self):
        """Test input < target_density returns single group."""
        # Create 3 propositions (less than target_density=4)
        propositions = ["Prop 0", "Prop 1", "Prop 2"]

        # Should return single group without calling LLM
        result = self.fracturer.fracture(propositions, target_density=4, max_density=6)

        # Verify result
        assert result == [[0, 1, 2]]
        # LLM should not be called for small inputs
        assert not self.mock_llm.call.called

    def test_fracture_exact_target_density(self):
        """Test input exactly equal to target_density returns single group."""
        # Create 4 propositions (exactly target_density=4)
        propositions = ["Prop 0", "Prop 1", "Prop 2", "Prop 3"]

        # Should return single group without calling LLM
        result = self.fracturer.fracture(propositions, target_density=4, max_density=6)

        # Verify result
        assert result == [[0, 1, 2, 3]]
        # LLM should not be called
        assert not self.mock_llm.call.called

    def test_fracture_llm_json_error(self):
        """Test LLM JSON error falls back to fixed chunking."""
        # Create 10 propositions
        propositions = [f"Proposition {i}" for i in range(10)]

        # Mock LLM returning invalid JSON
        self.mock_llm.call.return_value = "Invalid JSON response"

        # Test with target_density=4
        result = self.fracturer.fracture(propositions, target_density=4, max_density=6)

        # Should fall back to fixed chunking: [[0,1,2,3], [4,5,6,7], [8,9]]
        expected = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9]
        ]
        assert result == expected
        assert self.mock_llm.call.called

    def test_fracture_llm_exception(self):
        """Test LLM exception falls back to fixed chunking."""
        # Create 10 propositions
        propositions = [f"Proposition {i}" for i in range(10)]

        # Mock LLM raising an exception
        self.mock_llm.call.side_effect = Exception("LLM call failed")

        # Test with target_density=5
        result = self.fracturer.fracture(propositions, target_density=5, max_density=6)

        # Should fall back to fixed chunking: [[0,1,2,3,4], [5,6,7,8,9]]
        expected = [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]
        ]
        assert result == expected

    def test_fracture_validation_missing_index(self):
        """Test missing index validation triggers fallback."""
        # Create 3 propositions
        propositions = ["Prop 0", "Prop 1", "Prop 2"]

        # Mock LLM returning groups missing index 2
        llm_response = [[0, 1]]  # Missing index 2
        self.mock_llm.call.return_value = json.dumps(llm_response)

        # Test with target_density=2 (so it will try LLM)
        result = self.fracturer.fracture(propositions, target_density=2, max_density=6)

        # Should fall back to fixed chunking: [[0,1], [2]]
        expected = [[0, 1], [2]]
        assert result == expected
        assert self.mock_llm.call.called

    def test_fracture_validation_duplicate_index(self):
        """Test duplicate index validation triggers fallback."""
        # Create 3 propositions
        propositions = ["Prop 0", "Prop 1", "Prop 2"]

        # Mock LLM returning groups with duplicate index 1
        llm_response = [[0, 1], [1, 2]]  # Index 1 appears twice
        self.mock_llm.call.return_value = json.dumps(llm_response)

        # Test with target_density=2
        result = self.fracturer.fracture(propositions, target_density=2, max_density=6)

        # Should fall back to fixed chunking: [[0,1], [2]]
        expected = [[0, 1], [2]]
        assert result == expected
        assert self.mock_llm.call.called

    def test_fracture_validation_out_of_range_index(self):
        """Test out-of-range index validation triggers fallback."""
        # Create 3 propositions (indices 0, 1, 2)
        propositions = ["Prop 0", "Prop 1", "Prop 2"]

        # Mock LLM returning groups with out-of-range index
        llm_response = [[0, 1], [2, 3]]  # Index 3 is out of range
        self.mock_llm.call.return_value = json.dumps(llm_response)

        # Test with target_density=2
        result = self.fracturer.fracture(propositions, target_density=2, max_density=6)

        # Should fall back to fixed chunking: [[0,1], [2]]
        expected = [[0, 1], [2]]
        assert result == expected
        assert self.mock_llm.call.called

    def test_fracture_validation_non_list_response(self):
        """Test non-list LLM response triggers fallback."""
        # Create 10 propositions
        propositions = [f"Proposition {i}" for i in range(10)]

        # Mock LLM returning a dict instead of list
        llm_response = {"groups": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]}
        self.mock_llm.call.return_value = json.dumps(llm_response)

        # Test with target_density=5
        result = self.fracturer.fracture(propositions, target_density=5, max_density=6)

        # Should fall back to fixed chunking
        expected = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert result == expected
        assert self.mock_llm.call.called

    def test_fracture_validation_non_list_inner_items(self):
        """Test non-list inner items trigger fallback."""
        # Create 10 propositions
        propositions = [f"Proposition {i}" for i in range(10)]

        # Mock LLM returning list with non-list items
        llm_response = [[0, 1, 2, 3, 4], "not a list", [5, 6, 7, 8, 9]]
        self.mock_llm.call.return_value = json.dumps(llm_response)

        # Test with target_density=5
        result = self.fracturer.fracture(propositions, target_density=5, max_density=6)

        # Should fall back to fixed chunking
        expected = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert result == expected
        assert self.mock_llm.call.called

    def test_fracture_empty_input(self):
        """Test empty input returns empty list."""
        propositions = []

        result = self.fracturer.fracture(propositions, target_density=4, max_density=6)

        # Should return empty list
        assert result == []
        # LLM should not be called
        assert not self.mock_llm.call.called

    def test_fracture_fallback_chunking_various_sizes(self):
        """Test fallback chunking with various input sizes."""
        # Test with 10 props, target_density=3
        propositions = [f"Prop {i}" for i in range(10)]
        self.mock_llm.call.side_effect = Exception("LLM failed")

        result = self.fracturer.fracture(propositions, target_density=3, max_density=6)
        expected = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9]
        ]
        assert result == expected

        # Test with 7 props, target_density=4
        propositions = [f"Prop {i}" for i in range(7)]
        result = self.fracturer.fracture(propositions, target_density=4, max_density=6)
        expected = [
            [0, 1, 2, 3],
            [4, 5, 6]
        ]
        assert result == expected

    def test_fracture_markdown_code_blocks_stripped(self):
        """Test that markdown code blocks are stripped from LLM response."""
        # Create 10 propositions
        propositions = [f"Proposition {i}" for i in range(10)]

        # Mock LLM returning JSON wrapped in markdown code blocks
        llm_response = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        wrapped_response = f"```json\n{json.dumps(llm_response)}\n```"
        self.mock_llm.call.return_value = wrapped_response

        # Test with target_density=5
        result = self.fracturer.fracture(propositions, target_density=5, max_density=6)

        # Should correctly parse despite markdown
        assert result == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert self.mock_llm.call.called

    def test_fracture_llm_returns_none(self):
        """Test that LLM returning None triggers fallback."""
        # Create 10 propositions
        propositions = [f"Proposition {i}" for i in range(10)]

        # Mock _fracture_with_llm to return None (simulating validation failure)
        # We'll do this by making the LLM return invalid data that passes JSON parsing
        # but fails validation
        llm_response = []  # Empty list fails validation
        self.mock_llm.call.return_value = json.dumps(llm_response)

        # Test with target_density=5
        result = self.fracturer.fracture(propositions, target_density=5, max_density=6)

        # Should fall back to fixed chunking
        expected = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert result == expected
        assert self.mock_llm.call.called

    def test_fracture_with_graph_parameter(self):
        """Test that fracture() accepts optional input_graph parameter."""
        propositions = [f"Prop {i}" for i in range(10)]

        # Test with None graph (should use LLM)
        llm_response = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        self.mock_llm.call.return_value = json.dumps(llm_response)

        result = self.fracturer.fracture(propositions, target_density=5, max_density=6, input_graph=None)
        assert result == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert self.mock_llm.call.called

    def test_fracture_backward_compatibility(self):
        """Test that fracture() works without input_graph parameter (backward compatibility)."""
        propositions = [f"Prop {i}" for i in range(10)]

        # Mock LLM response
        llm_response = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        self.mock_llm.call.return_value = json.dumps(llm_response)

        # Call without input_graph parameter (should still work)
        result = self.fracturer.fracture(propositions, target_density=5, max_density=6)
        assert result == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        assert self.mock_llm.call.called


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


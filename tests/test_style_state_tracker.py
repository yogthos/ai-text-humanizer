"""Tests for GlobalStyleTracker to ensure state tracking works correctly."""

import unittest
from unittest.mock import patch, MagicMock
import json
import tempfile
import os

from src.processing.style_state import GlobalStyleTracker


class TestGlobalStyleTracker(unittest.TestCase):
    """Test suite for GlobalStyleTracker."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a minimal config
        self.config = {
            'style_state': {
                'history_window': 5,
                'connector_filter_threshold': 1,
                'opener_extraction_method': 'first_clause',
                'enabled': True
            }
        }
        self.tracker = GlobalStyleTracker(self.config)

    def test_initialization(self):
        """Test tracker initializes correctly."""
        self.assertEqual(self.tracker.history_window, 5)
        self.assertEqual(self.tracker.connector_threshold, 1)
        self.assertEqual(self.tracker.opener_method, 'first_clause')
        self.assertTrue(self.tracker.enabled)
        self.assertEqual(len(self.tracker.phrase_history), 0)
        self.assertEqual(len(self.tracker.connector_history), 0)
        self.assertEqual(len(self.tracker.structure_history), 0)

    def test_initialization_with_defaults(self):
        """Test tracker uses defaults when config is missing."""
        tracker = GlobalStyleTracker({})
        self.assertEqual(tracker.history_window, 5)
        self.assertEqual(tracker.connector_threshold, 1)
        self.assertTrue(tracker.enabled)

    def test_initialization_disabled(self):
        """Test tracker can be disabled."""
        config = {'style_state': {'enabled': False}}
        tracker = GlobalStyleTracker(config)
        self.assertFalse(tracker.enabled)

    def test_extract_opener_first_clause(self):
        """Test opener extraction from first clause."""
        text = "It is not a realm of strange mysticism, but a practical toolset."
        opener = self.tracker._extract_opener(text)
        self.assertEqual(opener, "It is not a realm of strange mysticism")

    def test_extract_opener_truncates_long_clause(self):
        """Test opener extraction truncates very long clauses."""
        text = "It is not a realm of strange mysticism or impenetrable jargon or complex terminology, but a toolset."
        opener = self.tracker._extract_opener(text)
        # Should truncate to 6 words + "..."
        self.assertTrue(opener.endswith("..."))
        words = opener.replace("...", "").split()
        self.assertLessEqual(len(words), 6)

    def test_extract_opener_no_punctuation(self):
        """Test opener extraction when no punctuation exists."""
        text = "This is a simple sentence without punctuation"
        opener = self.tracker._extract_opener(text)
        self.assertEqual(opener, text[:50])  # Should truncate to 50 chars

    def test_extract_connectors_from_text(self):
        """Test connector extraction from text."""
        text = "It is not X, but Y. However, Z is true. Because of this, we see A."
        connectors = self.tracker._extract_connectors_from_text(text)
        # Should find: but, however, because
        self.assertIn("but", connectors)
        self.assertIn("however", connectors)
        self.assertIn("because", connectors)

    def test_extract_connectors_case_insensitive(self):
        """Test connector extraction is case insensitive."""
        text = "However, this is true. BUT that is false."
        connectors = self.tracker._extract_connectors_from_text(text)
        self.assertIn("however", connectors)
        self.assertIn("but", connectors)

    def test_register_usage_with_opener(self):
        """Test registering usage with explicit opener."""
        text = "It is not X, but Y."
        self.tracker.register_usage(text, connectors=["but"], structure="CONTRAST", opener="It is not X")
        self.assertEqual(len(self.tracker.phrase_history), 1)
        self.assertEqual(len(self.tracker.connector_history), 1)
        self.assertEqual(len(self.tracker.structure_history), 1)
        self.assertIn("it is not x", self.tracker.phrase_history)
        self.assertIn("but", self.tracker.connector_history)
        self.assertIn("CONTRAST", self.tracker.structure_history)

    def test_register_usage_extracts_opener(self):
        """Test registering usage extracts opener if not provided."""
        text = "It is not a realm, but a toolset."
        self.tracker.register_usage(text, connectors=["but"], structure="CONTRAST")
        self.assertEqual(len(self.tracker.phrase_history), 1)
        self.assertIn("it is not a realm", self.tracker.phrase_history)

    def test_register_usage_extracts_connectors(self):
        """Test registering usage extracts connectors if not provided."""
        text = "It is not X, but Y. However, Z is true."
        self.tracker.register_usage(text, structure="CONTRAST")
        # Should extract connectors from text
        self.assertGreater(len(self.tracker.connector_history), 0)
        self.assertIn("but", self.tracker.connector_history)

    def test_register_usage_when_disabled(self):
        """Test registering usage does nothing when tracker is disabled."""
        self.tracker.enabled = False
        text = "It is not X, but Y."
        self.tracker.register_usage(text, connectors=["but"], structure="CONTRAST")
        self.assertEqual(len(self.tracker.phrase_history), 0)
        self.assertEqual(len(self.tracker.connector_history), 0)

    def test_filter_available_connectors(self):
        """Test filtering connectors based on usage history."""
        # Register usage of "but"
        self.tracker.register_usage("Text", connectors=["but"], structure="CONTRAST")

        # Filter connectors
        candidates = ["but", "however", "yet"]
        filtered = self.tracker.filter_available_connectors(candidates, "CONTRAST")

        # "but" should be filtered out (used once, threshold is 1)
        self.assertNotIn("but", filtered)
        # "however" and "yet" should remain
        self.assertIn("however", filtered)
        self.assertIn("yet", filtered)

    def test_filter_available_connectors_all_filtered_returns_original(self):
        """Test that if all connectors are filtered, original list is returned."""
        # Register usage of all connectors
        self.tracker.register_usage("Text", connectors=["but", "however", "yet"], structure="CONTRAST")

        candidates = ["but", "however", "yet"]
        filtered = self.tracker.filter_available_connectors(candidates, "CONTRAST")

        # Should return original list to avoid breaking generation
        self.assertEqual(set(filtered), set(candidates))

    def test_filter_available_connectors_when_disabled(self):
        """Test filtering returns original list when tracker is disabled."""
        self.tracker.enabled = False
        candidates = ["but", "however"]
        filtered = self.tracker.filter_available_connectors(candidates, "CONTRAST")
        self.assertEqual(filtered, candidates)

    def test_filter_available_connectors_empty_list(self):
        """Test filtering empty list returns empty list."""
        filtered = self.tracker.filter_available_connectors([], "CONTRAST")
        self.assertEqual(filtered, [])

    def test_get_forbidden_openers(self):
        """Test getting forbidden openers."""
        self.tracker.register_usage("It is not X", structure="CONTRAST")
        self.tracker.register_usage("It is not Y", structure="CONTRAST")

        forbidden = self.tracker.get_forbidden_openers()
        # Should have 2 unique openers
        self.assertEqual(len(forbidden), 2)
        self.assertIn("it is not x", forbidden)
        self.assertIn("it is not y", forbidden)

    def test_get_forbidden_openers_when_disabled(self):
        """Test getting forbidden openers returns empty when disabled."""
        self.tracker.enabled = False
        self.tracker.register_usage("It is not X", structure="CONTRAST")
        forbidden = self.tracker.get_forbidden_openers()
        self.assertEqual(forbidden, [])

    def test_history_window_limit(self):
        """Test that history window limits the number of tracked items."""
        # Register more items than history window
        for i in range(10):
            self.tracker.register_usage(f"Text {i}", connectors=["but"], structure="CONTRAST")

        # History should be limited to history_window (5)
        self.assertEqual(len(self.tracker.phrase_history), 5)
        self.assertEqual(len(self.tracker.structure_history), 5)
        # Connector history should be limited to 3x window (15), but we only added 10 connectors
        self.assertEqual(len(self.tracker.connector_history), 10)  # We added 10 "but" connectors
        self.assertLessEqual(len(self.tracker.connector_history), 15)  # But max is 15

    def test_reset(self):
        """Test reset clears all history."""
        self.tracker.register_usage("Text", connectors=["but"], structure="CONTRAST")
        self.tracker.reset()
        self.assertEqual(len(self.tracker.phrase_history), 0)
        self.assertEqual(len(self.tracker.connector_history), 0)
        self.assertEqual(len(self.tracker.structure_history), 0)

    def test_connector_threshold_configurable(self):
        """Test that connector threshold is configurable."""
        config = {
            'style_state': {
                'connector_filter_threshold': 2,  # Filter if count >= 2
                'enabled': True
            }
        }
        tracker = GlobalStyleTracker(config)

        # Register "but" once
        tracker.register_usage("Text 1", connectors=["but"], structure="CONTRAST")

        # Should still be available (threshold is 2, we've used it once, count=1 < 2)
        candidates = ["but", "however"]
        filtered = tracker.filter_available_connectors(candidates, "CONTRAST")
        self.assertIn("but", filtered)

        # Register one more time (now count=2)
        tracker.register_usage("Text 2", connectors=["but"], structure="CONTRAST")

        # Now should be filtered (used 2 times, threshold is 2, so count >= threshold)
        filtered = tracker.filter_available_connectors(candidates, "CONTRAST")
        self.assertNotIn("but", filtered)

    def test_normalization_lowercase(self):
        """Test that registered items are normalized to lowercase."""
        self.tracker.register_usage("IT IS NOT X", connectors=["BUT"], structure="CONTRAST")
        self.assertIn("it is not x", self.tracker.phrase_history)
        self.assertIn("but", self.tracker.connector_history)

    def test_multiple_connectors_per_sentence(self):
        """Test that multiple connectors in one sentence are all registered."""
        text = "It is not X, but Y. However, Z is true. Because of this, we see A."
        self.tracker.register_usage(text, structure="CONTRAST")

        # Should extract and register multiple connectors
        connectors_in_history = list(self.tracker.connector_history)
        self.assertGreater(len(connectors_in_history), 1)
        # Check that common connectors are present
        all_connectors = " ".join(connectors_in_history)
        self.assertTrue("but" in all_connectors or "however" in all_connectors or "because" in all_connectors)


if __name__ == '__main__':
    unittest.main()


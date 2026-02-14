"""Tests for generator factory module.

Tests cover:
- Bug 5: Silent exception swallowing in _set_fiction_markers
"""

import pytest
from unittest.mock import patch, MagicMock


class TestFictionMarkerLogging:
    """Tests for Bug 5: Fiction marker loading failure should be logged."""

    def test_fiction_marker_failure_logged(self):
        """When get_adapter_config raises, warning should be logged."""
        from src.generation.factory import _set_fiction_markers

        mock_generator = MagicMock()

        with patch('src.config.get_adapter_config', side_effect=RuntimeError("config error")):
            with patch('src.generation.factory.logger') as mock_logger:
                _set_fiction_markers(mock_generator, "some/adapter/path")

        mock_logger.warning.assert_called_once()
        assert "fiction markers" in str(mock_logger.warning.call_args).lower()

    def test_fiction_marker_success_no_warning(self):
        """When get_adapter_config succeeds, no warning should be logged."""
        from src.generation.factory import _set_fiction_markers
        from src.config import LoRAAdapterConfig

        mock_generator = MagicMock()

        with patch('src.config.get_adapter_config',
                   return_value=LoRAAdapterConfig(fiction_markers=["marker1"])):
            with patch('src.generation.factory.logger') as mock_logger:
                _set_fiction_markers(mock_generator, "some/adapter/path")

        mock_logger.warning.assert_not_called()
        assert mock_generator.fiction_markers == ["marker1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

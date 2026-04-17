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


class TestFictionMarkersFusedModel:
    """Tests for M8: fused-model path was not threading fiction markers."""

    def test_fused_model_config_has_fiction_markers_field(self):
        """FusedModelConfig needs a fiction_markers field to match LoRAAdapterConfig."""
        from src.config import FusedModelConfig

        cfg = FusedModelConfig(fiction_markers=["foo", "bar"])
        assert cfg.fiction_markers == ["foo", "bar"]

    def test_set_fiction_markers_falls_back_to_fused_config(self):
        """When adapter config has no markers, _set_fiction_markers should consult
        the fused-model config. A path pointing at a fused model has no adapter
        entry, so the adapter lookup returns defaults (empty markers)."""
        from src.generation.factory import _set_fiction_markers
        from src.config import LoRAAdapterConfig, FusedModelConfig

        mock_generator = MagicMock()
        # Explicitly drop any pre-existing attribute so the assertion is meaningful.
        mock_generator.fiction_markers = []

        with patch('src.config.get_adapter_config',
                   return_value=LoRAAdapterConfig(fiction_markers=[])):
            with patch('src.config.get_fused_model_config',
                       return_value=FusedModelConfig(fiction_markers=["fused_marker"])):
                _set_fiction_markers(mock_generator, "fused/model/path")

        assert mock_generator.fiction_markers == ["fused_marker"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for generator factory module.

Tests cover:
- Fiction marker loading: config errors handled internally by get_adapter_config
  / get_fused_model_config (they catch and return defaults). The factory's own
  exception handling is narrowed to (ImportError, AttributeError) so that
  unexpected errors surface instead of being silently swallowed.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestFictionMarkerLogging:
    """Tests for _set_fiction_markers exception handling."""

    def test_fiction_marker_attribute_error_logged(self):
        """AttributeError when setting fiction_markers should be caught and logged."""
        from src.generation.factory import _set_fiction_markers
        from src.config import ModelConfig

        # Generator that raises AttributeError on attribute assignment
        class RigidGenerator:
            __slots__ = ()

        generator = RigidGenerator()

        with patch('src.config.get_adapter_config',
                   return_value=ModelConfig(fiction_markers=["marker1"])):
            with patch('src.generation.factory.logger') as mock_logger:
                _set_fiction_markers(generator, "some/adapter/path")

        mock_logger.warning.assert_called_once()
        assert "fiction markers" in str(mock_logger.warning.call_args).lower()

    def test_fiction_marker_unexpected_errors_propagate(self):
        """Non-(ImportError, AttributeError) exceptions must surface as real bugs,
        not be silently swallowed. Config errors are already handled internally."""
        from src.generation.factory import _set_fiction_markers

        mock_generator = MagicMock()

        with patch('src.config.get_adapter_config', side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                _set_fiction_markers(mock_generator, "some/adapter/path")

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

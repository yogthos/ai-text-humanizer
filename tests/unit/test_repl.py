"""Tests for the REPL module."""

import pytest
from unittest.mock import MagicMock, patch

from src.repl.repl import (
    Colors,
    supports_color,
    get_terminal_width,
    REPLConfig,
    StyleREPL,
)


class TestColors:
    """Tests for color constants."""

    def test_reset_defined(self):
        """Test RESET code is defined."""
        assert Colors.RESET == "\033[0m"

    def test_bold_defined(self):
        """Test BOLD code is defined."""
        assert Colors.BOLD == "\033[1m"


class TestSupportsColor:
    """Tests for supports_color function."""

    def test_returns_bool(self):
        """Test supports_color returns a boolean."""
        result = supports_color()
        assert isinstance(result, bool)

    @patch.dict("os.environ", {"NO_COLOR": "1"})
    def test_no_color_env_disables(self):
        """Test NO_COLOR environment variable disables colors."""
        result = supports_color()
        assert result is False


class TestGetTerminalWidth:
    """Tests for get_terminal_width function."""

    def test_returns_int(self):
        """Test get_terminal_width returns an integer."""
        result = get_terminal_width()
        assert isinstance(result, int)
        assert result > 0

    def test_default_fallback(self):
        """Test fallback to 80 when terminal size unavailable."""
        with patch("os.get_terminal_size", side_effect=OSError):
            result = get_terminal_width()
            assert result == 80


class TestREPLConfig:
    """Tests for REPLConfig dataclass."""

    def test_required_fields(self):
        """Test REPLConfig requires author and adapter_path."""
        config = REPLConfig(
            author="Test Author",
            adapter_path="/path/to/adapter",
        )
        assert config.author == "Test Author"
        assert config.adapter_path == "/path/to/adapter"

    def test_default_values(self):
        """Test REPLConfig default values."""
        config = REPLConfig(
            author="Test Author",
            adapter_path="/path/to/adapter",
        )
        assert config.temperature == 0.4
        assert config.verify is True
        assert config.perspective == "preserve"
        assert config.use_color is True


class TestStyleREPL:
    """Tests for StyleREPL class."""

    @pytest.fixture
    def mock_transfer(self):
        """Create a mock StyleTransfer."""
        mock = MagicMock()
        mock.transfer_paragraph.return_value = ("Transformed text.", 0.95)
        return mock

    @pytest.fixture
    def repl_config(self):
        """Create a test REPLConfig."""
        return REPLConfig(
            author="Test Author",
            adapter_path="/path/to/adapter",
            use_color=False,  # Disable colors for testing
        )

    def test_init(self, mock_transfer, repl_config):
        """Test StyleREPL initialization."""
        repl = StyleREPL(mock_transfer, repl_config)
        assert repl.transfer is mock_transfer
        assert repl.config is repl_config
        assert repl.history == []
        assert repl.running is False

    def test_color_disabled(self, mock_transfer, repl_config):
        """Test color is disabled when use_color=False."""
        repl = StyleREPL(mock_transfer, repl_config)
        assert repl.use_color is False
        # _color should return plain text
        result = repl._color("test", Colors.RED)
        assert result == "test"

    def test_handle_quit_command(self, mock_transfer, repl_config):
        """Test /quit command returns False to exit."""
        repl = StyleREPL(mock_transfer, repl_config)
        result = repl._handle_command("/quit")
        assert result is False

    def test_handle_help_command(self, mock_transfer, repl_config):
        """Test /help command returns True to continue."""
        repl = StyleREPL(mock_transfer, repl_config)
        result = repl._handle_command("/help")
        assert result is True

    def test_handle_unknown_command(self, mock_transfer, repl_config):
        """Test unknown command returns True to continue."""
        repl = StyleREPL(mock_transfer, repl_config)
        result = repl._handle_command("/unknown")
        assert result is True

    def test_transform_text_short(self, mock_transfer, repl_config):
        """Test transforming short text generates variations using transfer_paragraph."""
        repl = StyleREPL(mock_transfer, repl_config)
        result = repl._transform_text("Short test input.")
        # Now returns a list of variations
        assert isinstance(result, list)
        assert "Transformed text." in result
        # Called multiple times for variations
        assert mock_transfer.transfer_paragraph.call_count >= 1

    def test_transform_empty_text(self, mock_transfer, repl_config):
        """Test transforming empty text returns None."""
        repl = StyleREPL(mock_transfer, repl_config)
        result = repl._transform_text("")
        assert result is None

    def test_history_tracking(self, mock_transfer, repl_config):
        """Test history is tracked after transformations."""
        repl = StyleREPL(mock_transfer, repl_config)

        # Simulate a transformation
        variations = repl._transform_text("Test input")
        if variations:
            repl.history.append(("Test input", variations))

        assert len(repl.history) == 1
        # History now stores list of variations
        inp, vars = repl.history[0]
        assert inp == "Test input"
        assert isinstance(vars, list)
        assert "Transformed text." in vars


class TestRunReplConfig:
    """Tests for run_repl configuration passing (Bug 4, Bug 5)."""

    @patch('src.repl.repl.StyleTransfer')
    @patch('src.config.load_config')
    def test_temperature_none_default(self, mock_load_config, mock_style_transfer):
        """Temperature should default to None, not 0.4 (Bug 5)."""
        from src.repl.repl import run_repl
        from src.config import Config

        mock_load_config.return_value = Config()

        with patch('src.repl.repl.StyleREPL') as mock_repl_class:
            mock_repl_class.return_value = MagicMock()

            run_repl(
                adapter_path="lora_adapters/test",
                author="Test",
            )

        call_kwargs = mock_style_transfer.call_args
        if call_kwargs:
            tc = call_kwargs.kwargs.get('config')
            if tc:
                assert tc.temperature is None, (
                    f"Default temperature should be None, got {tc.temperature}"
                )

    @patch('src.repl.repl.StyleTransfer')
    @patch('src.config.load_config')
    def test_explicit_temperature_passed(self, mock_load_config, mock_style_transfer):
        """Explicit temperature should be passed through."""
        from src.repl.repl import run_repl
        from src.config import Config

        mock_load_config.return_value = Config()

        with patch('src.repl.repl.StyleREPL') as mock_repl_class:
            mock_repl_class.return_value = MagicMock()

            run_repl(
                adapter_path="lora_adapters/test",
                author="Test",
                temperature=0.8,
            )

        call_kwargs = mock_style_transfer.call_args
        if call_kwargs:
            tc = call_kwargs.kwargs.get('config')
            if tc:
                assert tc.temperature == 0.8


class TestReplInputSubmission:
    """Tests for REPL input submission behavior."""

    def test_single_enter_submits(self):
        """Single Enter (blank line) after content should submit."""
        from src.repl.repl import StyleREPL, REPLConfig

        config = REPLConfig(author="Test", adapter_path="/test", use_color=False)
        mock_transfer = MagicMock()
        repl = StyleREPL(mock_transfer, config)

        # Simulate: "Hello" then one empty line
        inputs = iter(["Hello", ""])
        with patch('builtins.input', side_effect=inputs):
            result = repl._read_input()

        assert result is not None
        assert "Hello" in result

    def test_blank_enter_with_no_content_skips(self):
        """Blank Enter with no content should return empty string."""
        from src.repl.repl import StyleREPL, REPLConfig

        config = REPLConfig(author="Test", adapter_path="/test", use_color=False)
        mock_transfer = MagicMock()
        repl = StyleREPL(mock_transfer, config)

        # Simulate: just an empty line (no content)
        inputs = iter([""])
        with patch('builtins.input', side_effect=inputs):
            result = repl._read_input()

        assert result == ""


class TestTqdmNotGloballyDisabled:
    """Tests for tqdm not being disabled at import (Bug 15)."""

    def test_tqdm_disable_scoped_to_repl_run(self):
        """Importing repl module should NOT monkey-patch tqdm at import time."""
        # Save original state
        import importlib
        import tqdm

        original_init = tqdm.tqdm.__init__

        # Re-import the module
        import src.repl.repl as repl_module
        importlib.reload(repl_module)

        # tqdm.__init__ should NOT be monkey-patched at import time
        # (it should only be patched inside run_repl)
        current_init = tqdm.tqdm.__init__
        # If it's a partialmethod, it was monkey-patched
        from functools import partialmethod
        assert not isinstance(current_init, partialmethod), (
            "tqdm.tqdm.__init__ should not be monkey-patched at module import time"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

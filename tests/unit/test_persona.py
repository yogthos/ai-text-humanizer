"""Tests for persona prompt_builder module.

Tests cover:
- Bug 2: _load_persona_file("") crashes
- Bug 3: adapter_path not threaded to worldview lookup
"""

import pytest
from unittest.mock import patch


class TestLoadPersonaFile:
    """Tests for _load_persona_file edge cases (Bug 2)."""

    def test_empty_string_returns_empty_result(self):
        """Call with empty string should return empty frames, not crash."""
        from src.persona.prompt_builder import _load_persona_file

        # Clear lru_cache to ensure fresh call
        _load_persona_file.cache_clear()
        result = _load_persona_file("")
        assert result == {"narrative_frames": [], "conceptual_frames": []}

    def test_nonexistent_file_returns_empty_result(self):
        """Call with nonexistent filename should return empty frames."""
        from src.persona.prompt_builder import _load_persona_file

        _load_persona_file.cache_clear()
        # Use a filename that definitely doesn't exist and no default_persona.txt either
        with patch("pathlib.Path.exists", return_value=False):
            result = _load_persona_file("nonexistent_persona_xyz.txt")

        assert result == {"narrative_frames": [], "conceptual_frames": []}


class TestGetPersonaFrame:
    """Tests for _get_persona_frame with adapter_path threading (Bug 3)."""

    @patch('src.persona.prompt_builder._get_worldview_filename')
    @patch('src.persona.prompt_builder._load_persona_file')
    def test_adapter_path_passed_to_worldview_lookup(self, mock_load, mock_get_worldview):
        """_get_persona_frame should pass adapter_path to _get_worldview_filename."""
        from src.persona.prompt_builder import _get_persona_frame

        mock_get_worldview.return_value = "test_worldview.txt"
        mock_load.return_value = {
            "narrative_frames": ["Test narrative frame"],
            "conceptual_frames": ["Test conceptual frame"],
        }

        _get_persona_frame(is_narrative=True, adapter_path="lora_adapters/test")
        mock_get_worldview.assert_called_once_with("lora_adapters/test")

    @patch('src.persona.prompt_builder._get_worldview_filename')
    @patch('src.persona.prompt_builder._load_persona_file')
    def test_adapter_path_none_still_works(self, mock_load, mock_get_worldview):
        """_get_persona_frame should work without adapter_path."""
        from src.persona.prompt_builder import _get_persona_frame

        mock_get_worldview.return_value = "default_persona.txt"
        mock_load.return_value = {
            "narrative_frames": ["Default frame"],
            "conceptual_frames": [],
        }

        result = _get_persona_frame(is_narrative=True)
        mock_get_worldview.assert_called_once_with(None)
        assert result == "Default frame"


class TestBuildPersonaPromptAdapterPath:
    """Tests for build_persona_prompt threading adapter_path (Bug 3)."""

    @patch('src.persona.prompt_builder._get_persona_frame')
    @patch('src.persona.prompt_builder._detect_content_type')
    def test_build_persona_prompt_threads_adapter_path(self, mock_detect, mock_get_frame):
        """build_persona_prompt should pass adapter_path to _get_persona_frame."""
        from src.persona.prompt_builder import build_persona_prompt
        from src.persona.config import PersonaConfig

        mock_detect.return_value = True  # narrative
        mock_get_frame.return_value = "You are recounting events."

        persona = PersonaConfig(
            archetype="Test Scholar",
            emotional_lens="curiosity",
            voice_mode="journal",
            adjective_themes=["dark", "ancient"],
        )

        build_persona_prompt(
            content="Test content for the prompt builder.",
            author="Test",
            persona=persona,
            adapter_path="lora_adapters/test",
        )

        mock_get_frame.assert_called_once_with(True, adapter_path="lora_adapters/test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

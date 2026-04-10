"""Tests for semantic fidelity validation."""

import json
import pytest
from unittest.mock import MagicMock


class TestFidelityResult:
    """Tests for FidelityResult dataclass."""

    def test_was_modified_true_when_changes(self):
        from src.validation.semantic_fidelity import FidelityResult

        result = FidelityResult(
            original="original",
            corrected="corrected",
            changes=[{"issue": "test", "fix": "fixed"}],
        )
        assert result.was_modified is True

    def test_was_modified_false_when_no_changes(self):
        from src.validation.semantic_fidelity import FidelityResult

        result = FidelityResult(
            original="original",
            corrected="original",
            changes=[],
        )
        assert result.was_modified is False

    def test_default_changes_empty(self):
        from src.validation.semantic_fidelity import FidelityResult

        result = FidelityResult(original="a", corrected="a")
        assert result.changes == []
        assert result.was_modified is False


class TestValidateSemanticFidelity:
    """Tests for validate_semantic_fidelity function."""

    def test_no_changes_needed(self):
        """When restyled text is faithful, return it unchanged."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [],
            "result": "The cat sat on the mat.",
        })

        result = validate_semantic_fidelity(
            original="The cat sat on the mat.",
            restyled="The cat sat on the mat.",
            critic_provider=mock_provider,
        )

        assert result.corrected == "The cat sat on the mat."
        assert result.was_modified is False
        assert result.changes == []

    def test_corrects_factual_error(self):
        """When restyled has a factual error, return corrected version."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [{"issue": "wrong subject", "fix": "fixed subject"}],
            "result": "The dog sat on the mat.",
        })

        result = validate_semantic_fidelity(
            original="The dog sat on the mat.",
            restyled="The cat sat on the mat.",
            critic_provider=mock_provider,
        )

        assert result.corrected == "The dog sat on the mat."
        assert result.was_modified is True
        assert len(result.changes) == 1

    def test_passes_correct_prompts(self):
        """Verify the function sends original and restyled in the user prompt."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [],
            "result": "restyled text",
        })

        validate_semantic_fidelity(
            original="original text",
            restyled="restyled text",
            critic_provider=mock_provider,
        )

        call_kwargs = mock_provider.call.call_args
        user_prompt = call_kwargs.kwargs.get("user_prompt", call_kwargs.args[1] if len(call_kwargs.args) > 1 else "")
        assert "ORIGINAL:" in user_prompt
        assert "original text" in user_prompt
        assert "RESTYLED:" in user_prompt
        assert "restyled text" in user_prompt

    def test_uses_low_temperature(self):
        """Validation should use low temperature for deterministic output."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [],
            "result": "text",
        })

        validate_semantic_fidelity(
            original="text",
            restyled="text",
            critic_provider=mock_provider,
        )

        call_kwargs = mock_provider.call.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.1

    def test_requests_json_format(self):
        """Validation should request JSON response format."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [],
            "result": "text",
        })

        validate_semantic_fidelity(
            original="text",
            restyled="text",
            critic_provider=mock_provider,
        )

        call_kwargs = mock_provider.call.call_args
        assert call_kwargs.kwargs.get("require_json") is True

    def test_fallback_on_json_parse_error(self):
        """If LLM returns invalid JSON, return restyled text unchanged."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = "This is not JSON at all"

        result = validate_semantic_fidelity(
            original="original",
            restyled="restyled unchanged",
            critic_provider=mock_provider,
        )

        assert result.corrected == "restyled unchanged"
        assert result.was_modified is False

    def test_fallback_on_api_error(self):
        """If LLM call fails, return restyled text unchanged."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.side_effect = RuntimeError("API timeout")

        result = validate_semantic_fidelity(
            original="original",
            restyled="restyled unchanged",
            critic_provider=mock_provider,
        )

        assert result.corrected == "restyled unchanged"
        assert result.was_modified is False

    def test_fallback_on_missing_result_key(self):
        """If JSON is missing 'result' key, use restyled text."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [{"issue": "something"}],
        })

        result = validate_semantic_fidelity(
            original="original",
            restyled="restyled fallback",
            critic_provider=mock_provider,
        )

        # Should use restyled as fallback since "result" key missing
        assert result.corrected == "restyled fallback"

    def test_max_tokens_scales_with_input(self):
        """max_tokens should scale with restyled text length."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": [],
            "result": "short",
        })

        # Short text — should use minimum of 1024
        validate_semantic_fidelity(
            original="short",
            restyled="short",
            critic_provider=mock_provider,
        )

        call_kwargs = mock_provider.call.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 1024

        # Long text — should scale up
        long_text = " ".join(["word"] * 500)
        validate_semantic_fidelity(
            original=long_text,
            restyled=long_text,
            critic_provider=mock_provider,
        )

        call_kwargs = mock_provider.call.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 500 * 4

    def test_multiple_changes_logged(self):
        """Multiple changes should all be present in result."""
        from src.validation.semantic_fidelity import validate_semantic_fidelity

        changes = [
            {"issue": "missing claim A", "fix": "added A"},
            {"issue": "reversed meaning B", "fix": "fixed B"},
        ]
        mock_provider = MagicMock()
        mock_provider.call.return_value = json.dumps({
            "changes": changes,
            "result": "corrected text with A and B",
        })

        result = validate_semantic_fidelity(
            original="original",
            restyled="restyled",
            critic_provider=mock_provider,
        )

        assert len(result.changes) == 2
        assert result.was_modified is True
        assert result.corrected == "corrected text with A and B"


class TestSemanticFidelityPrompt:
    """Tests for the semantic fidelity prompt file."""

    def test_prompt_file_exists(self):
        """The prompt file should exist in prompts/."""
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity")
        assert len(prompt) > 0

    def test_prompt_describes_json_format(self):
        """Prompt should specify JSON output format."""
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity")
        assert '"changes"' in prompt
        assert '"result"' in prompt

    def test_prompt_emphasizes_conservatism(self):
        """Prompt should emphasize minimal changes."""
        from src.utils.prompts import load_prompt

        prompt = load_prompt("semantic_fidelity")
        assert "INTENTIONAL" in prompt
        assert "smallest" in prompt.lower()


class TestTransferPipelineIntegration:
    """Tests for semantic fidelity integration in the transfer pipeline."""

    def test_verify_semantic_fidelity_config_default(self):
        """verify_semantic_fidelity should default to True."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig()
        assert config.verify_semantic_fidelity is True

    def test_verify_semantic_fidelity_can_disable(self):
        """verify_semantic_fidelity can be set to False."""
        from src.generation.transfer import TransferConfig

        config = TransferConfig(verify_semantic_fidelity=False)
        assert config.verify_semantic_fidelity is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for deepseek provider module.

Tests cover:
- Bug 10: call_with_logit_bias should use retry logic
"""

import pytest
from unittest.mock import patch, MagicMock


class TestLogitBiasRetry:
    """Tests for call_with_logit_bias retry behavior (Bug 10)."""

    def test_call_with_logit_bias_retries_on_rate_limit(self):
        """call_with_logit_bias should retry on rate limit errors."""
        from src.llm.deepseek import DeepSeekProvider
        from src.llm.provider import LLMRateLimitError, LLMResponse

        provider = DeepSeekProvider.__new__(DeepSeekProvider)
        provider.config = MagicMock()
        provider.config.model = "deepseek-chat"
        provider.base_url = "https://api.deepseek.com"
        # provider_name is a property on DeepSeekProvider, no need to set it
        provider.retry_config = {
            "max_retries": 3,
            "base_delay": 0.01,
            "max_delay": 0.1,
        }
        provider._total_input_tokens = 0
        provider._total_output_tokens = 0
        provider._total_calls = 0

        # Mock _call_api to fail once then succeed
        call_count = 0
        def mock_call_api(messages, temperature=None, max_tokens=None,
                          require_json=False, logit_bias=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise LLMRateLimitError("Rate limited")
            return LLMResponse(
                content="Generated text",
                model="deepseek-chat",
                input_tokens=10,
                output_tokens=20,
            )

        provider._call_api = mock_call_api

        result = provider.call_with_logit_bias(
            system_prompt="System",
            user_prompt="User",
            logit_bias={"123": -100},
            temperature=0.7,
        )

        assert result == "Generated text"
        assert call_count == 2  # First failed, second succeeded


class TestThinkingMode:
    """Tests for thinking-mode control (deepseek-v4-flash defaults to thinking on).

    Thinking mode emits reasoning_content before content and can exhaust a
    small max_tokens budget, yielding an empty-content error. The provider must
    disable thinking by default and allow enabling it via config.
    """

    def _make_provider(self, thinking):
        from src.llm.deepseek import DeepSeekProvider
        from src.config import LLMProviderConfig

        provider = DeepSeekProvider.__new__(DeepSeekProvider)
        provider.config = LLMProviderConfig(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            thinking=thinking,
        )
        provider.base_url = provider.config.base_url
        provider.headers = {
            "Authorization": f"Bearer {provider.config.api_key}",
            "Content-Type": "application/json",
        }
        provider._tokenizer = None
        provider._total_input_tokens = 0
        provider._total_output_tokens = 0
        provider._total_calls = 0
        return provider

    def _mock_response(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "deepseek-v4-flash",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        return mock_resp

    def test_thinking_disabled_by_default(self):
        """Provider must send thinking:{type:disabled} when config.thinking is False."""
        provider = self._make_provider(thinking=False)
        with patch("src.llm.deepseek.requests.post", return_value=self._mock_response()) as mock_post:
            from src.models import Message, MessageRole
            provider._call_api([Message(role=MessageRole.USER, content="hi")])
            payload = mock_post.call_args.kwargs["json"]
            assert payload["thinking"] == {"type": "disabled"}

    def test_thinking_enabled_when_configured(self):
        """Provider must send thinking:{type:enabled} when config.thinking is True."""
        provider = self._make_provider(thinking=True)
        with patch("src.llm.deepseek.requests.post", return_value=self._mock_response()) as mock_post:
            from src.models import Message, MessageRole
            provider._call_api([Message(role=MessageRole.USER, content="hi")])
            payload = mock_post.call_args.kwargs["json"]
            assert payload["thinking"] == {"type": "enabled"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

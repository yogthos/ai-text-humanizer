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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

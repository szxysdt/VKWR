from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vkwr.engine.outputs import CompletionOutput, RequestOutput, RequestStats
from vkwr.engine.request import SamplingParams


class TestNormalizePrompts:
    """Test LLM._normalize_prompts without requiring engine init."""

    def _get_llm(self):
        """Create LLM without triggering __init__ (skips engine init)."""
        from vkwr.entrypoints.llm import LLM

        llm = LLM.__new__(LLM)
        llm._engine_args = MagicMock()
        llm.llm_engine = None
        return llm

    def test_single_string_prompt(self):
        llm = self._get_llm()
        assert llm._normalize_prompts("hello") == ["hello"]

    def test_list_of_strings(self):
        llm = self._get_llm()
        assert llm._normalize_prompts(["hello", "world"]) == ["hello", "world"]

    def test_single_int_list(self):
        llm = self._get_llm()
        assert llm._normalize_prompts([1, 2, 3]) == [[1, 2, 3]]

    def test_nested_int_lists(self):
        llm = self._get_llm()
        assert llm._normalize_prompts([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]

    def test_empty_list(self):
        llm = self._get_llm()
        assert llm._normalize_prompts([]) == []

    def test_invalid_prompts_type(self):
        llm = self._get_llm()
        with pytest.raises(TypeError, match="prompts must be str or list"):
            llm._normalize_prompts(123)  # type: ignore


class TestLLMGenerateSync:
    @patch("vkwr.engine.llm_engine.LLMEngine")
    def test_generate_returns_outputs(self, MockEngine):
        mock_engine = MagicMock()
        MockEngine.from_engine_args.return_value = mock_engine

        mock_engine.has_unfinished_requests.side_effect = [True, False]
        mock_engine.step.return_value = [
            RequestOutput(
                request_id="req-0",
                prompt="hello",
                prompt_token_ids=[1, 2],
                outputs=[
                    CompletionOutput(
                        index=0,
                        token_ids=[10, 20],
                        text="world",
                        cumlogprob=0.0,
                        finish_reason="length",
                    )
                ],
                finished=True,
                finish_reason="length",
                stats=RequestStats(),
            )
        ]

        from vkwr.entrypoints.llm import LLM

        llm = LLM(model="/fake/model.pth")
        outputs = llm.generate("hello", SamplingParams(max_tokens=2))

        assert len(outputs) == 1
        assert outputs[0].request_id == "req-0"
        assert outputs[0].finished is True

        mock_engine.add_request.assert_called_once()
        call_args = mock_engine.add_request.call_args
        assert call_args[0][0].startswith("req-0-")
        assert call_args[0][1] == "hello"


class TestLLMGenerateStreaming:
    @patch("vkwr.engine.llm_engine.LLMEngine")
    def test_streaming_yields_outputs(self, MockEngine):
        mock_engine = MagicMock()
        MockEngine.from_engine_args.return_value = mock_engine

        finished_output = RequestOutput(
            request_id="req-0",
            prompt="hello",
            prompt_token_ids=[1, 2],
            outputs=[
                CompletionOutput(
                    index=0,
                    token_ids=[10, 20],
                    text="world",
                    cumlogprob=0.0,
                    finish_reason="length",
                )
            ],
            finished=True,
            finish_reason="length",
            stats=RequestStats(),
        )

        mock_engine.has_unfinished_requests.side_effect = [True, True, False]
        mock_engine.step.side_effect = [[finished_output], []]

        from vkwr.entrypoints.llm import LLM

        llm = LLM(model="/fake/model.pth")
        results = list(llm.generate("hello", SamplingParams(max_tokens=2), streaming=True))

        assert len(results) == 1
        assert results[0].request_id == "req-0"


class TestLLMGenerateEmptyPrompts:
    @patch("vkwr.engine.llm_engine.LLMEngine")
    def test_empty_prompts_returns_empty(self, MockEngine):
        mock_engine = MagicMock()
        MockEngine.from_engine_args.return_value = mock_engine

        from vkwr.entrypoints.llm import LLM

        llm = LLM(model="/fake/model.pth")
        outputs = llm.generate([], SamplingParams())
        assert outputs == []


class TestLLMDefaultSamplingParams:
    @patch("vkwr.engine.llm_engine.LLMEngine")
    def test_default_sampling_params(self, MockEngine):
        mock_engine = MagicMock()
        MockEngine.from_engine_args.return_value = mock_engine
        mock_engine.has_unfinished_requests.return_value = False
        mock_engine.step.return_value = []

        from vkwr.entrypoints.llm import LLM

        llm = LLM(model="/fake/model.pth")
        llm.generate("hello")

        mock_engine.add_request.assert_called_once()
        call_args = mock_engine.add_request.call_args
        assert isinstance(call_args[0][2], SamplingParams)
        assert call_args[0][2].temperature == 1.0
        assert call_args[0][2].max_tokens is None

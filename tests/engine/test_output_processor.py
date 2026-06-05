from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vkwr.config.model import ModelConfig
from vkwr.engine.output_processor import OutputProcessor
from vkwr.engine.outputs import (
    EngineCoreOutput,
    EngineCoreOutputs,
)
from vkwr.engine.request import SamplingParams, VkwrRequest


@pytest.fixture
def model_config():
    return ModelConfig(model="/fake/path.pth", tokenizer="/fake/vocab.txt")


@pytest.fixture
def sample_request():
    return VkwrRequest(
        request_id="req-1",
        prompt="hello",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.decode = MagicMock(return_value="Hello world!")
    return tok


class TestOutputProcessorInit:
    def test_init(self, model_config):
        processor = OutputProcessor(model_config)
        assert processor.model_config is model_config


class TestAddRequest:
    def test_add_request(self, model_config, sample_request):
        processor = OutputProcessor(model_config)
        processor.add_request(sample_request)

        req_output = processor._requests["req-1"]
        assert req_output.request_id == "req-1"
        assert req_output.prompt == "hello"
        assert req_output.prompt_token_ids == [1, 2, 3]
        assert req_output.finished is False
        assert req_output.finish_reason is None
        assert len(req_output.outputs) == 1
        assert req_output.outputs[0].token_ids == []
        assert req_output.outputs[0].text == ""
        assert req_output.stats is not None


class TestProcessOutputsIncremental:
    @patch("vkwr.engine.tokenizer.get_tokenizer")
    def test_incremental_token_accumulation(self, mock_get_tok, model_config, sample_request, mock_tokenizer):
        mock_get_tok.return_value = mock_tokenizer
        processor = OutputProcessor(model_config)
        processor.add_request(sample_request)

        step1 = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="req-1",
                    new_token_ids=[10, 20],
                    new_logprobs=None,
                    finish_reason=None,
                )
            ]
        )
        finished = processor.process_outputs(step1)
        assert len(finished) == 0
        req_output = processor._requests["req-1"]
        assert req_output.outputs[0].token_ids == [10, 20]

        step2 = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="req-1",
                    new_token_ids=[30],
                    new_logprobs=None,
                    finish_reason=None,
                )
            ]
        )
        finished = processor.process_outputs(step2)
        assert len(finished) == 0
        assert req_output.outputs[0].token_ids == [10, 20, 30]


class TestProcessOutputsFinish:
    @patch("vkwr.engine.tokenizer.get_tokenizer")
    def test_finish_returns_output(self, mock_get_tok, model_config, sample_request, mock_tokenizer):
        mock_get_tok.return_value = mock_tokenizer
        processor = OutputProcessor(model_config)
        processor.add_request(sample_request)

        engine_out = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="req-1",
                    new_token_ids=[10, 20],
                    new_logprobs=None,
                    finish_reason="length",
                )
            ]
        )
        finished = processor.process_outputs(engine_out)
        assert len(finished) == 1
        assert finished[0].request_id == "req-1"
        assert finished[0].finished is True
        assert finished[0].finish_reason == "length"


class TestProcessOutputsUnknownRequest:
    @patch("vkwr.engine.tokenizer.get_tokenizer")
    def test_unknown_request_id_ignored(self, mock_get_tok, model_config, mock_tokenizer):
        mock_get_tok.return_value = mock_tokenizer
        processor = OutputProcessor(model_config)
        engine_out = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="nonexistent",
                    new_token_ids=[1],
                    new_logprobs=None,
                    finish_reason=None,
                )
            ]
        )
        finished = processor.process_outputs(engine_out)
        assert len(finished) == 0


class TestProcessOutputsLogprobs:
    @patch("vkwr.engine.tokenizer.get_tokenizer")
    def test_logprobs_accumulation(self, mock_get_tok, model_config, sample_request, mock_tokenizer):
        mock_get_tok.return_value = mock_tokenizer
        processor = OutputProcessor(model_config)
        processor.add_request(sample_request)

        engine_out = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="req-1",
                    new_token_ids=[10],
                    new_logprobs=[{10: -0.5}],
                    finish_reason=None,
                )
            ]
        )
        processor.process_outputs(engine_out)

        engine_out2 = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="req-1",
                    new_token_ids=[20],
                    new_logprobs=[{20: -1.0}],
                    finish_reason="length",
                )
            ]
        )
        finished = processor.process_outputs(engine_out2)
        assert len(finished) == 1
        completion = finished[0].outputs[0]
        assert completion.logprobs is not None
        assert len(completion.logprobs) == 2


class TestRemoveRequest:
    def test_remove_request(self, model_config, sample_request):
        processor = OutputProcessor(model_config)
        processor.add_request(sample_request)
        processor.remove_request("req-1")
        assert "req-1" not in processor._requests

    def test_remove_nonexistent_request(self, model_config):
        processor = OutputProcessor(model_config)
        processor.remove_request("does-not-exist")

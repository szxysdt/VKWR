from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vkwr.config.model import ModelConfig
from vkwr.config.scheduler import SchedulerConfig
from vkwr.engine.input_processor import InputProcessor
from vkwr.engine.request import SamplingParams


@pytest.fixture
def model_config():
    return ModelConfig(model="/fake/path.pth", tokenizer="/fake/vocab.txt")


@pytest.fixture
def scheduler_config():
    return SchedulerConfig(max_model_len=8192)


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.encode = MagicMock(side_effect=lambda text: list(range(1, len(text) + 1)))
    tok.decode = MagicMock(side_effect=lambda ids: "".join(chr(i) for i in ids))
    tok.eos_token_id = 0
    return tok


class TestInputProcessorInit:
    def test_init_without_scheduler_config(self, model_config, mock_tokenizer):
        processor = InputProcessor(model_config, tokenizer=mock_tokenizer)
        assert processor.model_config is model_config
        assert processor.scheduler_config is None

    def test_init_with_scheduler_config(self, model_config, scheduler_config, mock_tokenizer):
        processor = InputProcessor(model_config, scheduler_config, tokenizer=mock_tokenizer)
        assert processor.scheduler_config is scheduler_config


class TestProcessInputText:
    def test_process_text_prompt(self, model_config, mock_tokenizer):
        processor = InputProcessor(model_config, tokenizer=mock_tokenizer)
        sampling_params = SamplingParams()
        req = processor.process_input("req-1", "hello", sampling_params)

        assert req.request_id.startswith("req-1-")
        assert req.prompt == "hello"
        assert req.prompt_token_ids == [1, 2, 3, 4, 5]
        assert req.sampling_params is sampling_params
        assert req.sampling_params.eos_token_id is not None

    def test_process_token_ids_prompt(self, model_config, mock_tokenizer):
        processor = InputProcessor(model_config, tokenizer=mock_tokenizer)
        sampling_params = SamplingParams()
        req = processor.process_input("req-2", [10, 20, 30], sampling_params)

        assert req.prompt_token_ids == [10, 20, 30]
        mock_tokenizer.encode.assert_not_called()


class TestProcessInputValidation:
    def test_empty_request_id_raises(self, model_config, mock_tokenizer):
        processor = InputProcessor(model_config, tokenizer=mock_tokenizer)
        with pytest.raises(ValueError, match="request_id cannot be empty"):
            processor.process_input("", "hello", SamplingParams())

    def test_empty_text_prompt_raises(self, model_config, mock_tokenizer):
        processor = InputProcessor(model_config, tokenizer=mock_tokenizer)
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            processor.process_input("req-1", "", SamplingParams())

    def test_empty_token_ids_raises(self, model_config):
        processor = InputProcessor(model_config)
        with pytest.raises(ValueError, match="prompt_token_ids cannot be empty"):
            processor.process_input("req-1", [], SamplingParams())

    def test_text_prompt_without_tokenizer_raises(self, model_config):
        processor = InputProcessor(model_config)
        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            processor.process_input("req-1", "hello", SamplingParams())


class TestProcessInputTruncation:
    def test_truncation_by_max_model_len(self, mock_tokenizer):
        mc = ModelConfig(model="/fake.pth", max_model_len=3)
        processor = InputProcessor(mc, tokenizer=mock_tokenizer)
        sampling_params = SamplingParams()
        req = processor.process_input("req-1", "hello world", sampling_params)
        assert len(req.prompt_token_ids) <= 3

    def test_truncation_by_scheduler_config(self, mock_tokenizer):
        mc = ModelConfig(model="/fake.pth")
        sc = SchedulerConfig(max_num_batched_tokens=4)
        processor = InputProcessor(mc, sc, tokenizer=mock_tokenizer)
        sampling_params = SamplingParams()
        req = processor.process_input("req-1", "hello world", sampling_params)
        assert len(req.prompt_token_ids) <= 4


class TestEosTokenIdInjection:
    def test_eos_from_tokenizer(self, model_config):
        tok = MagicMock()
        tok.encode = MagicMock(return_value=[1])
        tok.eos_token_id = 107
        processor = InputProcessor(model_config, tokenizer=tok)
        sp = SamplingParams()
        processor.process_input("req-1", "a", sp)
        assert sp.eos_token_id == 107

    def test_eos_from_scheduler_config(self, model_config):
        tok = MagicMock()
        tok.encode = MagicMock(return_value=[1])
        tok.eos_token_id = None
        sc = SchedulerConfig(eos_token_id=42)
        processor = InputProcessor(model_config, sc, tokenizer=tok)
        sp = SamplingParams()
        processor.process_input("req-1", "a", sp)
        assert sp.eos_token_id == 42

    def test_eos_fallback_to_zero(self, model_config):
        processor = InputProcessor(model_config)
        sp = SamplingParams()
        processor.process_input("req-1", [1, 2], sp)
        assert sp.eos_token_id == 0

    def test_eos_not_overwritten_if_set(self, model_config):
        tok = MagicMock()
        tok.encode = MagicMock(return_value=[1])
        tok.eos_token_id = 100
        processor = InputProcessor(model_config, tokenizer=tok)
        sp = SamplingParams()
        sp.eos_token_id = 99
        processor.process_input("req-1", "a", sp)
        assert sp.eos_token_id == 99

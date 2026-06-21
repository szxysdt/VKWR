"""Tests for GPUModelRunner._is_uniform_decode detection logic."""

from unittest.mock import MagicMock

import torch

from vkwr.engine.request import SamplingParams
from vkwr.scheduler.output import RequestRunData, SchedulerOutput
from vkwr.worker.gpu_model_runner import GPUModelRunner

# ── Helpers ───────────────────────────────────────────────────────────


def _make_runner_config(max_num_seqs: int = 8) -> MagicMock:
    """Create a minimal mocked config for GPUModelRunner."""
    config = MagicMock()
    config.model_config.dtype = "float16"
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = 2048
    return config


def _make_request_run_data(req_id: str, is_decode: bool, input_token_ids: list[int]) -> RequestRunData:
    """Create a RequestRunData instance."""
    return RequestRunData(
        request_id=req_id,
        prompt_token_ids=input_token_ids,
        num_computed_tokens=len(input_token_ids),
        num_tokens=len(input_token_ids),
        sampling_params=SamplingParams(),
        input_token_ids=input_token_ids,
        is_decode=is_decode,
    )


def _make_scheduler_output(req_configs: list[dict]) -> SchedulerOutput:
    """Build a SchedulerOutput from a list of request configuration dicts.

    Each dict should contain:
        req_id: str
        is_decode: bool
        num_tokens: int  (tokens scheduled for this request)
        input_token_ids: list[int]
    """
    scheduled_req_ids = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    request_data = {}

    for cfg in req_configs:
        req_id = cfg["req_id"]
        is_decode = cfg["is_decode"]
        num_tokens = cfg["num_tokens"]
        input_token_ids = cfg["input_token_ids"]

        scheduled_req_ids.append(req_id)
        num_scheduled_tokens[req_id] = num_tokens
        total_num_scheduled_tokens += num_tokens
        request_data[req_id] = _make_request_run_data(req_id, is_decode, input_token_ids)

    return SchedulerOutput(
        scheduled_req_ids=scheduled_req_ids,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        finished_req_ids=set(),
        request_data=request_data,
    )


# ── Tests ─────────────────────────────────────────────────────────────


class TestIsUniformDecode:
    """Test GPUModelRunner._is_uniform_decode method."""

    def test_uniform_decode_single_request(self):
        """Single decode request with 1 token returns True."""
        config = _make_runner_config()
        runner = GPUModelRunner(config, torch.device("cpu"))

        scheduler_output = _make_scheduler_output(
            req_configs=[
                {"req_id": "req-1", "is_decode": True, "num_tokens": 1, "input_token_ids": [1]},
            ]
        )

        assert runner._is_uniform_decode(scheduler_output) is True

    def test_uniform_decode_multiple_requests(self):
        """Four decode requests each with 1 token returns True."""
        config = _make_runner_config()
        runner = GPUModelRunner(config, torch.device("cpu"))

        scheduler_output = _make_scheduler_output(
            req_configs=[
                {"req_id": "req-1", "is_decode": True, "num_tokens": 1, "input_token_ids": [1]},
                {"req_id": "req-2", "is_decode": True, "num_tokens": 1, "input_token_ids": [2]},
                {"req_id": "req-3", "is_decode": True, "num_tokens": 1, "input_token_ids": [3]},
                {"req_id": "req-4", "is_decode": True, "num_tokens": 1, "input_token_ids": [4]},
            ]
        )

        assert runner._is_uniform_decode(scheduler_output) is True

    def test_not_uniform_prefill_present(self):
        """Mix of decode and prefill requests returns False."""
        config = _make_runner_config()
        runner = GPUModelRunner(config, torch.device("cpu"))

        scheduler_output = _make_scheduler_output(
            req_configs=[
                {"req_id": "req-1", "is_decode": True, "num_tokens": 1, "input_token_ids": [1]},
                {"req_id": "req-2", "is_decode": False, "num_tokens": 1, "input_token_ids": [2]},
            ]
        )

        assert runner._is_uniform_decode(scheduler_output) is False

    def test_not_uniform_multi_token(self):
        """A request with 2 tokens (chunked prefill) returns False."""
        config = _make_runner_config()
        runner = GPUModelRunner(config, torch.device("cpu"))

        scheduler_output = _make_scheduler_output(
            req_configs=[
                {"req_id": "req-1", "is_decode": True, "num_tokens": 1, "input_token_ids": [1]},
                {"req_id": "req-2", "is_decode": True, "num_tokens": 2, "input_token_ids": [2, 3]},
            ]
        )

        assert runner._is_uniform_decode(scheduler_output) is False

    def test_not_uniform_zero_tokens(self):
        """Zero scheduled tokens returns False."""
        config = _make_runner_config()
        runner = GPUModelRunner(config, torch.device("cpu"))

        scheduler_output = SchedulerOutput(
            scheduled_req_ids=[],
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=set(),
            request_data={},
        )

        assert runner._is_uniform_decode(scheduler_output) is False

    def test_not_uniform_all_not_decode(self):
        """All requests present but is_decode=False returns False."""
        config = _make_runner_config()
        runner = GPUModelRunner(config, torch.device("cpu"))

        scheduler_output = _make_scheduler_output(
            req_configs=[
                {"req_id": "req-1", "is_decode": False, "num_tokens": 1, "input_token_ids": [1]},
                {"req_id": "req-2", "is_decode": False, "num_tokens": 1, "input_token_ids": [2]},
                {"req_id": "req-3", "is_decode": False, "num_tokens": 1, "input_token_ids": [3]},
            ]
        )

        assert runner._is_uniform_decode(scheduler_output) is False

    def test_uniform_decode_max_batch(self):
        """max_num_seqs requests all decoding 1 token returns True."""
        max_seqs = 8
        config = _make_runner_config(max_num_seqs=max_seqs)
        runner = GPUModelRunner(config, torch.device("cpu"))

        req_configs = [{"req_id": f"req-{i}", "is_decode": True, "num_tokens": 1, "input_token_ids": [i]} for i in range(max_seqs)]
        scheduler_output = _make_scheduler_output(req_configs)

        assert runner._is_uniform_decode(scheduler_output) is True

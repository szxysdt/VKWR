"""Tests for GPUModelRunner static buffer allocation and _prepare_input_ids."""

from unittest.mock import MagicMock

import torch

from vkwr.engine.request import SamplingParams
from vkwr.scheduler.output import RequestRunData, SchedulerOutput


def _make_runner(max_num_seqs=4, max_num_batched_tokens=128, L=32, C=2048):
    """Create a GPUModelRunner with mocked config and manually allocated CPU buffers."""
    from vkwr.worker.gpu_model_runner import GPUModelRunner

    config = MagicMock()
    config.model_config.dtype = "float16"
    config.model_config.model = "/fake/model.pth"
    config.model_config.load_format = "auto"
    config.model_config.seed = 42
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens

    runner = GPUModelRunner(config, torch.device("cpu"))
    runner._input_ids = torch.empty(max_num_batched_tokens, dtype=torch.long, device="cpu")
    runner._query_start_loc = torch.empty(max_num_seqs + 1, dtype=torch.int32, device="cpu")
    runner._input_ids_host = torch.empty(max_num_batched_tokens, dtype=torch.long)
    runner._query_start_loc_host = torch.empty(max_num_seqs + 1, dtype=torch.int32)
    runner._uniform_query_start_loc = torch.empty(max_num_seqs + 1, dtype=torch.int32, device="cpu")
    runner._uniform_req_id = torch.empty(max_num_seqs, dtype=torch.int32, device="cpu")
    runner._decode_state_shift = torch.empty((L, 2, max_num_seqs, C), dtype=torch.float16, device="cpu")
    return runner


def _make_request_run_data(req_id, input_token_ids, num_tokens, is_decode=False):
    """Build a RequestRunData for a single request."""
    return RequestRunData(
        request_id=req_id,
        prompt_token_ids=input_token_ids,
        num_computed_tokens=0,
        num_tokens=num_tokens,
        sampling_params=SamplingParams(),
        input_token_ids=input_token_ids,
        state=None,
        is_decode=is_decode,
        is_last_prefill=not is_decode,
    )


def _make_scheduler_output(requests):
    """Build SchedulerOutput from list of (req_id, input_token_ids, num_tokens)."""
    scheduled_req_ids = []
    request_data = {}
    num_scheduled_tokens = {}
    total = 0
    for req_id, input_token_ids, num_tokens, *rest in requests:
        is_decode = rest[0] if rest else False
        scheduled_req_ids.append(req_id)
        num_scheduled_tokens[req_id] = num_tokens
        total += num_tokens
        request_data[req_id] = _make_request_run_data(req_id, input_token_ids, num_tokens, is_decode)
    return SchedulerOutput(
        scheduled_req_ids=scheduled_req_ids,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total,
        finished_req_ids=set(),
        request_data=request_data,
    )


class TestPrepareInputIds:
    """Test _prepare_input_ids correctness with mocked buffers."""

    def test_single_request(self):
        runner = _make_runner()
        scheduler_output = _make_scheduler_output([("r1", [1, 2, 3], 3)])
        input_ids, query_start_loc, max_t = runner._prepare_input_ids(scheduler_output)
        assert input_ids.tolist() == [1, 2, 3]
        assert query_start_loc.tolist() == [0, 3]
        assert max_t == 3

    def test_multiple_requests(self):
        runner = _make_runner()
        scheduler_output = _make_scheduler_output(
            [
                ("r1", [1, 2], 2),
                ("r2", [3, 4, 5], 3),
            ]
        )
        input_ids, query_start_loc, max_t = runner._prepare_input_ids(scheduler_output)
        assert input_ids.tolist() == [1, 2, 3, 4, 5]
        assert query_start_loc.tolist() == [0, 2, 5]
        assert max_t == 3

    def test_decode_single_token(self):
        runner = _make_runner()
        scheduler_output = _make_scheduler_output(
            [
                ("r1", [10], 1, True),
                ("r2", [20], 1, True),
            ]
        )
        input_ids, query_start_loc, max_t = runner._prepare_input_ids(scheduler_output)
        assert input_ids.tolist() == [10, 20]
        assert query_start_loc.tolist() == [0, 1, 2]
        assert max_t == 1

    def test_buffer_reuse(self):
        runner = _make_runner()
        first = _make_scheduler_output([("r1", [1, 2, 3], 3)])
        input_ids1, query_start_loc1, max_t1 = runner._prepare_input_ids(first)
        assert input_ids1.tolist() == [1, 2, 3]
        assert query_start_loc1.tolist() == [0, 3]
        assert max_t1 == 3

        second = _make_scheduler_output([("r2", [10, 20], 2)])
        input_ids2, query_start_loc2, max_t2 = runner._prepare_input_ids(second)
        assert input_ids2.tolist() == [10, 20]
        assert query_start_loc2.tolist() == [0, 2]
        assert max_t2 == 2


class TestBufferShapes:
    """Verify static buffer shapes after manual allocation."""

    def test_buffer_shapes_after_init(self):
        max_num_seqs = 4
        max_num_batched_tokens = 128
        L, C = 32, 2048
        runner = _make_runner(max_num_seqs, max_num_batched_tokens, L, C)
        assert runner._input_ids.shape == (max_num_batched_tokens,)
        assert runner._query_start_loc.shape == (max_num_seqs + 1,)
        assert runner._uniform_query_start_loc.shape == (max_num_seqs + 1,)
        assert runner._uniform_req_id.shape == (max_num_seqs,)
        assert runner._decode_state_shift.shape == (L, 2, max_num_seqs, C)
        assert runner._input_ids.dtype == torch.long
        assert runner._query_start_loc.dtype == torch.int32
        assert runner._uniform_query_start_loc.dtype == torch.int32
        assert runner._uniform_req_id.dtype == torch.int32
        assert runner._decode_state_shift.dtype == torch.float16

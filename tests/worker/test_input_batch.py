"""Test for InputBatch."""

import pytest
import torch

from vkwr.engine.request import SamplingParams
from vkwr.scheduler.output import RequestRunData, SchedulerOutput
from vkwr.worker.gpu.input_batch import InputBatch


class TestInputBatch:
    @pytest.fixture
    def scheduler_output(self):
        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0

        run_data = RequestRunData(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3, 4, 5],
            num_computed_tokens=0,
            num_tokens=5,
            sampling_params=sp,
            input_token_ids=[1, 2, 3, 4, 5],
            is_decode=False,
            is_last_prefill=True,
        )

        return SchedulerOutput(
            scheduled_req_ids=["req-1"],
            num_scheduled_tokens={"req-1": 5},
            total_num_scheduled_tokens=5,
            finished_req_ids=set(),
            request_data={"req-1": run_data},
        )

    def test_from_scheduler_output_basic(self, scheduler_output):
        state_shift = torch.zeros((4, 2, 8, 256))
        state_wkv = torch.zeros((4, 8, 16, 64, 64))
        state_elapsed = torch.zeros((8,), dtype=torch.int32)

        batch = InputBatch.from_scheduler_output(
            scheduler_output,
            device=torch.device("cpu"),
            state_shift=state_shift,
            state_wkv=state_wkv,
            state_elapsed=state_elapsed,
        )

        assert batch.batch_size == 1
        assert batch.seq_len == 5
        assert batch.input_ids.shape == (1, 5)
        assert batch.input_ids[0].tolist() == [1, 2, 3, 4, 5]
        assert batch.request_ids == ["req-1"]
        assert batch.is_last_prefill is True
        assert batch.is_decode is False

    def test_state_slicing(self, scheduler_output):
        state_shift = torch.zeros((4, 2, 8, 256))
        state_wkv = torch.zeros((4, 8, 16, 64, 64))
        state_elapsed = torch.zeros((8,), dtype=torch.int32)

        batch = InputBatch.from_scheduler_output(
            scheduler_output,
            device=torch.device("cpu"),
            state_shift=state_shift,
            state_wkv=state_wkv,
            state_elapsed=state_elapsed,
        )

        assert batch.state[0].shape == (4, 2, 1, 256)
        assert batch.state[1].shape == (4, 1, 16, 64, 64)
        assert batch.state[2].shape == (1,)

    def test_decode_flag(self):
        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0

        run_data = RequestRunData(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3],
            num_computed_tokens=3,
            num_tokens=1,
            sampling_params=sp,
            input_token_ids=[42],
            is_decode=True,
            is_last_prefill=False,
        )

        scheduler_output = SchedulerOutput(
            scheduled_req_ids=["req-1"],
            num_scheduled_tokens={"req-1": 1},
            total_num_scheduled_tokens=1,
            finished_req_ids=set(),
            request_data={"req-1": run_data},
        )

        state_shift = torch.zeros((4, 2, 8, 256))
        state_wkv = torch.zeros((4, 8, 16, 64, 64))
        state_elapsed = torch.zeros((8,), dtype=torch.int32)

        batch = InputBatch.from_scheduler_output(
            scheduler_output,
            device=torch.device("cpu"),
            state_shift=state_shift,
            state_wkv=state_wkv,
            state_elapsed=state_elapsed,
        )

        assert batch.is_decode is True
        assert batch.input_ids[0].tolist() == [42]

    def test_ragged_input_raises(self):
        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0

        run_data1 = RequestRunData(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3],
            num_computed_tokens=0,
            num_tokens=3,
            sampling_params=sp,
            input_token_ids=[1, 2, 3],
            is_decode=False,
            is_last_prefill=False,
        )
        run_data2 = RequestRunData(
            request_id="req-2",
            prompt_token_ids=[4, 5],
            num_computed_tokens=0,
            num_tokens=2,
            sampling_params=sp,
            input_token_ids=[4, 5],
            is_decode=False,
            is_last_prefill=False,
        )

        scheduler_output = SchedulerOutput(
            scheduled_req_ids=["req-1", "req-2"],
            num_scheduled_tokens={"req-1": 3, "req-2": 2},
            total_num_scheduled_tokens=5,
            finished_req_ids=set(),
            request_data={"req-1": run_data1, "req-2": run_data2},
        )

        state_shift = torch.zeros((4, 2, 8, 256))
        state_wkv = torch.zeros((4, 8, 16, 64, 64))
        state_elapsed = torch.zeros((8,), dtype=torch.int32)

        with pytest.raises(ValueError, match="seq_len"):
            InputBatch.from_scheduler_output(
                scheduler_output,
                device=torch.device("cpu"),
                state_shift=state_shift,
                state_wkv=state_wkv,
                state_elapsed=state_elapsed,
            )

"""Test for GPUModelRunner."""

from unittest.mock import MagicMock

import pytest
import torch

from vkwr.engine.request import SamplingParams
from vkwr.scheduler.output import RequestRunData, SchedulerOutput


class TestGPUModelRunner:
    """Test GPUModelRunner without actual CUDA (mocked model)."""

    @pytest.fixture
    def runner_config(self):
        """Create a minimal VkwrConfig-like object for testing."""
        config = MagicMock()
        config.model_config.dtype = "float16"
        config.model_config.model = "/fake/model.pth"
        config.model_config.load_format = "auto"
        config.model_config.seed = 42
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 2048
        config.worker_config.device = "cuda"
        config.worker_config.gpu_memory_utilization = 0.92
        return config

    @pytest.fixture
    def scheduler_output(self):
        """Create a SchedulerOutput for testing."""
        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0

        run_data = RequestRunData(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3, 4, 5],
            start_pos=0,
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

    def test_prepare_input_ids_single_request(self, runner_config, scheduler_output):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        input_ids = runner._prepare_input_ids(scheduler_output)
        assert input_ids.shape == (1, 5)
        assert input_ids[0].tolist() == [1, 2, 3, 4, 5]

    def test_prepare_input_ids_multiple_same_length(self, runner_config):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0

        run_data1 = RequestRunData(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3],
            start_pos=0,
            num_tokens=3,
            sampling_params=sp,
            input_token_ids=[1, 2, 3],
            is_decode=False,
            is_last_prefill=True,
        )
        run_data2 = RequestRunData(
            request_id="req-2",
            prompt_token_ids=[4, 5, 6],
            start_pos=0,
            num_tokens=3,
            sampling_params=sp,
            input_token_ids=[4, 5, 6],
            is_decode=False,
            is_last_prefill=True,
        )

        scheduler_output = SchedulerOutput(
            scheduled_req_ids=["req-1", "req-2"],
            num_scheduled_tokens={"req-1": 3, "req-2": 3},
            total_num_scheduled_tokens=6,
            finished_req_ids=set(),
            request_data={"req-1": run_data1, "req-2": run_data2},
        )

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        input_ids = runner._prepare_input_ids(scheduler_output)
        assert input_ids.shape == (2, 3)

    def test_prepare_input_ids_ragged_raises(self, runner_config):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0

        run_data1 = RequestRunData(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3],
            start_pos=0,
            num_tokens=3,
            sampling_params=sp,
            input_token_ids=[1, 2, 3],
            is_decode=False,
            is_last_prefill=False,
        )
        run_data2 = RequestRunData(
            request_id="req-2",
            prompt_token_ids=[4, 5],
            start_pos=0,
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

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        with pytest.raises(ValueError, match="seq_len"):
            runner._prepare_input_ids(scheduler_output)

    def test_prepare_state_raises_before_load(self, runner_config, scheduler_output):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        with pytest.raises(RuntimeError, match="State buffers not allocated"):
            runner._prepare_state(scheduler_output)

    def test_execute_model_raises_before_load(self, runner_config, scheduler_output):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.execute_model(scheduler_output)

    def test_warmup_raises_before_load(self, runner_config):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.warmup()

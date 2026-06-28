from unittest.mock import patch

import pytest

from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.engine.outputs import ModelRunnerOutput
from vkwr.engine.request import SamplingParams, VkwrRequest
from vkwr.scheduler.output import RequestRunData, SchedulerOutput

# ── Helpers ───────────────────────────────────────────────────────────


def _make_config() -> VkwrConfig:
    return VkwrConfig(
        model_config=ModelConfig(model="fake-model", max_model_len=8192),
    )


def _make_request(request_id="req-1", prompt_token_ids=None, sampling_params=None):
    return VkwrRequest(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
        sampling_params=sampling_params or SamplingParams(),
    )


def _make_scheduler_output(
    req_id="req-1",
    input_token_ids=None,
    is_decode=False,
    is_last_prefill=False,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_req_ids=[req_id],
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=set(),
        request_data={
            req_id: RequestRunData(
                request_id=req_id,
                prompt_token_ids=[1, 2, 3],
                num_computed_tokens=0,
                num_tokens=1,
                sampling_params=SamplingParams(),
                input_token_ids=input_token_ids or [1],
                state=None,
                is_decode=is_decode,
                is_last_prefill=is_last_prefill,
            )
        },
    )


def _make_model_output(req_id="req-1", token_id=42):
    return ModelRunnerOutput(
        sampled_token_ids={req_id: [token_id]},
        sampled_logprobs=None,
        logits=None,
    )


# ── ExecutorInterface ────────────────────────────────────────────────


class TestExecutorInterface:
    def test_get_class_default_returns_uniproc(self):
        from vkwr.executor.abstract import ExecutorInterface

        config = _make_config()
        cls = ExecutorInterface.get_class(config)
        assert cls.__name__ == "UniprocExecutor"

    def test_get_class_ray(self):
        from vkwr.executor.abstract import ExecutorInterface

        config = VkwrConfig(
            model_config=ModelConfig(model="m"),
        )
        config.parallel_config.distributed_executor_backend = "ray"
        cls = ExecutorInterface.get_class(config)
        assert cls.__name__ == "RayExecutor"

    def test_get_class_mp(self):
        from vkwr.executor.abstract import ExecutorInterface

        config = VkwrConfig(
            model_config=ModelConfig(model="m"),
        )
        config.parallel_config.distributed_executor_backend = "mp"
        cls = ExecutorInterface.get_class(config)
        assert cls.__name__ == "MultiprocExecutor"


# ── UniprocExecutor ──────────────────────────────────────────────────


class TestUniprocExecutor:
    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_initialize_creates_worker(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        config = _make_config()
        executor = UniprocExecutor(config)
        assert executor.worker is None

        executor.initialize()
        MockGPUWorker.assert_called_once_with(config)
        MockGPUWorker.return_value.init_device.assert_called_once_with(None)

    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_load_model_before_init_raises(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        executor = UniprocExecutor(_make_config())
        with pytest.raises(RuntimeError, match="not initialized"):
            executor.load_model()

    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_load_model(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        config = _make_config()
        executor = UniprocExecutor(config)
        executor.initialize()
        executor.load_model()
        MockGPUWorker.return_value.load_model.assert_called_once()

    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_execute_model_sync(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        config = _make_config()
        executor = UniprocExecutor(config)
        executor.initialize()

        scheduler_output = _make_scheduler_output()
        expected_output = _make_model_output()
        MockGPUWorker.return_value.execute_model.return_value = expected_output

        result = executor.execute_model(scheduler_output)
        assert result == expected_output

    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_execute_model_nonblock_returns_future(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        config = _make_config()
        executor = UniprocExecutor(config)
        executor.initialize()

        scheduler_output = _make_scheduler_output()
        expected_output = _make_model_output()
        MockGPUWorker.return_value.execute_model.return_value = expected_output

        result = executor.execute_model(scheduler_output, non_block=True)
        from concurrent.futures import Future

        assert isinstance(result, Future)
        assert result.result() == expected_output

    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_execute_model_before_init_raises(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        executor = UniprocExecutor(_make_config())
        with pytest.raises(RuntimeError, match="not initialized"):
            executor.execute_model(_make_scheduler_output())

    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_sample_tokens(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        config = _make_config()
        executor = UniprocExecutor(config)
        executor.initialize()

        model_runner_output = ModelRunnerOutput(
            sampled_token_ids={},
            sampled_logprobs=None,
            logits=None,
        )
        scheduler_output = _make_scheduler_output()
        expected = _make_model_output()
        MockGPUWorker.return_value.sample_tokens.return_value = expected

        result = executor.sample_tokens(model_runner_output, scheduler_output)
        assert result == expected

    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_determine_available_memory(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        config = _make_config()
        executor = UniprocExecutor(config)
        executor.initialize()
        MockGPUWorker.return_value.determine_available_memory.return_value = 1024

        result = executor.determine_available_memory()
        assert result == 1024

    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_compile_or_warm_up_model(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        config = _make_config()
        executor = UniprocExecutor(config)
        executor.initialize()
        executor.compile_or_warm_up_model()
        MockGPUWorker.return_value.compile_or_warm_up_model.assert_called_once()

    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_condense_calls_runner(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        config = _make_config()
        executor = UniprocExecutor(config)
        executor.initialize()

        moves = [(2, 0), (3, 1)]
        executor.condense(moves)

        MockGPUWorker.return_value.model_runner._condense_slots.assert_called_once_with(moves)

    @patch("vkwr.executor.uniproc_executor.GPUWorker")
    def test_condense_empty(self, MockGPUWorker):
        from vkwr.executor.uniproc_executor import UniprocExecutor

        config = _make_config()
        executor = UniprocExecutor(config)
        executor.initialize()

        executor.condense([])

        MockGPUWorker.return_value.model_runner._condense_slots.assert_not_called()

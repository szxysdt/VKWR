from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.config.scheduler import SchedulerConfig
from vkwr.engine.core import EngineCore
from vkwr.engine.outputs import EngineCoreOutput, EngineCoreOutputs, ModelRunnerOutput
from vkwr.engine.request import SamplingParams
from vkwr.scheduler.output import RequestRunData, SchedulerOutput

# ── Helpers ───────────────────────────────────────────────────────────


def _make_config(enable_async: bool = False, batch_queue_size: int = 2) -> VkwrConfig:
    sched_config = SchedulerConfig(
        enable_async_scheduling=enable_async,
        batch_queue_size=batch_queue_size,
    )
    return VkwrConfig(
        model_config=ModelConfig(model="fake-model", max_model_len=8192),
        scheduler_config=sched_config,
    )


def _make_scheduler_output(
    req_id: str = "req-1",
    input_token_ids: list[int] | None = None,
    finished_req_ids: set[str] | None = None,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_req_ids=[req_id],
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        finished_req_ids=finished_req_ids or set(),
        request_data={
            req_id: RequestRunData(
                request_id=req_id,
                prompt_token_ids=[1, 2, 3],
                num_computed_tokens=0,
                num_tokens=1,
                sampling_params=SamplingParams(),
                input_token_ids=input_token_ids or [1],
                state=None,
                is_decode=False,
                is_last_prefill=False,
            )
        },
    )


def _make_model_output(req_id: str = "req-1", token_id: int = 42) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        sampled_token_ids={req_id: [token_id]},
    )


def _make_engine_outputs(req_id: str = "req-1", token_id: int = 42) -> dict:
    return {
        req_id: EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=[token_id],
                )
            ],
        ),
    }


# ── EngineCore async scheduling init ─────────────────────────────────


class TestEngineCoreAsyncInit:
    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_fn_is_batch_queue_when_async_enabled(self, _mock: MagicMock) -> None:
        """With enable_async_scheduling=True, step_fn == step_with_batch_queue."""
        _mock.return_value = MagicMock()
        config = _make_config(enable_async=True, batch_queue_size=3)
        core = EngineCore(config)

        assert core.step_fn.__name__ == "step_with_batch_queue"
        assert core.batch_queue is not None
        assert core.batch_queue_size == 3

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_fn_is_normal_step_when_sync(self, _mock: MagicMock) -> None:
        """Default: step_fn == step."""
        _mock.return_value = MagicMock()
        config = _make_config(enable_async=False)
        core = EngineCore(config)

        assert core.step_fn.__name__ == "step"
        assert core.batch_queue is None
        assert core.batch_queue_size == 0


# ── step_with_batch_queue ────────────────────────────────────────────


class TestStepWithBatchQueue:
    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_with_batch_queue_returns_none_signal(self, _mock: MagicMock) -> None:
        """When batch_queue is filling (pending future, not full), returns (None, True)."""
        _mock.return_value = MagicMock()
        config = _make_config(enable_async=True, batch_queue_size=3)
        core = EngineCore(config)

        scheduler_output = _make_scheduler_output()
        pending_future: Future = Future()

        core.scheduler.has_requests = MagicMock(return_value=True)
        core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        core.model_executor.execute_model = MagicMock(return_value=pending_future)

        outputs_dict, model_executed = core.step_with_batch_queue()

        assert outputs_dict is None
        assert model_executed is True
        core.model_executor.execute_model.assert_called_once_with(scheduler_output, non_block=True)

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_with_batch_queue_returns_output(self, _mock: MagicMock) -> None:
        """When batch_queue has a done future, first step processes it, second schedules."""
        _mock.return_value = MagicMock()
        config = _make_config(enable_async=True, batch_queue_size=2)
        core = EngineCore(config)

        scheduler_output = _make_scheduler_output()
        model_output = _make_model_output()
        engine_outputs = _make_engine_outputs()

        done_future: Future = Future()
        done_future.set_result(model_output)
        core.batch_queue.appendleft((done_future, scheduler_output))

        new_future: Future = Future()

        core.scheduler.has_requests = MagicMock(return_value=True)
        core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        core.model_executor.execute_model = MagicMock(return_value=new_future)
        core.scheduler.update_from_output = MagicMock(return_value=engine_outputs)

        outputs_dict, model_executed = core.step_with_batch_queue()

        assert model_executed is True
        assert 0 in outputs_dict
        result = outputs_dict[0]
        assert len(result.outputs) == 1
        assert result.outputs[0].new_token_ids == [42]

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_with_batch_queue_drains_existing(self, _mock: MagicMock) -> None:
        """Drains existing queue when no new requests."""
        _mock.return_value = MagicMock()
        config = _make_config(enable_async=True, batch_queue_size=2)
        core = EngineCore(config)

        scheduler_output = _make_scheduler_output("req-A")
        model_output = _make_model_output("req-A", 10)

        future_a: Future = Future()
        future_a.set_result(model_output)

        core.batch_queue.appendleft((future_a, scheduler_output))

        core.scheduler.has_requests = MagicMock(return_value=False)
        core.scheduler.update_from_output = MagicMock(return_value=_make_engine_outputs("req-A", 10))

        outputs_dict, model_executed = core.step_with_batch_queue()

        assert 0 in outputs_dict
        assert outputs_dict[0].outputs[0].new_token_ids == [10]
        assert model_executed is True
        assert len(core.batch_queue) == 0


# ── abort_requests and _process_aborts_queue ────────────────────────


class TestAbortRequests:
    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_abort_requests(self, _mock: MagicMock) -> None:
        """abort_requests() calls scheduler.finish_requests with FINISHED_ABORTED."""
        _mock.return_value = MagicMock()
        config = _make_config()
        core = EngineCore(config)
        core.scheduler.finish_requests = MagicMock()

        core.abort_requests(["req-1", "req-2"])

        core.scheduler.finish_requests.assert_called_once_with({"req-1", "req-2"})


class TestProcessAbortsQueue:
    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_process_aborts_queue(self, _mock: MagicMock) -> None:
        """_process_aborts_queue drains aborts_queue, batches into single call."""
        _mock.return_value = MagicMock()
        config = _make_config()
        core = EngineCore(config)
        core.scheduler.finish_requests = MagicMock()

        core.aborts_queue.put(["req-1"])
        core.aborts_queue.put(["req-2", "req-3"])
        core._process_aborts_queue()

        core.scheduler.finish_requests.assert_called_once_with({"req-1", "req-2", "req-3"})

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_process_aborts_queue_empty_noop(self, _mock: MagicMock) -> None:
        """Empty aborts_queue results in no finish_requests call."""
        _mock.return_value = MagicMock()
        config = _make_config()
        core = EngineCore(config)
        core.scheduler.finish_requests = MagicMock()

        core._process_aborts_queue()

        core.scheduler.finish_requests.assert_not_called()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_process_aborts_queue_single_entry(self, _mock: MagicMock) -> None:
        """Single abort entry drained correctly."""
        _mock.return_value = MagicMock()
        config = _make_config()
        core = EngineCore(config)
        core.scheduler.finish_requests = MagicMock()

        core.aborts_queue.put(["req-1"])
        core._process_aborts_queue()

        core.scheduler.finish_requests.assert_called_once_with({"req-1"})

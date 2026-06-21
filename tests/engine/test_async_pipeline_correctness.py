from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.config.scheduler import SchedulerConfig
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


def _make_scheduler_output(req_id: str = "req-1") -> SchedulerOutput:
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
                input_token_ids=[1],
                state=None,
                is_decode=False,
                is_last_prefill=False,
            )
        },
    )


def _make_model_output(token_id: int = 42) -> ModelRunnerOutput:
    return ModelRunnerOutput(sampled_token_ids={"req-1": [token_id]})


def _make_engine_outputs(token_id: int = 42) -> dict:
    return {
        "req-1": EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="req-1",
                    new_token_ids=[token_id],
                )
            ],
        ),
    }


# ── Async pipeline correctness ──────────────────────────────────────


class TestAsyncPipelineCorrectness:
    """Integration-level tests for async pipeline correctness.

    Verifies:
    - Async mode output matches sync mode output
    - batch_queue full correctly blocks new scheduling
    - (None, True) signal works correctly in background loop
    """

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_async_output_matches_sync_output(self, _mock: MagicMock) -> None:
        """Async mode output should match sync mode output for same input."""
        from vkwr.engine.core import EngineCore

        _mock.return_value = MagicMock()

        # Sync mode
        sync_config = _make_config(enable_async=False)
        sync_core = EngineCore(sync_config)

        scheduler_output = _make_scheduler_output()
        model_output = _make_model_output(token_id=42)
        engine_outputs = _make_engine_outputs(token_id=42)

        sync_core.scheduler.has_requests = MagicMock(return_value=True)
        sync_core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        sync_core.model_executor.execute_model = MagicMock(return_value=model_output)
        sync_core.scheduler.update_from_output = MagicMock(return_value=engine_outputs)
        sync_core._initialized = True

        sync_outputs, sync_executed = sync_core.step()

        # Async mode with immediately-done future pre-filled in queue
        async_config = _make_config(enable_async=True, batch_queue_size=2)
        async_core = EngineCore(async_config)

        done_future: Future = Future()
        done_future.set_result(model_output)
        core_sched_output = _make_scheduler_output()
        async_core.batch_queue.appendleft((done_future, core_sched_output))

        pending_future: Future = Future()

        async_core.scheduler.has_requests = MagicMock(return_value=True)
        async_core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        async_core.model_executor.execute_model = MagicMock(return_value=pending_future)
        async_core.scheduler.update_from_output = MagicMock(return_value=engine_outputs)

        async_outputs, async_executed = async_core.step_with_batch_queue()

        # Both should produce the same output
        assert sync_executed is True
        assert async_executed is True
        assert 0 in sync_outputs
        assert 0 in async_outputs
        assert sync_outputs[0].outputs[0].new_token_ids == [42]
        assert async_outputs[0].outputs[0].new_token_ids == [42]

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_batch_queue_full_blocks_new_scheduling(self, _mock: MagicMock) -> None:
        """When batch_queue is full and future pending, should return (None, True)."""
        from vkwr.engine.core import EngineCore

        _mock.return_value = MagicMock()

        config = _make_config(enable_async=True, batch_queue_size=2)
        core = EngineCore(config)

        scheduler_output = _make_scheduler_output()

        # Pre-fill with a done future so drain path doesn't block
        done_future: Future = Future()
        done_future.set_result(_make_model_output())
        core.batch_queue.appendleft((done_future, scheduler_output))

        # step_with_batch_queue will add a pending future; queue becomes full
        pending_future: Future = Future()

        core.scheduler.has_requests = MagicMock(return_value=True)
        core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        core.model_executor.execute_model = MagicMock(return_value=pending_future)
        core.scheduler.update_from_output = MagicMock(return_value=_make_engine_outputs())

        outputs_dict, model_executed = core.step_with_batch_queue()

        # Should drain the done future, not block
        assert 0 in outputs_dict
        assert model_executed is True

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_none_true_signal_in_background_loop(self, _mock: MagicMock) -> None:
        """(None, True) signal allows background loop to continue scheduling without blocking."""
        from vkwr.engine.core import EngineCore

        _mock.return_value = MagicMock()

        config = _make_config(enable_async=True, batch_queue_size=3)
        core = EngineCore(config)

        scheduler_output = _make_scheduler_output()
        pending_future: Future = Future()

        core.scheduler.has_requests = MagicMock(return_value=True)
        core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        core.model_executor.execute_model = MagicMock(return_value=pending_future)

        # First call: pipeline filling, returns (None, True)
        outputs_dict, model_executed = core.step_with_batch_queue()
        assert outputs_dict is None
        assert model_executed is True

        # When background loop sees (None, True), it should NOT try to process outputs
        # but should continue the loop (signal means "I'm working, come back later")

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_pipeline_drains_in_order(self, _mock: MagicMock) -> None:
        """Batch queue drains in FIFO order (first submitted, first completed)."""
        from vkwr.engine.core import EngineCore

        _mock.return_value = MagicMock()

        config = _make_config(enable_async=True, batch_queue_size=3)
        core = EngineCore(config)

        scheduler_output_a = _make_scheduler_output("req-A")
        scheduler_output_b = _make_scheduler_output("req-B")

        future_a: Future = Future()
        future_b: Future = Future()
        future_a.set_result(_make_model_output(token_id=10))
        future_b.set_result(_make_model_output(token_id=20))

        # Pre-fill: B was submitted first (left of deque), then A (right)
        core.batch_queue.appendleft((future_b, scheduler_output_b))
        core.batch_queue.appendleft((future_a, scheduler_output_a))

        core.scheduler.has_requests = MagicMock(return_value=False)
        core.scheduler.update_from_output = MagicMock(
            side_effect=[
                _make_engine_outputs(token_id=10),
                _make_engine_outputs(token_id=20),
            ]
        )

        # First drain should get the rightmost (most recently added)
        out1, _ = core.step_with_batch_queue()
        assert 0 in out1

        # Second drain should get the remaining
        out2, _ = core.step_with_batch_queue()
        assert 0 in out2

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_abort_queue_processed_during_step(self, _mock: MagicMock) -> None:
        """Abort queue is processed during step, cleaning up aborted requests."""
        from vkwr.engine.core import EngineCore

        _mock.return_value = MagicMock()

        config = _make_config()
        core = EngineCore(config)
        core.scheduler.finish_requests = MagicMock()

        # Enqueue aborts before step
        core.aborts_queue.put(["abort-1", "abort-2"])

        scheduler_output = _make_scheduler_output()
        model_output = _make_model_output()
        engine_outputs = _make_engine_outputs()

        core.scheduler.has_requests = MagicMock(return_value=True)
        core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        core.model_executor.execute_model = MagicMock(return_value=model_output)
        core.scheduler.update_from_output = MagicMock(return_value=engine_outputs)
        core._initialized = True

        core.step()

        # _process_aborts_queue is called inside step()
        core.scheduler.finish_requests.assert_called_once_with({"abort-1", "abort-2"})

from unittest.mock import MagicMock, patch

import pytest

from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.engine.outputs import EngineCoreOutputs, ModelRunnerOutput
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
    finished_req_ids=None,
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


# ── EngineCore ───────────────────────────────────────────────────────


class TestEngineCoreInit:
    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_creates_executor_and_scheduler(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_cls = MagicMock()
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)

        MockGetClass.assert_called_once_with(config)
        assert mock_executor_cls.call_count == 1
        call_args = mock_executor_cls.call_args[0]
        assert call_args[0] is config
        assert not core._initialized

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_scheduler_exists(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_cls = MagicMock()
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        assert hasattr(core, "scheduler")


class TestEngineCoreInitialize:
    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_initialize_calls_executor_methods(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core.initialize()

        mock_executor_instance.initialize.assert_called_once()
        mock_executor_instance.load_model.assert_called_once()
        mock_executor_instance.compile_or_warm_up_model.assert_called_once()
        assert core._initialized

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_initialize_idempotent(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core.initialize()
        core.initialize()

        assert mock_executor_instance.initialize.call_count == 1
        assert mock_executor_instance.load_model.call_count == 1


class TestEngineCoreNotInitialized:
    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_add_request_raises_before_init(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_cls = MagicMock()
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)

        with pytest.raises(RuntimeError, match="not initialized"):
            core.add_request(_make_request())

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_raises_before_init(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_cls = MagicMock()
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)

        with pytest.raises(RuntimeError, match="not initialized"):
            core.step()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_abort_request_raises_before_init(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_cls = MagicMock()
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)

        core.abort_request("req-1")


class TestEngineCoreStep:
    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_no_requests_returns_empty(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core.scheduler.has_requests = MagicMock(return_value=False)
        core._initialized = True

        outputs_dict, model_executed = core.step()
        assert outputs_dict == {}
        assert model_executed is False

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_empty_schedule_returns_empty(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core.scheduler.has_requests = MagicMock(return_value=True)
        core.scheduler.schedule = MagicMock(
            return_value=SchedulerOutput(
                scheduled_req_ids=[],
                num_scheduled_tokens={},
                total_num_scheduled_tokens=0,
                finished_req_ids=set(),
                request_data={},
            )
        )
        core._initialized = True

        outputs_dict, model_executed = core.step()
        assert outputs_dict == {}
        assert model_executed is False

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_full_flow(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)

        scheduler_output = _make_scheduler_output()
        model_output = _make_model_output()
        from vkwr.engine.outputs import EngineCoreOutput

        engine_outputs = {
            "req-1": EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="req-1",
                        new_token_ids=[42],
                        new_logprobs=None,
                        finish_reason=None,
                    )
                ],
            ),
        }

        core.scheduler.has_requests = MagicMock(return_value=True)
        core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        core.model_executor.execute_model = MagicMock(return_value=model_output)
        core.model_executor.sample_tokens = MagicMock(return_value=model_output)
        core.scheduler.update_from_output = MagicMock(return_value=engine_outputs)
        core._initialized = True

        outputs_dict, model_executed = core.step()
        assert model_executed is True
        assert 0 in outputs_dict
        result = outputs_dict[0]
        assert len(result.outputs) == 1
        assert result.outputs[0].new_token_ids == [42]

        core.scheduler.schedule.assert_called_once()
        core.model_executor.execute_model.assert_called_once_with(scheduler_output)
        core.model_executor.sample_tokens.assert_called_once_with(model_output, scheduler_output)
        core.scheduler.update_from_output.assert_called_once_with(scheduler_output, model_output)


class TestEngineCoreAddAndAbort:
    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_add_request(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_cls = MagicMock()
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core._initialized = True

        req = _make_request()
        core.add_request(req)
        assert "req-1" in core.scheduler.running or len(core.scheduler.waiting) > 0

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_abort_request_uses_queue(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_cls = MagicMock()
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)

        core.abort_request("req-1")
        val = core.aborts_queue.get_nowait()
        assert val == ["req-1"]
        assert core.aborts_queue.empty()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_has_unfinished_requests(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_cls = MagicMock()
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core._initialized = True

        core.scheduler.has_requests = MagicMock(return_value=True)
        assert core.has_unfinished_requests() is True

        core.scheduler.has_requests = MagicMock(return_value=False)
        assert core.has_unfinished_requests() is False


class TestEngineCoreBatchCondense:
    """Phase 2: EngineCore condense integration."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_batch_condense_empty(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core.slot_manager.batch_condense = MagicMock(return_value=[])

        core._batch_condense([])
        core.slot_manager.batch_condense.assert_not_called()
        core.model_executor.condense.assert_not_called()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_batch_condense_merges_abort_and_eos(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core.slot_manager.batch_condense = MagicMock(return_value=[(2, 1)])

        freed = [("req-a", 1), ("req-b", 2)]
        core._batch_condense(freed)

        core.slot_manager.batch_condense.assert_called_once_with(freed)
        core.model_executor.condense.assert_called_once_with([(2, 1)])

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_condense_after_update(self, MockGetClass):
        from vkwr.engine.core import EngineCore
        from vkwr.engine.outputs import EngineCoreOutput

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)

        scheduler_output = _make_scheduler_output(
            req_id="req-1",
            is_last_prefill=True,
        )
        scheduler_output.freed_slots = [("req-1", 0)]

        model_output = _make_model_output()

        engine_outputs = {
            "req-1": EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="req-1",
                        new_token_ids=[42],
                        new_logprobs=None,
                        finish_reason="length",
                    )
                ],
            ),
        }

        core.scheduler.has_requests = MagicMock(return_value=True)
        core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        core.model_executor.execute_model = MagicMock(return_value=model_output)
        core.model_executor.sample_tokens = MagicMock(return_value=model_output)
        core.scheduler.update_from_output = MagicMock(return_value=engine_outputs)
        core.slot_manager.batch_condense = MagicMock(return_value=[])
        core._initialized = True

        core.step()

        core.slot_manager.batch_condense.assert_called_once()
        call_args = core.slot_manager.batch_condense.call_args[0][0]
        assert ("req-1", 0) in call_args

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_process_aborts_queue_returns_freed(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core.slot_manager.req_to_slot = {"req-abort": 2}
        core.scheduler.finish_requests = MagicMock(return_value=["req-abort"])
        core.aborts_queue.put(["req-abort"])

        freed = core._process_aborts_queue()
        assert freed == [("req-abort", 2)]

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_abort_freed_reaches_condense(self, MockGetClass):
        from vkwr.engine.core import EngineCore
        from vkwr.engine.outputs import EngineCoreOutput

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core.slot_manager.req_to_slot = {"req-abort": 1}
        core.scheduler.finish_requests = MagicMock(return_value=["req-abort"])

        scheduler_output = _make_scheduler_output()
        model_output = _make_model_output()

        engine_outputs = {
            "req-1": EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="req-1",
                        new_token_ids=[42],
                        new_logprobs=None,
                        finish_reason=None,
                    )
                ],
            ),
        }

        core.scheduler.has_requests = MagicMock(return_value=True)
        core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        core.model_executor.execute_model = MagicMock(return_value=model_output)
        core.model_executor.sample_tokens = MagicMock(return_value=model_output)
        core.scheduler.update_from_output = MagicMock(return_value=engine_outputs)
        core.slot_manager.batch_condense = MagicMock(return_value=[])
        core._initialized = True

        core.aborts_queue.put(["req-abort"])
        core.step()

        call_args = core.slot_manager.batch_condense.call_args[0][0]
        assert ("req-abort", 1) in call_args

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_zero_tokens_condense(self, MockGetClass):
        from vkwr.engine.core import EngineCore
        from vkwr.engine.outputs import EngineCoreOutput

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)

        scheduler_output = _make_scheduler_output(
            req_id="req-1",
            finished_req_ids={"req-1"},
        )
        scheduler_output.scheduled_req_ids = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.total_num_scheduled_tokens = 0
        scheduler_output.freed_slots = [("req-1", 0)]

        engine_outputs = {
            "req-1": EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="req-1",
                        new_token_ids=[],
                        new_logprobs=None,
                        finish_reason="length",
                    )
                ],
            ),
        }

        core.scheduler.has_requests = MagicMock(return_value=True)
        core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        core.scheduler.update_from_output = MagicMock(return_value=engine_outputs)
        core.slot_manager.batch_condense = MagicMock(return_value=[])
        core._initialized = True

        core.step()

        core.slot_manager.batch_condense.assert_called_once()
        core.model_executor.execute_model.assert_not_called()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_finish_requests_signature_unchanged(self, MockGetClass):
        from vkwr.engine.core import EngineCore

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)
        core.scheduler.finish_requests = MagicMock(return_value=["req-1"])

        result = core.abort_requests(["req-1"])
        core.scheduler.finish_requests.assert_called_once_with({"req-1"})
        assert 0 in result


class TestEngineCoreAsyncCondense:
    """Phase 2: step_with_batch_queue and _handle_zero_tokens condense."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_step_with_batch_queue_condense(self, MockGetClass):
        from vkwr.engine.core import EngineCore
        from vkwr.engine.outputs import EngineCoreOutput

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        config.scheduler_config.enable_async_scheduling = True
        config.scheduler_config.batch_queue_size = 2
        core = EngineCore(config)

        from concurrent.futures import Future

        future = Future()
        model_output = _make_model_output()
        future.set_result(model_output)

        scheduler_output = _make_scheduler_output()
        scheduler_output.freed_slots = [("req-1", 0)]

        engine_outputs = {
            "req-1": EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="req-1",
                        new_token_ids=[42],
                        new_logprobs=None,
                        finish_reason=None,
                    )
                ],
            ),
        }

        core.scheduler.has_requests = MagicMock(side_effect=[True, False])
        core.scheduler.schedule = MagicMock(return_value=scheduler_output)
        core.model_executor.execute_model = MagicMock(return_value=future)
        core.model_executor.sample_tokens = MagicMock(return_value=model_output)
        core.scheduler.update_from_output = MagicMock(return_value=engine_outputs)
        core.slot_manager.batch_condense = MagicMock(return_value=[])
        core._initialized = True

        core.step_with_batch_queue()
        core.step_with_batch_queue()

        core.slot_manager.batch_condense.assert_called_once()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_handle_zero_tokens_condense(self, MockGetClass):
        from vkwr.engine.core import EngineCore
        from vkwr.engine.outputs import EngineCoreOutput

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        core = EngineCore(config)

        scheduler_output = _make_scheduler_output(
            req_id="req-1",
            finished_req_ids={"req-1"},
        )
        scheduler_output.scheduled_req_ids = []
        scheduler_output.num_scheduled_tokens = {}
        scheduler_output.total_num_scheduled_tokens = 0
        scheduler_output.freed_slots = [("req-1", 0)]

        engine_outputs = {
            "req-1": EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id="req-1",
                        new_token_ids=[],
                        new_logprobs=None,
                        finish_reason="length",
                    )
                ],
            ),
        }

        core.scheduler.update_from_output = MagicMock(return_value=engine_outputs)
        core.slot_manager.batch_condense = MagicMock(return_value=[])

        core._handle_zero_tokens(scheduler_output)

        core.slot_manager.batch_condense.assert_called_once()

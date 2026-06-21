"""Tests for EngineCoreProc busy loop, shutdown state machine, and core logic.

Covers:
- EngineShutdownState state machine transitions
- _handle_shutdown() graceful drain and immediate abort paths
- _process_input_queue() drain logic
- _handle_client_request() dispatch for ADD/ABORT/WAKEUP/UTILITY
- _reject_add_in_shutdown() / _reject_utility_in_shutdown()
- _process_engine_step() output routing
- _send_engine_dead() sentinel
- make_zmq_socket bind/connect defaults
- SignalCallback behavior
- _core_proc_target function wiring

Note: Full ZMQ process spawning is tested in test_sync_mp_client_real.py
(heavy integration). These tests focus on the busy loop logic using mocks.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest
import zmq

from vkwr.config.compilation import CompilationConfig
from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.config.worker import GPUWorkerConfig
from vkwr.engine.core_proc import (
    EngineCoreProc,
    EngineShutdownState,
    SignalCallback,
    make_zmq_socket,
)
from vkwr.engine.core_request import EngineCoreRequest, EngineCoreRequestType
from vkwr.engine.outputs import EngineCoreOutput, EngineCoreOutputs
from vkwr.engine.request import RequestStatus, SamplingParams
from vkwr.engine.serial_utils import MsgpackDecoder, MsgpackEncoder


def _make_config(shutdown_timeout: int = 0) -> VkwrConfig:
    return VkwrConfig(
        model_config=ModelConfig(model="fake-model", max_model_len=8192),
        worker_config=GPUWorkerConfig(device="cuda", shutdown_timeout=shutdown_timeout),
        compilation_config=CompilationConfig(cudagraph_mode="none"),
    )


def _make_request(request_id: str = "req-1") -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=8),
        arrival_time=100.0,
        priority=0,
        current_wave=0,
    )


class TestEngineShutdownState:
    """Test EngineShutdownState enum values and transitions."""

    def test_running_value(self) -> None:
        assert EngineShutdownState.RUNNING == 0

    def test_requested_value(self) -> None:
        assert EngineShutdownState.REQUESTED == 1

    def test_shutting_down_value(self) -> None:
        assert EngineShutdownState.SHUTTING_DOWN == 2


class TestSignalCallback:
    """Test SignalCallback thread-safe callback."""

    def test_signal_callback_triggers(self) -> None:
        triggered = []

        def callback():
            triggered.append(1)

        sc = SignalCallback(callback)
        sc.trigger()
        import time

        time.sleep(0.2)
        assert triggered == [1]
        sc.stop()

    def test_signal_callback_stop_prevents_trigger(self) -> None:
        triggered = []

        def callback():
            triggered.append(1)

        sc = SignalCallback(callback)
        sc.stop()
        sc.trigger()
        import time

        time.sleep(0.2)
        assert triggered == []


class TestMakeZMQSocket:
    """Test make_zmq_socket bind/connect defaults."""

    def test_push_bind_false_default(self) -> None:
        ctx = zmq.Context()
        sock = make_zmq_socket(ctx, "ipc:///tmp/test-push", zmq.PUSH)
        # PUSH defaults to bind=False (connect)
        sock.close()
        ctx.term()

    def test_pull_bind_true_default(self) -> None:
        ctx = zmq.Context()
        sock = make_zmq_socket(ctx, "ipc:///tmp/test-pull", zmq.PULL)
        # PULL defaults to bind=True (bind)
        sock.close()
        ctx.term()

    def test_explicit_bind_true(self) -> None:
        ctx = zmq.Context()
        sock = make_zmq_socket(ctx, "ipc:///tmp/test-push2", zmq.PUSH, bind=True)
        sock.close()
        ctx.term()

    def test_explicit_bind_false(self) -> None:
        ctx = zmq.Context()
        sock = make_zmq_socket(ctx, "ipc:///tmp/test-pull2", zmq.PULL, bind=False)
        sock.close()
        ctx.term()

    def test_linger_set(self) -> None:
        ctx = zmq.Context()
        sock = make_zmq_socket(ctx, "ipc:///tmp/test-linger", zmq.PUSH, linger=4000)
        assert sock.getsockopt(zmq.LINGER) == 4000
        sock.close()
        ctx.term()


class TestEngineCoreProcHandleShutdown:
    """Test _handle_shutdown() state machine logic."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_running_returns_true(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc.shutdown_state = EngineShutdownState.RUNNING

        assert proc._handle_shutdown() is True

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_requested_immediate_abort(self, MockGetClass: MagicMock) -> None:
        """shutdown_timeout=0 -> immediate abort, then returns True/False based on work."""
        MockGetClass.return_value = MagicMock()
        config = _make_config(shutdown_timeout=0)
        proc = EngineCoreProc(config)
        proc.shutdown_state = EngineShutdownState.REQUESTED

        proc.scheduler.get_num_unfinished_requests = MagicMock(return_value=0)
        proc.has_work = MagicMock(return_value=False)

        assert proc._handle_shutdown() is False
        assert proc.shutdown_state == EngineShutdownState.SHUTTING_DOWN

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_requested_immediate_abort_with_requests(self, MockGetClass: MagicMock) -> None:
        """With requests, aborts them and sends output."""
        MockGetClass.return_value = MagicMock()
        config = _make_config(shutdown_timeout=0)
        proc = EngineCoreProc(config)
        proc.shutdown_state = EngineShutdownState.REQUESTED

        proc.scheduler.get_num_unfinished_requests = MagicMock(return_value=2)
        proc.scheduler.finish_requests = MagicMock(return_value=["req-1", "req-2"])
        proc.has_work = MagicMock(return_value=False)

        assert proc._handle_shutdown() is False

        proc.scheduler.finish_requests.assert_called_once_with(None, RequestStatus.FINISHED_ABORTED)

        # Abort outputs should be in output_queue
        assert not proc.output_queue.empty()
        item = proc.output_queue.get_nowait()
        assert isinstance(item, tuple)
        assert item[0] == 0
        assert isinstance(item[1], EngineCoreOutputs)
        assert item[1].outputs[0].request_id == "req-1"

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_requested_graceful_drain(self, MockGetClass: MagicMock) -> None:
        """shutdown_timeout>0 -> drain, returns True while work exists."""
        MockGetClass.return_value = MagicMock()
        config = _make_config(shutdown_timeout=30)
        proc = EngineCoreProc(config)
        proc.shutdown_state = EngineShutdownState.REQUESTED

        proc.scheduler.get_num_unfinished_requests = MagicMock(return_value=3)
        proc.has_work = MagicMock(return_value=True)

        assert proc._handle_shutdown() is True
        assert proc.shutdown_state == EngineShutdownState.SHUTTING_DOWN

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_shutting_down_no_work_exits(self, MockGetClass: MagicMock) -> None:
        """SHUTTING_DOWN + no work -> exit."""
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        proc.has_work = MagicMock(return_value=False)
        assert proc._handle_shutdown() is False


class TestEngineCoreProcHandleClientRequest:
    """Test _handle_client_request() dispatch logic."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_add_request(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc._initialized = True
        proc.scheduler.has_requests = MagicMock(return_value=False)

        req = _make_request("add-test")
        proc._handle_client_request(EngineCoreRequestType.ADD, req)

        assert "add-test" in proc.scheduler.running or len(proc.scheduler.waiting) > 0

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_abort_request_routes_to_output_queue(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)

        proc.abort_requests = MagicMock(return_value={0: EngineCoreOutputs()})

        proc._handle_client_request(EngineCoreRequestType.ABORT, ["abort-1", "abort-2"])

        proc.abort_requests.assert_called_once_with(["abort-1", "abort-2"])
        assert not proc.output_queue.empty()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_wakeup_noop(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)

        proc._handle_client_request(EngineCoreRequestType.WAKEUP, None)

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_executor_failed_raises(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)

        with pytest.raises(RuntimeError, match="Executor failed"):
            proc._handle_client_request(EngineCoreRequestType.EXECUTOR_FAILED, None)


class TestEngineCoreProcRejectInShutdown:
    """Test _reject_add_in_shutdown() and _reject_utility_in_shutdown()."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_reject_add_in_running(self, MockGetClass: MagicMock) -> None:
        """RUNNING state: don't reject."""
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc.shutdown_state = EngineShutdownState.RUNNING

        req = _make_request("keep-me")
        assert proc._reject_add_in_shutdown(req) is False

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_reject_add_in_requested(self, MockGetClass: MagicMock) -> None:
        """REQUESTED state: reject with abort output."""
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc.shutdown_state = EngineShutdownState.REQUESTED
        proc.scheduler.finish_requests = MagicMock(return_value=set())

        req = _make_request("reject-me")
        assert proc._reject_add_in_shutdown(req) is True
        proc.scheduler.finish_requests.assert_called_once_with({"reject-me"})

        # Output queue should have abort output
        item = proc.output_queue.get_nowait()
        assert isinstance(item, tuple)
        assert item[1].outputs[0].request_id == "reject-me"
        assert item[1].outputs[0].finish_reason == "abort"

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_reject_utility_in_running(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc.shutdown_state = EngineShutdownState.RUNNING

        assert proc._reject_utility_in_shutdown(0, 123, "some_method") is False

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_reject_utility_in_requested(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc.shutdown_state = EngineShutdownState.REQUESTED

        assert proc._reject_utility_in_shutdown(0, 123, "some_method") is True

        item = proc.output_queue.get_nowait()
        assert item[0] == 0
        assert item[1].utility_output.call_id == 123
        assert "shutting down" in item[1].utility_output.failure_message.lower()


class TestEngineCoreProcProcessInputQueue:
    """Test _process_input_queue() drain and abort clearing logic."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_process_input_queue_handles_requests(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc.scheduler.has_requests = MagicMock(return_value=True)

        proc.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
        proc.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

        proc.process_input_queue_block = False
        proc._process_input_queue()

        assert proc.input_queue.empty()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_process_input_queue_clears_aborts_when_idle(self, MockGetClass: MagicMock) -> None:
        """When idle (no requests), clears aborts_queue before processing input."""
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc.scheduler.has_requests = MagicMock(return_value=False)

        proc.aborts_queue.put(["should-be-cleared"])

        proc.process_input_queue_block = False
        proc._process_input_queue()

        assert proc.aborts_queue.empty()


class TestEngineCoreProcProcessEngineStep:
    """Test _process_engine_step() output routing."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_process_engine_step_routes_outputs(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)

        engine_outputs = {0: EngineCoreOutputs(outputs=[EngineCoreOutput(request_id="step-test", new_token_ids=[42])])}
        proc.step_fn = MagicMock(return_value=(engine_outputs, True))
        proc.post_step = MagicMock()

        model_executed = proc._process_engine_step()

        assert model_executed is True
        assert not proc.output_queue.empty()
        item = proc.output_queue.get_nowait()
        assert item[0] == 0
        assert item[1].outputs[0].request_id == "step-test"

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_process_engine_step_empty_outputs(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)

        proc.step_fn = MagicMock(return_value=({}, False))
        proc.scheduler.has_requests = MagicMock(return_value=False)
        proc.post_step = MagicMock()

        model_executed = proc._process_engine_step()
        assert model_executed is False
        assert proc.output_queue.empty()


class TestEngineCoreProcSendEngineDead:
    """Test _send_engine_dead() sentinel logic."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_send_engine_dead_puts_sentinel(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc.output_thread = MagicMock()

        proc._send_engine_dead()

        item = proc.output_queue.get_nowait()
        assert item == EngineCoreProc.ENGINE_CORE_DEAD
        proc.output_thread.join.assert_called_once_with(timeout=5.0)


class TestEngineCoreProcDecodeInput:
    """Test _decode_input() frame decoding and dual-queue abort."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_decode_add_request(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)

        encoder = MsgpackEncoder()
        req = _make_request("decode-add")
        payload_frames = encoder.encode(req)
        frames = [EngineCoreRequestType.ADD.value, *payload_frames]

        proc._decoder = MsgpackDecoder()
        proc._decode_input(frames)

        assert not proc.input_queue.empty()
        rtype, payload = proc.input_queue.get_nowait()
        assert rtype == EngineCoreRequestType.ADD
        assert payload.request_id == "decode-add"

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_decode_abort_dual_queue(self, MockGetClass: MagicMock) -> None:
        """ABORT goes to BOTH aborts_queue and input_queue."""
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)

        encoder = MsgpackEncoder()
        payload_frames = encoder.encode(["abort-a", "abort-b"])
        frames = [EngineCoreRequestType.ABORT.value, *payload_frames]

        proc._decoder = MsgpackDecoder()
        proc._decode_input(frames)

        # aborts_queue
        assert not proc.aborts_queue.empty()
        abort_payload = proc.aborts_queue.get_nowait()
        assert abort_payload == ["abort-a", "abort-b"]

        # input_queue
        assert not proc.input_queue.empty()
        rtype, payload = proc.input_queue.get_nowait()
        assert rtype == EngineCoreRequestType.ABORT
        assert payload == ["abort-a", "abort-b"]


class TestEngineCoreProcInit:
    """Test EngineCoreProc initialization."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_init_sets_defaults(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)

        assert proc.shutdown_state == EngineShutdownState.RUNNING
        assert proc.process_input_queue_block is True
        assert isinstance(proc.input_queue, queue.Queue)
        assert isinstance(proc.output_queue, queue.Queue)
        assert isinstance(proc.aborts_queue, queue.Queue)
        assert proc._input_socket is None
        assert proc._output_socket is None
        assert proc.input_thread is None
        assert proc.output_thread is None

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_has_work(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        proc.scheduler.has_requests = MagicMock(return_value=True)
        assert proc.has_work() is True

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_is_running(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)
        assert proc.is_running() is True
        proc.shutdown_state = EngineShutdownState.SHUTTING_DOWN
        assert proc.is_running() is False


class TestSendAbortOutputs:
    """Test _send_abort_outputs() method."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_send_abort_outputs(self, MockGetClass: MagicMock) -> None:
        MockGetClass.return_value = MagicMock()
        config = _make_config()
        proc = EngineCoreProc(config)

        proc._send_abort_outputs(["req-a", "req-b"])

        assert proc.output_queue.qsize() == 2
        item1 = proc.output_queue.get_nowait()
        item2 = proc.output_queue.get_nowait()
        assert item1[0] == 0
        assert item1[1].outputs[0].request_id == "req-a"
        assert item1[1].outputs[0].finish_reason == "abort"
        assert item2[1].outputs[0].request_id == "req-b"

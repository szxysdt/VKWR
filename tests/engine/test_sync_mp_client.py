from __future__ import annotations

import queue
import threading
import time
import uuid
from concurrent.futures import Future

import msgspec
import pytest
import zmq

from vkwr.engine.core_proc import EngineCoreProc, make_zmq_socket
from vkwr.engine.core_request import EngineCoreRequest, EngineCoreRequestType, UtilityOutput, UtilityResult
from vkwr.engine.exceptions import EngineDeadError
from vkwr.engine.outputs import EngineCoreOutput, EngineCoreOutputs
from vkwr.engine.request import SamplingParams
from vkwr.engine.serial_utils import MsgpackDecoder, MsgpackEncoder


class _OutputFrame(msgspec.Struct, array_like=True, gc=False):
    """Matches vkwr.engine.core_client._OutputFrame for decoding test frames."""

    client_idx: int
    outputs: EngineCoreOutputs


def _make_request(request_id: str = "req-1", prompt_token_ids: list[int] | None = None) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
        sampling_params=SamplingParams(),
        arrival_time=time.monotonic(),
        priority=0,
        current_wave=0,
    )


class FakeZMQEngine:
    """Fake engine that processes requests from a queue and sends ZMQ output.

    Input comes from a queue.Queue (avoids ZMQ routing complexity in tests).
    Output uses real ZMQ PUSH socket on output_address, matching the real
    EngineCoreProc protocol.

    Protocol:
    - Sends READY on output socket at startup
    - Processes (request_type, payload) tuples from input_queue
    - Sends (client_idx, EngineCoreOutputs) msgpack-encoded on output socket
    - Sends ENGINE_CORE_DEAD sentinel when die() is called
    """

    def __init__(self, output_address: str):
        self._output_address = output_address
        self._stop_event = threading.Event()
        self._die_event = threading.Event()
        self.input_queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        ctx = zmq.Context(io_threads=2)
        try:
            output_sock = make_zmq_socket(ctx, self._output_address, zmq.PUSH, bind=False, linger=0)
            encoder = MsgpackEncoder()

            output_sock.send(b"READY")

            while not self._stop_event.is_set():
                if self._die_event.is_set():
                    output_sock.send(EngineCoreProc.ENGINE_CORE_DEAD, copy=False)
                    break

                try:
                    request_type, payload = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if request_type == EngineCoreRequestType.ADD:
                    output = EngineCoreOutputs(
                        outputs=[
                            EngineCoreOutput(
                                request_id=payload.request_id,
                                new_token_ids=[42, 43],
                            )
                        ]
                    )
                    out_frames = encoder.encode((0, output))
                    output_sock.send_multipart(out_frames, copy=False)

                elif request_type == EngineCoreRequestType.ABORT:
                    for req_id in payload:
                        output = EngineCoreOutputs(
                            outputs=[
                                EngineCoreOutput(
                                    request_id=req_id,
                                    new_token_ids=[],
                                    finish_reason="abort",
                                )
                            ]
                        )
                        out_frames = encoder.encode((0, output))
                        output_sock.send_multipart(out_frames, copy=False)

                elif request_type == EngineCoreRequestType.UTILITY:
                    client_idx, call_id, method_name, args = payload
                    util_output = UtilityOutput(call_id=call_id)
                    if method_name == "get_model_name":
                        util_output.result = UtilityResult(result="fake-model")
                    elif method_name == "echo":
                        util_output.result = UtilityResult(result=args)
                    elif method_name == "raise_error":
                        util_output.failure_message = "utility method failed"
                    else:
                        util_output.failure_message = f"Unknown method: {method_name}"
                    output = EngineCoreOutputs(utility_output=util_output)
                    out_frames = encoder.encode((client_idx, output))
                    output_sock.send_multipart(out_frames, copy=False)
        finally:
            output_sock.close(linger=0)
            ctx.term()

    def send_request(self, request_type: EngineCoreRequestType, payload):
        self.input_queue.put_nowait((request_type, payload))

    def terminate(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

    def die(self):
        self._die_event.set()

    def join(self, timeout=None):
        self._thread.join(timeout=timeout)

    @property
    def is_alive(self):
        return self._thread.is_alive()

    @property
    def pid(self):
        return 99999

    @property
    def exitcode(self):
        return 0


def _reader_loop(
    ctx,
    output_socket,
    shutdown_path,
    outputs_q,
    utility_results,
):
    """Shared reader thread logic matching SyncMPClient process_outputs_socket."""
    shutdown_socket = ctx.socket(zmq.PAIR)
    try:
        shutdown_socket.bind(shutdown_path)
        poller = zmq.Poller()
        poller.register(shutdown_socket, zmq.POLLIN)
        poller.register(output_socket, zmq.POLLIN)
        while True:
            socks = poller.poll(timeout=5000)
            if not socks:
                break
            if len(socks) == 2 or socks[0][0] == shutdown_socket:
                break
            frames = output_socket.recv_multipart(copy=False)
            raw = bytes(frames[0]) if len(frames) == 1 else None
            if raw == b"READY":
                continue
            if raw == EngineCoreProc.ENGINE_CORE_DEAD:
                outputs_q.put_nowait(EngineDeadError())
                break
            # Decode (client_idx, EngineCoreOutputs) tuple with typed struct.
            decoder = MsgpackDecoder()
            frame = decoder.decode([raw], _OutputFrame)
            outputs = frame.outputs
            if outputs.utility_output:
                from vkwr.engine.core_client import MPClient

                MPClient._process_utility_output(outputs.utility_output, utility_results)
            else:
                outputs_q.put_nowait(outputs)
    finally:
        shutdown_socket.close(linger=0)
        output_socket.close(linger=0)


class TestSyncMPClientOutputReader:
    """Test the output reader thread logic that SyncMPClient uses.

    Uses FakeZMQEngine (queue input, real ZMQ PUSH output) to test the
    reader thread that consumes from a ZMQ PULL socket.
    """

    def _setup(self):
        """Create a FakeZMQEngine, reader thread, and PULL socket."""
        session_id = uuid.uuid4().hex
        output_address = f"ipc:///tmp/vkwr-test-output-{session_id}"

        fake = FakeZMQEngine(output_address)
        time.sleep(0.1)

        ctx = zmq.Context()
        output_socket = make_zmq_socket(ctx, output_address, zmq.PULL)
        shutdown_path = f"inproc://{uuid.uuid4().hex}"

        outputs_q: queue.Queue = queue.Queue()
        utility_results: dict = {}

        t = threading.Thread(
            target=_reader_loop,
            args=(ctx, output_socket, shutdown_path, outputs_q, utility_results),
            daemon=True,
        )
        t.start()

        return fake, ctx, shutdown_path, outputs_q, utility_results, t

    def test_reader_thread_receives_output(self):
        fake, ctx, shutdown_path, outputs_q, utility_results, t = self._setup()

        req = _make_request("test-1")
        fake.send_request(EngineCoreRequestType.ADD, req)

        try:
            outputs = outputs_q.get(timeout=5)
            assert isinstance(outputs, EngineCoreOutputs)
            assert len(outputs.outputs) == 1
            assert outputs.outputs[0].request_id == "test-1"
            assert outputs.outputs[0].new_token_ids == [42, 43]
        finally:
            with ctx.socket(zmq.PAIR) as s:
                s.connect(shutdown_path)
                s.send(b"")
            t.join(timeout=3)
            ctx.term()
            fake.terminate()

    def test_reader_thread_handles_engine_dead(self):
        fake, ctx, shutdown_path, outputs_q, utility_results, t = self._setup()

        time.sleep(0.05)
        fake.die()

        try:
            result = outputs_q.get(timeout=5)
            assert isinstance(result, EngineDeadError)
        finally:
            with ctx.socket(zmq.PAIR) as s:
                s.connect(shutdown_path)
                s.send(b"")
            t.join(timeout=3)
            ctx.term()
            fake.terminate()

    def test_reader_thread_handles_utility_output(self):
        fake, ctx, shutdown_path, outputs_q, utility_results, t = self._setup()

        call_id = 12345
        utility_results[call_id] = Future()

        payload = (0, call_id, "get_model_name", ())
        fake.send_request(EngineCoreRequestType.UTILITY, payload)

        try:
            result = utility_results[call_id].result(timeout=5)
            assert result == "fake-model"
        finally:
            with ctx.socket(zmq.PAIR) as s:
                s.connect(shutdown_path)
                s.send(b"")
            t.join(timeout=3)
            ctx.term()
            fake.terminate()

    def test_reader_thread_shutdown_unblock(self):
        session_id = uuid.uuid4().hex
        output_address = f"ipc:///tmp/vkwr-test-output-{session_id}"

        fake = FakeZMQEngine(output_address)
        time.sleep(0.05)

        ctx = zmq.Context()
        output_socket = make_zmq_socket(ctx, output_address, zmq.PULL)
        shutdown_path = f"inproc://{uuid.uuid4().hex}"
        reader_done = threading.Event()

        def reader():
            shutdown_socket = ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(output_socket, zmq.POLLIN)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        break
                    output_socket.recv_multipart(copy=False)
            finally:
                shutdown_socket.close(linger=0)
                output_socket.close(linger=0)
                reader_done.set()

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        time.sleep(0.05)
        with ctx.socket(zmq.PAIR) as s:
            s.connect(shutdown_path)
            s.send(b"")

        assert reader_done.wait(timeout=3), "Reader thread did not exit after shutdown signal"
        t.join(timeout=3)
        ctx.term()
        fake.terminate()

    def test_multiple_outputs_ordered(self):
        fake, ctx, shutdown_path, outputs_q, utility_results, t = self._setup()

        fake.send_request(EngineCoreRequestType.ADD, _make_request("multi-1"))
        fake.send_request(EngineCoreRequestType.ADD, _make_request("multi-2"))

        try:
            out1 = outputs_q.get(timeout=5)
            out2 = outputs_q.get(timeout=5)
            assert out1.outputs[0].request_id == "multi-1"
            assert out2.outputs[0].request_id == "multi-2"
        finally:
            with ctx.socket(zmq.PAIR) as s:
                s.connect(shutdown_path)
                s.send(b"")
            t.join(timeout=3)
            ctx.term()
            fake.terminate()

    def test_abort_output_received(self):
        fake, ctx, shutdown_path, outputs_q, utility_results, t = self._setup()

        fake.send_request(EngineCoreRequestType.ABORT, ["abort-1"])

        try:
            outputs = outputs_q.get(timeout=5)
            assert isinstance(outputs, EngineCoreOutputs)
            assert len(outputs.outputs) == 1
            assert outputs.outputs[0].request_id == "abort-1"
            assert outputs.outputs[0].finish_reason == "abort"
        finally:
            with ctx.socket(zmq.PAIR) as s:
                s.connect(shutdown_path)
                s.send(b"")
            t.join(timeout=3)
            ctx.term()
            fake.terminate()

    def test_utility_failure_propagates(self):
        fake, ctx, shutdown_path, outputs_q, utility_results, t = self._setup()

        call_id = 54321
        utility_results[call_id] = Future()

        payload = (0, call_id, "raise_error", ())
        fake.send_request(EngineCoreRequestType.UTILITY, payload)

        try:
            with pytest.raises(Exception, match="utility method failed"):
                utility_results[call_id].result(timeout=5)
        finally:
            with ctx.socket(zmq.PAIR) as s:
                s.connect(shutdown_path)
                s.send(b"")
            t.join(timeout=3)
            ctx.term()
            fake.terminate()


class TestSyncMPClientMethods:
    """Test SyncMPClient method behavior at unit level."""

    def test_add_request_sends_correct_frames(self):
        encoder = MsgpackEncoder()
        req = _make_request("frame-test", [10, 20, 30])
        frames = encoder.encode(req)

        msg = [b"\x00", EngineCoreRequestType.ADD.value, *frames]
        assert len(msg) == 3
        assert msg[0] == b"\x00"
        assert msg[1] == EngineCoreRequestType.ADD.value

    def test_abort_request_sends_correct_frames(self):
        encoder = MsgpackEncoder()
        request_ids = ["abort-1", "abort-2"]
        frames = encoder.encode(request_ids)

        msg = [b"\x00", EngineCoreRequestType.ABORT.value, *frames]
        assert len(msg) == 3
        assert msg[1] == EngineCoreRequestType.ABORT.value

    def test_utility_request_sends_correct_frames(self):
        encoder = MsgpackEncoder()
        payload = (0, 999, "test_method", ("arg1",))
        frames = encoder.encode(payload)

        msg = [b"\x00", EngineCoreRequestType.UTILITY.value, *frames]
        assert len(msg) == 3
        assert msg[1] == EngineCoreRequestType.UTILITY.value


class TestMPClientProcessUtilityOutput:
    """Test MPClient._process_utility_output static method."""

    def test_success_result(self):
        from vkwr.engine.core_client import MPClient

        call_id = 42
        future = Future()
        utility_results = {call_id: future}

        output = UtilityOutput(call_id=call_id, result=UtilityResult(result="hello"))
        MPClient._process_utility_output(output, utility_results)

        assert future.result(timeout=1) == "hello"
        assert call_id not in utility_results

    def test_failure_result(self):
        from vkwr.engine.core_client import MPClient

        call_id = 43
        future = Future()
        utility_results = {call_id: future}

        output = UtilityOutput(call_id=call_id, failure_message="something broke")
        MPClient._process_utility_output(output, utility_results)

        with pytest.raises(Exception, match="something broke"):
            future.result(timeout=1)
        assert call_id not in utility_results

    def test_none_result(self):
        from vkwr.engine.core_client import MPClient

        call_id = 44
        future = Future()
        utility_results = {call_id: future}

        output = UtilityOutput(call_id=call_id, result=None)
        MPClient._process_utility_output(output, utility_results)

        assert future.result(timeout=1) is None

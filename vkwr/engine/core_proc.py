"""Inspired by vLLM."""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import zmq

from vkwr.engine.core import EngineCore
from vkwr.engine.core_request import (
    EngineCoreRequestType,
    UtilityOutput,
)
from vkwr.engine.outputs import EngineCoreOutput, EngineCoreOutputs
from vkwr.engine.request import RequestStatus
from vkwr.engine.serial_utils import MsgpackDecoder, MsgpackEncoder

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig

logger = logging.getLogger(__name__)


class EngineShutdownState(IntEnum):
    RUNNING = 0
    REQUESTED = 1
    SHUTTING_DOWN = 2


class SignalCallback:
    """Thread-safe callback for signal handlers.

    Signal handler calls trigger(), which wakes a dedicated daemon thread
    that executes the callback fn().
    """

    def __init__(self, callback: callable):
        self._callback = callback
        self._event = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="signal-callback",
        )
        self._thread.start()

    def _run(self):
        self._event.wait()
        if not self._stopped and self._callback:
            self._callback()

    def trigger(self):
        self._event.set()

    def stop(self):
        self._stopped = True
        self._event.set()


def make_zmq_socket(
    ctx: zmq.Context | zmq.asyncio.Context,
    path: str,
    socket_type: int,
    bind: bool | None = None,
    linger: int | None = None,
    identity: bytes | None = None,
) -> zmq.Socket | zmq.asyncio.Socket:
    socket = ctx.socket(socket_type)
    socket.setsockopt(zmq.RCVHWM, 0)
    socket.setsockopt(zmq.SNDHWM, 0)
    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)
    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)
    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)
    if bind:
        socket.bind(path)
    else:
        socket.connect(path)
    return socket


class EngineCoreProc(EngineCore):
    """EngineCore that runs in a background process with ZMQ IPC."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    def __init__(self, config: VkwrConfig):
        super().__init__(config)
        self._input_socket: zmq.Socket | None = None
        self._output_socket: zmq.Socket | None = None
        self._encoder: MsgpackEncoder | None = None
        self._decoder: MsgpackDecoder | None = None

        self.input_queue: queue.Queue = queue.Queue()
        self.output_queue: queue.Queue = queue.Queue()
        self.aborts_queue: queue.Queue[list[str]] = queue.Queue()

        self.shutdown_state = EngineShutdownState.RUNNING
        self.process_input_queue_block: bool = True

        self.input_thread: threading.Thread | None = None
        self.output_thread: threading.Thread | None = None

    def has_work(self) -> bool:
        return self.scheduler.has_requests() or (self.batch_queue is not None and len(self.batch_queue) > 0)

    def is_running(self) -> bool:
        return self.shutdown_state == EngineShutdownState.RUNNING

    def run_busy_loop(self):
        def wakeup_engine():
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

        signal_callback = SignalCallback(wakeup_engine)

        def signal_handler(signum, frame):
            self.shutdown_state = EngineShutdownState.REQUESTED
            signal_callback.trigger()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        self.input_thread = threading.Thread(target=self.process_input_sockets, daemon=True)
        self.input_thread.start()

        self.output_thread = threading.Thread(target=self.process_output_sockets, daemon=True)
        self.output_thread.start()

        try:
            while self._handle_shutdown():
                self._process_input_queue()
                self._process_engine_step()
        except Exception:
            logger.exception("EngineCore encountered a fatal error")
            self._send_engine_dead()
            raise
        finally:
            signal_callback.stop()
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if self.output_thread:
                self.output_thread.join(timeout=5.0)
            if self.input_thread:
                self.input_thread.join(timeout=5.0)
            raise SystemExit

    def _handle_shutdown(self) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True

        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.config.worker_config.shutdown_timeout

            logger.info("Shutdown initiated (timeout=%d)", shutdown_timeout)

            if shutdown_timeout == 0:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info("Aborting %d requests", num_requests)
                aborted_reqs = self.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
                if aborted_reqs:
                    self._send_abort_outputs(aborted_reqs)
            else:
                num_requests = self.scheduler.get_num_unfinished_requests()
                if num_requests > 0:
                    logger.info(
                        "Draining %d in-flight requests (timeout=%ds)",
                        num_requests,
                        shutdown_timeout,
                    )

            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN

        if not self.has_work():
            logger.info("Shutdown complete")
            return False

        return True

    def _process_input_queue(self):
        while not self.scheduler.has_requests() and self.is_running():
            if self.input_queue.empty():
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
            try:
                req = self.input_queue.get(block=self.process_input_queue_block)
                self._handle_client_request(*req)
                if not self.process_input_queue_block:
                    break
            except queue.Empty:
                break

        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _handle_client_request(self, request_type: EngineCoreRequestType, payload: Any) -> None:
        if request_type == EngineCoreRequestType.ADD:
            req = payload
            if self._reject_add_in_shutdown(req):
                return
            internal_req, request_wave = self.preprocess_add_request(req)
            self.add_request(internal_req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            abort_outputs = self.abort_requests(payload)
            for worker_id, outputs in abort_outputs.items():
                self.output_queue.put_nowait((worker_id, outputs))
        elif request_type == EngineCoreRequestType.WAKEUP:
            pass
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed")
        elif request_type == EngineCoreRequestType.UTILITY:
            from vkwr.engine.core_client import MPClient

            client_idx, call_id, method_name, args = payload
            if self._reject_utility_in_shutdown(client_idx, call_id, method_name):
                return
            util_output = UtilityOutput(call_id=call_id)

            def enqueue_output(out):
                self.output_queue.put_nowait((client_idx, EngineCoreOutputs(utility_output=out)))

            MPClient._invoke_utility_method(
                method_name,
                lambda: getattr(self, method_name)(*args),
                util_output,
                enqueue_output,
            )

    def _reject_add_in_shutdown(self, request: Any) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False
        logger.info("Rejecting request %s (shutting down)", request.request_id)
        self.scheduler.finish_requests({request.request_id})
        output = EngineCoreOutput(
            request_id=request.request_id,
            new_token_ids=[],
            finish_reason="abort",
        )
        self.output_queue.put_nowait((0, EngineCoreOutputs(outputs=[output])))
        return True

    def _reject_utility_in_shutdown(self, client_idx: int, call_id: int, method_name: str) -> bool:
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return False
        logger.warning("Rejecting utility call %s (shutting down)", method_name)
        output = UtilityOutput(call_id=call_id, failure_message="Server shutting down")
        self.output_queue.put_nowait((client_idx, EngineCoreOutputs(utility_output=output)))
        return True

    def _send_abort_outputs(self, request_ids: list[str]):
        for request_id in request_ids:
            output = EngineCoreOutput(
                request_id=request_id,
                new_token_ids=[],
                finish_reason="abort",
            )
            self.output_queue.put_nowait((0, EngineCoreOutputs(outputs=[output])))

    def _process_engine_step(self) -> bool:
        outputs, model_executed = self.step_fn()

        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)

        self.post_step(model_executed)

        if not model_executed and self.scheduler.has_requests():
            time.sleep(0.001)

        return model_executed

    def _send_engine_dead(self):
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal("VKWR shutdown signal from EngineCore failed to send")

    def process_input_sockets(self):
        poller = zmq.Poller()
        poller.register(self._input_socket, zmq.POLLIN)

        while True:
            events = dict(poller.poll(timeout=100))
            if self._input_socket not in events:
                continue
            while True:
                try:
                    frames = self._input_socket.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                self._decode_input(frames)

    def _decode_input(self, frames: list[zmq.Frame]):
        # DEALER strips the ROUTER identity frame, so frames are shifted by 1.
        # Client sends [identity, request_type, *msgpack_frames] → DEALER sees [request_type, *msgpack_frames]
        request_type = EngineCoreRequestType(frames[0])
        if request_type == EngineCoreRequestType.ABORT:
            payload = self._decoder.decode(frames[1:])
        elif request_type == EngineCoreRequestType.UTILITY:
            payload = self._decoder.decode(frames[1:])
        elif request_type in (EngineCoreRequestType.ADD, EngineCoreRequestType.START_DP_WAVE):
            from vkwr.engine.core_request import EngineCoreRequest

            payload = self._decoder.decode(frames[1:], EngineCoreRequest)
        else:
            payload = self._decoder.decode(frames[1:])

        if request_type == EngineCoreRequestType.ABORT:
            self.aborts_queue.put_nowait(payload)

        self.input_queue.put_nowait((request_type, payload))

    def process_output_sockets(self):
        while True:
            item = self.output_queue.get()
            if item == EngineCoreProc.ENGINE_CORE_DEAD:
                self._output_socket.send(item, copy=False)
                break
            frames = self._encoder.encode(item)
            self._output_socket.send_multipart(frames, copy=False)


def _core_proc_target(
    config: VkwrConfig,
    device: str,
    input_address: str,
    output_address: str,
):
    """Target function for background EngineCore process."""
    import torch

    if isinstance(device, str):
        device_idx = 0 if device == "cuda" else int(device.split(":")[-1])
    else:
        device_idx = device
    torch.cuda.set_device(device_idx)

    ctx = zmq.Context(io_threads=2)

    proc = EngineCoreProc(config)
    proc.initialize()

    proc._input_socket = make_zmq_socket(ctx, input_address, zmq.DEALER, bind=False, identity=b"\x00")
    proc._output_socket = make_zmq_socket(ctx, output_address, zmq.PUSH, bind=False, linger=4000)
    proc._encoder = MsgpackEncoder()
    proc._decoder = MsgpackDecoder()

    proc._output_socket.send(b"READY")
    proc.run_busy_loop()

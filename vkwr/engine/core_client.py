"""Inspired by vLLM."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import queue
import signal
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from concurrent import futures
from threading import Thread
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
import zmq.asyncio

from vkwr.engine.core_request import EngineCoreRequest, EngineCoreRequestType
from vkwr.engine.exceptions import EngineDeadError
from vkwr.engine.outputs import EngineCoreOutputs


# msgspec Struct for decoding (client_idx, EngineCoreOutputs) output frames.
# EngineCoreProc sends these as msgpack-encoded tuples; msgpack represents
# tuples as arrays, so we need a typed decoder to reconstruct the struct.
class _OutputFrame(msgspec.Struct, array_like=True, gc=False):
    client_idx: int
    outputs: EngineCoreOutputs


if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.core import EngineCore
    from vkwr.executor.abstract import ExecutorInterface


logger = logging.getLogger(__name__)


class EngineCoreClient(ABC):
    """Abstract interface for EngineCore access.

    Subclasses handle different transport methods:
    * InprocClient: Direct method calls (same process)
    * SyncMPClient: ZMQ + background thread (multiprocessing)
    * AsyncMPClient: ZMQ asyncio (API server)
    """

    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    def abort_requests(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    def get_output(self, timeout: float | None = None) -> EngineCoreOutputs:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None: ...

    def call_utility(self, method: str, *args: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def make_client(
        multiprocess_mode: bool = False,
        asyncio_mode: bool = False,
        vkwr_config: VkwrConfig | None = None,
        executor_class: type[ExecutorInterface] | None = None,
        log_stats: bool = True,
        **kwargs: Any,
    ) -> EngineCoreClient:
        """Factory method that returns the appropriate client."""
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError("asyncio without multiprocessing not supported")
        if multiprocess_mode and asyncio_mode:
            return AsyncMPClient(vkwr_config, executor_class, log_stats)
        if multiprocess_mode:
            return SyncMPClient(vkwr_config, executor_class, log_stats)
        return InprocClient(vkwr_config)  # type: ignore[arg-type]


class _BackgroundResources:
    """Finalizer to clean up ZMQ resources on client death.

    Uses `weakref.finalize` pattern. Handles both sync and async paths.
    """

    def __init__(self, sync_ctx: zmq.Context):
        self.sync_ctx = sync_ctx
        self.input_socket: zmq.Socket | zmq.asyncio.Socket | None = None
        self.output_socket: zmq.Socket | zmq.asyncio.Socket | None = None
        self.output_queue_task: asyncio.Task | None = None
        self.shutdown_path: str | None = None
        self.engine_dead = False
        self.process: multiprocessing.Process | None = None

    def __call__(self):
        self.engine_dead = True
        # Terminate engine core process to release GPU memory.
        if self.process is not None:
            try:
                self.process.terminate()
            except Exception:
                pass
            self.process.join(timeout=5)
            if self.process.is_alive():
                try:
                    os.kill(self.process.pid, signal.SIGKILL)
                except Exception:
                    pass
                self.process.join(timeout=5)

        if self.output_queue_task is not None:
            loop = self.output_queue_task._loop if self.output_queue_task else None
            sockets = (self.output_socket, self.input_socket)

            def close_sockets_and_tasks():
                for sock in sockets:
                    if sock is not None:
                        try:
                            sock.close(linger=0)
                        except Exception:
                            pass
                if self.output_queue_task is not None and not self.output_queue_task.done():
                    try:
                        self.output_queue_task.cancel()
                    except Exception:
                        pass

            if loop is not None:
                try:
                    if asyncio.get_running_loop() is loop:
                        close_sockets_and_tasks()
                    elif not loop.is_closed():
                        loop.call_soon_threadsafe(close_sockets_and_tasks)
                    else:
                        close_sockets_and_tasks()
                except RuntimeError:
                    close_sockets_and_tasks()
            else:
                for sock in sockets:
                    if sock is not None:
                        try:
                            sock.close(linger=0)
                        except Exception:
                            pass
        else:
            for sock in (self.output_socket, self.input_socket):
                if sock is not None:
                    try:
                        sock.close(linger=0)
                    except Exception:
                        pass
            if self.shutdown_path is not None:
                try:
                    with self.sync_ctx.socket(zmq.PAIR) as shutdown_sender:
                        shutdown_sender.connect(self.shutdown_path)
                        shutdown_sender.send(b"")
                except Exception:
                    pass
        # Do not call sync_ctx.term() here. vLLM also skips this — the
        # sockets are already closed and the context will be collected by GC.
        # Calling term() races with pending async socket callbacks and causes
        # zmq.error.ContextTerminated on shutdown.


class MPClient(EngineCoreClient):
    """Base client for multi-proc EngineCore.

    Manages:
    - ZMQ context (sync or async, based on asyncio_mode)
    - ZMQ sockets (ROUTER for input, PULL for output)
    - Background EngineCore process
    - Encoder/decoder
    - Process shutdown via proc.terminate() -> SIGTERM -> EngineShutdownState
    - Engine core monitor thread

    Subclasses implement output consumption:
    - SyncMPClient: background thread -> queue.Queue
    - AsyncMPClient: async task -> asyncio.Queue
    """

    def __init__(
        self,
        asyncio_mode: bool,
        vkwr_config: VkwrConfig,
        executor_class=None,
        log_stats=True,
    ):
        self.vkwr_config = vkwr_config

        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx
        self._sync_ctx = sync_ctx

        self._resources = _BackgroundResources(sync_ctx)
        self._finalizer = weakref.finalize(self, self._resources)

        self._input_address, self._output_address = self._generate_addresses()
        device = vkwr_config.worker_config.device

        from vkwr.engine.core_proc import make_zmq_socket

        # Bind client-side sockets BEFORE spawning the engine process so the
        # engine's DEALER/PUSH can successfully connect to a listening peer.
        self._input_socket = self._resources.input_socket = make_zmq_socket(self.ctx, self._input_address, zmq.ROUTER, bind=True)
        self._output_socket = self._resources.output_socket = make_zmq_socket(self.ctx, self._output_address, zmq.PULL, bind=True)

        from vkwr.engine.core_proc import _core_proc_target

        ctx = multiprocessing.get_context("spawn")
        self._process = ctx.Process(
            target=_core_proc_target,
            args=(vkwr_config, device, self._input_address, self._output_address),
            daemon=True,
        )
        self._process.start()
        self._resources.process = self._process

        from vkwr.config.engine import ENGINE_READY_TIMEOUT_S

        timeout_ms = ENGINE_READY_TIMEOUT_S * 1000

        if asyncio_mode:
            sync_output = zmq.Socket.shadow(self._output_socket)
            sync_output.setsockopt(zmq.RCVTIMEO, timeout_ms)
            ready_frame = sync_output.recv()
        else:
            self._output_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            ready_frame = self._output_socket.recv()

        if ready_frame != b"READY":
            raise TimeoutError(
                f"Timed out waiting for engine core process to start. "
                f"This is often caused by slow weight loading for large models. "
                f"Waited {ENGINE_READY_TIMEOUT_S}s (configured by "
                f"VKWR_ENGINE_READY_TIMEOUT_S). To increase the timeout, set "
                f"VKWR_ENGINE_READY_TIMEOUT_S=<seconds>."
            )

        from vkwr.engine.serial_utils import MsgpackDecoder, MsgpackEncoder

        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder()

        self.utility_results: dict[int, futures.Future] = {}

        self.core_engine = b"\x00"

        self._start_engine_core_monitor()
        self._engine_dead = False

    def _start_engine_core_monitor(self):
        import multiprocessing.connection as conn

        proc = self._process
        self_ref = weakref.ref(self)

        def _monitor():
            tried = conn.wait([proc.sentinel], timeout=None)
            if tried:
                _self = self_ref()
                if not _self or _self._engine_dead:
                    return
                _self._engine_dead = True
                logger.warning(
                    "Engine core process (PID %d) exited with code %s. Shutting down client.",
                    proc.pid,
                    proc.exitcode,
                )
                if _self._finalizer.alive:
                    _self._finalizer()

        Thread(
            target=_monitor,
            daemon=True,
            name="VKWR-EngineMonitor",
        ).start()

    def ensure_alive(self):
        if self._engine_dead or not self._process.is_alive():
            self._engine_dead = True
            raise EngineDeadError(f"Engine process (PID {self._process.pid}) exited with code {self._process.exitcode}")

    def _format_exception(self, e: Exception) -> Exception:
        return EngineDeadError() if self._engine_dead else e

    @staticmethod
    def _process_utility_output(output, utility_results):
        future = utility_results.pop(output.call_id, None)
        if future is None:
            logger.warning("Utility call_id %d not found, dropping response", output.call_id)
            return
        try:
            if output.failure_message is not None:
                future.set_exception(Exception(output.failure_message))
            else:
                future.set_result(output.result.result if output.result else None)
        except Exception:
            pass

    @staticmethod
    def _invoke_utility_method(name, get_result, output, enqueue_output):
        from concurrent.futures import Future as StdFuture

        from vkwr.engine.core_request import UtilityResult

        try:
            result = get_result()
            if isinstance(result, StdFuture):

                def callback(f):
                    MPClient._invoke_utility_method(name, f.result, output, enqueue_output)

                result.add_done_callback(callback)
                return
            output.result = UtilityResult(result=result)
        except Exception as e:
            output.failure_message = f"Call to {name} method failed: {str(e)}"
        enqueue_output(output)

    def _send_input(self, request_type: EngineCoreRequestType, request):
        self.ensure_alive()
        frames = self.encoder.encode(request) if request is not None else []
        msg = [self.core_engine, request_type.value, *frames]
        self._input_socket.send_multipart(msg, copy=False)

    def shutdown(self):
        if self._finalizer.detach() is None:
            return
        try:
            self._process.terminate()
        except Exception:
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():
            try:
                os.kill(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.join(timeout=5)
        self._resources()

    @staticmethod
    def _generate_addresses():
        session_id = uuid.uuid4().hex
        return (
            f"ipc:///tmp/vkwr-input-{session_id}",
            f"ipc:///tmp/vkwr-output-{session_id}",
        )


class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore.

    Background thread reads from ZMQ PULL socket, decodes msgpack frames,
    and pushes results into a queue.Queue for get_output() to consume.

    Uses an inproc:// PAIR socket pair to unblock the reader thread during
    shutdown.
    """

    def __init__(
        self,
        vkwr_config: VkwrConfig,
        executor_class=None,
        log_stats=True,
    ):
        super().__init__(asyncio_mode=False, vkwr_config=vkwr_config, executor_class=executor_class, log_stats=log_stats)

        self.outputs_queue: queue.Queue[EngineCoreOutputs | EngineDeadError | Exception] = queue.Queue()

        # Capture local refs to avoid keeping a strong ref to self in the
        # reader thread (which would prevent GC of the client).
        ctx = self.ctx
        out_socket = self._output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self.outputs_queue
        resources = self._resources

        shutdown_path = f"inproc://{uuid.uuid4().hex}"
        resources.shutdown_path = shutdown_path

        from vkwr.engine.core_proc import EngineCoreProc

        def process_outputs_socket():
            assert isinstance(out_socket, zmq.Socket)
            shutdown_socket = ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(out_socket, zmq.POLLIN)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        break

                    frames = out_socket.recv_multipart(copy=False)
                    raw = bytes(frames[0]) if len(frames) == 1 else None
                    if raw == EngineCoreProc.ENGINE_CORE_DEAD:
                        outputs_queue.put_nowait(EngineDeadError())
                        break
                    # EngineCoreProc sends (client_idx, EngineCoreOutputs) tuples.
                    # Decode with _OutputFrame type to reconstruct EngineCoreOutputs.
                    frame = decoder.decode([raw], _OutputFrame)
                    outputs = frame.outputs
                    if outputs.utility_output:
                        MPClient._process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            finally:
                shutdown_socket.close(linger=0)
                out_socket.close(linger=0)

        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="VKWR-EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()

        self._resources.output_socket = None

    def add_request(self, request: EngineCoreRequest) -> None:
        self._send_input(EngineCoreRequestType.ADD, request)

    def get_output(self, timeout: float | None = None) -> EngineCoreOutputs:
        try:
            result = self.outputs_queue.get(timeout=timeout)
        except queue.Empty:
            if self._engine_dead or not self._process.is_alive():
                self._engine_dead = True
                raise EngineDeadError()
            return EngineCoreOutputs()
        if isinstance(result, Exception):
            raise self._format_exception(result) from None
        return result

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids and not self._resources.engine_dead:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    def call_utility(self, method: str, *args: Any) -> Any:
        call_id = uuid.uuid1().int >> 64
        future: futures.Future[Any] = futures.Future()
        self.utility_results[call_id] = future
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))
        return future.result()

    def shutdown(self):
        try:
            if self.output_queue_thread.is_alive():
                with self._sync_ctx.socket(zmq.PAIR) as shutdown_sender:
                    shutdown_sender.connect(self._resources.shutdown_path)
                    shutdown_sender.send(b"")
                self.output_queue_thread.join(timeout=5)
        except Exception:
            pass
        super().shutdown()


class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore.

    Background asyncio task reads from ZMQ async PULL socket, decodes
    msgpack frames, and pushes results into an asyncio.Queue for
    get_output_async() to consume.

    Uses task.cancel() to unblock the reader during shutdown (no PAIR socket).
    """

    def __init__(
        self,
        vkwr_config: VkwrConfig,
        executor_class=None,
        log_stats=True,
    ):
        super().__init__(asyncio_mode=True, vkwr_config=vkwr_config, executor_class=executor_class, log_stats=log_stats)

        self._outputs_queue: asyncio.Queue[EngineCoreOutputs | Exception] | None = None
        self._output_task: asyncio.Task | None = None
        self._outputs_queue_ready = False

    def _ensure_output_queue(self):
        if self._outputs_queue_ready:
            return
        self._outputs_queue = asyncio.Queue()

        out_socket = self._output_socket
        decoder = self.decoder
        utility_results = self.utility_results
        outputs_queue = self._outputs_queue
        resources = self._resources

        from vkwr.engine.core_proc import EngineCoreProc

        async def process_outputs_socket():
            try:
                assert isinstance(out_socket, zmq.asyncio.Socket)
                while True:
                    try:
                        frames = await out_socket.recv_multipart(copy=False)
                    except zmq.Again:
                        continue
                    raw = bytes(frames[0]) if len(frames) == 1 else None
                    if raw == EngineCoreProc.ENGINE_CORE_DEAD:
                        outputs_queue.put_nowait(EngineDeadError())
                        break
                    frame = decoder.decode([raw], _OutputFrame)
                    outputs = frame.outputs
                    if outputs.utility_output:
                        MPClient._process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())
            except Exception as e:
                outputs_queue.put_nowait(e)

        self._output_task = asyncio.create_task(
            process_outputs_socket(),
            name="VKWR-EngineCoreOutputQueueTask",
        )
        resources.output_queue_task = self._output_task
        resources.output_socket = out_socket
        self._outputs_queue_ready = True

    def _ensure_output_queue_task(self):
        if self._output_task is None or self._output_task.done():
            raise EngineDeadError()

    async def get_output_async(self, timeout: float | None = None) -> EngineCoreOutputs:
        self._ensure_output_queue()
        self._ensure_output_queue_task()
        try:
            if timeout is not None:
                result = await asyncio.wait_for(self._outputs_queue.get(), timeout=timeout)
            else:
                result = await self._outputs_queue.get()
        except asyncio.TimeoutError:
            if self._engine_dead or not self._process.is_alive():
                self._engine_dead = True
                raise EngineDeadError()
            return EngineCoreOutputs()
        if isinstance(result, Exception):
            raise self._format_exception(result) from None
        return result

    async def add_request(self, request: EngineCoreRequest) -> None:
        self._ensure_output_queue()
        self.ensure_alive()
        frames = self.encoder.encode(request) if request is not None else []
        msg = [self.core_engine, EngineCoreRequestType.ADD.value, *frames]
        await self._input_socket.send_multipart(msg, copy=False)

    async def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids and not self._resources.engine_dead:
            self.ensure_alive()
            frames = self.encoder.encode(request_ids) if request_ids is not None else []
            msg = [self.core_engine, EngineCoreRequestType.ABORT.value, *frames]
            await self._input_socket.send_multipart(msg, copy=False)

    async def call_utility_async(self, method: str, *args: Any) -> Any:
        self._ensure_output_queue()
        call_id = uuid.uuid1().int >> 64
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self.utility_results[call_id] = future
        self._ensure_output_queue_task()
        payload = (0, call_id, method, args)
        frames = self.encoder.encode(payload)
        msg = [self.core_engine, EngineCoreRequestType.UTILITY.value, *frames]
        await self._input_socket.send_multipart(msg, copy=False)
        return await future

    def shutdown(self):
        try:
            if self._output_task is not None and not self._output_task.done():
                self._output_task.cancel()
        except Exception:
            pass
        super().shutdown()


class InprocClient(EngineCoreClient):
    """EngineCore runs in the same process. Direct method calls."""

    def __init__(self, vkwr_config: VkwrConfig) -> None:
        from vkwr.engine.core import EngineCore

        self.engine_core: EngineCore = EngineCore(vkwr_config)
        self.engine_core.initialize()
        self._outputs_queue: queue.Queue[EngineCoreOutputs] = queue.Queue()

    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    def get_output(self, timeout: float | None = None) -> EngineCoreOutputs:
        # Drain any abort outputs first
        while not self._outputs_queue.empty():
            try:
                return self._outputs_queue.get_nowait()
            except queue.Empty:
                break

        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            outputs, model_executed = self.engine_core.step_fn()
            self.engine_core.post_step(model_executed=model_executed)
            result = outputs and outputs.get(0) or EngineCoreOutputs()
            if result.outputs:
                return result
            # No output yet (e.g., batch queue filling). Loop or block briefly.
            if not self.engine_core.scheduler.has_requests():
                return result
            if deadline is not None and time.monotonic() >= deadline:
                return result
            if not model_executed:
                time.sleep(0.001)

    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            abort_outputs = self.engine_core.abort_requests(request_ids)
            for outputs in abort_outputs.values():
                self._outputs_queue.put_nowait(outputs)

    def shutdown(self, timeout: float | None = None) -> None:
        self.engine_core.shutdown()

    def call_utility(self, method: str, *args: Any) -> Any:
        fn = getattr(self.engine_core, method)
        return fn(*args)

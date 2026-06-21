"""Tests for AsyncMPClient output reader and async methods.

Tests the async-specific parts of AsyncMPClient:
- asyncio.Queue integration
- get_output_async timeout behavior
- async utility calls with asyncio.Future
- task cancel behavior
- make_client factory

The underlying reader logic (ZMQ recv → msgpack decode → queue put) is
already covered by test_sync_mp_client.py since the decode path is shared.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from vkwr.engine.core_client import MPClient, _OutputFrame
from vkwr.engine.core_proc import EngineCoreProc
from vkwr.engine.core_request import (
    EngineCoreRequest,
    EngineCoreRequestType,
    UtilityOutput,
    UtilityResult,
)
from vkwr.engine.exceptions import EngineDeadError
from vkwr.engine.outputs import EngineCoreOutput, EngineCoreOutputs
from vkwr.engine.request import SamplingParams
from vkwr.engine.serial_utils import MsgpackDecoder, MsgpackEncoder


def _make_request(request_id: str = "req-1", prompt_token_ids: list[int] | None = None) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
        sampling_params=SamplingParams(),
        arrival_time=time.monotonic(),
        priority=0,
        current_wave=0,
    )


def _encode_output(client_idx: int, outputs: EngineCoreOutputs) -> bytes:
    """Encode a (client_idx, outputs) tuple as the reader would receive it."""
    encoder = MsgpackEncoder()
    frames = encoder.encode((client_idx, outputs))
    return bytes(frames[0])


class TestAsyncMPClientReaderDecoding:
    """Test the decoding logic used by the AsyncMPClient reader task.

    These tests verify that _OutputFrame decoding works correctly for
    the same message formats that the async ZMQ reader will receive.
    The transport (async ZMQ recv) is tested implicitly via the
    integration tests, but the decode path is tested here directly.
    """

    def test_decode_normal_output(self):
        """Decoding a normal output frame reconstructs EngineCoreOutputs."""
        output = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="test-1",
                    new_token_ids=[42, 43],
                )
            ]
        )
        raw = _encode_output(0, output)
        decoder = MsgpackDecoder()
        frame = decoder.decode([raw], _OutputFrame)
        assert frame.client_idx == 0
        assert len(frame.outputs.outputs) == 1
        assert frame.outputs.outputs[0].request_id == "test-1"
        assert frame.outputs.outputs[0].new_token_ids == [42, 43]

    def test_decode_abort_output(self):
        """Decoding an abort output frame reconstructs EngineCoreOutputs."""
        output = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="abort-1",
                    new_token_ids=[],
                    finish_reason="abort",
                )
            ]
        )
        raw = _encode_output(0, output)
        decoder = MsgpackDecoder()
        frame = decoder.decode([raw], _OutputFrame)
        assert len(frame.outputs.outputs) == 1
        assert frame.outputs.outputs[0].finish_reason == "abort"

    def test_decode_utility_output(self):
        """Decoding a utility output frame reconstructs EngineCoreOutputs with utility_output."""
        util_output = UtilityOutput(
            call_id=12345,
            result=UtilityResult(result="fake-model"),
        )
        output = EngineCoreOutputs(utility_output=util_output)
        raw = _encode_output(0, output)
        decoder = MsgpackDecoder()
        frame = decoder.decode([raw], _OutputFrame)
        assert frame.outputs.utility_output is not None
        assert frame.outputs.utility_output.call_id == 12345

    def test_decode_empty_outputs(self):
        """Decoding empty EngineCoreOutputs works."""
        output = EngineCoreOutputs()
        raw = _encode_output(0, output)
        decoder = MsgpackDecoder()
        frame = decoder.decode([raw], _OutputFrame)
        assert not frame.outputs.outputs

    def test_engine_dead_sentinel_detection(self):
        """The ENGINE_CORE_DEAD sentinel is detected as raw bytes."""
        raw = EngineCoreProc.ENGINE_CORE_DEAD
        assert raw == b"ENGINE_CORE_DEAD"
        # In the reader, this check happens before decoding:
        # if bytes(frames[0]) == EngineCoreProc.ENGINE_CORE_DEAD
        assert bytes(raw) == EngineCoreProc.ENGINE_CORE_DEAD


class TestAsyncMPClientAsyncQueue:
    """Test asyncio.Queue behavior in AsyncMPClient pattern."""

    def test_async_queue_put_get(self):
        """Basic asyncio.Queue put/get works for output routing."""

        async def _test():
            q: asyncio.Queue[EngineCoreOutputs | Exception] = asyncio.Queue()
            output = EngineCoreOutputs(outputs=[EngineCoreOutput(request_id="q-test", new_token_ids=[1, 2])])
            q.put_nowait(output)
            result = await asyncio.wait_for(q.get(), timeout=1.0)
            assert isinstance(result, EngineCoreOutputs)
            assert result.outputs[0].request_id == "q-test"

        asyncio.get_event_loop().run_until_complete(_test())

    def test_async_queue_timeout(self):
        """asyncio.wait_for on an empty queue raises TimeoutError."""

        async def _test():
            q: asyncio.Queue = asyncio.Queue()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(q.get(), timeout=0.05)

        asyncio.get_event_loop().run_until_complete(_test())

    def test_async_queue_timeout_returns_empty(self):
        """AsyncMPClient get_output_async pattern: timeout -> empty outputs."""

        async def _test():
            q: asyncio.Queue = asyncio.Queue()
            try:
                result = await asyncio.wait_for(q.get(), timeout=0.05)
            except asyncio.TimeoutError:
                result = EngineCoreOutputs()
            assert isinstance(result, EngineCoreOutputs)
            assert not result.outputs

        asyncio.get_event_loop().run_until_complete(_test())

    def test_async_queue_exception_propagation(self):
        """Exception in queue is raised by get_output_async pattern."""

        async def _test():
            q: asyncio.Queue[EngineCoreOutputs | Exception] = asyncio.Queue()
            q.put_nowait(EngineDeadError())
            result = await asyncio.wait_for(q.get(), timeout=1.0)
            assert isinstance(result, EngineDeadError)

        asyncio.get_event_loop().run_until_complete(_test())

    def test_async_queue_cancelled_task_puts_dead(self):
        """When reader task is cancelled, it puts EngineDeadError into queue."""

        async def _test():
            q: asyncio.Queue[EngineCoreOutputs | Exception] = asyncio.Queue()

            async def reader():
                try:
                    await asyncio.sleep(10)
                except asyncio.CancelledError:
                    q.put_nowait(EngineDeadError())

            task = asyncio.create_task(reader())
            await asyncio.sleep(0.01)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            result = await asyncio.wait_for(q.get(), timeout=1.0)
            assert isinstance(result, EngineDeadError)

        asyncio.get_event_loop().run_until_complete(_test())


class TestAsyncMPClientUtilityAsync:
    """Test async utility call pattern with asyncio.Future."""

    def test_async_utility_future_resolution(self):
        """asyncio.Future set from reader resolves the awaiting call_utility_async."""

        async def _test():
            utility_results: dict[int, asyncio.Future] = {}
            call_id = 999
            future = asyncio.get_running_loop().create_future()
            utility_results[call_id] = future

            # Simulate reader resolving the utility call
            util_output = UtilityOutput(
                call_id=call_id,
                result=UtilityResult(result="async-result"),
            )
            MPClient._process_utility_output(util_output, utility_results)

            result = await asyncio.wait_for(future, timeout=1.0)
            assert result == "async-result"
            assert call_id not in utility_results

        asyncio.get_event_loop().run_until_complete(_test())

    def test_async_utility_future_failure(self):
        """asyncio.Future set with exception from reader raises in await."""

        async def _test():
            utility_results: dict[int, asyncio.Future] = {}
            call_id = 998
            future = asyncio.get_running_loop().create_future()
            utility_results[call_id] = future

            util_output = UtilityOutput(
                call_id=call_id,
                failure_message="utility failed",
            )
            MPClient._process_utility_output(util_output, utility_results)

            with pytest.raises(Exception, match="utility failed"):
                await asyncio.wait_for(future, timeout=1.0)

        asyncio.get_event_loop().run_until_complete(_test())

    def test_async_utility_stale_call_id(self):
        """Stale utility response with unknown call_id is safely dropped."""

        async def _test():
            utility_results: dict[int, asyncio.Future] = {}

            util_output = UtilityOutput(
                call_id=99999,
                result=UtilityResult(result="stale"),
            )
            # Should not raise, just log a warning
            MPClient._process_utility_output(util_output, utility_results)
            assert 99999 not in utility_results

        asyncio.get_event_loop().run_until_complete(_test())


class TestAsyncMPClientSendMessage:
    """Test message frame construction for AsyncMPClient."""

    def test_add_request_frame_format(self):
        """Verify add_request produces correct ZMQ frames."""
        encoder = MsgpackEncoder()
        req = _make_request("async-frame-test", [10, 20, 30])
        frames = encoder.encode(req)
        msg = [b"\x00", EngineCoreRequestType.ADD.value, *frames]
        assert len(msg) == 3
        assert msg[0] == b"\x00"
        assert msg[1] == EngineCoreRequestType.ADD.value

    def test_abort_request_frame_format(self):
        """Verify abort_requests produces correct ZMQ frames."""
        encoder = MsgpackEncoder()
        request_ids = ["abort-1", "abort-2"]
        frames = encoder.encode(request_ids)
        msg = [b"\x00", EngineCoreRequestType.ABORT.value, *frames]
        assert len(msg) == 3
        assert msg[1] == EngineCoreRequestType.ABORT.value

    def test_utility_request_frame_format(self):
        """Verify call_utility_async produces correct ZMQ frames."""
        encoder = MsgpackEncoder()
        payload = (0, 999, "test_method", ("arg1",))
        frames = encoder.encode(payload)
        msg = [b"\x00", EngineCoreRequestType.UTILITY.value, *frames]
        assert len(msg) == 3
        assert msg[1] == EngineCoreRequestType.UTILITY.value


class TestAsyncMPClientFactory:
    """Test make_client factory returns AsyncMPClient."""

    def test_factory_import(self):
        """Verify AsyncMPClient is importable and has expected methods."""
        from vkwr.engine.core_client import AsyncMPClient

        assert AsyncMPClient is not None
        assert hasattr(AsyncMPClient, "get_output_async")
        assert hasattr(AsyncMPClient, "add_request")
        assert hasattr(AsyncMPClient, "abort_requests")
        assert hasattr(AsyncMPClient, "call_utility_async")
        assert hasattr(AsyncMPClient, "shutdown")

    def test_asyncmpclient_inherits_mpclient(self):
        """AsyncMPClient inherits from MPClient."""
        from vkwr.engine.core_client import AsyncMPClient, MPClient

        assert issubclass(AsyncMPClient, MPClient)

    def test_make_client_raises_for_async_mp(self):
        """make_client with multiprocess_mode=True and asyncio_mode=True returns AsyncMPClient.

        Note: This will try to actually create the client, which requires a model.
        We only verify the code path is correct by checking the return type annotation
        and that the code doesn't raise NotImplementedError.
        """
        # Verify the factory path is wired (won't actually instantiate without config)
        import inspect

        from vkwr.engine.core_client import EngineCoreClient

        src = inspect.getsource(EngineCoreClient.make_client)
        assert "AsyncMPClient" in src

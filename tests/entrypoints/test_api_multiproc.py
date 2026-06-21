"""Integration tests for api.py against AsyncLLMEngine.generate().

Covers:
- Stream completion and chat completion via async engine
- Non-streaming completion via async engine
- Concurrent streaming requests without race conditions
- EngineDeadError propagation through API layer
- CancelledError -> abort -> request cleanup
- Non-streaming completion resilience
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.responses import StreamingResponse

from vkwr.config.vkwr import EngineArgs
from vkwr.engine.async_llm import AsyncLLMEngine
from vkwr.engine.exceptions import EngineDeadError, EngineGenerateError
from vkwr.engine.outputs import CompletionOutput, RequestOutput, RequestStats
from vkwr.engine.request import SamplingParams

# ── Helpers ─────────────────────────────────────────────────────────


def _make_output(
    request_id: str = "req-1",
    finished: bool = False,
    finish_reason: str | None = None,
    text: str = "",
    token_ids: list[int] | None = None,
) -> RequestOutput:
    return RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=[1, 2],
        outputs=[
            CompletionOutput(
                index=0,
                token_ids=token_ids or [42],
                text=text,
                cumlogprob=0.0,
                finish_reason=finish_reason,
            )
        ],
        finished=finished,
        finish_reason=finish_reason,
        stats=RequestStats(generated_tokens=1),
    )


async def _seq_gen(*items: RequestOutput) -> AsyncGenerator[RequestOutput, None]:
    for item in items:
        yield item


async def _error_gen(exc: Exception) -> AsyncGenerator[RequestOutput, None]:
    raise exc
    yield  # unreachable


async def _drain_sse(response: StreamingResponse) -> str:
    body = b""
    async for chunk in response.body_iterator:
        body += chunk if isinstance(chunk, bytes) else chunk.encode()
    return body.decode()


# ── EngineArgs multiprocess flag ────────────────────────────────────


class TestEngineArgsMultiprocess:
    def test_enable_multiprocessing_default_true(self) -> None:
        """EngineArgs defaults enable_multiprocessing to True."""
        args = EngineArgs(model="test-model")
        assert args.enable_multiprocessing is True

    def test_disable_multiprocessing_can_be_false(self) -> None:
        """--disable-multiprocessing on CLI sets enable_multiprocessing to False."""
        args = EngineArgs(model="test-model", enable_multiprocessing=False)
        assert args.enable_multiprocessing is False

    def test_from_engine_args_passes_multiprocess_mode(self) -> None:
        """AsyncLLMEngine.from_engine_args forces multiprocess_mode=True regardless of EngineArgs."""
        args = EngineArgs(model="test-model", enable_multiprocessing=False)
        with patch.object(AsyncLLMEngine, "__init__", return_value=None) as mock_init:
            args.create_engine_config = MagicMock()
            AsyncLLMEngine.from_engine_args(args)
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args
            assert call_kwargs[1]["multiprocess_mode"] is True

    def test_from_engine_args_forces_multiprocess_true(self) -> None:
        args = EngineArgs(model="test-model")
        with patch.object(AsyncLLMEngine, "__init__", return_value=None) as mock_init:
            args.create_engine_config = MagicMock()
            AsyncLLMEngine.from_engine_args(args)
            call_kwargs = mock_init.call_args
            assert call_kwargs[1]["multiprocess_mode"] is True


# ── Stream completion integration ───────────────────────────────────


class TestStreamCompletionIntegration:
    @patch("vkwr.utils.generate_request_id")
    def test_stream_yields_delta_chunks_then_done(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_completion

        mock_id.return_value = "req-1"

        out1 = _make_output("req-1", finished=False, text="He")
        out2 = _make_output("req-1", finished=True, text="llo", finish_reason="length")

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _seq_gen(out1, out2)

        async def run():
            resp = await _stream_completion(mock_engine, "hello", SamplingParams(), "m")
            body = await _drain_sse(resp)
            parts = [line for line in body.split("\n\n") if line.startswith("data: ")]
            chunks = []
            for p in parts:
                payload = p[6:]
                if payload == "[DONE]":
                    continue
                chunks.append(json.loads(payload))
            assert len(chunks) == 2
            assert chunks[0]["choices"][0]["text"] == "He"
            assert chunks[1]["choices"][0]["text"] == "llo"
            assert chunks[1]["choices"][0]["finish_reason"] == "length"

        asyncio.get_event_loop().run_until_complete(run())

    @patch("vkwr.utils.generate_request_id")
    def test_stream_handles_empty_output_text(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_completion

        mock_id.return_value = "req-1"

        out = _make_output("req-1", finished=True, text="", finish_reason="stop")

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _seq_gen(out)

        async def run():
            resp = await _stream_completion(mock_engine, "", SamplingParams(), "m")
            body = await _drain_sse(resp)
            assert '"text": ""' in body

        asyncio.get_event_loop().run_until_complete(run())

    @patch("vkwr.utils.generate_request_id")
    def test_stream_handles_no_outputs(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_completion

        mock_id.return_value = "req-1"

        out = _make_output("req-1", finished=True, text="ok")
        out.outputs = []

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _seq_gen(out)

        async def run():
            resp = await _stream_completion(mock_engine, "hi", SamplingParams(), "m")
            body = await _drain_sse(resp)
            assert '"text": ""' in body

        asyncio.get_event_loop().run_until_complete(run())


# ── Stream chat completion integration ──────────────────────────────


class TestStreamChatCompletionIntegration:
    @patch("vkwr.utils.generate_request_id")
    def test_chat_stream_yields_initial_role_then_deltas(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_chat_completion

        mock_id.return_value = "req-1"

        out1 = _make_output("req-1", finished=False, text="He")
        out2 = _make_output("req-1", finished=True, text="llo", finish_reason="stop")

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _seq_gen(out1, out2)

        async def run():
            resp = await _stream_chat_completion(mock_engine, "hello", SamplingParams(), "m")
            body = await _drain_sse(resp)
            assert '"role": "assistant"' in body
            assert '"content": "He"' in body
            assert '"content": "llo"' in body
            assert '"finish_reason": "stop"' in body

        asyncio.get_event_loop().run_until_complete(run())


# ── Concurrent streaming requests ───────────────────────────────────


class TestConcurrentStreaming:
    @patch("vkwr.utils.generate_request_id")
    def test_concurrent_streams_no_race(self, mock_id) -> None:
        """Two concurrent stream requests should not interfere."""
        from vkwr.entrypoints.api import _stream_completion

        call_ids: list[str] = []

        async def tracked_gen(req_id: str, *a, **kw):
            call_ids.append(req_id)
            yield _make_output(req_id, finished=True, text=f"from-{req_id}")

        mock_engine = MagicMock()
        mock_engine.generate = tracked_gen

        async def run():
            resp1 = await _stream_completion(mock_engine, "p1", SamplingParams(), "m")
            resp2 = await _stream_completion(mock_engine, "p2", SamplingParams(), "m")
            b1 = await _drain_sse(resp1)
            b2 = await _drain_sse(resp2)
            return b1, b2

        b1, b2 = asyncio.get_event_loop().run_until_complete(run())
        assert len(call_ids) == 2
        assert "from-" in b1
        assert "from-" in b2


# ── EngineDeadError propagation ────────────────────────────────────


class TestEngineDeadPropagation:
    @patch("vkwr.utils.generate_request_id")
    def test_stream_completion_catches_engine_dead(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_completion

        mock_id.return_value = "req-1"

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _error_gen(EngineDeadError())

        async def run():
            resp = await _stream_completion(mock_engine, "hi", SamplingParams(), "m")
            body = await _drain_sse(resp)
            assert '"finish_reason": "error"' in body
            assert "data: [DONE]" not in body

        asyncio.get_event_loop().run_until_complete(run())

    @patch("vkwr.utils.generate_request_id")
    def test_stream_chat_catches_engine_dead(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_chat_completion

        mock_id.return_value = "req-1"

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _error_gen(EngineDeadError())

        async def run():
            resp = await _stream_chat_completion(mock_engine, "hi", SamplingParams(), "m")
            body = await _drain_sse(resp)
            assert '"finish_reason": "error"' in body

        asyncio.get_event_loop().run_until_complete(run())

    @patch("vkwr.utils.generate_request_id")
    def test_collect_completion_propagates_engine_dead(self, mock_id) -> None:
        from vkwr.entrypoints.api import _collect_completion

        mock_id.return_value = "req-1"

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _error_gen(EngineDeadError())

        async def run():
            with pytest.raises(EngineDeadError):
                await _collect_completion(mock_engine, "hi", SamplingParams())

        asyncio.get_event_loop().run_until_complete(run())

    @patch("vkwr.utils.generate_request_id")
    def test_collect_completion_propagates_generate_error(self, mock_id) -> None:
        from vkwr.entrypoints.api import _collect_completion

        mock_id.return_value = "req-1"

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _error_gen(EngineGenerateError())

        async def run():
            with pytest.raises(EngineGenerateError):
                await _collect_completion(mock_engine, "hi", SamplingParams())

        asyncio.get_event_loop().run_until_complete(run())

    def test_nonstream_completion_httpexception_on_engine_dead(self) -> None:
        """Non-streaming endpoint should raise HTTPException 503 on EngineDeadError."""
        from vkwr.entrypoints.api import _collect_completion

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _error_gen(EngineDeadError())

        async def run():
            with pytest.raises(EngineDeadError):
                await _collect_completion(mock_engine, "hi", SamplingParams())

        asyncio.get_event_loop().run_until_complete(run())

    def test_nonstream_completion_httpexception_on_generate_error(self) -> None:
        """Non-streaming endpoint should raise HTTPException 500 on EngineGenerateError."""
        from vkwr.entrypoints.api import _collect_completion

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _error_gen(EngineGenerateError())

        async def run():
            with pytest.raises(EngineGenerateError):
                await _collect_completion(mock_engine, "hi", SamplingParams())

        asyncio.get_event_loop().run_until_complete(run())


# ── CancelledError -> abort ────────────────────────────────────────


class TestCancelledErrorAbort:
    def test_generate_aborts_on_cancel(self) -> None:
        """When generate() receives CancelledError, it should abort the request."""
        from vkwr.engine.input_processor import InputProcessor
        from vkwr.engine.output_processor import OutputProcessor

        config = MagicMock()
        config.model_config = MagicMock()
        config.scheduler_config = MagicMock()

        mock_core = AsyncMock()
        mock_core.add_request = AsyncMock()
        mock_core.abort_requests = AsyncMock()
        mock_core.get_output_async = AsyncMock(return_value=MagicMock(outputs=[]))

        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        engine._engine_core = mock_core
        engine._input_processor = InputProcessor(config.model_config, config.scheduler_config)
        engine._output_processor = OutputProcessor(config.model_config)
        engine._config = config
        engine._output_handler = None
        engine._errored = False

        sampling_params = SamplingParams()
        req_id = "test-cancel-1"
        abort_called = False

        async def fake_abort(rid):
            nonlocal abort_called
            abort_called = True

        with patch.object(engine, "add_request", new_callable=AsyncMock) as mock_add:
            collector = MagicMock()
            collector.request_id = req_id
            collector.get_nowait.return_value = None
            collector.get = AsyncMock(side_effect=asyncio.CancelledError())
            mock_add.return_value = collector

            with patch.object(engine, "abort", new=fake_abort):

                async def consume():
                    gen = engine.generate("hi", sampling_params, req_id)
                    try:
                        async for _ in gen:
                            pass
                    except asyncio.CancelledError:
                        await gen.aclose()
                        raise

                with pytest.raises(asyncio.CancelledError):
                    asyncio.get_event_loop().run_until_complete(consume())

        assert abort_called


# ── create_app integration ─────────────────────────────────────────


class TestCreateApp:
    def test_create_app_no_ensure_engine_call(self) -> None:
        """create_app should not call _ensure_engine (method doesn't exist)."""
        from vkwr.entrypoints.api import create_app

        mock_args = MagicMock(spec=EngineArgs)
        mock_args.model = "test-model"
        mock_args.create_engine_config = MagicMock()

        with patch("vkwr.entrypoints.api.AsyncLLMEngine") as MockEngine:
            mock_engine_instance = MagicMock()
            MockEngine.from_engine_args.return_value = mock_engine_instance

            create_app(mock_args)

            MockEngine.from_engine_args.assert_called_once_with(mock_args)
            assert not hasattr(mock_engine_instance, "_ensure_engine") or not mock_engine_instance._ensure_engine.called

    def test_create_app_engine_init_receives_multiprocess_mode(self) -> None:
        """create_app passes engine_args through from_engine_args which reads enable_multiprocessing."""
        from vkwr.entrypoints.api import create_app

        args = MagicMock(spec=EngineArgs)
        args.model = "test"
        args.enable_multiprocessing = False
        args.create_engine_config = MagicMock()

        with patch("vkwr.entrypoints.api.AsyncLLMEngine") as MockEngine:
            MockEngine.from_engine_args.return_value = MagicMock()
            create_app(args)
            MockEngine.from_engine_args.assert_called_once_with(args)


# ── create_app shutdown ────────────────────────────────────────────


class TestCreateAppShutdown:
    def test_create_app_no_on_event_deprecation(self) -> None:
        """create_app uses lifespan, not deprecated on_event."""
        import warnings

        from vkwr.entrypoints.api import create_app

        args = MagicMock(spec=EngineArgs)
        args.model = "test"
        args.create_engine_config = MagicMock()

        with patch("vkwr.entrypoints.api.AsyncLLMEngine") as MockEngine:
            MockEngine.from_engine_args.return_value = MagicMock()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                create_app(args)
                deprecation_warnings = [w for w in caught if "on_event" in str(w.message).lower()]
                assert len(deprecation_warnings) == 0, f"on_event deprecation warning present: {deprecation_warnings}"


# ── Prompt as token IDs ────────────────────────────────────────────


class TestTokenIdsPrompt:
    @patch("vkwr.utils.generate_request_id")
    def test_stream_completion_with_token_ids_prompt(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_completion

        mock_id.return_value = "req-1"

        out = _make_output("req-1", finished=True, text="Hello")
        prompt_received: list = []

        def capture_gen(prompt, *a, **kw):
            prompt_received.append(prompt)
            return _seq_gen(out)

        mock_engine = MagicMock()
        mock_engine.generate = capture_gen

        async def run():
            resp = await _stream_completion(mock_engine, [1, 2, 3], SamplingParams(), "m")
            await _drain_sse(resp)

        asyncio.get_event_loop().run_until_complete(run())
        assert prompt_received == [[1, 2, 3]]

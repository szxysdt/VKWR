from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.responses import StreamingResponse

from vkwr.engine.exceptions import EngineDeadError, EngineGenerateError
from vkwr.engine.outputs import CompletionOutput, RequestOutput, RequestStats
from vkwr.engine.request import RequestOutputKind

# ── Helpers ───────────────────────────────────────────────────────────


def _make_request_output(
    request_id: str = "req-1",
    finished: bool = False,
    finish_reason: str | None = None,
    token_ids: list[int] | None = None,
    text: str = "",
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


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _erroring_generator(exc: Exception) -> AsyncGenerator[RequestOutput, None]:
    """Async generator that raises an exception on first iteration."""
    raise exc
    yield  # type: ignore[unreachable] -- makes this an async generator


async def _seq_generator(*items: RequestOutput) -> AsyncGenerator[RequestOutput, None]:
    """Async generator that yields each item in order."""
    for item in items:
        yield item


async def _collect_sse(response: StreamingResponse) -> str:
    """Helper: drain StreamingResponse body into a single string."""
    body = b""
    async for chunk in response.body_iterator:
        body += chunk if isinstance(chunk, bytes) else chunk.encode()
    return body.decode()


# ── _make_sampling_params ────────────────────────────────────────────


class TestMakeSamplingParams:
    def test_streaming_output_kind(self) -> None:
        from vkwr.entrypoints.api import _make_sampling_params

        params = _make_sampling_params(stream=True)
        assert params.output_kind == RequestOutputKind.DELTA

    def test_non_streaming_output_kind(self) -> None:
        from vkwr.entrypoints.api import _make_sampling_params

        params = _make_sampling_params(stream=False)
        assert params.output_kind == RequestOutputKind.FINAL_ONLY

    def test_passes_through_sampling_params(self) -> None:
        from vkwr.entrypoints.api import _make_sampling_params

        params = _make_sampling_params(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            max_tokens=100,
            seed=42,
        )
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.min_p == 0.1
        assert params.max_tokens == 100
        assert params.seed == 42


# ── Response helpers ─────────────────────────────────────────────────


class TestMakeCompletionResponse:
    def test_make_completion_response(self) -> None:
        from vkwr.entrypoints.api import _make_completion_response

        resp = _make_completion_response("r1", "hello", "length", "model", 3, 2)
        assert resp["id"] == "cmpl-r1"
        assert resp["object"] == "text_completion"
        assert resp["choices"][0]["text"] == "hello"
        assert resp["choices"][0]["finish_reason"] == "length"
        assert resp["usage"]["prompt_tokens"] == 3
        assert resp["usage"]["completion_tokens"] == 2

    def test_make_chat_completion_response(self) -> None:
        from vkwr.entrypoints.api import _make_chat_completion_response

        resp = _make_chat_completion_response("r1", "hello", "length", "model", 3, 2)
        assert resp["id"] == "chatcmpl-r1"
        assert resp["object"] == "chat.completion"
        assert resp["choices"][0]["message"]["role"] == "assistant"
        assert resp["choices"][0]["message"]["content"] == "hello"


class TestMakeCompletionChunk:
    def test_make_completion_chunk(self) -> None:
        from vkwr.entrypoints.api import _make_completion_chunk

        chunk = _make_completion_chunk("r1", "hello", None, "model")
        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")
        data = json.loads(chunk[6:])
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "hello"

    def test_make_chat_completion_chunk_first(self) -> None:
        from vkwr.entrypoints.api import _make_chat_completion_chunk

        chunk = _make_chat_completion_chunk("r1", "", None, "model", is_first=True)
        data = json.loads(chunk[6:])
        assert data["choices"][0]["delta"]["role"] == "assistant"

    def test_make_chat_completion_chunk_delta(self) -> None:
        from vkwr.entrypoints.api import _make_chat_completion_chunk

        chunk = _make_chat_completion_chunk("r1", "hello", None, "model", is_first=False)
        data = json.loads(chunk[6:])
        assert data["choices"][0]["delta"]["role"] is None
        assert data["choices"][0]["delta"]["content"] == "hello"


# ── Streaming generators ────────────────────────────────────────────


class TestStreamCompletion:
    @patch("vkwr.utils.generate_request_id")
    def test_stream_completion_yields_chunks(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_completion

        mock_id.return_value = "req-1"

        out1 = _make_request_output(request_id="req-1", finished=False, text="He")
        out2 = _make_request_output(request_id="req-1", finished=True, text="llo", finish_reason="length")

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _seq_generator(out1, out2)

        async def _test() -> None:
            response = await _stream_completion(mock_engine, "hello", MagicMock(), "model")
            text_body = await _collect_sse(response)
            assert "data: [DONE]" in text_body
            assert '"text": "He"' in text_body
            assert '"text": "llo"' in text_body

        _run(_test())

    @patch("vkwr.utils.generate_request_id")
    def test_stream_completion_handles_engine_dead(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_completion

        mock_id.return_value = "req-1"

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _erroring_generator(EngineDeadError())

        async def _test() -> None:
            response = await _stream_completion(mock_engine, "hello", MagicMock(), "model")
            text_body = await _collect_sse(response)
            assert '"finish_reason": "error"' in text_body

        _run(_test())


class TestStreamChatCompletion:
    @patch("vkwr.utils.generate_request_id")
    def test_stream_chat_yields_initial_chunk(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_chat_completion

        mock_id.return_value = "req-1"

        out1 = _make_request_output(request_id="req-1", finished=False, text="He")

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _seq_generator(out1)

        async def _test() -> None:
            response = await _stream_chat_completion(mock_engine, "hello", MagicMock(), "model")
            text_body = await _collect_sse(response)
            assert '"role": "assistant"' in text_body
            assert '"content": "He"' in text_body

        _run(_test())

    @patch("vkwr.utils.generate_request_id")
    def test_stream_chat_handles_engine_dead(self, mock_id) -> None:
        from vkwr.entrypoints.api import _stream_chat_completion

        mock_id.return_value = "req-1"

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _erroring_generator(EngineDeadError())

        async def _test() -> None:
            response = await _stream_chat_completion(mock_engine, "hello", MagicMock(), "model")
            text_body = await _collect_sse(response)
            assert '"finish_reason": "error"' in text_body

        _run(_test())


# ── _collect_completion ──────────────────────────────────────────────


class TestCollectCompletion:
    @patch("vkwr.utils.generate_request_id")
    def test_collect_completion_returns_last_output(self, mock_id) -> None:
        from vkwr.entrypoints.api import _collect_completion

        mock_id.return_value = "req-1"

        final_out = _make_request_output(
            request_id="req-1",
            finished=True,
            text="Hello world",
            finish_reason="length",
        )

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _seq_generator(final_out)

        async def _test() -> None:
            result = await _collect_completion(mock_engine, "hello", MagicMock())
            assert result is not None
            assert result.finished is True
            assert result.outputs[0].text == "Hello world"

        _run(_test())

    @patch("vkwr.utils.generate_request_id")
    def test_collect_completion_propagates_engine_dead(self, mock_id) -> None:
        from vkwr.entrypoints.api import _collect_completion

        mock_id.return_value = "req-1"

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _erroring_generator(EngineDeadError())

        async def _test() -> None:
            with pytest.raises(EngineDeadError):
                await _collect_completion(mock_engine, "hello", MagicMock())

        _run(_test())

    @patch("vkwr.utils.generate_request_id")
    def test_collect_completion_propagates_generate_error(self, mock_id) -> None:
        from vkwr.entrypoints.api import _collect_completion

        mock_id.return_value = "req-1"

        mock_engine = MagicMock()
        mock_engine.generate = lambda *a, **kw: _erroring_generator(EngineGenerateError())

        async def _test() -> None:
            with pytest.raises(EngineGenerateError):
                await _collect_completion(mock_engine, "hello", MagicMock())

        _run(_test())


# ── Request models ───────────────────────────────────────────────────


class TestCompletionRequest:
    def test_default_values(self) -> None:
        from vkwr.entrypoints.api import CompletionRequest

        req = CompletionRequest(model="test", prompt="hello")
        assert req.temperature == 1.0
        assert req.top_p == 1.0
        assert req.top_k == -1
        assert req.min_p == 0.0
        assert req.stream is False
        assert req.max_tokens is None
        assert req.stop is None

    def test_stream_true(self) -> None:
        from vkwr.entrypoints.api import CompletionRequest

        req = CompletionRequest(model="test", prompt="hello", stream=True)
        assert req.stream is True

    def test_prompt_as_int_list(self) -> None:
        from vkwr.entrypoints.api import CompletionRequest

        req = CompletionRequest(model="test", prompt=[1, 2, 3])
        assert req.prompt == [1, 2, 3]


class TestChatCompletionRequest:
    def test_default_values(self) -> None:
        from vkwr.entrypoints.api import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hello")],
        )
        assert req.temperature == 1.0
        assert req.stream is False

    def test_empty_messages_validation(self) -> None:
        from vkwr.entrypoints.api import ChatCompletionRequest

        req = ChatCompletionRequest(model="test", messages=[])
        assert not req.messages


# ── API layer uses AsyncLLMEngine.generate() ────────────────────────


class TestApiUsesAsyncGenerate:
    @patch("vkwr.utils.generate_request_id")
    def test_api_uses_async_generate_for_streaming(self, mock_id) -> None:
        """API correctly calls AsyncLLMEngine.generate() for streaming."""
        from vkwr.entrypoints.api import _stream_completion

        mock_id.return_value = "req-1"

        generate_call_count = 0

        def make_generator():
            nonlocal generate_call_count

            async def gen(*a, **kw):
                nonlocal generate_call_count
                generate_call_count += 1
                yield _make_request_output(request_id="req-1", finished=True, text="Hi")

            return gen

        mock_engine = MagicMock()
        mock_engine.generate = make_generator()

        async def _test() -> None:
            response = await _stream_completion(mock_engine, "hello", MagicMock(), "model")
            await _collect_sse(response)

        _run(_test())
        assert generate_call_count == 1

    @patch("vkwr.utils.generate_request_id")
    def test_api_uses_async_generate_for_non_streaming(self, mock_id) -> None:
        """API correctly calls AsyncLLMEngine.generate() for non-streaming."""
        from vkwr.entrypoints.api import _collect_completion

        mock_id.return_value = "req-1"

        generate_call_count = 0

        def make_generator():
            nonlocal generate_call_count

            async def gen(*a, **kw):
                nonlocal generate_call_count
                generate_call_count += 1
                yield _make_request_output(request_id="req-1", finished=True, text="Hi")

            return gen

        mock_engine = MagicMock()
        mock_engine.generate = make_generator()

        async def _test() -> None:
            result = await _collect_completion(mock_engine, "hello", MagicMock())
            assert result is not None

        _run(_test())
        assert generate_call_count == 1

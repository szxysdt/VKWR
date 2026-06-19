from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.engine.async_llm import AsyncLLMEngine
from vkwr.engine.exceptions import EngineDeadError
from vkwr.engine.output_processor import RequestOutputCollector
from vkwr.engine.outputs import RequestOutput
from vkwr.engine.request import RequestOutputKind, SamplingParams

# ── Helpers ───────────────────────────────────────────────────────────


def _make_config() -> VkwrConfig:
    return VkwrConfig(
        model_config=ModelConfig(model="fake-model", max_model_len=8192),
    )


def _make_request_output(
    request_id="req-1",
    finished=False,
    finish_reason=None,
    token_ids=None,
    text="",
):
    from vkwr.engine.outputs import CompletionOutput, RequestStats

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


# ── Properties ───────────────────────────────────────────────────────


class TestAsyncLLMEngineProperties:
    def test_is_running_property_before_start(self):
        engine = AsyncLLMEngine(_make_config())
        assert engine.is_running is False

    def test_is_stopped_property(self):
        engine = AsyncLLMEngine(_make_config())
        assert engine.is_stopped is False
        engine._errored = True
        assert engine.is_stopped is True

    def test_errored_property(self):
        engine = AsyncLLMEngine(_make_config())
        assert engine.errored is False
        engine._errored = True
        assert engine.errored is True

    def test_errored_when_task_done(self):
        engine = AsyncLLMEngine(_make_config())
        loop = asyncio.new_event_loop()
        task = loop.create_task(asyncio.sleep(0))
        task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        loop.close()
        engine._engine_loop_task = task
        assert engine.errored is True

    def test_dead_error_property(self):
        engine = AsyncLLMEngine(_make_config())
        err = engine.dead_error
        assert isinstance(err, EngineDeadError)


# ── check_health ─────────────────────────────────────────────────────


class TestCheckHealth:
    def test_check_health_raises_when_errored(self):
        async def _test():
            engine = AsyncLLMEngine(_make_config())
            engine._errored = True
            with pytest.raises(EngineDeadError):
                await engine.check_health()

        asyncio.get_event_loop().run_until_complete(_test())

    def test_check_health_ok_when_not_errored(self):
        async def _test():
            engine = AsyncLLMEngine(_make_config())
            await engine.check_health()

        asyncio.get_event_loop().run_until_complete(_test())


# ── abort ────────────────────────────────────────────────────────────


class TestAbort:
    @patch("vkwr.engine.llm_engine.LLMEngine")
    def test_abort_single_string(self, MockLLMEngine):
        mock_engine = MagicMock()
        MockLLMEngine.return_value = mock_engine

        engine = AsyncLLMEngine(_make_config())
        engine._ensure_engine()

        async def _test():
            await engine.abort("req-1")

        asyncio.get_event_loop().run_until_complete(_test())
        mock_engine.engine_core.abort_request.assert_called_once_with("req-1")

    @patch("vkwr.engine.llm_engine.LLMEngine")
    def test_abort_batch(self, MockLLMEngine):
        mock_engine = MagicMock()
        MockLLMEngine.return_value = mock_engine

        engine = AsyncLLMEngine(_make_config())
        engine._ensure_engine()

        async def _test():
            await engine.abort(["req-1", "req-2", "req-3"])

        asyncio.get_event_loop().run_until_complete(_test())
        assert mock_engine.engine_core.abort_request.call_count == 3
        calls = [c[0][0] for c in mock_engine.engine_core.abort_request.call_args_list]
        assert calls == ["req-1", "req-2", "req-3"]


# ── add_request ──────────────────────────────────────────────────────


class TestAddRequest:
    @patch("vkwr.engine.llm_engine.LLMEngine")
    def test_add_request_errored_raises(self, MockLLMEngine):
        mock_engine = MagicMock()
        MockLLMEngine.return_value = mock_engine

        engine = AsyncLLMEngine(_make_config())
        engine._errored = True

        async def _test():
            with pytest.raises(EngineDeadError):
                await engine.add_request("req-1", "prompt", SamplingParams())

        asyncio.get_event_loop().run_until_complete(_test())

    @patch("vkwr.engine.llm_engine.LLMEngine")
    def test_add_request_returns_collector(self, MockLLMEngine):
        mock_engine = MagicMock()
        MockLLMEngine.return_value = mock_engine

        engine = AsyncLLMEngine(_make_config())

        async def _test():
            sampling_params = SamplingParams(output_kind=RequestOutputKind.DELTA)
            collector = await engine.add_request("req-1", "hello", sampling_params)
            assert isinstance(collector, RequestOutputCollector)
            assert collector.request_id == "req-1"
            mock_engine.input_processor.process_input.assert_called_once()
            mock_engine.output_processor.add_request.assert_called_once()
            mock_engine.engine_core.add_request.assert_called_once()

        asyncio.get_event_loop().run_until_complete(_test())


# ── generate lifecycle ───────────────────────────────────────────────


def _make_mock_async_engine() -> MagicMock:
    """Create a mock AsyncLLMEngine with the real generate() method bound.

    Pattern from vLLM: AsyncLLM.generate.__get__(mock, AsyncLLM) — binds the
    real async generator logic to a mock object so add_request/abort can be
    controlled without starting an engine loop.
    """
    mock = MagicMock(spec=AsyncLLMEngine)
    mock.generate = AsyncLLMEngine.generate.__get__(mock, AsyncLLMEngine)
    mock.abort = AsyncMock()
    return mock


class TestGenerateLifecycle:
    def test_generate_lifecycle(self) -> None:
        """Full generate() lifecycle: add -> yield -> finish -> close."""
        mock_engine = _make_mock_async_engine()

        async def _test() -> None:
            collector = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, "req-1")

            async def feed() -> None:
                await asyncio.sleep(0.01)
                collector.put(_make_request_output(finished=False, text="partial"))
                await asyncio.sleep(0.01)
                collector.put(_make_request_output(finished=True, text="final", finish_reason="length"))

            asyncio.create_task(feed())

            async def _mock_add(*a, **kw) -> RequestOutputCollector:
                return collector

            mock_engine.add_request = _mock_add  # type: ignore[assignment]

            results: list[RequestOutput] = []
            async for out in mock_engine.generate("hello", SamplingParams(), "req-1"):
                results.append(out)

            assert len(results) == 2
            assert results[0].finished is False
            assert results[1].finished is True

        asyncio.new_event_loop().run_until_complete(_test())

    def test_generate_handles_cancelled_error(self) -> None:
        """When generator is cancelled, request is aborted."""
        mock_engine = _make_mock_async_engine()

        async def _test() -> None:
            collector = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, "req-1")

            async def _mock_add(*a, **kw) -> RequestOutputCollector:
                return collector

            mock_engine.add_request = _mock_add  # type: ignore[assignment]

            gen = mock_engine.generate("hello", SamplingParams(), "req-1")

            cancel_task = asyncio.create_task(gen.__anext__())
            await asyncio.sleep(0.05)
            cancel_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await cancel_task

        loop = asyncio.new_event_loop()
        loop.run_until_complete(_test())
        loop.close()
        assert mock_engine.abort.called

    def test_generate_propagates_engine_dead_error(self) -> None:
        """EngineDeadError is re-raised, no abort."""
        engine = AsyncLLMEngine(_make_config())
        engine._errored = True

        async def _test() -> None:
            with pytest.raises(EngineDeadError):
                async for _ in engine.generate("hello", SamplingParams(output_kind=RequestOutputKind.DELTA), "req-1"):
                    pass

        asyncio.new_event_loop().run_until_complete(_test())

    def test_generate_raises_engine_generate_error(self) -> None:
        """Unexpected exception wraps to EngineGenerateError, aborts."""
        from vkwr.engine.exceptions import EngineGenerateError

        mock_engine = _make_mock_async_engine()

        async def _test() -> None:
            collector = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, "req-1")

            async def _mock_add(*a, **kw) -> RequestOutputCollector:
                return collector

            mock_engine.add_request = _mock_add  # type: ignore[assignment]

            collector.put(Exception("unexpected"))

            with pytest.raises(EngineGenerateError):
                async for _ in mock_engine.generate("hello", SamplingParams(), "req-1"):
                    pass

        asyncio.new_event_loop().run_until_complete(_test())
        assert mock_engine.abort.called

    @patch("vkwr.engine.llm_engine.LLMEngine")
    def test_add_request_with_collector(self, mock_llm: MagicMock) -> None:
        """add_request creates a collector and registers it with output_processor."""
        mock_engine_obj = MagicMock()
        mock_llm.return_value = mock_engine_obj
        mock_req = MagicMock(request_id="req-1", prompt="test", sampling_params=SamplingParams())
        mock_engine_obj.input_processor.process_input.return_value = mock_req

        engine = AsyncLLMEngine(_make_config())

        async def _test() -> None:
            collector = await engine.add_request("req-1", "test", SamplingParams())
            assert isinstance(collector, RequestOutputCollector)
            mock_engine_obj.output_processor.add_request.assert_called_once_with(mock_req, collector)

        asyncio.new_event_loop().run_until_complete(_test())


# ── propagate_error ──────────────────────────────────────────────────


class TestPropagateError:
    def test_propagate_error_pushes_to_queues(self):
        """propagate_error pushes exception to all request queues."""
        from vkwr.engine.output_processor import OutputProcessor
        from vkwr.engine.request import VkwrRequest

        model_config = ModelConfig(model="fake", tokenizer=None)
        proc = OutputProcessor(model_config)

        req = VkwrRequest(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(output_kind=RequestOutputKind.DELTA),
        )
        collector = RequestOutputCollector(RequestOutputKind.DELTA, "req-1")
        proc.add_request(req, collector)

        test_error = RuntimeError("test error")
        proc.propagate_error(test_error)

        try:
            out = collector.get_nowait()
            assert out is None
        except RuntimeError as e:
            assert str(e) == "test error"

"""Integration tests for LLMEngine and AsyncLLMEngine with EngineCoreClient.

Covers:
- LLMEngine inproc mode (InprocClient)
- LLMEngine multiprocess mode flag passes through
- AsyncLLMEngine inproc mode (multiprocess_mode=False)
- AsyncLLMEngine generate() full lifecycle
- Concurrent streaming requests
- EngineDeadError propagation to collector queue
- LLMEngine step() output equivalence

These are unit-level integration tests using mocks for the underlying
EngineCore/Executor. Real model tests live in test_sync_mp_client_real.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.config.scheduler import SchedulerConfig
from vkwr.engine.async_llm import AsyncLLMEngine
from vkwr.engine.core_client import EngineCoreClient, InprocClient
from vkwr.engine.exceptions import EngineDeadError
from vkwr.engine.output_processor import OutputProcessor, RequestOutputCollector
from vkwr.engine.outputs import (
    CompletionOutput,
    EngineCoreOutput,
    EngineCoreOutputs,
    RequestOutput,
    RequestStats,
)
from vkwr.engine.request import RequestOutputKind, SamplingParams, VkwrRequest


def _make_config() -> VkwrConfig:
    return VkwrConfig(
        model_config=ModelConfig(model="fake-model", max_model_len=8192),
        scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=256),
    )


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


def _run(coroutine):
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coroutine)
    finally:
        loop.close()


async def _seq_gen(*items) -> AsyncGenerator[RequestOutput, None]:
    for item in items:
        yield item


class TestLLMEngineInprocMode:
    """Test LLMEngine with InprocClient (multiprocess_mode=False)."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_llm_engine_inproc_creates_inproc_client(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.llm_engine import LLMEngine

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        engine = LLMEngine(config, multiprocess_mode=False)

        assert isinstance(engine.engine_core, InprocClient)
        engine.shutdown()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_llm_engine_add_request_and_step(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.llm_engine import LLMEngine
        from vkwr.tokenizers.rwkv7 import get_tokenizer

        tok = get_tokenizer()
        if tok is None:
            pytest.skip("RWKV vocab file not found — cannot encode text prompt")

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        engine = LLMEngine(config, multiprocess_mode=False)

        req_id = engine.add_request("req-1", "hello", SamplingParams())
        assert req_id.startswith("req-1")

        step_outputs = engine.step()
        assert isinstance(step_outputs, list)

        engine.shutdown()

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_llm_engine_abort_requests(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.llm_engine import LLMEngine

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        engine = LLMEngine(config, multiprocess_mode=False)
        engine.engine_core.abort_requests = MagicMock()

        engine.abort_requests(["req-1"])
        engine.engine_core.abort_requests.assert_called_once_with(["req-1"])
        engine.shutdown()

    @patch("vkwr.engine.core_client.SyncMPClient")
    def test_llm_engine_multiproc_mode_flag(self, MockSyncClient: MagicMock) -> None:
        """LLMEngine with multiprocess_mode=True creates SyncMPClient."""
        MockSyncClient.return_value = MagicMock()

        engine_core = EngineCoreClient.make_client(multiprocess_mode=True, asyncio_mode=False, vkwr_config=_make_config())

        assert isinstance(engine_core, MockSyncClient.return_value.__class__) or MockSyncClient.called

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_llm_engine_shutdown(self, MockGetClass: MagicMock) -> None:
        from vkwr.engine.llm_engine import LLMEngine

        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        engine = LLMEngine(config, multiprocess_mode=False)
        engine.engine_core.shutdown = MagicMock()

        engine.shutdown()
        engine.engine_core.shutdown.assert_called_once()


class TestAsyncLLMEngineInprocMode:
    """Test AsyncLLMEngine with InprocClient (multiprocess_mode=False)."""

    def test_async_engine_inproc_raises(self) -> None:
        """AsyncLLMEngine always sets asyncio_mode=True, so multiprocess_mode=False raises.

        AsyncLLMEngine.__init__ passes asyncio_mode=True + multiprocess_mode=param.
        make_client raises NotImplementedError for asyncio_mode=True + multiprocess_mode=False.
        """
        config = _make_config()
        with pytest.raises(NotImplementedError, match="asyncio without multiprocessing"):
            AsyncLLMEngine(config, multiprocess_mode=False)


class TestAsyncLLMEngineGenerateLifecycle:
    """Test AsyncLLMEngine.generate() full lifecycle."""

    def test_generate_streams_to_finished(self) -> None:
        """generate() yields outputs until finished=True."""
        mock_engine = MagicMock(spec=AsyncLLMEngine)
        mock_engine.generate = AsyncLLMEngine.generate.__get__(mock_engine, AsyncLLMEngine)
        mock_engine.abort = AsyncMock()

        async def _test() -> None:
            collector = RequestOutputCollector(RequestOutputKind.DELTA, "req-1")

            async def feed() -> None:
                await asyncio.sleep(0.01)
                collector.put(_make_request_output(finished=False, text="He"))
                await asyncio.sleep(0.01)
                collector.put(_make_request_output(finished=True, text="llo", finish_reason="length"))

            asyncio.create_task(feed())

            async def _mock_add(*a, **kw) -> RequestOutputCollector:
                return collector

            mock_engine.add_request = _mock_add  # type: ignore

            results: list[RequestOutput] = []
            async for out in mock_engine.generate("hello", SamplingParams(), "req-1"):
                results.append(out)

            assert len(results) == 2
            assert results[0].finished is False
            assert results[1].finished is True
            assert results[1].finish_reason == "length"

        _run(_test())

    def test_generate_abort_on_cancel(self) -> None:
        """CancelledError triggers abort."""
        mock_engine = MagicMock(spec=AsyncLLMEngine)
        mock_engine.generate = AsyncLLMEngine.generate.__get__(mock_engine, AsyncLLMEngine)
        mock_engine.abort = AsyncMock()

        async def _test() -> None:
            collector = RequestOutputCollector(RequestOutputKind.DELTA, "req-1")

            async def _mock_add(*a, **kw) -> RequestOutputCollector:
                return collector

            mock_engine.add_request = _mock_add  # type: ignore

            gen = mock_engine.generate("hello", SamplingParams(), "req-1")
            cancel_task = asyncio.create_task(gen.__anext__())
            await asyncio.sleep(0.05)
            cancel_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await cancel_task

        _run(_test())
        assert mock_engine.abort.called

    def test_generate_propagates_engine_dead(self) -> None:
        """EngineDeadError from add_request propagates through generate()."""
        mock_engine = MagicMock(spec=AsyncLLMEngine)
        mock_engine.generate = AsyncLLMEngine.generate.__get__(mock_engine, AsyncLLMEngine)
        mock_engine.abort = AsyncMock()

        async def _test() -> None:
            async def _mock_add(*a, **kw) -> RequestOutputCollector:
                raise EngineDeadError()

            mock_engine.add_request = _mock_add  # type: ignore

            with pytest.raises(EngineDeadError):
                async for _ in mock_engine.generate("hello", SamplingParams(), "req-1"):
                    pass

        _run(_test())

    def test_generate_wraps_unexpected_error(self) -> None:
        """Unexpected exception from collector wraps to EngineGenerateError."""
        from vkwr.engine.exceptions import EngineGenerateError

        mock_engine = MagicMock(spec=AsyncLLMEngine)
        mock_engine.generate = AsyncLLMEngine.generate.__get__(mock_engine, AsyncLLMEngine)
        mock_engine.abort = AsyncMock()

        async def _test() -> None:
            collector = RequestOutputCollector(RequestOutputKind.FINAL_ONLY, "req-1")

            async def _mock_add(*a, **kw) -> RequestOutputCollector:
                return collector

            mock_engine.add_request = _mock_add  # type: ignore
            collector.put(Exception("unexpected"))

            with pytest.raises(EngineGenerateError):
                async for _ in mock_engine.generate("hello", SamplingParams(), "req-1"):
                    pass

        _run(_test())
        assert mock_engine.abort.called


class TestAsyncLLMEngineOutputHandler:
    """Test _run_output_handler() output routing."""

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_output_handler_routes_outputs(self, mock_make: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_make.return_value = mock_client

        outputs = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="req-1-abc",
                    new_token_ids=[42],
                    finish_reason=None,
                )
            ]
        )
        mock_client.get_output_async = AsyncMock(side_effect=[outputs, EngineDeadError()])

        engine = AsyncLLMEngine(_make_config())

        from vkwr.engine.request import VkwrRequest

        req = VkwrRequest(
            request_id="req-1-abc",
            prompt="test",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(output_kind=RequestOutputKind.DELTA),
        )
        req.external_req_id = "req-1"
        engine._output_processor.add_request(
            req,
            RequestOutputCollector(RequestOutputKind.DELTA, "req-1"),
        )

        async def _test():
            engine._run_output_handler()
            await asyncio.sleep(0.2)
            assert mock_client.get_output_async.called

        _run(_test())

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_output_handler_error_propagation(self, mock_make: MagicMock) -> None:
        """EngineDeadError in output handler sets _errored and propagates."""
        mock_client = AsyncMock()
        mock_make.return_value = mock_client
        mock_client.get_output_async = AsyncMock(side_effect=EngineDeadError())

        engine = AsyncLLMEngine(_make_config())

        engine._output_processor.add_request(
            MagicMock(
                request_id="req-1",
                external_req_id="req-1",
                sampling_params=SamplingParams(output_kind=RequestOutputKind.DELTA),
                prompt="test",
                prompt_token_ids=[1, 2],
            ),
            RequestOutputCollector(RequestOutputKind.DELTA, "req-1"),
        )

        async def _test():
            engine._run_output_handler()
            await asyncio.sleep(0.2)
            assert engine._errored is True

        _run(_test())


class TestConcurrentRequests:
    """Test concurrent request handling through AsyncLLMEngine."""

    def test_two_collectors_independent(self) -> None:
        """Two collectors receive their own outputs independently."""
        collector1 = RequestOutputCollector(RequestOutputKind.DELTA, "req-1")
        collector2 = RequestOutputCollector(RequestOutputKind.DELTA, "req-2")

        collector1.put(_make_request_output("req-1", text="A"))
        collector2.put(_make_request_output("req-2", text="B"))

        out1 = collector1.get_nowait()
        out2 = collector2.get_nowait()

        assert out1.request_id == "req-1"
        assert out1.outputs[0].text == "A"
        assert out2.request_id == "req-2"
        assert out2.outputs[0].text == "B"

    def test_engine_dead_reaches_all_collectors(self) -> None:
        """propagate_error pushes EngineDeadError to all collectors."""
        model_config = ModelConfig(model="fake", tokenizer=None)
        proc = OutputProcessor(model_config)

        collectors = {}
        for i in range(3):
            req = VkwrRequest(
                request_id=f"req-{i}",
                prompt="test",
                prompt_token_ids=[1],
                sampling_params=SamplingParams(output_kind=RequestOutputKind.DELTA),
            )
            c = RequestOutputCollector(RequestOutputKind.DELTA, f"req-{i}")
            collectors[i] = c
            proc.add_request(req, c)

        proc.propagate_error(EngineDeadError())

        for i in range(3):
            with pytest.raises(EngineDeadError):
                collectors[i].get_nowait()


class TestEquivalence:
    """Test output equivalence between InprocClient and direct EngineCore."""

    @patch("vkwr.engine.core.ExecutorInterface.get_class")
    def test_inproc_client_equivalence(self, MockGetClass: MagicMock) -> None:
        """InprocClient.get_output() returns same type as direct step()."""
        mock_executor_instance = MagicMock()
        mock_executor_cls = MagicMock(return_value=mock_executor_instance)
        MockGetClass.return_value = mock_executor_cls

        config = _make_config()
        client = InprocClient(config)

        result = client.get_output(timeout=0.1)
        assert isinstance(result, EngineCoreOutputs)
        assert isinstance(result.timestamp, float)
        assert result.timestamp > 0

        client.shutdown()

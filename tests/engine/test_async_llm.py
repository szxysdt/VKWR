from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.engine.async_llm import AsyncLLMEngine
from vkwr.engine.exceptions import EngineDeadError
from vkwr.engine.output_processor import RequestOutputCollector
from vkwr.engine.outputs import EngineCoreOutputs, RequestOutput
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


def _run(coroutine):
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coroutine)
    finally:
        loop.close()


# ── Properties ───────────────────────────────────────────────────────


class TestAsyncLLMEngineProperties:
    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_is_running_property_before_start(self, mock_make: MagicMock):
        mock_make.return_value = MagicMock()
        engine = AsyncLLMEngine(_make_config())
        assert engine.is_running is True

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_is_stopped_property(self, mock_make: MagicMock):
        mock_make.return_value = MagicMock()
        engine = AsyncLLMEngine(_make_config())
        assert engine.is_stopped is False
        engine._errored = True
        assert engine.is_stopped is True

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_errored_property(self, mock_make: MagicMock):
        mock_make.return_value = MagicMock()
        engine = AsyncLLMEngine(_make_config())
        assert engine.errored is False
        engine._errored = True
        assert engine.errored is True

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_errored_when_task_done(self, mock_make: MagicMock):
        mock_make.return_value = MagicMock()
        engine = AsyncLLMEngine(_make_config())
        loop = asyncio.new_event_loop()
        task = loop.create_task(asyncio.sleep(0))
        task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
        loop.close()
        engine._output_handler = task
        assert engine.errored is True

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_dead_error_property(self, mock_make: MagicMock):
        mock_make.return_value = MagicMock()
        engine = AsyncLLMEngine(_make_config())
        err = engine.dead_error
        assert isinstance(err, EngineDeadError)


# ── check_health ─────────────────────────────────────────────────────


class TestCheckHealth:
    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_check_health_raises_when_errored(self, mock_make: MagicMock):
        mock_make.return_value = MagicMock()

        async def _test():
            engine = AsyncLLMEngine(_make_config())
            engine._errored = True
            with pytest.raises(EngineDeadError):
                await engine.check_health()

        _run(_test())

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_check_health_ok_when_not_errored(self, mock_make: MagicMock):
        mock_make.return_value = MagicMock()

        async def _test():
            engine = AsyncLLMEngine(_make_config())
            await engine.check_health()

        _run(_test())


# ── abort ────────────────────────────────────────────────────────────


class TestAbort:
    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_abort_single_string(self, mock_make: MagicMock):
        mock_client = AsyncMock()
        mock_make.return_value = mock_client

        engine = AsyncLLMEngine(_make_config())
        engine._output_processor.add_request(
            MagicMock(request_id="req-1", external_req_id="req-1", sampling_params=SamplingParams(), prompt="test", prompt_token_ids=[1]),
            None,
        )

        _run(engine.abort("req-1"))
        mock_client.abort_requests.assert_called_once()

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_abort_batch(self, mock_make: MagicMock):
        mock_client = AsyncMock()
        mock_make.return_value = mock_client

        engine = AsyncLLMEngine(_make_config())
        for rid in ["req-1", "req-2", "req-3"]:
            engine._output_processor.add_request(
                MagicMock(request_id=rid, external_req_id=rid, sampling_params=SamplingParams(), prompt="test", prompt_token_ids=[1]),
                None,
            )

        _run(engine.abort(["req-1", "req-2", "req-3"]))
        mock_client.abort_requests.assert_called_once()

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_abort_produces_finished_abort_output(self, mock_make: MagicMock):
        """abort() pushes FINISHED_ABORTED output to collector, unblocking generate()."""
        mock_client = AsyncMock()
        mock_make.return_value = mock_client

        engine = AsyncLLMEngine(_make_config())
        from vkwr.engine.request import VkwrRequest

        req = VkwrRequest(
            request_id="req-1-abc",
            prompt="test",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(output_kind=RequestOutputKind.DELTA),
        )
        req.external_req_id = "req-1"
        collector = RequestOutputCollector(RequestOutputKind.DELTA, "req-1")
        engine._output_processor.add_request(req, collector)

        _run(engine.abort("req-1"))

        out = collector.get_nowait()
        assert out is not None
        assert out.finished is True
        assert out.finish_reason == "abort"

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_abort_resolves_external_id(self, mock_make: MagicMock):
        """abort() resolves external request ID to internal ID(s)."""
        mock_client = AsyncMock()
        mock_make.return_value = mock_client

        engine = AsyncLLMEngine(_make_config())
        from vkwr.engine.request import VkwrRequest

        req = VkwrRequest(
            request_id="req-1-xyz",
            prompt="test",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(),
        )
        req.external_req_id = "ext-id"
        engine._output_processor.add_request(req, None)

        _run(engine.abort("ext-id"))
        mock_client.abort_requests.assert_called_once_with(["req-1-xyz"])


# ── add_request ──────────────────────────────────────────────────────


class TestAddRequest:
    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_add_request_errored_raises(self, mock_make: MagicMock):
        mock_make.return_value = MagicMock()

        engine = AsyncLLMEngine(_make_config())
        engine._errored = True

        async def _test():
            with pytest.raises(EngineDeadError):
                await engine.add_request("req-1", "prompt", SamplingParams())

        _run(_test())

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_add_request_returns_collector(self, mock_make: MagicMock):
        """add_request creates collector, sends EngineCoreRequest, starts output handler."""
        mock_client = AsyncMock()
        mock_make.return_value = mock_client

        # Make get_output_async raise EngineDeadError immediately so the
        # background _run_output_handler task exits and doesn't hang.
        mock_client.get_output_async = AsyncMock(side_effect=EngineDeadError())

        engine = AsyncLLMEngine(_make_config())

        async def _test():
            sampling_params = SamplingParams(output_kind=RequestOutputKind.DELTA)
            collector = await engine.add_request("req-1", [1, 2, 3], sampling_params)
            assert isinstance(collector, RequestOutputCollector)
            assert collector.request_id == "req-1"
            mock_client.add_request.assert_called_once()

            # Let the background task terminate
            await asyncio.sleep(0.05)

        _run(_test())


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

        _run(_test())

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

        _run(_test())
        assert mock_engine.abort.called

    def test_generate_propagates_engine_dead_error(self) -> None:
        """EngineDeadError is re-raised when add_request fails due to errored state."""
        mock_engine = _make_mock_async_engine()

        async def _test() -> None:
            async def _mock_add(*a, **kw) -> RequestOutputCollector:
                raise EngineDeadError()

            mock_engine.add_request = _mock_add  # type: ignore[assignment]

            with pytest.raises(EngineDeadError):
                async for _ in mock_engine.generate("hello", SamplingParams(output_kind=RequestOutputKind.DELTA), "req-1"):
                    pass

        _run(_test())

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

        _run(_test())
        assert mock_engine.abort.called


# ── _run_output_handler ──────────────────────────────────────────────


class TestRunOutputHandler:
    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_output_handler_processes_outputs(self, mock_make: MagicMock) -> None:
        """_run_output_handler processes EngineCoreOutputs and pushes to collectors."""
        mock_client = AsyncMock()
        mock_make.return_value = mock_client

        engine = AsyncLLMEngine(_make_config())

        from vkwr.engine.outputs import EngineCoreOutput

        outputs = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(
                    request_id="req-1-abc",
                    new_token_ids=[42],
                    finish_reason=None,
                )
            ]
        )
        # Return one valid output, then raise EngineDeadError to terminate
        mock_client.get_output_async = AsyncMock(side_effect=[outputs, EngineDeadError()])

        engine._output_processor.add_request(
            MagicMock(
                request_id="req-1-abc",
                external_req_id="req-1",
                sampling_params=SamplingParams(output_kind=RequestOutputKind.DELTA),
                prompt="test",
                prompt_token_ids=[1, 2],
            ),
            RequestOutputCollector(RequestOutputKind.DELTA, "req-1"),
        )

        async def _test():
            engine._run_output_handler()
            await asyncio.sleep(0.1)
            assert engine._output_handler is not None
            assert mock_client.get_output_async.called

        _run(_test())

    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_output_handler_handles_engine_dead_error(self, mock_make: MagicMock) -> None:
        """When EngineDeadError is raised, _errored is set and error propagated."""
        mock_client = AsyncMock()
        mock_make.return_value = mock_client

        engine = AsyncLLMEngine(_make_config())
        mock_client.get_output_async = AsyncMock(side_effect=EngineDeadError())

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
            await asyncio.sleep(0.1)
            assert engine._errored is True

        _run(_test())


# ── shutdown ─────────────────────────────────────────────────────────


class TestShutdown:
    @patch("vkwr.engine.core_client.EngineCoreClient.make_client")
    def test_shutdown_cancels_output_handler(self, mock_make: MagicMock) -> None:
        mock_client = AsyncMock()
        mock_client.shutdown = MagicMock()
        mock_make.return_value = mock_client

        engine = AsyncLLMEngine(_make_config())

        async def _test():
            async def _never_ending():
                while True:
                    await asyncio.sleep(1)

            engine._output_handler = asyncio.create_task(_never_ending())
            assert engine._output_handler is not None
            await engine.shutdown()
            assert engine._output_handler is None
            mock_client.shutdown.assert_called_once()

        _run(_test())


# ── OutputProcessor.abort_requests ───────────────────────────────────


class TestOutputProcessorAbort:
    def test_abort_requests_produces_finished_output(self):
        """abort_requests produces FINISHED_ABORTED output to collector."""
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

        internal_ids = proc.abort_requests(["req-1"])
        assert internal_ids == ["req-1"]
        assert "req-1" not in proc._requests

        out = collector.get_nowait()
        assert out is not None
        assert out.finished is True
        assert out.finish_reason == "abort"

    def test_abort_requests_no_collector(self):
        """abort_requests works when there is no collector queue."""
        from vkwr.engine.output_processor import OutputProcessor
        from vkwr.engine.request import VkwrRequest

        model_config = ModelConfig(model="fake", tokenizer=None)
        proc = OutputProcessor(model_config)

        req = VkwrRequest(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(),
        )
        proc.add_request(req, None)

        internal_ids = proc.abort_requests(["req-1"])
        assert internal_ids == ["req-1"]
        assert "req-1" not in proc._requests

    def test_abort_requests_nonexistent_id(self):
        """abort_requests ignores nonexistent request IDs."""
        from vkwr.engine.output_processor import OutputProcessor

        model_config = ModelConfig(model="fake", tokenizer=None)
        proc = OutputProcessor(model_config)

        internal_ids = proc.abort_requests(["nonexistent"])
        assert internal_ids == []

    def test_abort_requests_multiple_ids(self):
        """abort_requests handles multiple request IDs."""
        from vkwr.engine.output_processor import OutputProcessor
        from vkwr.engine.request import VkwrRequest

        model_config = ModelConfig(model="fake", tokenizer=None)
        proc = OutputProcessor(model_config)

        collectors = {}
        for i in range(3):
            req = VkwrRequest(
                request_id=f"req-{i}",
                prompt="test",
                prompt_token_ids=[1, 2],
                sampling_params=SamplingParams(output_kind=RequestOutputKind.DELTA),
            )
            collector = RequestOutputCollector(RequestOutputKind.DELTA, f"req-{i}")
            collectors[f"req-{i}"] = collector
            proc.add_request(req, collector)

        internal_ids = proc.abort_requests(["req-0", "req-1"])
        assert sorted(internal_ids) == ["req-0", "req-1"]
        assert "req-0" not in proc._requests
        assert "req-1" not in proc._requests
        assert "req-2" in proc._requests

        out0 = collectors["req-0"].get_nowait()
        out1 = collectors["req-1"].get_nowait()
        assert out0.finished is True
        assert out1.finished is True

    def test_abort_unblocks_generate(self):
        """abort_requests output can unblock a waiting generate() consumer."""
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

        async def _test():
            get_task = asyncio.create_task(collector.get())
            await asyncio.sleep(0.01)
            proc.abort_requests(["req-1"])
            out = await get_task
            assert out.finished is True
            assert out.finish_reason == "abort"

        _run(_test())


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

"""Inspired by vLLM."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Iterable
from typing import TYPE_CHECKING

from vkwr.engine.core_request import EngineCoreRequest
from vkwr.engine.exceptions import EngineDeadError, EngineGenerateError
from vkwr.engine.output_processor import RequestOutputCollector
from vkwr.engine.outputs import RequestOutput
from vkwr.engine.request import SamplingParams

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig

logger = logging.getLogger(__name__)


class AsyncLLMEngine:
    """Async LLM engine — holds EngineCoreClient, InputProcessor, OutputProcessor directly.

    - One background asyncio.Task pulls outputs via EngineCoreClient.get_output_async()
    - Each request gets a RequestOutputCollector queue
    - HTTP streams consume from their own queue
    """

    def __init__(self, config: VkwrConfig, multiprocess_mode: bool = True):
        from vkwr.engine.core_client import EngineCoreClient
        from vkwr.engine.input_processor import InputProcessor
        from vkwr.engine.output_processor import OutputProcessor

        tokenizer = None
        if not config.model_config.skip_tokenizer_init:
            from vkwr.engine.tokenizer import get_tokenizer

            tokenizer = get_tokenizer(config.model_config.tokenizer)
        else:
            logger.warning("Tokenizer initialization skipped. Text prompts will fail, and outputs will not contain decoded text.")

        self._engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=True,
            vkwr_config=config,
            log_stats=True,
        )
        self._input_processor = InputProcessor(config.model_config, config.scheduler_config, tokenizer=tokenizer)
        self._output_processor = OutputProcessor(config.model_config, tokenizer=tokenizer)
        self._config = config
        self._output_handler: asyncio.Task | None = None

        self._errored: bool = False

    @classmethod
    def from_engine_args(cls, engine_args) -> AsyncLLMEngine:
        """Create an AsyncLLMEngine instance from EngineArgs."""
        config = engine_args.create_engine_config()
        multiprocess_mode = True  # AsyncLLMEngine requires AsyncMPClient
        return cls(config, multiprocess_mode=multiprocess_mode)

    @property
    def is_running(self) -> bool:
        return self._output_handler is None or not self._output_handler.done()

    @property
    def is_stopped(self) -> bool:
        """Alias for errored."""
        return self.errored

    @property
    def errored(self) -> bool:
        if self._errored:
            return True
        if self._output_handler is not None and self._output_handler.done():
            return True
        return False

    @property
    def dead_error(self) -> EngineDeadError:
        return EngineDeadError()

    async def add_request(
        self,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> RequestOutputCollector:
        """Add request, return per-request collector.

        On success, registers with both OutputProcessor and EngineCore.
        If EngineCore.add_request() fails, cleans up OutputProcessor to prevent leaks.
        """
        if self.errored:
            raise self.dead_error
        collector = RequestOutputCollector(sampling_params.output_kind, request_id)
        vkwr_request = self._input_processor.process_input(request_id, prompt, sampling_params)
        core_request = EngineCoreRequest(
            request_id=vkwr_request.request_id,
            prompt_token_ids=vkwr_request.prompt_token_ids,
            sampling_params=sampling_params,
        )
        # Register in output_processor first, then send to engine.
        # If engine_core.add_request fails, we clean up the registration.
        self._output_processor.add_request(vkwr_request, collector)
        try:
            await self._engine_core.add_request(core_request)
        except Exception:
            self._output_processor.remove_request(vkwr_request.request_id)
            raise
        self._run_output_handler()
        return collector

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Main streaming generator.

        1. Calls add_request() to obtain a RequestOutputCollector
        2. Loops: get_nowait() or await get() from the collector
        3. Yields each RequestOutput until finished

        On cancel/abort (CancelledError / GeneratorExit), aborts the
        request in both OutputProcessor and EngineCore.
        """
        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(request_id, prompt, sampling_params)

            finished = False
            while not finished:
                out = q.get_nowait() or await q.get()
                assert isinstance(out, RequestOutput)
                finished = out.finished
                yield out

        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id)
            raise

        except EngineDeadError:
            raise

        except ValueError:
            raise

        except Exception as e:
            if q is not None:
                await self.abort(q.request_id)
            raise EngineGenerateError() from e

        finally:
            if q is not None:
                q.close()

    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort request(s) in OutputProcessor AND EngineCore.

        1. OutputProcessor.abort_requests(): produces FINISHED_ABORTED terminal output
           to collector queue (unblocks generate()), then cleans up request state
        2. EngineCore.abort_requests(): notifies backend process to stop scheduling
        """
        request_ids = (request_id,) if isinstance(request_id, str) else list(request_id)
        # Resolve external IDs to internal IDs
        all_internal_ids: list[str] = []
        for rid in request_ids:
            internals = self._output_processor.external_req_ids.get(rid)
            if internals:
                all_internal_ids.extend(internals)
            else:
                all_internal_ids.append(rid)

        # Step 1: OutputProcessor side — push FINISHED_ABORTED to collectors
        self._output_processor.abort_requests(all_internal_ids)
        # Step 2: EngineCore side — notify engine to stop processing
        await self._engine_core.abort_requests(all_internal_ids)

    def _run_output_handler(self) -> None:
        """Background loop: pulls from EngineCore and pushes to collectors."""
        if self._output_handler is not None:
            return
        engine_core = self._engine_core
        output_processor = self._output_processor

        async def output_handler() -> None:
            try:
                while True:
                    outputs = await engine_core.get_output_async()
                    if not outputs.outputs:
                        continue
                    processed = output_processor.process_outputs(
                        outputs.outputs,
                        engine_core_timestamp=outputs.timestamp,
                    )
                    output_processor.update_scheduler_stats(outputs.scheduler_stats)
                    if processed.reqs_to_abort:
                        await engine_core.abort_requests(processed.reqs_to_abort)
                    for req_id in output_processor.get_and_clear_finished_ids():
                        output_processor.remove_request(req_id)
            except EngineDeadError:
                logger.exception("Engine core is dead")
                self._errored = True
                output_processor.propagate_error(EngineDeadError())
            except Exception as e:
                logger.exception("AsyncLLMEngine output_handler failed")
                self._errored = True
                output_processor.propagate_error(e)

        self._output_handler = asyncio.create_task(output_handler())

    async def check_health(self) -> None:
        if self.errored:
            raise self.dead_error

    async def shutdown(self) -> None:
        if self._output_handler is not None and not self._output_handler.done():
            self._output_handler.cancel()
            try:
                await self._output_handler
            except asyncio.CancelledError:
                pass
            self._output_handler = None
        await asyncio.to_thread(self._engine_core.shutdown)

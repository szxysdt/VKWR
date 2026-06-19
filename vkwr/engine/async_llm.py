from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from vkwr.engine.exceptions import EngineDeadError, EngineGenerateError
from vkwr.engine.output_processor import RequestOutputCollector
from vkwr.engine.outputs import EngineCoreOutputs, RequestOutput
from vkwr.engine.request import SamplingParams

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.llm_engine import LLMEngine

logger = logging.getLogger(__name__)


class AsyncLLMEngine:
    """Async wrapper around LLMEngine — single background task drives step().

    Mirrors vLLM AsyncLLMEngine pattern:
    - One background asyncio.Task calls step() in a loop
    - Each request gets a RequestOutputCollector queue
    - HTTP streams consume from their own queue
    """

    def __init__(self, config: VkwrConfig):
        self.engine: LLMEngine | None = None
        self._config = config
        self._engine_loop_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._errored = False

    @classmethod
    def from_engine_args(cls, engine_args) -> AsyncLLMEngine:
        config = engine_args.create_engine_config()
        return cls(config)

    def _ensure_engine(self) -> LLMEngine:
        if self.engine is None:
            from vkwr.engine.llm_engine import LLMEngine

            self.engine = LLMEngine(self._config)
        return self.engine

    @property
    def is_running(self) -> bool:
        return self._engine_loop_task is not None and not self._engine_loop_task.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        if self._errored:
            return True
        if self._engine_loop_task is not None and self._engine_loop_task.done():
            return True
        return False

    @property
    def dead_error(self):
        return EngineDeadError()

    async def add_request(
        self,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> RequestOutputCollector:
        """Add request, return per-request collector."""
        if self.errored:
            raise self.dead_error
        if self._shutdown_event.is_set():
            raise RuntimeError("Engine is shut down")
        engine = self._ensure_engine()
        request = engine.input_processor.process_input(request_id, prompt, sampling_params)
        collector = RequestOutputCollector(sampling_params.output_kind, request.request_id)
        engine.output_processor.add_request(request, collector)
        engine.engine_core.add_request(request)
        self._ensure_loop_started()
        return collector

    def _ensure_loop_started(self) -> None:
        if self._engine_loop_task is None:
            self._engine_loop_task = asyncio.create_task(self._engine_loop())
        elif self._engine_loop_task.done():
            self._engine_loop_task = asyncio.create_task(self._engine_loop())

    async def _engine_loop(self) -> None:
        """Background loop: single-threaded step() driver."""
        engine = self._ensure_engine()
        try:
            while not self._shutdown_event.is_set():
                if not engine.has_unfinished_requests():
                    await asyncio.sleep(0.1)
                    continue
                try:
                    outputs_dict, model_executed = await asyncio.to_thread(engine.engine_core.step_fn)
                    engine.engine_core.post_step(model_executed)
                    if outputs_dict is not None:
                        engine_core_outputs = outputs_dict.get(0) or EngineCoreOutputs()
                        processed = engine.output_processor.process_outputs(
                            engine_core_outputs.outputs,
                            engine_core_timestamp=engine_core_outputs.timestamp,
                        )
                        engine.output_processor.update_scheduler_stats(engine_core_outputs.scheduler_stats)
                        if processed.reqs_to_abort:
                            for rid in processed.reqs_to_abort:
                                engine.engine_core.abort_request(rid)
                        for req_id in engine.output_processor.get_and_clear_finished_ids():
                            engine.output_processor.remove_request(req_id)
                except Exception as e:
                    self._errored = True
                    engine.output_processor.propagate_error(e)
                    logger.exception("Engine loop crashed")
                await asyncio.sleep(0)
        except Exception as e:
            self._errored = True
            engine.output_processor.propagate_error(e)
            logger.exception("Engine loop crashed")

    def _resolve_internal_ids(self, engine: LLMEngine, request_id: str) -> list[str]:
        """Resolve a request ID to internal ID(s).

        If request_id is an external ID, look up the internal ID(s) via
        output_processor.external_req_ids. Otherwise treat it as internal.
        """
        internals = engine.output_processor.external_req_ids.get(request_id)
        if internals:
            return internals
        return [request_id]

    async def abort(self, request_id: str | Iterable[str]) -> None:
        engine = self._ensure_engine()
        if isinstance(request_id, str):
            for rid in self._resolve_internal_ids(engine, request_id):
                engine.engine_core.abort_request(rid)
                engine.output_processor.remove_request(rid)
        else:
            for rid in request_id:
                for internal_id in self._resolve_internal_ids(engine, rid):
                    engine.engine_core.abort_request(internal_id)
                    engine.output_processor.remove_request(internal_id)

    async def abort_request(self, request_id: str) -> None:
        engine = self._ensure_engine()
        for rid in self._resolve_internal_ids(engine, request_id):
            engine.engine_core.abort_request(rid)
            engine.output_processor.remove_request(rid)

    async def check_health(self) -> None:
        if self.errored:
            raise self.dead_error

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
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

    async def shutdown(self) -> None:
        self._shutdown_event.set()
        if self._engine_loop_task:
            self._engine_loop_task.cancel()
            try:
                await self._engine_loop_task
            except asyncio.CancelledError:
                pass
            self._engine_loop_task = None

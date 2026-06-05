from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vkwr.engine.request import SamplingParams

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vkwr.engine.llm_engine import LLMEngine
    from vkwr.engine.outputs import RequestOutput

logger = logging.getLogger(__name__)


class LLM:
    """Programmatic API, modeled after vLLM LLM (vllm/entrypoints/llm.py:100+).

    Wraps LLMEngine and provides a simple generate() method with support for:
    - Text or token IDs as prompts
    - Single or batched requests
    - Synchronous blocking until all results are ready
    - Streaming iterator output (streaming=True)
    """

    def __init__(self, model: str, **kwargs):
        from vkwr.config.vkwr import EngineArgs

        engine_args = EngineArgs(model=model, **kwargs)
        self._engine_args = engine_args
        self.llm_engine: LLMEngine | None = None
        self._lazy_init()

    def _lazy_init(self) -> None:
        """Lazy-init LLMEngine to avoid loading the model until needed."""
        if self.llm_engine is None:
            from vkwr.engine.llm_engine import LLMEngine

            self.llm_engine = LLMEngine.from_engine_args(self._engine_args)

    def generate(
        self,
        prompts: str | list[str] | list[int] | list[list[int]],
        sampling_params: SamplingParams | None = None,
        streaming: bool = False,
    ) -> list[RequestOutput] | Iterator[RequestOutput]:
        """Execute inference generation.

        Args:
            prompts: Single or batched prompts. Accepts str, list[int] (token IDs),
                     or a nested list of multiple prompts.
            sampling_params: Sampling parameters. Defaults to SamplingParams().
            streaming: Whether to return partial results as an iterator.

        Returns:
            streaming=False: A list of RequestOutput after all requests complete.
            streaming=True: An iterator yielding completed RequestOutput one by one.
        """
        self._lazy_init()
        engine = self.llm_engine

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Normalize prompts to a list
        prompts_list = self._normalize_prompts(prompts)
        if len(prompts_list) == 0:
            return [] if not streaming else iter([])

        # Register all requests
        for i, prompt in enumerate(prompts_list):
            req_id = f"req-{i}"
            engine.add_request(req_id, prompt, sampling_params)

        if streaming:
            return self._stream_results(engine)
        else:
            return self._collect_results(engine)

    def _normalize_prompts(
        self,
        prompts: str | list[str] | list[int] | list[list[int]],
    ) -> list[str | list[int]]:
        """Normalize various input formats to list[str | list[int]]."""
        if isinstance(prompts, str):
            return [prompts]
        if not isinstance(prompts, list):
            raise TypeError(f"prompts must be str or list, got {type(prompts).__name__}")
        if len(prompts) == 0:
            return []
        first = prompts[0]
        if isinstance(first, int):
            return [prompts]
        return list(prompts)

    def _collect_results(self, engine: LLMEngine) -> list[RequestOutput]:
        """Block until all requests finish, then return the full results list."""
        outputs: list[RequestOutput] = []
        while engine.has_unfinished_requests():
            finished = engine.step()
            outputs.extend(finished)
        return outputs

    def _stream_results(self, engine: LLMEngine) -> Iterator[RequestOutput]:
        """Stream completed request results as they finish."""
        while engine.has_unfinished_requests():
            finished = engine.step()
            yield from finished

    def abort_request(self, request_id: str) -> None:
        """Abort the request with the given ID."""
        self._lazy_init()
        self.llm_engine.abort_request(request_id)

    def has_unfinished_requests(self) -> bool:
        """Check whether there are unfinished requests."""
        self._lazy_init()
        return self.llm_engine.has_unfinished_requests()

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.core_client import EngineCoreClient
    from vkwr.engine.outputs import RequestOutput
    from vkwr.engine.request import SamplingParams

logger = logging.getLogger(__name__)


class LLMEngine:
    """Public API for the LLM inference engine.

    Responsibilities:
    1. Manage lifecycle of EngineCoreClient, InputProcessor, OutputProcessor
    2. Provide add_request / step / abort_request / has_unfinished_requests interface
    3. Handle request input (tokenize, EOS injection) and output (detokenize, text concatenation)

    Inspired by vLLM's architecture: LLMEngine owns InputProcessor/OutputProcessor
    at the API boundary, while EngineCoreClient handles only the compute loop.
    """

    def __init__(self, config: VkwrConfig, multiprocess_mode: bool = False):
        self.config = config

        from vkwr.engine.core_client import EngineCoreClient
        from vkwr.engine.input_processor import InputProcessor
        from vkwr.engine.output_processor import OutputProcessor

        tokenizer = None
        if not config.model_config.skip_tokenizer_init:
            from vkwr.engine.tokenizer import get_tokenizer

            tokenizer = get_tokenizer(config.model_config.tokenizer)
        else:
            logger.warning("Tokenizer initialization skipped. Text prompts will fail, and outputs will not contain decoded text.")

        self.engine_core: EngineCoreClient = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vkwr_config=config,
            log_stats=True,
        )
        self.input_processor = InputProcessor(config.model_config, config.scheduler_config, tokenizer=tokenizer)
        self.output_processor = OutputProcessor(config.model_config, tokenizer=tokenizer)

    @classmethod
    def from_engine_args(cls, engine_args) -> LLMEngine:
        """Create an LLMEngine instance from EngineArgs."""
        config = engine_args.create_engine_config()
        multiprocess_mode = getattr(engine_args, "enable_multiprocessing", False)
        return cls(config, multiprocess_mode=multiprocess_mode)

    def add_request(
        self,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        collector=None,
    ) -> str:
        """Add an inference request.

        Args:
            request_id: User-provided request identifier.
            prompt: Original prompt text or token IDs.
            sampling_params: Sampling parameters.
            collector: Optional async output collector.

        Returns:
            The internal (mangled) request ID.

        Raises:
            ValueError: If prompt is empty or request_id is empty.
            RuntimeError: If attempting text encoding when tokenizer is unavailable.
        """
        vkwr_request = self.input_processor.process_input(request_id, prompt, sampling_params)
        self.output_processor.add_request(vkwr_request, collector)

        from vkwr.engine.core_request import EngineCoreRequest

        core_request = EngineCoreRequest(
            request_id=vkwr_request.request_id,
            prompt_token_ids=vkwr_request.prompt_token_ids,
            sampling_params=sampling_params,
        )
        self.engine_core.add_request(core_request)
        return vkwr_request.request_id

    def step(self) -> list[RequestOutput]:
        """Execute one inference step, returning streaming request outputs.

        Returns RequestOutput for each request that produced new tokens this step.
        In sync mode (no queue), these are collected and returned directly.
        In async mode (with queue), they are pushed to per-request collectors.

        Callers should check output.finished to know if a request has completed.
        """
        engine_core_outputs = self.engine_core.get_output()

        if not engine_core_outputs.outputs:
            return []

        processed = self.output_processor.process_outputs(
            engine_core_outputs.outputs,
            engine_core_timestamp=engine_core_outputs.timestamp,
        )
        self.output_processor.update_scheduler_stats(engine_core_outputs.scheduler_stats)

        if processed.reqs_to_abort:
            self.engine_core.abort_requests(processed.reqs_to_abort)

        for req_id in self.output_processor.get_and_clear_finished_ids():
            self.output_processor.remove_request(req_id)

        return processed.request_outputs

    def get_and_clear_finished_ids(self) -> list[str]:
        """Return and clear the list of request IDs that finished in the last step()."""
        return self.output_processor.get_and_clear_finished_ids()

    def _resolve_internal_ids(self, request_id: str) -> list[str]:
        """Resolve a request ID to internal ID(s)."""
        internals = self.output_processor.external_req_ids.get(request_id)
        if internals:
            return internals
        return [request_id]

    def abort_request(self, request_id: str) -> None:
        """Abort the specified request (by internal or external ID).

        Bilateral abort: first produce FINISHED_ABORTED output in OutputProcessor
        (unblocks any waiting collectors), then notify EngineCore to stop scheduling.
        """
        internal_ids = self._resolve_internal_ids(request_id)
        self.output_processor.abort_requests(internal_ids)
        self.engine_core.abort_requests(internal_ids)

    def abort_requests(self, request_ids: list[str]) -> None:
        """Abort multiple requests (by internal or external IDs).

        Bilateral abort: first produce FINISHED_ABORTED output in OutputProcessor,
        then notify EngineCore to stop scheduling.
        """
        all_internal_ids: list[str] = []
        for rid in request_ids:
            all_internal_ids.extend(self._resolve_internal_ids(rid))
        self.output_processor.abort_requests(all_internal_ids)
        self.engine_core.abort_requests(all_internal_ids)

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return len(self.output_processor._requests) > 0

    def remove_request(self, request_id: str) -> None:
        """Remove finished request record from output processor (by internal or external ID)."""
        for rid in self._resolve_internal_ids(request_id):
            self.output_processor.remove_request(rid)

    def shutdown(self) -> None:
        """Shutdown the engine."""
        self.engine_core.shutdown()

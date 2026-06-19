from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.outputs import EngineCoreOutputs, RequestOutput
    from vkwr.engine.request import SamplingParams

logger = logging.getLogger(__name__)


class LLMEngine:
    """Public API for the LLM inference engine.

    Responsibilities:
    1. Manage lifecycle of EngineCore, InputProcessor, OutputProcessor
    2. Provide add_request / step / abort_request / has_unfinished_requests interface
    3. Handle request input (tokenize, EOS injection) and output (detokenize, text concatenation)

    Inspired by vLLM's architecture: LLMEngine owns InputProcessor/OutputProcessor
    at the API boundary, while EngineCore handles only the compute loop.
    """

    def __init__(self, config: VkwrConfig):
        self.config = config

        from vkwr.engine.core import EngineCore
        from vkwr.engine.input_processor import InputProcessor
        from vkwr.engine.output_processor import OutputProcessor

        self.engine_core = EngineCore(config)
        self.input_processor = InputProcessor(config.model_config, config.scheduler_config)
        self.output_processor = OutputProcessor(config.model_config)
        self.engine_core.initialize()

    @classmethod
    def from_engine_args(cls, engine_args) -> LLMEngine:
        """Create an LLMEngine instance from EngineArgs."""
        config = engine_args.create_engine_config()
        return cls(config)

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
        request = self.input_processor.process_input(request_id, prompt, sampling_params)
        self.output_processor.add_request(request, collector)
        self.engine_core.add_request(request)
        return request.request_id

    def step(self) -> list[RequestOutput]:
        """Execute one inference step, returning streaming request outputs.

        Returns RequestOutput for each request that produced new tokens this step.
        In sync mode (no queue), these are collected and returned directly.
        In async mode (with queue), they are pushed to per-request collectors.

        Callers should check output.finished to know if a request has completed.
        """
        outputs_dict, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed)

        if outputs_dict is None:
            return []
        if not outputs_dict:
            return []

        engine_core_outputs = outputs_dict.get(0) or EngineCoreOutputs()
        processed = self.output_processor.process_outputs(
            engine_core_outputs.outputs,
            engine_core_timestamp=engine_core_outputs.timestamp,
        )
        self.output_processor.update_scheduler_stats(engine_core_outputs.scheduler_stats)

        if processed.reqs_to_abort:
            for rid in processed.reqs_to_abort:
                self.engine_core.abort_request(rid)

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
        """Abort the specified request (by internal or external ID)."""
        for rid in self._resolve_internal_ids(request_id):
            self.engine_core.abort_request(rid)
            self.output_processor.remove_request(rid)

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return self.engine_core.has_unfinished_requests()

    def remove_request(self, request_id: str) -> None:
        """Remove finished request record from output processor (by internal or external ID)."""
        for rid in self._resolve_internal_ids(request_id):
            self.output_processor.remove_request(rid)

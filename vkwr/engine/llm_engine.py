from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.outputs import RequestOutput
    from vkwr.engine.request import SamplingParams

logger = logging.getLogger(__name__)


class LLMEngine:
    """Public API for the LLM inference engine.

    Responsibilities:
    1. Manage lifecycle of EngineCore, InputProcessor, OutputProcessor
    2. Provide add_request / step / abort_request / has_unfinished_requests interface
    3. Handle request input (tokenize, EOS injection) and output (detokenize, text concatenation)

    Aligned with vLLM architecture:
    - LLMEngine holds InputProcessor/OutputProcessor at the API boundary
    - EngineCore handles only the compute loop (scheduler dispatch, execution, sampling)
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
    ) -> str:
        """Add an inference request.

        Args:
            request_id: Unique request identifier.
            prompt: Original prompt text or token IDs.
            sampling_params: Sampling parameters.

        Returns:
            The request ID.

        Raises:
            ValueError: If prompt is empty or request_id is empty.
            RuntimeError: If attempting text encoding when tokenizer is unavailable.
        """
        request = self.input_processor.process_input(request_id, prompt, sampling_params)
        self.output_processor.add_request(request)
        self.engine_core.add_request(request)
        return request_id

    def step(self) -> list[RequestOutput]:
        """Execute one inference step, returning completed request outputs."""
        engine_outputs = self.engine_core.step()
        finished = self.output_processor.process_outputs(engine_outputs)

        # Clean up finished request records
        for req_output in finished:
            self.output_processor.remove_request(req_output.request_id)

        return finished

    def abort_request(self, request_id: str) -> None:
        """Abort the specified request."""
        self.engine_core.abort_request(request_id)
        self.output_processor.remove_request(request_id)

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return self.engine_core.has_unfinished_requests()

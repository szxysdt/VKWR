from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vkwr.config.model import ModelConfig
from vkwr.engine.outputs import (
    CompletionOutput,
    EngineCoreOutputs,
    RequestOutput,
    RequestStats,
)
from vkwr.engine.request import VkwrRequest

if TYPE_CHECKING:
    from vkwr.engine.tokenizer import RWKVTokenizer

logger = logging.getLogger(__name__)


class OutputProcessor:
    """Convert engine core output to user-visible request output.

    Responsibilities:
    1. Maintain request output history (one RequestOutput per request)
    2. Append EngineCoreOutput to the corresponding request's CompletionOutput
    3. Trigger detokenize after completion detection
    4. Return finished request outputs

    Aligned with vLLM architecture: OutputProcessor is held only in LLMEngine, not in EngineCore.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._requests: dict[str, RequestOutput] = {}
        self._tokenizer: RWKVTokenizer | None = None

    def _get_tokenizer(self) -> RWKVTokenizer | None:
        if self._tokenizer is None:
            from vkwr.engine.tokenizer import get_tokenizer

            self._tokenizer = get_tokenizer(self.model_config.tokenizer)
        return self._tokenizer

    def add_request(self, request: VkwrRequest) -> None:
        """Register a new request, initialize RequestOutput object (R3 fix)."""
        self._requests[request.request_id] = RequestOutput(
            request_id=request.request_id,
            prompt=request.prompt if isinstance(request.prompt, str) else "",
            prompt_token_ids=request.prompt_token_ids,
            outputs=[
                CompletionOutput(
                    index=0,
                    token_ids=[],
                    text="",
                    cumlogprob=0.0,
                    logprobs=None,
                    finish_reason=None,
                )
            ],
            finished=False,
            finish_reason=None,
            stats=RequestStats(),
        )

    def process_outputs(
        self,
        engine_outputs: EngineCoreOutputs,
    ) -> list[RequestOutput]:
        """Process engine core output, return finished request outputs.

        For each EngineCoreOutput:
        1. Append new_token_ids to the corresponding request's CompletionOutput
        2. Detect finish_reason
        3. Execute detokenize on completion, mark as finished

        Returns:
            All RequestOutput finished this step (unfinished are not returned).
        """
        finished_outputs: list[RequestOutput] = []

        for core_output in engine_outputs.outputs:
            req_output = self._requests.get(core_output.request_id)
            if req_output is None:
                logger.warning(
                    "Unknown request_id %s in engine output, skipping",
                    core_output.request_id,
                )
                continue

            completion = req_output.outputs[0]

            completion.token_ids.extend(core_output.new_token_ids)
            if core_output.new_logprobs is not None:
                if completion.logprobs is None:
                    completion.logprobs = []
                completion.logprobs.extend(core_output.new_logprobs)

            if core_output.finish_reason is not None:
                completion.finish_reason = core_output.finish_reason
                completion.text = self._decode_tokens(completion.token_ids)
                req_output.finished = True
                req_output.finish_reason = core_output.finish_reason
                finished_outputs.append(req_output)
            else:
                completion.text = self._decode_tokens(completion.token_ids)

        return finished_outputs

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text (D3 fix)."""
        if not token_ids:
            return ""
        tok = self._get_tokenizer()
        if tok is not None:
            return tok.decode(token_ids)
        return ""

    def remove_request(self, request_id: str) -> None:
        """Remove finished request record (called by LLMEngine after consuming finished output)."""
        self._requests.pop(request_id, None)

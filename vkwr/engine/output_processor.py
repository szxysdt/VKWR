from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vkwr.config.model import ModelConfig
from vkwr.engine.outputs import (
    CompletionOutput,
    EngineCoreOutput,
    RequestOutput,
    RequestStats,
)
from vkwr.engine.request import RequestOutputKind, VkwrRequest

if TYPE_CHECKING:
    from vkwr.engine.tokenizer import RWKVTokenizer

logger = logging.getLogger(__name__)


@dataclass
class OutputProcessorOutput:
    request_outputs: list[RequestOutput]
    reqs_to_abort: list[str] = field(default_factory=list)


@dataclass
class _RequestState:
    """Internal per-request tracking state."""

    request: VkwrRequest
    external_req_id: str
    token_ids: list[int]
    text: str
    logprobs: list[dict[int, float]] | None
    cumlogprob: float
    finished: bool
    finish_reason: str | None
    stop_reason: int | str | None
    queue: RequestOutputCollector | None
    output_kind: RequestOutputKind
    sent_tokens_offset: int
    is_prefilling: bool

    def __init__(self, request: VkwrRequest, queue: RequestOutputCollector | None = None):
        self.request = request
        self.external_req_id = request.external_req_id or request.request_id
        self.token_ids: list[int] = []
        self.text = ""
        self.logprobs: list[dict[int, float]] | None = None
        self.cumlogprob = 0.0
        self.finished = False
        self.finish_reason: str | None = None
        self.stop_reason: int | str | None = None
        self.queue = queue
        self.output_kind = request.sampling_params.output_kind
        self.sent_tokens_offset = 0
        self.is_prefilling = True

    def make_request_output(self, prev_text: str = "") -> RequestOutput:
        """Build a RequestOutput snapshot.

        Args:
            prev_text: Previous full text (for DELTA mode text slicing).

        For DELTA mode, only includes tokens/text since the last sent output.
        For CUMULATIVE mode, includes all tokens from the start.
        """
        if self.output_kind == RequestOutputKind.DELTA:
            delta_ids = self.token_ids[self.sent_tokens_offset :]
            delta_text = self.text[len(prev_text) :] if delta_ids else ""
            delta_logprobs = self.logprobs[self.sent_tokens_offset :] if self.logprobs else None
        else:
            delta_ids = list(self.token_ids)
            delta_text = self.text
            delta_logprobs = list(self.logprobs) if self.logprobs else None

        return RequestOutput(
            request_id=self.external_req_id,
            prompt=self.request.prompt if isinstance(self.request.prompt, str) else "",
            prompt_token_ids=self.request.prompt_token_ids,
            outputs=[
                CompletionOutput(
                    index=0,
                    token_ids=delta_ids,
                    text=delta_text,
                    cumlogprob=self.cumlogprob,
                    logprobs=delta_logprobs,
                    finish_reason=self.finish_reason,
                    stop_reason=self.stop_reason,
                )
            ],
            finished=self.finished,
            finish_reason=self.finish_reason,
            stats=RequestStats(generated_tokens=len(self.token_ids)),
        )


class RequestOutputCollector:
    """Single-slot async queue per request (vLLM-aligned).

    If the producer (engine loop) pushes faster than the consumer (stream reader),
    outputs are merged via RequestOutput.add() rather than buffered.
    """

    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.request_id = request_id
        self._output: RequestOutput | Exception | None = None
        self._ready = asyncio.Event()

    def put(self, output: RequestOutput | Exception) -> None:
        if self._output is None or isinstance(output, Exception):
            self._output = output
            self._ready.set()
        elif isinstance(self._output, RequestOutput) and isinstance(output, RequestOutput):
            self._output.add(output, aggregate=self.aggregate)
        elif isinstance(self._output, Exception) and isinstance(output, Exception):
            pass

    async def get(self) -> RequestOutput:
        while (output := self._output) is None:
            await self._ready.wait()
        self._output = None
        self._ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    def get_nowait(self) -> RequestOutput | None:
        output = self._output
        if output is not None:
            self._output = None
            self._ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    def close(self):
        """Close the collector, cleaning up resources."""
        pass


class OutputProcessor:
    """Convert engine core output to user-visible request output.

    Responsibilities:
    1. Maintain per-request state (_RequestState) keyed by internal request_id
    2. Append EngineCoreOutput tokens to the corresponding request
    3. Build streaming RequestOutput snapshots each step (with external_req_id)
    4. Support DELTA and CUMULATIVE output modes
    5. Push to RequestOutputCollector queues for async mode

    Inspired by vLLM's architecture:
    - _requests dict keyed by internal request_id
    - external_req_ids maps external -> [internal] for abort support
    - Final RequestOutput.request_id is the external (user-provided) ID
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._requests: dict[str, _RequestState] = {}
        self.external_req_ids: dict[str, list[str]] = {}
        self._tokenizer: RWKVTokenizer | None = None
        self._finished_ids: list[str] = []

    def _get_tokenizer(self) -> RWKVTokenizer | None:
        if self._tokenizer is None:
            from vkwr.engine.tokenizer import get_tokenizer

            self._tokenizer = get_tokenizer(self.model_config.tokenizer)
        return self._tokenizer

    def add_request(
        self,
        request: VkwrRequest,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        """Register a new request with optional async collector queue.

        Uses internal request_id as key. Tracks external -> [internal] mapping
        for abort support.
        """
        request_id = request.request_id
        external_req_id = request.external_req_id or request_id
        self._requests[request_id] = _RequestState(request, queue)
        self.external_req_ids.setdefault(external_req_id, []).append(request_id)

    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
    ) -> OutputProcessorOutput:
        """Process engine core output, return streaming request outputs.

        EngineCoreOutput.request_id is the internal ID. Lookup is by internal ID.
        The resulting RequestOutput.request_id is the external (user-provided) ID.
        """
        self._finished_ids.clear()
        streaming_outputs: list[RequestOutput] = []

        for core_output in engine_core_outputs:
            req_state = self._requests.get(core_output.request_id)
            if req_state is None:
                logger.warning(
                    "Unknown request_id %s in engine output, skipping",
                    core_output.request_id,
                )
                continue

            # Accumulate new tokens
            req_state.token_ids.extend(core_output.new_token_ids)
            if core_output.new_logprobs is not None:
                if req_state.logprobs is None:
                    req_state.logprobs = []
                req_state.logprobs.extend(core_output.new_logprobs)

            # Detect finish
            if core_output.finish_reason is not None:
                req_state.finish_reason = core_output.finish_reason
                req_state.stop_reason = core_output.stop_reason
                req_state.finished = True

            # Decode full text (save previous for delta computation)
            prev_text = req_state.text
            req_state.text = self._decode_tokens(req_state.token_ids)

            # Build RequestOutput snapshot (uses external_req_id)
            req_output = req_state.make_request_output(prev_text=prev_text)

            # FINAL_ONLY: only emit on completion
            if req_state.output_kind == RequestOutputKind.FINAL_ONLY and not req_state.finished:
                continue

            # Push to async queue or collect for sync return
            if req_state.queue is not None:
                req_state.queue.put(req_output)
            else:
                streaming_outputs.append(req_output)

            # Update sent offset for next delta
            if not req_state.finished:
                req_state.sent_tokens_offset = len(req_state.token_ids)

            # Flip is_prefilling AFTER stats computation for this request
            req_state.is_prefilling = False

            # Track finished requests (by internal ID)
            if req_state.finished:
                self._finished_ids.append(req_state.request.request_id)

        return OutputProcessorOutput(request_outputs=streaming_outputs)

    def _decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        if not token_ids:
            return ""
        tok = self._get_tokenizer()
        if tok is not None:
            return tok.decode(token_ids)
        return ""

    def get_and_clear_finished_ids(self) -> list[str]:
        """Return and clear the list of finished internal request IDs."""
        finished = self._finished_ids
        self._finished_ids = []
        return finished

    def remove_request(self, request_id: str) -> None:
        """Remove finished request record by internal request_id."""
        req_state = self._requests.pop(request_id, None)
        if req_state:
            external = req_state.external_req_id
            if external in self.external_req_ids:
                self.external_req_ids[external].remove(request_id)
                if not self.external_req_ids[external]:
                    del self.external_req_ids[external]

    def update_scheduler_stats(self, scheduler_stats: dict | None):
        pass

    def propagate_error(self, e: Exception):
        for _, state in self._requests.items():
            if state.queue is not None:
                state.queue.put(e)

import enum
import time
from dataclasses import dataclass

import msgspec


class RequestStatus(enum.IntEnum):
    WAITING = 0
    RUNNING = 1
    FINISHED_STOPPED = 2
    FINISHED_ABORTED = 3


class RequestOutputKind(enum.IntEnum):
    """Controls what each streaming RequestOutput contains.

    CUMULATIVE: Each output contains full text/token_ids from start.
    DELTA: Each output contains only new tokens since last output.
    FINAL_ONLY: Only the final completed output is emitted.
    """

    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


class SamplingParams(msgspec.Struct, dict=True, omit_defaults=True):
    """Sampling parameters, based on vLLM SamplingParams + Albatross app.py:39-56"""

    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty_decay: float = 1.0
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    eos_token_id: int | None = None
    ignore_eos: bool = False
    seed: int | None = None
    use_beam_search: bool = False
    output_kind: int = 0

    def __post_init__(self):
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.n > 1:
            raise ValueError("n > 1 (parallel sampling) is not supported")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not (0 <= self.min_p <= 1):
            raise ValueError("min_p must be in [0, 1]")
        if self.top_k < -1:
            raise ValueError("top_k must be >= -1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.seed == -1:
            self.seed = None
        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]
        if self.stop and any(not s for s in self.stop):
            raise ValueError("stop cannot contain an empty string")
        if self.temperature < 1e-5:
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0


@dataclass
class VkwrRequest:
    """Internal request object for the engine.

    Inspired by vLLM's EngineCoreRequest:
    - request_id: internal ID (may have random suffix appended)
    - external_req_id: user-provided ID, preserved as-is for output
    """

    request_id: str
    prompt: str | list[int]
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float = 0.0
    status: RequestStatus = RequestStatus.WAITING
    priority: int = 0
    lora_request: None = None
    # The user-provided request ID. Set by InputProcessor.assign_request_id().
    # Used in final RequestOutput and to support abort by external ID.
    external_req_id: str | None = None
    # Phase 2: tracking fields for continuous batching
    num_computed_tokens: int = 0
    num_output_tokens: int = 0

    def __post_init__(self):
        if self.arrival_time == 0.0:
            self.arrival_time = time.monotonic()

    def __hash__(self) -> int:
        return hash(self.request_id)

    def __lt__(self, other: "VkwrRequest") -> bool:
        return (self.priority, self.arrival_time) < (other.priority, other.arrival_time)

    @property
    def is_decode(self) -> bool:
        """True if request has finished prefill and is in decode phase."""
        return self.num_computed_tokens >= len(self.prompt_token_ids)

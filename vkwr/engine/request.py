import enum
import time
from dataclasses import dataclass


class RequestStatus(enum.IntEnum):
    WAITING = 0
    RUNNING = 1
    FINISHED_STOPPED = 2
    FINISHED_ABORTED = 3


@dataclass
class SamplingParams:
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
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    _eos_token_id: int | None = None
    ignore_eos: bool = False
    seed: int | None = None
    use_beam_search: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be in [0, 1]")
        if self.top_k < -1:
            raise ValueError("top_k must be >= -1")
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @eos_token_id.setter
    def eos_token_id(self, value: int | None):
        self._eos_token_id = value


@dataclass
class VkwrRequest:
    """Internal request object for the engine."""

    request_id: str
    prompt: str | list[int]
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float = 0.0
    status: RequestStatus = RequestStatus.WAITING
    priority: int = 0
    lora_request: None = None

    def __post_init__(self):
        if self.arrival_time == 0.0:
            self.arrival_time = time.time()

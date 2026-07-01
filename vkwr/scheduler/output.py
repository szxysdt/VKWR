from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkwr.engine.request import SamplingParams


@dataclass
class RequestRunData:
    request_id: str
    prompt_token_ids: list[int]
    num_computed_tokens: int
    num_tokens: int
    sampling_params: SamplingParams
    input_token_ids: list[int] | None
    state: list | None = None
    is_decode: bool = False
    is_last_prefill: bool = False
    slot_index: int = 0


@dataclass
class SchedulerOutput:
    scheduled_req_ids: list[str]
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    finished_req_ids: set[str]
    request_data: dict[str, RequestRunData]
    freed_slots: list[tuple[str, int]] = field(default_factory=list)
    slot_indices: list[int] | None = None

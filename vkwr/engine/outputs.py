from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class CompletionOutput:
    index: int
    token_ids: list[int]
    text: str
    cumlogprob: float
    logprobs: list[dict[int, float]] | None = None
    finish_reason: str | None = None


@dataclass
class RequestStats:
    waiting_time: float = 0.0
    running_time: float = 0.0
    total_tokens: int = 0
    generated_tokens: int = 0


@dataclass
class RequestOutput:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool
    finish_reason: str | None
    stats: RequestStats | None = None


@dataclass
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
    new_logprobs: list[dict[int, float]] | None = None
    finish_reason: str | None = None


@dataclass
class EngineCoreOutputs:
    outputs: list[EngineCoreOutput] = field(default_factory=list)
    scheduler_stats: dict | None = None
    timestamp: float = 0.0


@dataclass
class ModelRunnerOutput:
    sampled_token_ids: dict[str, list[int]]
    sampled_logprobs: dict[str, list[dict[int, float]]] | None = None
    logits: torch.Tensor | None = None

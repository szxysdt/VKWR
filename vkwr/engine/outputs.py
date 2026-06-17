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
    stop_reason: int | str | None = None


@dataclass
class RequestStats:
    waiting_time: float = 0.0
    running_time: float = 0.0
    total_tokens: int = 0
    generated_tokens: int = 0


class RequestOutput:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool
    finish_reason: str | None
    stats: RequestStats | None

    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: list[int],
        outputs: list[CompletionOutput],
        finished: bool,
        finish_reason: str | None,
        stats: RequestStats | None = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished
        self.finish_reason = finish_reason
        self.stats = stats

    def add(self, next_output: RequestOutput, aggregate: bool) -> None:
        """Merge subsequent RequestOutput into this one (vLLM-aligned).

        Args:
            next_output: The newer RequestOutput to merge in.
            aggregate: If True (DELTA mode), concatenate new tokens onto
                existing ones. If False (CUMULATIVE mode), replace with
                the newer output entirely.
        """
        self.finished |= next_output.finished
        self.finish_reason = next_output.finish_reason or self.finish_reason

        for next_completion in next_output.outputs:
            for i, completion in enumerate(self.outputs):
                if completion.index == next_completion.index:
                    if aggregate:
                        completion.text += next_completion.text
                        completion.token_ids.extend(next_completion.token_ids)
                        if next_completion.logprobs:
                            if completion.logprobs is None:
                                completion.logprobs = []
                            completion.logprobs.extend(next_completion.logprobs)
                        completion.cumlogprob = next_completion.cumlogprob
                        completion.finish_reason = next_completion.finish_reason
                        completion.stop_reason = next_completion.stop_reason
                    else:
                        self.outputs[i] = next_completion
                    break
            else:
                self.outputs.append(next_completion)


# Sentinel to indicate request is finished, used with streaming inputs.
STREAM_FINISHED = RequestOutput(
    request_id="",
    prompt="",
    prompt_token_ids=[],
    outputs=[],
    finished=True,
    finish_reason=None,
)


@dataclass
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
    new_logprobs: list[dict[int, float]] | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None


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

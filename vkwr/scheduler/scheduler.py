from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vkwr.engine.outputs import EngineCoreOutput, EngineCoreOutputs
from vkwr.engine.request import RequestStatus, VkwrRequest
from vkwr.scheduler.interface import SchedulerInterface
from vkwr.scheduler.output import RequestRunData, SchedulerOutput

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.outputs import ModelRunnerOutput


class SimpleScheduler(SchedulerInterface):
    """Phase 1 minimal scheduler: processes one chunk per request at a time"""

    def __init__(self, config: VkwrConfig):
        self.config = config
        self.waiting: list[VkwrRequest] = []
        self.running: dict[str, VkwrRequest] = {}
        self.running_tokens: dict[str, list[int]] = {}
        self.prompt_pos: dict[str, int] = {}
        self.start_decode_pos: dict[str, int] = {}
        self.completed: dict[str, list[int]] = {}

    # ── SchedulerInterface ──────────────────────────────────────────

    def add_request(self, request: VkwrRequest) -> None:
        self.waiting.append(request)

    def has_requests(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    def finish_requests(self, request_ids: set[str]) -> None:
        for req_id in request_ids:
            self.running.pop(req_id, None)
            self.running_tokens.pop(req_id, None)
            self.start_decode_pos.pop(req_id, None)
            self.prompt_pos.pop(req_id, None)

    def schedule(self) -> SchedulerOutput:
        """
        Phase 1 simplified logic:
        1. If there are running requests, continue processing (remaining prefill chunks or decode 1 token)
        2. If no running requests, take a new one from waiting and start prefill
        """
        is_decode = False
        is_last_prefill = False

        req: VkwrRequest | None = None
        rid: str | None = None
        num_tokens = 0
        start_pos = 0

        # ── Priority: continue running requests ─────────────────────
        if self.running:
            req = next(iter(self.running.values()))
            rid = req.request_id
            consumed = self.prompt_pos.get(rid, 0)
            remaining_prompt = len(req.prompt_token_ids) - consumed

            if remaining_prompt > 0:
                num_tokens = min(
                    remaining_prompt,
                    self.config.scheduler_config.chunked_prefill_threshold,
                )
                start_pos = consumed
                new_pos = consumed + num_tokens
                self.prompt_pos[rid] = new_pos
                is_last_prefill = new_pos >= len(req.prompt_token_ids)
            else:
                is_decode = True
                num_tokens = 1
                start_pos = len(self.running_tokens.get(rid, []))

        # ── Otherwise pick a new request from waiting ──────────────
        elif self.waiting:
            req = self.waiting.pop(0)
            req.status = RequestStatus.RUNNING
            rid = req.request_id
            self.running[rid] = req
            self.running_tokens[rid] = []
            self.start_decode_pos[rid] = 0
            remaining = len(req.prompt_token_ids)
            num_tokens = min(
                remaining,
                self.config.scheduler_config.chunked_prefill_threshold,
            )
            start_pos = 0
            self.prompt_pos[rid] = num_tokens
            is_last_prefill = num_tokens >= remaining
        else:
            return SchedulerOutput(
                scheduled_req_ids=[],
                num_scheduled_tokens={},
                total_num_scheduled_tokens=0,
                finished_req_ids=set(),
                request_data={},
            )

        # ── Completion check ───────────────────────────────────────
        finished_req_ids: set[str] = set()

        if is_last_prefill:
            self.start_decode_pos[rid] = len(self.running_tokens.get(rid, [])) + 1

        # G3: when max_tokens=0, mark as finished after prefill without running the model
        if is_last_prefill and (req.sampling_params.max_tokens is not None and req.sampling_params.max_tokens == 0):
            finished_req_ids.add(rid)
            return SchedulerOutput(
                scheduled_req_ids=[],
                num_scheduled_tokens={},
                total_num_scheduled_tokens=0,
                finished_req_ids=finished_req_ids,
                request_data={},
            )
        if is_last_prefill:
            # If the prompt already fills max_model_len, the request will also finish after the last prefill chunk generates a token
            max_model_len = self.config.model_config.max_model_len
            if max_model_len is not None and len(req.prompt_token_ids) + 1 >= max_model_len:
                finished_req_ids.add(rid)
        elif is_decode:
            existing = len(self.running_tokens.get(rid, []))
            max_tok = req.sampling_params.max_tokens
            if max_tok is not None and existing >= max_tok:
                finished_req_ids.add(rid)
                return SchedulerOutput(
                    scheduled_req_ids=[],
                    num_scheduled_tokens={},
                    total_num_scheduled_tokens=0,
                    finished_req_ids=finished_req_ids,
                    request_data={},
                )
            max_model_len = self.config.model_config.max_model_len
            if max_model_len is not None:
                total_tokens = len(req.prompt_token_ids) + existing + 1
                if total_tokens >= max_model_len:
                    finished_req_ids.add(rid)

        # ── Build input_token_ids ──────────────────────────────────
        if not is_decode:
            input_token_ids = req.prompt_token_ids[start_pos : start_pos + num_tokens]
        else:
            generated = self.running_tokens.get(rid, [])
            input_token_ids = generated[-1:] if generated else req.prompt_token_ids[-1:]

        return SchedulerOutput(
            scheduled_req_ids=[rid],
            num_scheduled_tokens={rid: num_tokens},
            total_num_scheduled_tokens=num_tokens,
            finished_req_ids=finished_req_ids,
            request_data={
                rid: RequestRunData(
                    request_id=rid,
                    prompt_token_ids=req.prompt_token_ids,
                    start_pos=start_pos,
                    num_tokens=num_tokens,
                    sampling_params=req.sampling_params,
                    input_token_ids=input_token_ids,
                    state=None,
                    is_decode=is_decode,
                    is_last_prefill=is_last_prefill,
                )
            },
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> dict[str, EngineCoreOutputs]:
        """
        Record sampling results to running_tokens / completed and check completion conditions.
        Return an engine outputs dictionary.
        """
        engine_outputs: dict[str, EngineCoreOutputs] = {}

        # ── Record sampling results & detect EOS/stop_token_ids ─────
        for req_id in scheduler_output.scheduled_req_ids:
            tokens = model_output.sampled_token_ids.get(req_id, [])
            self.running_tokens[req_id].extend(tokens)

            if req_id not in scheduler_output.finished_req_ids:
                req = self.running.get(req_id)
                if req:
                    # EOS detection
                    eos_id = req.sampling_params.eos_token_id
                    if not req.sampling_params.ignore_eos and tokens and tokens[0] == eos_id:
                        scheduler_output.finished_req_ids.add(req_id)

            # stop_token_ids detection
            if req_id not in scheduler_output.finished_req_ids:
                req = self.running.get(req_id)
                if req and req.sampling_params.stop_token_ids:
                    stop_ids = set(req.sampling_params.stop_token_ids)
                    if any(tid in stop_ids for tid in tokens):
                        scheduler_output.finished_req_ids.add(req_id)

        # ── Process finished requests ───────────────────────────────
        for req_id in scheduler_output.finished_req_ids:
            req = self.running.get(req_id)
            self.completed[req_id] = self.running_tokens.pop(req_id, [])
            self.running.pop(req_id, None)
            self.prompt_pos.pop(req_id, None)
            self.start_decode_pos.pop(req_id, None)

            run_data = scheduler_output.request_data.get(req_id)
            finish_reason = _classify_finish_reason(req, run_data, self.completed.get(req_id, []))

            engine_outputs[req_id] = EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=model_output.sampled_token_ids.get(req_id, []),
                        new_logprobs=None,
                        finish_reason=finish_reason,
                    )
                ],
                timestamp=time.time(),
            )

        # ── Process unfinished requests that produced tokens ────────
        for req_id in scheduler_output.scheduled_req_ids:
            if req_id not in engine_outputs:
                tokens = model_output.sampled_token_ids.get(req_id, [])
                if not tokens:
                    continue
                engine_outputs[req_id] = EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=tokens,
                            new_logprobs=(model_output.sampled_logprobs.get(req_id) if model_output.sampled_logprobs else None),
                            finish_reason=None,
                        )
                    ],
                    timestamp=time.time(),
                )

        return engine_outputs


def _classify_finish_reason(
    req: VkwrRequest | None,
    run_data: RequestRunData | None,
    completed_tokens: list[int],
) -> str:
    """Determine finish reason: "stop" | "eos" | "length".
    Only applies to the decode phase or the last prefill chunk.
    """
    if run_data and (run_data.is_decode or run_data.is_last_prefill):
        finish_reason = "length"
        if req and completed_tokens:
            if not req.sampling_params.ignore_eos and completed_tokens[-1] == req.sampling_params.eos_token_id:
                finish_reason = "eos"
            elif req.sampling_params.stop_token_ids:
                stop_ids = set(req.sampling_params.stop_token_ids)
                if any(tid in stop_ids for tid in completed_tokens):
                    finish_reason = "stop"
        return finish_reason
    return "length"

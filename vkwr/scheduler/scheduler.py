from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vkwr.engine.outputs import EngineCoreOutput, EngineCoreOutputs
from vkwr.engine.request import RequestStatus, VkwrRequest
from vkwr.scheduler.batch_reorder import reorder_batch_to_split_decodes_and_prefills
from vkwr.scheduler.interface import SchedulerInterface
from vkwr.scheduler.output import RequestRunData, SchedulerOutput
from vkwr.scheduler.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vkwr.scheduler.utils import _get_input_tokens, _should_finish

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.outputs import ModelRunnerOutput
    from vkwr.state.state_slot_manager import StateSlotManager


class SimpleScheduler(SchedulerInterface):
    """Multi-request Continuous Batching scheduler with two-phase scheduling."""

    def __init__(self, config: VkwrConfig, slot_manager: StateSlotManager | None = None):
        self.config = config
        self.scheduler_config = config.scheduler_config
        self.model_config = config.model_config
        self.slot_manager = slot_manager

        policy = SchedulingPolicy(config.scheduler_config.scheduling_policy)
        self.waiting = create_request_queue(policy)
        self.skipped_waiting = create_request_queue(policy)

        self.running: list[VkwrRequest] = []
        self._running_map: dict[str, VkwrRequest] = {}

        self.running_output_tokens: dict[str, list[int]] = {}

    def add_request(self, request: VkwrRequest) -> None:
        request.status = RequestStatus.WAITING
        self.waiting.add_request(request)

    def has_requests(self) -> bool:
        return bool(self.waiting) or bool(self.skipped_waiting) or bool(self.running)

    def get_num_unfinished_requests(self) -> int:
        """Return number of unfinished requests (waiting + running)."""
        return len(self.running) + len(self.waiting) + len(self.skipped_waiting)

    def finish_requests(
        self,
        request_ids: set[str] | None,
        status: RequestStatus = RequestStatus.FINISHED_ABORTED,
    ) -> list[str]:
        """Force-terminate requests.

        When request_ids is None, finish ALL unfinished requests with the
        given status. Returns list of finished request IDs.

        When request_ids is a set, finish only those requests with
        FINISHED_ABORTED status (legacy path).
        """
        if request_ids is not None:
            finished_ids: list[str] = []
            for req_id in request_ids:
                req = self._running_map.get(req_id)
                if req:
                    req.status = RequestStatus.FINISHED_ABORTED
                    self.running.remove(req)
                    self._running_map.pop(req_id, None)
                    self.running_output_tokens.pop(req_id, None)
                    if self.slot_manager:
                        self.slot_manager.free(req_id)
                    finished_ids.append(req_id)
                    continue

                for q in (self.waiting, self.skipped_waiting):
                    req = next((r for r in q if r.request_id == req_id), None)
                    if req:
                        req.status = RequestStatus.FINISHED_ABORTED
                        q.remove_request(req)
                        finished_ids.append(req_id)
                        break
            return finished_ids

        finished_ids = []
        while self.running:
            req = self.running.pop()
            req.status = status
            self._running_map.pop(req.request_id, None)
            self.running_output_tokens.pop(req.request_id, None)
            if self.slot_manager:
                self.slot_manager.free(req.request_id)
            finished_ids.append(req.request_id)

        while self.waiting:
            req = self.waiting.pop_request()
            req.status = status
            finished_ids.append(req.request_id)

        while self.skipped_waiting:
            req = self.skipped_waiting.pop_request()
            req.status = status
            finished_ids.append(req.request_id)

        return finished_ids

    def schedule(self) -> SchedulerOutput:
        """Two-phase scheduling: Phase 1 schedules RUNNING, Phase 2 schedules WAITING."""
        max_model_len = self.config.model_config.max_model_len
        chunked_prefill_threshold = self.config.scheduler_config.chunked_prefill_threshold
        max_num_batched_tokens = self.config.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.config.scheduler_config.max_num_seqs
        max_num_partial_prefills = self.config.scheduler_config.max_num_partial_prefills
        max_long_partial_prefills = self.config.scheduler_config.max_long_partial_prefills

        token_budget = max_num_batched_tokens
        scheduled_req_ids: list[str] = []
        num_scheduled_tokens: dict[str, int] = {}
        request_data: dict[str, RequestRunData] = {}
        finished_req_ids: set[str] = set()

        # ── Phase 1: Schedule RUNNING requests ────────────────────
        for req in self.running:
            rid = req.request_id
            is_decode = req.is_decode

            if is_decode:
                num_new = 1
            else:
                num_new = len(req.prompt_token_ids) - req.num_computed_tokens

            if num_new <= 0:
                continue

            num_new = min(num_new, chunked_prefill_threshold)
            num_new = min(num_new, token_budget)

            if max_model_len is not None:
                num_new = min(num_new, max_model_len - 1 - req.num_computed_tokens)

            if num_new <= 0:
                if is_decode:
                    finished_req_ids.add(rid)
                continue

            output_tokens = self.running_output_tokens.get(rid, [])
            input_token_ids = _get_input_tokens(req, num_new, is_decode, output_tokens)

            if _should_finish(req, is_decode, output_tokens):
                finished_req_ids.add(rid)
                continue

            is_last_prefill = not is_decode and req.num_computed_tokens + num_new >= len(req.prompt_token_ids)

            if is_last_prefill and max_model_len is not None:
                if len(req.prompt_token_ids) + 1 >= max_model_len:
                    finished_req_ids.add(rid)
                    continue

            slot_index = 0
            if self.slot_manager:
                slot_index = self.slot_manager.get_slot(rid)

            request_data[rid] = RequestRunData(
                request_id=rid,
                prompt_token_ids=req.prompt_token_ids,
                num_computed_tokens=req.num_computed_tokens,
                num_tokens=num_new,
                sampling_params=req.sampling_params,
                input_token_ids=input_token_ids,
                state=None,
                is_decode=is_decode,
                is_last_prefill=is_last_prefill,
                slot_index=slot_index,
            )

            scheduled_req_ids.append(rid)
            num_scheduled_tokens[rid] = num_new
            token_budget -= num_new

        # Remove finished from scheduled
        if finished_req_ids:
            scheduled_req_ids = [r for r in scheduled_req_ids if r not in finished_req_ids]
            request_data = {k: v for k, v in request_data.items() if k not in finished_req_ids}
            num_scheduled_tokens = {k: v for k, v in num_scheduled_tokens.items() if k not in finished_req_ids}

        # ── Phase 2: Schedule WAITING requests ────────────────────
        policy = SchedulingPolicy(self.config.scheduler_config.scheduling_policy)
        step_skipped_waiting = create_request_queue(policy)

        partial_prefills = 0
        for r in self.running:
            if r.num_computed_tokens < len(r.prompt_token_ids):
                remaining = len(r.prompt_token_ids) - r.num_computed_tokens
                if remaining > chunked_prefill_threshold:
                    partial_prefills += 1

        while (self.waiting or self.skipped_waiting) and token_budget > 0 and len(self.running) < max_num_seqs:
            request_queue = None
            for q in (self.skipped_waiting, self.waiting):
                if not q:
                    continue
                req_head = q.peek_request()
                is_dec_h = req_head.is_decode
                remaining_h = len(req_head.prompt_token_ids) - req_head.num_computed_tokens if not is_dec_h else 0
                is_lp_h = not is_dec_h and remaining_h <= chunked_prefill_threshold
                if is_dec_h or is_lp_h:
                    request_queue = q
                    break
                is_long_h = 0 < chunked_prefill_threshold < remaining_h
                lim_h = max_long_partial_prefills if is_long_h else max_num_partial_prefills
                if lim_h <= 0 or partial_prefills < lim_h:
                    request_queue = q
                    break

            if not request_queue:
                moved = False
                for q in (self.skipped_waiting, self.waiting):
                    if not q:
                        continue
                    req_head = q.peek_request()
                    is_dec_h = req_head.is_decode
                    remaining_h = len(req_head.prompt_token_ids) - req_head.num_computed_tokens if not is_dec_h else 0
                    is_lp_h = not is_dec_h and remaining_h <= chunked_prefill_threshold
                    if is_dec_h or is_lp_h:
                        continue
                    is_long_h = 0 < chunked_prefill_threshold < remaining_h
                    lim_h = max_long_partial_prefills if is_long_h else max_num_partial_prefills
                    if lim_h > 0 and partial_prefills >= lim_h:
                        q.pop_request()
                        step_skipped_waiting.prepend_request(req_head)
                        moved = True
                if not moved:
                    break
                continue

            req = request_queue.pop_request()
            rid = req.request_id
            is_decode = req.is_decode

            if is_decode:
                num_new = 1
            else:
                num_new = len(req.prompt_token_ids) - req.num_computed_tokens

            num_new = min(num_new, chunked_prefill_threshold)
            num_new = min(num_new, token_budget)

            if max_model_len is not None:
                num_new = min(num_new, max_model_len - 1 - req.num_computed_tokens)

            if num_new <= 0:
                step_skipped_waiting.prepend_request(req)
                continue

            is_last_prefill = not is_decode and req.num_computed_tokens + num_new >= len(req.prompt_token_ids)

            if not is_decode and not is_last_prefill:
                remaining_prompt = len(req.prompt_token_ids) - req.num_computed_tokens
                is_long_prefill = 0 < chunked_prefill_threshold < remaining_prompt
                limit = max_long_partial_prefills if is_long_prefill else max_num_partial_prefills
                if limit > 0 and partial_prefills >= limit:
                    step_skipped_waiting.prepend_request(req)
                    continue

            req.status = RequestStatus.RUNNING
            if self.slot_manager:
                self.slot_manager.allocate(rid)

            self.running_output_tokens[rid] = []
            self.running.append(req)
            self._running_map[rid] = req

            if not is_decode and not is_last_prefill:
                partial_prefills += 1

            output_tokens = self.running_output_tokens.get(rid, [])
            input_token_ids = _get_input_tokens(req, num_new, is_decode, output_tokens)

            slot_index = 0
            if self.slot_manager:
                slot_index = self.slot_manager.get_slot(rid)

            request_data[rid] = RequestRunData(
                request_id=rid,
                prompt_token_ids=req.prompt_token_ids,
                num_computed_tokens=req.num_computed_tokens,
                num_tokens=num_new,
                sampling_params=req.sampling_params,
                input_token_ids=input_token_ids,
                state=None,
                is_decode=is_decode,
                is_last_prefill=is_last_prefill,
                slot_index=slot_index,
            )

            scheduled_req_ids.append(rid)
            num_scheduled_tokens[rid] = num_new
            token_budget -= num_new

        if step_skipped_waiting:
            self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # ── Advance num_computed_tokens ───────────────────────────
        for req_id in scheduled_req_ids:
            n = num_scheduled_tokens[req_id]
            if req_id in self._running_map:
                self._running_map[req_id].num_computed_tokens += n

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())

        scheduler_output = SchedulerOutput(
            scheduled_req_ids=scheduled_req_ids,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            finished_req_ids=finished_req_ids,
            request_data=request_data,
        )

        slot_indices = [self.slot_manager.get_slot(req_id) if self.slot_manager else 0 for req_id in scheduler_output.scheduled_req_ids]

        sorted_indices = reorder_batch_to_split_decodes_and_prefills(scheduler_output, decode_threshold=1)

        scheduler_output.sorted_indices = sorted_indices
        scheduler_output.slot_indices = slot_indices

        return scheduler_output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> dict[str, EngineCoreOutputs]:
        """Record sampling results and check completion conditions."""
        engine_outputs: dict[str, EngineCoreOutputs] = {}

        for req_id in scheduler_output.scheduled_req_ids:
            if req_id not in self.running_output_tokens:
                continue
            tokens = model_output.sampled_token_ids.get(req_id, [])
            self.running_output_tokens[req_id].extend(tokens)

            if req_id not in scheduler_output.finished_req_ids:
                req = self._running_map.get(req_id)
                if req:
                    req.num_output_tokens = len(self.running_output_tokens[req_id])
                    eos_id = req.sampling_params.eos_token_id
                    if not req.sampling_params.ignore_eos and tokens and tokens[0] == eos_id:
                        scheduler_output.finished_req_ids.add(req_id)

            if req_id not in scheduler_output.finished_req_ids:
                req = self._running_map.get(req_id)
                if req and req.sampling_params.stop_token_ids:
                    stop_ids = set(req.sampling_params.stop_token_ids)
                    if any(tid in stop_ids for tid in tokens):
                        scheduler_output.finished_req_ids.add(req_id)

        for req_id in scheduler_output.finished_req_ids:
            req = self._running_map.get(req_id)
            if req:
                req.status = RequestStatus.FINISHED_STOPPED
            self._running_map.pop(req_id, None)
            completed_tokens = self.running_output_tokens.pop(req_id, [])
            self.running = [r for r in self.running if r.request_id != req_id]
            if self.slot_manager:
                try:
                    self.slot_manager.free(req_id)
                except KeyError:
                    pass

            run_data = scheduler_output.request_data.get(req_id)
            finish_reason, stop_reason = _classify_finish_reason(req, run_data, completed_tokens)

            engine_outputs[req_id] = EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=model_output.sampled_token_ids.get(req_id, []),
                        new_logprobs=None,
                        finish_reason=finish_reason,
                        stop_reason=stop_reason,
                    )
                ],
                timestamp=time.monotonic(),
            )

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
                    timestamp=time.monotonic(),
                )

        return engine_outputs


def _classify_finish_reason(
    req: VkwrRequest | None,
    run_data: RequestRunData | None,
    completed_tokens: list[int],
) -> tuple[str | None, int | str | None]:
    """Determine finish reason and stop reason.

    Returns:
        (finish_reason, stop_reason)
    """
    if not run_data or not (run_data.is_decode or run_data.is_last_prefill):
        return ("length", None)

    finish_reason: str | None = "length"
    stop_reason: int | str | None = None

    if req and completed_tokens:
        if not req.sampling_params.ignore_eos and req.sampling_params.eos_token_id is not None:
            if completed_tokens[-1] == req.sampling_params.eos_token_id:
                finish_reason = "eos"
                stop_reason = req.sampling_params.eos_token_id
        elif req.sampling_params.stop_token_ids:
            stop_ids = set(req.sampling_params.stop_token_ids)
            for tid in reversed(completed_tokens):
                if tid in stop_ids:
                    finish_reason = "stop"
                    stop_reason = tid
                    break

    return (finish_reason, stop_reason)

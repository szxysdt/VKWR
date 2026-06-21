from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vkwr.engine.request import SamplingParams


@dataclass
class InputBatch:
    """Input batch data structure, constructed from SchedulerOutput.

    Encapsulates all inputs required for one model forward pass:
    token IDs, state, request ID mapping, and sampling parameters.
    """

    input_ids: torch.Tensor
    state: list[torch.Tensor]
    request_ids: list[str]
    sampling_params: list[SamplingParams]
    is_decode: bool = False
    is_last_prefill: bool = False

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1] if self.input_ids.dim() > 1 else 1

    @classmethod
    def from_scheduler_output(
        cls,
        scheduler_output,
        device: torch.device,
        state_shift: torch.Tensor,
        state_wkv: torch.Tensor,
        state_elapsed: torch.Tensor,
    ) -> InputBatch:
        """Construct an InputBatch from SchedulerOutput.

        Args:
            scheduler_output: Scheduler output containing scheduled request info and input tokens.
            device: Target device.
            state_shift: Shift state buffer [L, 2, max_bsz, C].
            state_wkv: WKV state buffer [L, max_bsz, H, N, N].
            state_elapsed: Elapsed time counter [max_bsz].

        Returns:
            An InputBatch instance.
        """
        req_ids = scheduler_output.scheduled_req_ids
        bsz = len(req_ids)

        input_ids_list = []
        sampling_params_list = []
        is_decode = False
        is_last_prefill = False

        for req_id in req_ids:
            run_data = scheduler_output.request_data[req_id]
            input_ids_list.append(run_data.input_token_ids)
            sampling_params_list.append(run_data.sampling_params)
            if run_data.is_decode:
                is_decode = True
            if run_data.is_last_prefill:
                is_last_prefill = True

        if len(input_ids_list) > 1:
            seq_len = len(input_ids_list[0])
            for i, ids in enumerate(input_ids_list[1:], 1):
                if len(ids) != seq_len:
                    raise ValueError(f"Request {i} has seq_len={len(ids)} != batch seq_len={seq_len}. Prefill and decode requests cannot be batched together.")

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)

        shift = state_shift[:, :, :bsz]
        wkv = state_wkv[:, :bsz]
        elapsed = state_elapsed[:bsz]
        state = [shift, wkv, elapsed]

        return cls(
            input_ids=input_ids,
            state=state,
            request_ids=req_ids,
            sampling_params=sampling_params_list,
            is_decode=is_decode,
            is_last_prefill=is_last_prefill,
        )

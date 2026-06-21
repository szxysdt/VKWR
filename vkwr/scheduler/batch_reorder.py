"""Batch reorder utilities for VKWR.

Reorders scheduled requests into four zones:
- decode (0): has_context AND num_tokens <= threshold AND done_prefilling
- short_extend (1): has_context AND num_tokens <= threshold AND NOT done_prefilling
- long_extend (2): has_context AND num_tokens > threshold
- prefill (3): NOT has_context
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vkwr.scheduler.output import SchedulerOutput


def reorder_batch_to_split_decodes_and_prefills(
    scheduler_output: SchedulerOutput,
    decode_threshold: int = 1,
) -> np.ndarray | None:
    """Reorder scheduled_req_ids so decode requests come first.

    Returns sorted_indices (np.ndarray) where sorted_indices[i] is the
    original position index of the request now at new position i.
    Returns None if no reorder is needed.
    """
    req_ids = scheduler_output.scheduled_req_ids
    num_reqs = len(req_ids)
    if num_reqs <= 1:
        return None

    regions = np.zeros(num_reqs, dtype=np.int32)
    for i, rid in enumerate(req_ids):
        rd = scheduler_output.request_data[rid]
        has_context = rd.num_computed_tokens > 0
        is_below_threshold = rd.num_tokens <= decode_threshold
        done_prefilling = rd.num_computed_tokens >= len(rd.prompt_token_ids)

        if not has_context:
            regions[i] = 3
        elif not is_below_threshold:
            regions[i] = 2
        elif not done_prefilling:
            regions[i] = 1
        else:
            regions[i] = 0

    sorted_indices = np.argsort(regions, kind="stable")

    if np.array_equal(sorted_indices, np.arange(num_reqs)):
        return None

    scheduler_output.scheduled_req_ids = [req_ids[i] for i in sorted_indices]
    return sorted_indices

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkwr.engine.request import VkwrRequest


def _get_input_tokens(
    req: VkwrRequest,
    num_new: int,
    is_decode: bool,
    output_tokens: list[int],
) -> list[int] | None:
    """Build input_token_ids for a scheduled request.

    Returns None for decode — model runner reads from GPU cache.
    """
    if not is_decode:
        return req.prompt_token_ids[req.num_computed_tokens : req.num_computed_tokens + num_new]
    return None


def _should_finish(req: VkwrRequest, is_decode: bool, output_tokens: list[int]) -> bool:
    """Check if request should finish before being scheduled."""
    if not is_decode:
        return False
    existing = len(output_tokens)
    max_tok = req.sampling_params.max_tokens
    if max_tok is not None and existing >= max_tok:
        return True
    max_model_len = getattr(req, "_max_model_len", None)
    if max_model_len is not None:
        total = len(req.prompt_token_ids) + existing + 1
        if total >= max_model_len:
            return True
    return False

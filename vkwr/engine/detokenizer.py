from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkwr.engine.tokenizer import RWKVTokenizer


class Detokenizer:
    """Simple token IDs to text conversion wrapper.

    Phase 1 uses full decode. Phase 2+ will replace with incremental detokenizer
    (see vLLM BaseIncrementalDetokenizer.update()).
    """

    def __init__(self, tokenizer: RWKVTokenizer):
        self.tokenizer = tokenizer

    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids)

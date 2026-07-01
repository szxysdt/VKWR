from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from vkwr.config.model import ModelConfig
from vkwr.config.scheduler import SchedulerConfig
from vkwr.engine.request import SamplingParams, VkwrRequest
from vkwr.utils import _random_uuid

if TYPE_CHECKING:
    from vkwr.tokenizers.rwkv7 import RWKVTokenizer

logger = logging.getLogger(__name__)


class InputProcessor:
    """Convert user input into engine internal requests.

    Responsibilities:
    1. Text to token IDs (via RWKVTokenizer)
    2. Inject eos_token_id (D4 fix)
    3. Optional length truncation (constrained by max_model_len / max_num_batched_tokens)
    4. Create VkwrRequest + assign internal request ID

    Inspired by vLLM's architecture: InputProcessor lives at the LLMEngine layer,
    not inside EngineCore.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig | None = None,
        tokenizer: RWKVTokenizer | None = None,
    ):
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self._tokenizer: RWKVTokenizer | None = tokenizer

    @staticmethod
    def assign_request_id(request: VkwrRequest) -> None:
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters to ensure uniqueness.

        The user-provided ID is preserved as external_req_id for output.
        """
        request.external_req_id = request.request_id
        request.request_id = f"{request.external_req_id}-{_random_uuid()[:8]}"

    def get_tokenizer(self) -> RWKVTokenizer | None:
        return self._tokenizer

    def process_input(
        self,
        request_id: str,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> VkwrRequest:
        """Process input and return VkwrRequest.

        Args:
            request_id: User-provided request identifier (becomes external_req_id).
            prompt: Original prompt (text or token IDs).
            sampling_params: Sampling parameters.

        Returns:
            VkwrRequest instance with internal request_id and external_req_id set.

        Raises:
            ValueError: If prompt is empty or request_id is empty.
            RuntimeError: If attempting text encoding when tokenizer is unavailable.
        """
        if not request_id:
            raise ValueError("request_id cannot be empty")

        prompt_token_ids: list[int]

        if isinstance(prompt, str):
            if not prompt:
                raise ValueError("prompt cannot be empty")
            tokenizer = self.get_tokenizer()
            if tokenizer is None:
                raise RuntimeError("Cannot encode text prompt — tokenizer unavailable. Set ModelConfig.tokenizer or use skip_tokenizer_init=False.")
            prompt_token_ids = tokenizer.encode(prompt)
        else:
            prompt_token_ids = list(prompt)
            if not prompt_token_ids:
                raise ValueError("prompt_token_ids cannot be empty")

        max_len = self._get_max_prompt_len()
        if max_len is not None and len(prompt_token_ids) > max_len:
            logger.warning(
                "Truncating prompt for request %s from %d to %d tokens",
                request_id,
                len(prompt_token_ids),
                max_len,
            )
            prompt_token_ids = prompt_token_ids[:max_len]

        self._inject_eos_token_id(sampling_params)
        self._infer_max_tokens(sampling_params, prompt_token_ids)

        request = VkwrRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            arrival_time=time.monotonic(),
        )

        # Assign internal request ID (appends random suffix).
        self.assign_request_id(request)

        return request

    def _get_max_prompt_len(self) -> int | None:
        """Get the maximum allowed prompt length.

        Prefers model_config.max_model_len, then scheduler_config.max_num_batched_tokens.
        """
        if self.model_config.max_model_len is not None:
            return self.model_config.max_model_len
        if self.scheduler_config is not None:
            return self.scheduler_config.max_num_batched_tokens
        return None

    def _inject_eos_token_id(self, sampling_params: SamplingParams) -> None:
        """Inject eos_token_id (D4 fix).

        Priority: tokenizer.eos_token_id -> SchedulerConfig.eos_token_id -> 0.
        """
        if sampling_params.eos_token_id is not None:
            return
        tokenizer = self.get_tokenizer()
        if tokenizer is not None and tokenizer.eos_token_id is not None:
            sampling_params.eos_token_id = tokenizer.eos_token_id
            return
        if self.scheduler_config is not None:
            sampling_params.eos_token_id = self.scheduler_config.eos_token_id
        else:
            sampling_params.eos_token_id = 0

    def _infer_max_tokens(self, sampling_params: SamplingParams, prompt_token_ids: list[int]) -> None:
        """Infer max_tokens value.

        Priority:
        1. Request explicitly provides max_tokens.
        2. Global default default_max_tokens (--max-tokens CLI arg).
        3. max_model_len - len(prompt_token_ids).
        4. Hard-coded 2048.
        """
        if sampling_params.max_tokens is not None:
            return
        if self.scheduler_config is not None and self.scheduler_config.default_max_tokens is not None:
            sampling_params.max_tokens = self.scheduler_config.default_max_tokens
            return
        max_model_len = self._get_max_prompt_len()
        if max_model_len is not None:
            remaining = max_model_len - len(prompt_token_ids)
            sampling_params.max_tokens = max(remaining, 1)
        else:
            sampling_params.max_tokens = 2048

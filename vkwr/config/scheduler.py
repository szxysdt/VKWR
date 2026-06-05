import logging
from dataclasses import InitVar

from vkwr.config.utils import config, get_hash_factors, hash_factors

logger = logging.getLogger(__name__)


@config
class SchedulerConfig:
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 128
    max_model_len: InitVar[int | None] = None
    chunked_prefill_threshold: int = 512
    max_num_partial_prefills: int = 2
    max_long_partial_prefills: int = 1
    scheduling_policy: str = "fcfs"
    runner_type: str = "generate"
    enable_chunked_prefill: bool = True
    eos_token_id: int = 0
    ignore_eos: bool = False
    default_max_tokens: int | None = None

    def __post_init__(self, max_model_len: int | None) -> None:
        if max_model_len and self.max_num_batched_tokens < max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) smaller than max_model_len (%d). Consider increasing max_num_batched_tokens or decreasing max_model_len.",
                self.max_num_batched_tokens,
                max_model_len,
            )

    def compute_hash(self) -> str:
        factors = get_hash_factors(self)
        return hash_factors(factors)

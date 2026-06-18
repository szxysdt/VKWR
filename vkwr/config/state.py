"""State lifecycle configuration."""

from vkwr.config.utils import config, get_hash_factors, hash_factors


@config
class StateConfig:
    state_cache_max_cpu_entries: int = 64
    max_checkpoints_per_req: int = 8
    checkpoint_interval: int | None = None

    def compute_hash(self) -> str:
        factors = get_hash_factors(self)
        return hash_factors(factors)

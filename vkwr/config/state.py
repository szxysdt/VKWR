"""State lifecycle configuration."""

from vkwr.config.utils import config, get_hash_factors, hash_factors


@config
class StateConfig:
    state_cache_max_cpu_entries: int = 64
    max_checkpoints_per_req: int = 8
    checkpoint_interval: int | None = None
    # TODO: Disk cache is not yet implemented. These fields are reserved for
    # a planned disk-backed state caching layer. Currently no module consumes
    # them — they will become functional when the disk cache backend is added.
    # See: dev_docs/code_reviews/v008-20260621-engine-cli-env-audit.md P0-3.1
    enable_disk_cache: bool = False
    disk_cache_path: str | None = None
    disk_cache_max_size_mb: int = 10240

    def compute_hash(self) -> str:
        factors = get_hash_factors(self)
        return hash_factors(factors)

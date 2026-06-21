from vkwr.config.utils import config, get_hash_factors, hash_factors


@config
class GPUWorkerConfig:
    device: str = "cuda"
    gpu_memory_utilization: float = 0.92
    enforce_eager: bool = False
    max_context_len_from_generator: int | None = None
    shutdown_timeout: int = 0

    def compute_hash(self) -> str:
        factors = get_hash_factors(self)
        return hash_factors(factors)

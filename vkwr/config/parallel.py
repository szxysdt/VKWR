from vkwr.config.utils import config, get_hash_factors, hash_factors


@config
class ParallelConfig:
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    distributed_executor_backend: str | None = None

    def __post_init__(self):
        if self.tp_size != 1:
            raise ValueError("V3 does not support TP yet")
        if self.pp_size != 1:
            raise ValueError("V3 does not support PP yet")

    def compute_hash(self) -> str:
        factors = get_hash_factors(self)
        return hash_factors(factors)

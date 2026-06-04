from vkwr.config.utils import config, get_hash_factors, hash_factors


@config
class CompilationConfig:
    cudagraph_capture_size: list[int] | None = None
    cudagraph_mode: str = "full"
    torch_compile_mode: str | None = None

    def compute_hash(self) -> str:
        factors = get_hash_factors(self)
        return hash_factors(factors)

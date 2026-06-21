from vkwr.config.utils import config, get_hash_factors, hash_factors


@config
class CompilationConfig:
    cudagraph_capture_size: list[int] | None = None
    cudagraph_mode: str = "full"
    # TODO: torch.compile mode is not yet implemented. Reserved for a future
    # compilation pipeline that would support "reduce-overhead", "max-autotune",
    # and other torch.compile modes beyond cudagraphs.
    # See: dev_docs/code_reviews/v008-20260621-engine-cli-env-audit.md P0-3.2
    torch_compile_mode: str | None = None

    def compute_hash(self) -> str:
        factors = get_hash_factors(self)
        return hash_factors(factors)

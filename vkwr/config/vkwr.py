from dataclasses import dataclass
from pathlib import Path

from vkwr.config.compilation import CompilationConfig
from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.config.parallel import ParallelConfig
from vkwr.config.scheduler import SchedulerConfig
from vkwr.config.worker import GPUWorkerConfig


@dataclass
class EngineArgs:
    model: str
    tokenizer: str | None = None
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = False
    dtype: str = "float16"
    max_model_len: int | None = None
    # TODO(download_dir): Currently not passed to ModelConfig in create_engine_config().
    # Either wire it through or remove once the download_dir flow is decided.
    # See: dev_docs/code_reviews/v001-20260621-engine-cli-env-audit.md P0-2.1
    # download_dir: str | None = None
    load_format: str = "auto"
    seed: int = 42
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.92
    enforce_eager: bool = False
    distributed_executor_backend: str | None = None
    cudagraph_capture_size: list[int] | None = None
    cudagraph_mode: str = "full"
    default_max_tokens: int | None = None
    enable_async_scheduling: bool = True
    batch_queue_size: int = 2
    enable_multiprocessing: bool = True

    def create_engine_config(self) -> VkwrConfig:
        tokenizer = self.tokenizer
        if tokenizer is None:
            tokenizer = str(Path(self.model).parent)

        mc_kwargs: dict = {
            "model": self.model,
            "tokenizer": tokenizer,
            "skip_tokenizer_init": self.skip_tokenizer_init,
            "trust_remote_code": self.trust_remote_code,
            "dtype": self.dtype,
            "load_format": self.load_format,
            "seed": self.seed,
        }
        if self.max_model_len is not None:
            mc_kwargs["max_model_len"] = self.max_model_len

        model_config = ModelConfig(**mc_kwargs)

        scheduler_config = SchedulerConfig(
            max_model_len=model_config.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            default_max_tokens=self.default_max_tokens,
            enable_async_scheduling=self.enable_async_scheduling,
            batch_queue_size=self.batch_queue_size,
        )

        worker_config = GPUWorkerConfig(
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
        )

        parallel_config = ParallelConfig(
            distributed_executor_backend=self.distributed_executor_backend,
        )

        compilation_config = CompilationConfig(
            cudagraph_capture_size=self.cudagraph_capture_size,
            cudagraph_mode=self.cudagraph_mode,
        )

        return VkwrConfig(
            model_config=model_config,
            scheduler_config=scheduler_config,
            worker_config=worker_config,
            parallel_config=parallel_config,
            compilation_config=compilation_config,
        )

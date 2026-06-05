from dataclasses import dataclass

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
    tokenizer_mode: str = "rwkv"
    trust_remote_code: bool = False
    dtype: str = "float16"
    max_model_len: int | None = None
    download_dir: str | None = None
    load_format: str = "auto"
    seed: int = 42
    max_num_batched_tokens: int | None = None
    max_num_seqs: int = 128
    gpu_memory_utilization: float = 0.92
    enforce_eager: bool = False
    distributed_executor_backend: str | None = None
    cudagraph_capture_size: list[int] | None = None
    cudagraph_mode: str = "full"
    default_max_tokens: int | None = None

    def create_engine_config(self) -> VkwrConfig:
        mc_kwargs: dict = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "tokenizer_mode": self.tokenizer_mode,
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
            max_num_batched_tokens=self.max_num_batched_tokens or 2048,
            max_num_seqs=self.max_num_seqs,
            default_max_tokens=self.default_max_tokens,
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

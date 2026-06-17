import hashlib

from pydantic import Field

from vkwr.config.compilation import CompilationConfig
from vkwr.config.model import ModelConfig
from vkwr.config.parallel import ParallelConfig
from vkwr.config.scheduler import SchedulerConfig
from vkwr.config.state import StateConfig
from vkwr.config.utils import config
from vkwr.config.worker import GPUWorkerConfig


@config
class VkwrConfig:
    model_config: ModelConfig
    scheduler_config: SchedulerConfig = Field(default_factory=SchedulerConfig)
    worker_config: GPUWorkerConfig = Field(default_factory=GPUWorkerConfig)
    parallel_config: ParallelConfig = Field(default_factory=ParallelConfig)
    compilation_config: CompilationConfig = Field(default_factory=CompilationConfig)
    state_config: StateConfig = Field(default_factory=StateConfig)

    def __post_init__(self):
        if self.model_config.dtype != "float16":
            raise ValueError("RWKV7 only supports float16")

        if self.scheduler_config.max_num_batched_tokens < self.scheduler_config.max_num_seqs:
            raise ValueError("max_num_batched_tokens must be >= max_num_seqs")

        # Dynamically compute CUDA Graph capture sizes (F1 fix)
        if self.compilation_config.cudagraph_capture_size is None and not self.worker_config.enforce_eager and self.compilation_config.cudagraph_mode != "none":
            from vkwr.config.utils import get_default_cudagraph_capture_sizes, replace

            object.__setattr__(
                self,
                "compilation_config",
                replace(
                    self.compilation_config,
                    cudagraph_capture_size=get_default_cudagraph_capture_sizes(
                        self.scheduler_config.max_num_seqs,
                        self.scheduler_config.max_num_batched_tokens,
                    ),
                ),
            )

    def compute_hash(self) -> str:
        factors = []
        factors.append(self.model_config.compute_hash())
        factors.append(self.scheduler_config.compute_hash())
        factors.append(self.worker_config.compute_hash())
        factors.append(self.parallel_config.compute_hash())
        factors.append(self.compilation_config.compute_hash())
        factors.append(self.state_config.compute_hash())
        return hashlib.sha256(str(factors).encode()).hexdigest()[:10]

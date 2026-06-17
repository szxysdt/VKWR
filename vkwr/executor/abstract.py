from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.outputs import ModelRunnerOutput
    from vkwr.scheduler.output import SchedulerOutput


class ExecutorInterface(ABC):
    """Abstract executor interface"""

    @staticmethod
    def get_class(config: VkwrConfig) -> type[ExecutorInterface]:
        """Select executor class based on config"""
        backend = config.parallel_config.distributed_executor_backend
        if backend == "ray":
            from vkwr.executor.ray_executor import RayExecutor

            return RayExecutor
        elif backend == "mp":
            from vkwr.executor.multiproc_executor import MultiprocExecutor

            return MultiprocExecutor
        else:
            from vkwr.executor.uniproc_executor import UniprocExecutor

            return UniprocExecutor

    def __init__(self, config: VkwrConfig, slot_manager=None):
        self.config = config
        self.slot_manager = slot_manager

    @abstractmethod
    def initialize(self) -> None:
        """Initialize worker and device"""

    @abstractmethod
    def execute_model(self, scheduler_output: SchedulerOutput, non_block: bool = False) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        """Execute model forward pass"""

    @abstractmethod
    def sample_tokens(
        self,
        model_runner_output: ModelRunnerOutput,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """Sample tokens"""

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights"""

    @abstractmethod
    def determine_available_memory(self) -> int:
        """Detect available GPU memory (bytes)"""

    @abstractmethod
    def compile_or_warm_up_model(self) -> None:
        """Compile or warm up the model"""

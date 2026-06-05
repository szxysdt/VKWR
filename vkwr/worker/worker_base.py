from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.outputs import ModelRunnerOutput
    from vkwr.scheduler.output import SchedulerOutput


class WorkerBase(ABC):
    """Abstract base class for workers"""

    def __init__(self, config: VkwrConfig):
        self.config = config

    @abstractmethod
    def init_device(self) -> None:
        """Initialize the device (GPU/CPU)"""

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights"""

    @abstractmethod
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        """Execute model forward pass"""

    @abstractmethod
    def sample_tokens(self, model_runner_output: ModelRunnerOutput, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Sample from logits"""

    @abstractmethod
    def determine_available_memory(self) -> int:
        """Detect available GPU memory (bytes)"""

    @abstractmethod
    def compile_or_warm_up_model(self) -> None:
        """Compile or warm up the model"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import TYPE_CHECKING

from vkwr.executor.abstract import ExecutorInterface
from vkwr.worker.gpu_worker import GPUWorker

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.outputs import ModelRunnerOutput
    from vkwr.scheduler.output import SchedulerOutput

logger = logging.getLogger(__name__)


class UniprocExecutor(ExecutorInterface):
    """Single-process executor — directly calls GPUWorker"""

    def __init__(self, config: VkwrConfig, slot_manager=None):
        super().__init__(config, slot_manager)
        self.worker: GPUWorker | None = None

    def initialize(self) -> None:
        self.worker = GPUWorker(self.config)
        self.worker.init_device(self.slot_manager)

    def load_model(self) -> None:
        if self.worker is None:
            raise RuntimeError("Executor not initialized. Call initialize() first.")
        self.worker.load_model()

    def execute_model(self, scheduler_output: SchedulerOutput, non_block: bool = False) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        if self.worker is None:
            raise RuntimeError("Executor not initialized. Call initialize() first.")
        output = self.worker.execute_model(scheduler_output)
        if non_block:
            future: Future[ModelRunnerOutput] = Future()
            future.set_result(output)
            return future
        return output

    def sample_tokens(
        self,
        model_runner_output: ModelRunnerOutput,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        if self.worker is None:
            raise RuntimeError("Executor not initialized. Call initialize() first.")
        return self.worker.sample_tokens(model_runner_output, scheduler_output)

    def determine_available_memory(self) -> int:
        if self.worker is None:
            raise RuntimeError("Executor not initialized. Call initialize() first.")
        return self.worker.determine_available_memory()

    def compile_or_warm_up_model(self) -> None:
        if self.worker is None:
            raise RuntimeError("Executor not initialized. Call initialize() first.")
        self.worker.compile_or_warm_up_model()

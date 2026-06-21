from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

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
        self._compute_stream = torch.cuda.Stream()
        self._executor = ThreadPoolExecutor(max_workers=1)

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
        if non_block:
            future: Future[ModelRunnerOutput] = self._executor.submit(self._run_forward, scheduler_output)
            return future
        return self._run_forward(scheduler_output)

    def _run_forward(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        with torch.cuda.stream(self._compute_stream):
            result = self.worker.execute_model(scheduler_output)
            return result

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

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()
            self.worker = None
        self._executor.shutdown(wait=False)

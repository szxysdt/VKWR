from __future__ import annotations

import logging
from concurrent.futures import Future
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
        # Synchronous execution on the calling thread.  This avoids the
        # cross-thread race where the worker thread reads _last_sampled_token
        # before the main thread's sample_tokens has written it.  When
        # non_block=True, we still return a completed Future so the caller's
        # pipeline logic (batch_queue + future.done() + future.result()) works
        # unchanged.
        result = self._run_forward(scheduler_output)
        if non_block:
            future: Future[ModelRunnerOutput] = Future()
            future.set_result(result)
            return future
        return result

    def _run_forward(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        with torch.cuda.stream(self._compute_stream):
            # Wait for the default stream so that any _last_sampled_token
            # writes from the previous batch's sample_tokens are visible.
            self._compute_stream.wait_stream(torch.cuda.default_stream())
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

    def condense(self, moves: list[tuple[int, int]]) -> None:
        """v2x: no-op. CPU-side slot_manager.batch_condense() already updated
        req_to_slot / slot_to_req maps. GPU state is addressed by slot_indices,
        physical contiguity is not required."""
        pass

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()
            self.worker = None

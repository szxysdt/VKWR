from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from vkwr.worker.gpu_model_runner import GPUModelRunner
from vkwr.worker.worker_base import WorkerBase

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.outputs import ModelRunnerOutput
    from vkwr.scheduler.output import SchedulerOutput

logger = logging.getLogger(__name__)


class GPUWorker(WorkerBase):
    """GPU Worker - manages GPU device and model execution"""

    def __init__(self, config: VkwrConfig, local_rank: int = 0):
        super().__init__(config)
        self.local_rank = local_rank
        self.device = torch.device(config.worker_config.device)
        self.model_runner: GPUModelRunner | None = None

    def init_device(self, slot_manager=None) -> None:
        """Initialize GPU device and create ModelRunner"""
        torch.cuda.set_device(self.local_rank)
        self.model_runner = GPUModelRunner(self.config, self.device, slot_manager)
        logger.info("GPU device initialized: %s (rank %d)", self.device, self.local_rank)

    def load_model(self) -> None:
        if self.model_runner is None:
            raise RuntimeError("Device not initialized. Call init_device() first.")
        self.model_runner.load_model()

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        if self.model_runner is None:
            raise RuntimeError("ModelRunner not initialized. Call init_device() first.")
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, model_runner_output: ModelRunnerOutput, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        if self.model_runner is None:
            raise RuntimeError("ModelRunner not initialized. Call init_device() first.")
        return self.model_runner.sample_tokens(model_runner_output, scheduler_output)

    def determine_available_memory(self) -> int:
        """Profile available GPU memory"""
        gpu_mem_total = torch.cuda.get_device_properties(self.device).total_memory
        available_mem = int(gpu_mem_total * self.config.worker_config.gpu_memory_utilization)
        return available_mem

    def compile_or_warm_up_model(self) -> None:
        """Warmup + CUDA Graph capture (Phase 3)."""
        if self.model_runner is None:
            raise RuntimeError("ModelRunner not initialized. Call init_device() first.")

        self.model_runner.warmup()

        if self.model_runner._cudagraph_enabled and self.model_runner.cudagraph_manager:
            n_shapes = len(self.model_runner.cudagraph_manager.capture_shapes)
            capture_sizes = self.model_runner.cudagraph_manager._capture_sizes
            logger.info(
                "Starting CUDA Graph capture: %d graphs for sizes %s",
                n_shapes,
                capture_sizes,
            )
            for shape in reversed(self.model_runner.cudagraph_manager.capture_shapes):
                self.model_runner._capture_for_shape(shape)
            n = len(self.model_runner.cudagraph_manager._entries)
            logger.info("CUDA Graph captured: %d graphs", n)

    def shutdown(self) -> None:
        if self.model_runner is not None:
            self.model_runner.shutdown()
            self.model_runner = None

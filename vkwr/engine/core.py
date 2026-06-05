from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from vkwr.executor.abstract import ExecutorInterface

if TYPE_CHECKING:
    from vkwr.config.engine import VkwrConfig
    from vkwr.engine.outputs import EngineCoreOutputs
    from vkwr.engine.request import VkwrRequest
    from vkwr.scheduler.interface import SchedulerInterface

logger = logging.getLogger(__name__)


class EngineCore:
    """Core engine: synchronous main loop.

    Responsibilities:
    1. Initialize executor (load model, warmup)
    2. Manage the schedule-execute-update compute loop
    3. Handle request addition, abortion, and completion status queries

    Aligned with vLLM architecture: EngineCore handles only the compute loop (schedule, execute, sample).
    InputProcessor / OutputProcessor are held at the LLMEngine layer, not inside EngineCore.
    """

    def __init__(self, config: VkwrConfig):
        self.config = config
        self.model_executor: ExecutorInterface = ExecutorInterface.get_class(config)(config)
        self._create_scheduler()
        self._initialized = False

    def _create_scheduler(self) -> None:
        """Create scheduler instance. Phase 1 uses SimpleScheduler, replaceable later."""
        from vkwr.scheduler.scheduler import SimpleScheduler

        self.scheduler: SchedulerInterface = SimpleScheduler(self.config)

    def initialize(self) -> None:
        """Initialize: load model and warmup."""
        if self._initialized:
            return

        logger.info("Initializing EngineCore...")
        self.model_executor.initialize()
        self.model_executor.load_model()
        self.model_executor.compile_or_warm_up_model()
        self._initialized = True
        logger.info("EngineCore initialized successfully.")

    def add_request(self, request: VkwrRequest) -> None:
        """Add a request to the scheduler."""
        if not self._initialized:
            raise RuntimeError("EngineCore not initialized. Call initialize() first.")
        self.scheduler.add_request(request)

    def step(self) -> EngineCoreOutputs:
        """Execute one inference step.

        Process:
        1. Schedule: SimpleScheduler.schedule() -> SchedulerOutput
        2. Execute: model_executor.execute_model() -> ModelRunnerOutput
        3. Update: scheduler.update_from_output() -> dict[req_id, EngineCoreOutputs]
        4. Merge: all EngineCoreOutput into a single EngineCoreOutputs

        Returns:
            EngineCoreOutputs containing all request outputs for this step (unfinished + finished).
            Returns empty EngineCoreOutputs if no requests to process.
        """
        from vkwr.engine.outputs import EngineCoreOutputs

        if not self._initialized:
            raise RuntimeError("EngineCore not initialized. Call initialize() first.")

        if not self.scheduler.has_requests():
            return EngineCoreOutputs()

        # 1. Schedule
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            if scheduler_output.finished_req_ids:
                from vkwr.engine.outputs import ModelRunnerOutput

                fake_model_output = ModelRunnerOutput(sampled_token_ids={})
                engine_outputs = self.scheduler.update_from_output(scheduler_output, fake_model_output)
                return EngineCoreOutputs(
                    outputs=[eo for eo_list in engine_outputs.values() for eo in eo_list.outputs],
                    timestamp=time.time(),
                )
            return EngineCoreOutputs()

        # 2. Execute model (including sampling)
        model_output = self.model_executor.execute_model(scheduler_output)

        # 3. Update scheduler state and generate engine outputs
        engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

        # 4. Merge all request outputs
        all_outputs = EngineCoreOutputs(
            outputs=[eo for eo_list in engine_outputs.values() for eo in eo_list.outputs],
            timestamp=time.time(),
        )
        return all_outputs

    def abort_request(self, request_id: str) -> None:
        """Abort the specified request."""
        if not self._initialized:
            raise RuntimeError("EngineCore not initialized. Call initialize() first.")
        self.scheduler.finish_requests({request_id})

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return self.scheduler.has_requests()

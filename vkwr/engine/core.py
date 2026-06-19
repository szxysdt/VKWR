from __future__ import annotations

import logging
import time
from collections import deque
from queue import Queue
from typing import TYPE_CHECKING

from vkwr.executor.abstract import ExecutorInterface
from vkwr.state.state_slot_manager import StateSlotManager

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

    Inspired by vLLM's design: EngineCore handles only the compute loop (schedule,
    execute, sample). InputProcessor / OutputProcessor live at the LLMEngine layer
    above, not inside EngineCore.
    """

    def __init__(self, config: VkwrConfig):
        self.config = config
        self.slot_manager = StateSlotManager(config.scheduler_config.max_num_seqs)
        self.model_executor: ExecutorInterface = ExecutorInterface.get_class(config)(config, self.slot_manager)
        self._create_scheduler()
        self._initialized = False
        self.batch_queue: deque | None = None
        self.batch_queue_size: int = 0
        self.step_fn = self.step

        if self.config.scheduler_config.enable_async_scheduling:
            self.batch_queue_size = self.config.scheduler_config.batch_queue_size
            self.batch_queue = deque(maxlen=self.batch_queue_size)
            self.step_fn = self.step_with_batch_queue

        self.aborts_queue: Queue = Queue()

    def _create_scheduler(self) -> None:
        """Create scheduler instance. Phase 1 uses SimpleScheduler, replaceable later."""
        from vkwr.scheduler.scheduler import SimpleScheduler

        self.scheduler: SchedulerInterface = SimpleScheduler(self.config, self.slot_manager)

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

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Execute one inference step.

        Process:
        1. Schedule: SimpleScheduler.schedule() -> SchedulerOutput
        2. Execute: model_executor.execute_model() -> ModelRunnerOutput
        3. Update: scheduler.update_from_output() -> dict[req_id, EngineCoreOutputs]
        4. Merge: all EngineCoreOutput into a single EngineCoreOutputs

        Returns:
            Tuple of (outputs_dict, model_executed). outputs_dict maps worker ID
            to EngineCoreOutputs. model_executed indicates if a forward pass ran.
        """
        from vkwr.engine.outputs import EngineCoreOutputs

        if not self._initialized:
            raise RuntimeError("EngineCore not initialized. Call initialize() first.")

        if not self.scheduler.has_requests():
            return {}, False

        # 1. Schedule
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            if scheduler_output.finished_req_ids:
                from vkwr.engine.outputs import ModelRunnerOutput

                fake_model_output = ModelRunnerOutput(sampled_token_ids={})
                self._save_finished_state(scheduler_output)
                self._process_aborts_queue()
                engine_outputs = self.scheduler.update_from_output(scheduler_output, fake_model_output)
                merged = EngineCoreOutputs(
                    outputs=[eo for eo_list in engine_outputs.values() for eo in eo_list.outputs],
                    timestamp=time.monotonic(),
                )
                return {0: merged}, False
            return {}, False

        # 2. Execute model (including sampling)
        model_output = self.model_executor.execute_model(scheduler_output)

        # 2b. Save state cache for finished requests (before slots are freed)
        self._save_finished_state(scheduler_output)

        # Process abort queue
        self._process_aborts_queue()

        # 3. Update scheduler state and generate engine outputs
        engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

        # 4. Merge all request outputs
        merged = EngineCoreOutputs(
            outputs=[eo for eo_list in engine_outputs.values() for eo in eo_list.outputs],
            timestamp=time.monotonic(),
        )
        return {0: merged}, True

    def post_step(self, model_executed: bool) -> None:
        pass

    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids: list[str] = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            self.abort_requests(request_ids)

    def abort_requests(self, request_ids: list[str]) -> None:
        self.scheduler.finish_requests(set(request_ids))

    def step_with_batch_queue(self) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Async step: schedule first, then process completed batches.

        Schedule-first ordering aligns with vLLM. Since decode input tokens
        are now read from GPU cache (_last_sampled_token), the scheduler no
        longer depends on running_output_tokens, eliminating the one-step lag.
        """
        from vkwr.engine.outputs import EngineCoreOutputs

        batch_queue = self.batch_queue
        assert batch_queue is not None
        assert len(batch_queue) <= self.batch_queue_size

        # 1. Schedule and launch new work (only if queue has room).
        model_executed = False
        if self.scheduler.has_requests() and len(batch_queue) < self.batch_queue_size:
            scheduler_output = self.scheduler.schedule()
            if scheduler_output.total_num_scheduled_tokens == 0:
                if not batch_queue:
                    return self._handle_zero_tokens(scheduler_output)
            else:
                exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
                batch_queue.appendleft((exec_future, scheduler_output))
                model_executed = True

        # 2. Process oldest completed batch.
        if batch_queue and batch_queue[-1][0].done():
            future, sched_out = batch_queue.pop()
            model_output = future.result()
            self._save_finished_state(sched_out)
            self._process_aborts_queue()
            engine_outputs = self.scheduler.update_from_output(sched_out, model_output)
            processed_merged = EngineCoreOutputs(
                outputs=[eo for eo_list in engine_outputs.values() for eo in eo_list.outputs],
                timestamp=time.monotonic(),
            )
            return {0: processed_merged}, True

        # 3. No completed batch to return. Signal caller if queue filling.
        if model_executed and len(batch_queue) < self.batch_queue_size:
            return None, True
        return {}, model_executed

    def _handle_zero_tokens(self, scheduler_output) -> tuple[dict | None, bool]:
        """Handle schedule result with zero scheduled tokens."""
        from vkwr.engine.outputs import EngineCoreOutputs, ModelRunnerOutput

        if not scheduler_output.finished_req_ids:
            return {}, False

        fake_model_output = ModelRunnerOutput(sampled_token_ids={})
        self._save_finished_state(scheduler_output)
        self._process_aborts_queue()
        engine_outputs = self.scheduler.update_from_output(scheduler_output, fake_model_output)
        merged = EngineCoreOutputs(
            outputs=[eo for eo_list in engine_outputs.values() for eo in eo_list.outputs],
            timestamp=time.monotonic(),
        )
        return {0: merged}, False

    def _save_finished_state(self, scheduler_output) -> None:
        """Save state cache for finished requests (before slots are freed)."""
        runner = self.model_executor.worker.model_runner
        if runner is None or runner.state_cache is None:
            return

        for req_id in scheduler_output.finished_req_ids:
            run_data = scheduler_output.request_data.get(req_id)
            if run_data is None:
                continue
            slot = run_data.slot_index
            token_pos = run_data.num_computed_tokens
            req = next((r for r in self.scheduler.running if r.request_id == req_id), None)
            if req is not None:
                state = runner._slice_state_from_slot(slot)
                runner.state_cache.save(req_id, state, token_pos)
            else:
                runner.state_cache.clear(req_id)

    def abort_request(self, request_id: str) -> None:
        """Abort a single request by enqueueing to aborts_queue."""
        self.aborts_queue.put([request_id])

    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests."""
        return self.scheduler.has_requests()

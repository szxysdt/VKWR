from vkwr.executor.abstract import ExecutorInterface


class RayExecutor(ExecutorInterface):
    """Ray distributed executor (placeholder — Phase 4 implementation)"""

    def __init__(self, config, slot_manager=None):
        super().__init__(config, slot_manager)

    def initialize(self) -> None:
        raise NotImplementedError("RayExecutor is not yet implemented")

    def execute_model(self, scheduler_output, non_block=False):
        raise NotImplementedError("RayExecutor is not yet implemented")

    def sample_tokens(self, model_runner_output, scheduler_output):
        raise NotImplementedError("RayExecutor is not yet implemented")

    def load_model(self) -> None:
        raise NotImplementedError("RayExecutor is not yet implemented")

    def determine_available_memory(self) -> int:
        raise NotImplementedError("RayExecutor is not yet implemented")

    def compile_or_warm_up_model(self) -> None:
        raise NotImplementedError("RayExecutor is not yet implemented")

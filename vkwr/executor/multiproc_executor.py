from vkwr.executor.abstract import ExecutorInterface


class MultiprocExecutor(ExecutorInterface):
    """Multi-process executor (placeholder — Phase 4 implementation)"""

    def initialize(self) -> None:
        raise NotImplementedError("MultiprocExecutor is not yet implemented")

    def execute_model(self, scheduler_output, non_block=False):
        raise NotImplementedError("MultiprocExecutor is not yet implemented")

    def sample_tokens(self, model_runner_output, scheduler_output):
        raise NotImplementedError("MultiprocExecutor is not yet implemented")

    def load_model(self) -> None:
        raise NotImplementedError("MultiprocExecutor is not yet implemented")

    def determine_available_memory(self) -> int:
        raise NotImplementedError("MultiprocExecutor is not yet implemented")

    def compile_or_warm_up_model(self) -> None:
        raise NotImplementedError("MultiprocExecutor is not yet implemented")

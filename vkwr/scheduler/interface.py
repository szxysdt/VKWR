from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vkwr.engine.outputs import EngineCoreOutputs, ModelRunnerOutput
    from vkwr.engine.request import VkwrRequest
    from vkwr.scheduler.output import SchedulerOutput

from vkwr.engine.request import RequestStatus


class SchedulerInterface(ABC):
    """Abstract scheduler interface"""

    @abstractmethod
    def add_request(self, request: VkwrRequest) -> None:
        """Add a request to the scheduler"""

    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        """Execute one scheduling step, returning the requests to process and metadata"""

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> dict[str, EngineCoreOutputs]:
        """Update scheduler state from model output, return engine outputs"""

    @abstractmethod
    def has_requests(self) -> bool:
        """Whether there are unfinished requests"""

    @abstractmethod
    def finish_requests(
        self,
        request_ids: set[str] | None,
        status: RequestStatus = RequestStatus.FINISHED_ABORTED,
    ) -> list[str]:
        """Forcefully terminate requests.

        When request_ids is None, finish ALL unfinished requests.
        Returns list of finished request IDs.
        """

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Return number of unfinished requests."""

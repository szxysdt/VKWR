import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vkwr.engine.request import VkwrRequest


class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    @abstractmethod
    def add_request(self, request: VkwrRequest) -> None:
        pass

    @abstractmethod
    def pop_request(self) -> VkwrRequest:
        pass

    @abstractmethod
    def peek_request(self) -> VkwrRequest:
        pass

    @abstractmethod
    def prepend_request(self, request: VkwrRequest) -> None:
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        pass

    @abstractmethod
    def remove_request(self, request: VkwrRequest) -> None:
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[VkwrRequest]) -> None:
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[VkwrRequest]:
        pass


class FCFSRequestQueue(deque[VkwrRequest], RequestQueue):
    def add_request(self, request: VkwrRequest) -> None:
        self.append(request)

    def pop_request(self) -> VkwrRequest:
        return self.popleft()

    def peek_request(self) -> VkwrRequest:
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: VkwrRequest) -> None:
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        self.extendleft(requests)

    def remove_request(self, request: VkwrRequest) -> None:
        self.remove(request)

    def remove_requests(self, requests: Iterable[VkwrRequest]) -> None:
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return super().__len__()

    def __iter__(self) -> Iterator[VkwrRequest]:
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    def __init__(self) -> None:
        self._heap: list[VkwrRequest] = []

    def add_request(self, request: VkwrRequest) -> None:
        heapq.heappush(self._heap, request)

    def pop_request(self) -> VkwrRequest:
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> VkwrRequest:
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: VkwrRequest) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: VkwrRequest) -> None:
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[VkwrRequest]) -> None:
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[VkwrRequest]:
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")

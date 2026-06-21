import pytest

from vkwr.engine.request import SamplingParams, VkwrRequest
from vkwr.scheduler.request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)


def _make_req(req_id: str, priority: int = 0, arrival_time: float = 0.0) -> VkwrRequest:
    return VkwrRequest(
        request_id=req_id,
        prompt="",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(),
        priority=priority,
        arrival_time=arrival_time,
    )


class TestFCFSRequestQueue:
    def test_add_and_pop_order(self):
        q = FCFSRequestQueue()
        r1, r2, r3 = _make_req("r1"), _make_req("r2"), _make_req("r3")
        q.add_request(r1)
        q.add_request(r2)
        q.add_request(r3)
        assert q.pop_request().request_id == "r1"
        assert q.pop_request().request_id == "r2"

    def test_peek(self):
        q = FCFSRequestQueue()
        r = _make_req("r1")
        q.add_request(r)
        assert q.peek_request().request_id == "r1"
        assert len(q) == 1

    def test_prepend(self):
        q = FCFSRequestQueue()
        q.add_request(_make_req("r2"))
        q.prepend_request(_make_req("r1"))
        assert q.pop_request().request_id == "r1"

    def test_prepend_requests(self):
        q = FCFSRequestQueue()
        q.add_request(_make_req("r3"))
        source = FCFSRequestQueue()
        source.add_request(_make_req("r1"))
        source.add_request(_make_req("r2"))
        q.prepend_requests(source)
        assert q.pop_request().request_id == "r2"
        assert q.pop_request().request_id == "r1"

    def test_remove_request(self):
        q = FCFSRequestQueue()
        r1, r2 = _make_req("r1"), _make_req("r2")
        q.add_request(r1)
        q.add_request(r2)
        q.remove_request(r1)
        assert q.pop_request().request_id == "r2"

    def test_remove_requests(self):
        q = FCFSRequestQueue()
        r1, r2, r3 = _make_req("r1"), _make_req("r2"), _make_req("r3")
        q.add_request(r1)
        q.add_request(r2)
        q.add_request(r3)
        q.remove_requests([r1, r3])
        assert len(q) == 1
        assert q.pop_request().request_id == "r2"

    def test_empty_pop_raises(self):
        q = FCFSRequestQueue()
        with pytest.raises(IndexError):
            q.pop_request()

    def test_empty_peek_raises(self):
        q = FCFSRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_bool_and_len(self):
        q = FCFSRequestQueue()
        assert not q
        assert len(q) == 0
        q.add_request(_make_req("r1"))
        assert q
        assert len(q) == 1

    def test_iter(self):
        q = FCFSRequestQueue()
        q.add_request(_make_req("r1"))
        q.add_request(_make_req("r2"))
        ids = [r.request_id for r in q]
        assert ids == ["r1", "r2"]


class TestPriorityRequestQueue:
    def test_priority_order(self):
        q = PriorityRequestQueue()
        q.add_request(_make_req("r1", priority=5))
        q.add_request(_make_req("r2", priority=1))
        q.add_request(_make_req("r3", priority=3))
        assert q.pop_request().request_id == "r2"
        assert q.pop_request().request_id == "r3"
        assert q.pop_request().request_id == "r1"

    def test_same_priority_arrival_order(self):
        q = PriorityRequestQueue()
        q.add_request(_make_req("r1", priority=1, arrival_time=2.0))
        q.add_request(_make_req("r2", priority=1, arrival_time=1.0))
        assert q.pop_request().request_id == "r2"

    def test_peek(self):
        q = PriorityRequestQueue()
        r = _make_req("r1", priority=1)
        q.add_request(r)
        assert q.peek_request().request_id == "r1"
        assert len(q) == 1

    def test_prepend_is_add(self):
        q = PriorityRequestQueue()
        q.add_request(_make_req("r1", priority=5))
        q.prepend_request(_make_req("r2", priority=1))
        assert q.pop_request().request_id == "r2"

    def test_remove_request(self):
        q = PriorityRequestQueue()
        r1 = _make_req("r1", priority=5)
        r2 = _make_req("r2", priority=1)
        q.add_request(r1)
        q.add_request(r2)
        q.remove_request(r1)
        assert q.pop_request().request_id == "r2"

    def test_remove_requests(self):
        q = PriorityRequestQueue()
        r1 = _make_req("r1", priority=5)
        r2 = _make_req("r2", priority=1)
        r3 = _make_req("r3", priority=3)
        q.add_request(r1)
        q.add_request(r2)
        q.add_request(r3)
        q.remove_requests({r1, r3})
        assert len(q) == 1
        assert q.pop_request().request_id == "r2"

    def test_empty_pop_raises(self):
        q = PriorityRequestQueue()
        with pytest.raises(IndexError):
            q.pop_request()

    def test_empty_peek_raises(self):
        q = PriorityRequestQueue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_iter_priority_order(self):
        q = PriorityRequestQueue()
        q.add_request(_make_req("r1", priority=5))
        q.add_request(_make_req("r2", priority=1))
        q.add_request(_make_req("r3", priority=3))
        ids = [r.request_id for r in q]
        assert ids == ["r2", "r3", "r1"]
        assert len(q) == 3


class TestCreateRequestQueue:
    def test_fcfs(self):
        q = create_request_queue(SchedulingPolicy.FCFS)
        assert isinstance(q, FCFSRequestQueue)

    def test_priority(self):
        q = create_request_queue(SchedulingPolicy.PRIORITY)
        assert isinstance(q, PriorityRequestQueue)

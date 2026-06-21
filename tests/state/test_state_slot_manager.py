from __future__ import annotations

import pytest

from vkwr.state.state_slot_manager import StateSlotManager


class TestStateSlotManager:
    def test_allocate_different_slots(self):
        mgr = StateSlotManager(4)
        s0 = mgr.allocate("req_a")
        s1 = mgr.allocate("req_b")
        s2 = mgr.allocate("req_c")
        assert s0 == 0
        assert s1 == 1
        assert s2 == 2
        assert len(mgr.free_slots) == 1

    def test_free_and_reuse(self):
        mgr = StateSlotManager(2)
        s0 = mgr.allocate("req_a")
        s1 = mgr.allocate("req_b")
        assert s0 == 0
        assert s1 == 1
        assert len(mgr.free_slots) == 0

        freed = mgr.free("req_a")
        assert freed == 0
        assert len(mgr.free_slots) == 1

        s2 = mgr.allocate("req_c")
        assert s2 == 0

    def test_get_indices_order(self):
        mgr = StateSlotManager(8)
        mgr.allocate("req_x")
        mgr.allocate("req_y")
        mgr.allocate("req_z")
        indices = mgr.get_indices(["req_z", "req_x", "req_y"])
        assert indices == [2, 0, 1]

    def test_exhaust_raises(self):
        mgr = StateSlotManager(2)
        mgr.allocate("req_a")
        mgr.allocate("req_b")
        with pytest.raises(RuntimeError, match="No free slots"):
            mgr.allocate("req_c")

    def test_free_returns_slot(self):
        mgr = StateSlotManager(4)
        s0 = mgr.allocate("req_a")
        s1 = mgr.allocate("req_b")
        assert mgr.free("req_b") == s1
        assert mgr.free("req_a") == s0

    def test_get_slot(self):
        mgr = StateSlotManager(4)
        s = mgr.allocate("req_a")
        assert mgr.get_slot("req_a") == s

    def test_get_slot_missing_raises(self):
        mgr = StateSlotManager(4)
        with pytest.raises(KeyError):
            mgr.get_slot("nonexistent")

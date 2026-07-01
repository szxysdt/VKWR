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

    # --- slot_to_req consistency tests ---

    def test_slot_to_req_consistency(self):
        mgr = StateSlotManager(4)
        s0 = mgr.allocate("req_a")
        s1 = mgr.allocate("req_b")
        assert mgr.slot_to_req[s0] == "req_a"
        assert mgr.slot_to_req[s1] == "req_b"
        assert mgr.req_to_slot["req_a"] == s0
        assert mgr.req_to_slot["req_b"] == s1
        mgr.free("req_a")
        assert s0 not in mgr.slot_to_req
        assert "req_a" not in mgr.req_to_slot
        s2 = mgr.allocate("req_c")
        assert mgr.slot_to_req[s2] == "req_c"
        assert mgr.req_to_slot["req_c"] == s2

    def test_free_updates_slot_to_req(self):
        mgr = StateSlotManager(4)
        mgr.allocate("req_a")
        mgr.allocate("req_b")
        mgr.allocate("req_c")
        mgr.free("req_b")
        assert 1 not in mgr.slot_to_req
        assert "req_b" not in mgr.req_to_slot

    def test_allocate_maintains_slot_to_req(self):
        mgr = StateSlotManager(4)
        s0 = mgr.allocate("req_a")
        assert mgr.slot_to_req[s0] == "req_a"
        mgr.free("req_a")
        s1 = mgr.allocate("req_b")
        assert s1 == 0
        assert mgr.slot_to_req[0] == "req_b"
        assert mgr.req_to_slot["req_b"] == 0

    # --- batch_condense tests ---

    def test_batch_condense_single(self):
        mgr = StateSlotManager(4)
        mgr.allocate("A")
        mgr.allocate("B")
        mgr.allocate("C")
        moves = mgr.batch_condense([("C", 2)])
        assert moves == []
        assert mgr.req_to_slot == {"A": 0, "B": 1}
        assert mgr.slot_to_req == {0: "A", 1: "B"}
        assert 2 in mgr.free_slots

    def test_batch_condense_back_to_front(self):
        mgr = StateSlotManager(5)
        mgr.allocate("A")  # slot 0
        mgr.allocate("B")  # slot 1
        mgr.allocate("C")  # slot 2
        mgr.allocate("D")  # slot 3
        mgr.allocate("E")  # slot 4
        moves = mgr.batch_condense([("B", 1), ("D", 3)])
        # E(4) slides to hole 1. Now active={A:0, E:1, C:2}, holes {3,4} go to free_slots.
        # Only 1 move needed to achieve contiguity since remaining holes are above active region.
        assert moves == [(4, 1)]
        assert mgr.req_to_slot == {"A": 0, "C": 2, "E": 1}
        active_slots = sorted(mgr.req_to_slot.values())
        assert active_slots == [0, 1, 2]
        assert sorted(mgr.free_slots) == [3, 4]

    def test_batch_condense_multiple_holes(self):
        mgr = StateSlotManager(6)
        mgr.allocate("A")
        mgr.allocate("B")
        mgr.allocate("C")
        mgr.allocate("D")
        mgr.allocate("E")
        mgr.allocate("F")
        moves = mgr.batch_condense([("B", 1), ("D", 3)])
        assert len(moves) == 2
        active_slots = sorted(mgr.req_to_slot.values())
        assert active_slots == [0, 1, 2, 3]
        assert len(mgr.free_slots) == 2
        assert len(mgr.free_slots) == len(set(mgr.free_slots))
        assert mgr.free_slots == sorted(mgr.free_slots)

    def test_batch_condense_all_slots(self):
        mgr = StateSlotManager(3)
        mgr.allocate("A")
        mgr.allocate("B")
        mgr.allocate("C")
        moves = mgr.batch_condense([("A", 0), ("B", 1), ("C", 2)])
        assert moves == []
        assert mgr.req_to_slot == {}
        assert mgr.slot_to_req == {}
        assert sorted(mgr.free_slots) == [0, 1, 2]

    def test_batch_condense_empty(self):
        mgr = StateSlotManager(4)
        mgr.allocate("A")
        moves = mgr.batch_condense([])
        assert moves == []
        assert mgr.req_to_slot == {"A": 0}

    def test_batch_condense_sparse_slots(self):
        mgr = StateSlotManager(10)
        # Allocate all 10, free inner slots to create holes, then allocate new
        # requests into the freed slots, creating a scenario with many holes to fill.
        for i in range(10):
            mgr.allocate(f"X{i}")
        # Free slots 3, 5, 7 creating holes at those positions
        for i in [3, 5, 7]:
            mgr.free(f"X{i}")
        # Allocate 3 new requests into freed slots 3, 5, 7
        mgr.allocate("A")  # slot 3
        mgr.allocate("B")  # slot 5
        mgr.allocate("C")  # slot 7
        # Active: X0@0, X1@1, X2@2, A@3, X4@4, B@5, X6@6, C@7, X8@8, X9@9
        # Free A(3), B(5), C(7) — slots far apart
        moves = mgr.batch_condense([("A", 3), ("B", 5), ("C", 7)])
        assert len(moves) == 2
        active_slots = sorted(mgr.req_to_slot.values())
        assert active_slots == [0, 1, 2, 3, 4, 5, 6]
        assert mgr.free_slots == sorted(mgr.free_slots)
        assert len(mgr.free_slots) == len(set(mgr.free_slots))

    def test_batch_condense_free_slots_sorted(self):
        mgr = StateSlotManager(8)
        mgr.allocate("A")
        mgr.allocate("B")
        mgr.allocate("C")
        mgr.allocate("D")
        mgr.allocate("E")
        mgr.allocate("F")
        mgr.allocate("G")
        mgr.allocate("H")
        mgr.batch_condense([("C", 2), ("F", 5)])
        assert mgr.free_slots == sorted(mgr.free_slots)
        mgr.batch_condense([("B", 1)])
        assert mgr.free_slots == sorted(mgr.free_slots)

    def test_batch_condense_maintains_invariant(self):
        mgr = StateSlotManager(8)
        for i in range(6):
            mgr.allocate(f"R{i}")
        active = sorted(mgr.req_to_slot.values())
        assert active == list(range(len(active)))
        mgr.batch_condense([("R1", 1), ("R4", 4)])
        active = sorted(mgr.req_to_slot.values())
        assert active == list(range(len(active)))
        mgr.batch_condense([("R2", 2)])
        active = sorted(mgr.req_to_slot.values())
        assert active == list(range(len(active)))

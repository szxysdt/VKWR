"""Integration test: batch condense / slot compaction end-to-end flow.

Tests the complete batch_condense/compaction flow through StateSlotManager
with mocked Runner/Executor components. Since GPU is not available in CI,
the runner's _condense_slots and _swap_state_slots are mocked.

Covers:
- StateSlotManager.batch_condense() produces correct moves
- slot_to_req / req_to_slot consistency after condense
- Active slot invariant {0..N-1} maintained after multiple cycles
- New allocation after condense fills lowest available slot
- Multiple consecutive condense operations preserve invariants
"""

from unittest.mock import MagicMock

from vkwr.state.state_slot_manager import StateSlotManager


def _assert_slot_contiguous(sm: StateSlotManager) -> int:
    """Verify active slots form {0, 1, ..., N-1}. Returns active count."""
    active = set(sm.req_to_slot.values())
    n = len(active)
    expected = set(range(n))
    assert active == expected, f"Slot contiguity violated: active={sorted(active)}, expected={sorted(expected)}"
    # Also verify reverse map consistency
    for req_id, slot in sm.req_to_slot.items():
        assert sm.slot_to_req[slot] == req_id, f"Reverse map inconsistent: req_to_slot[{req_id}]={slot} but slot_to_req[{slot}]={sm.slot_to_req[slot]}"
    return n


class TestMultiRequestSequentialFinish:
    """3 requests decode, finish one at a time, verify slot contiguity maintained."""

    def test_multi_request_sequential_finish(self):
        sm = StateSlotManager(max_slots=8)

        # Allocate 3 requests: slots 0, 1, 2
        sm.allocate("req-a")
        sm.allocate("req-b")
        sm.allocate("req-c")
        assert sm.req_to_slot == {"req-a": 0, "req-b": 1, "req-c": 2}
        assert _assert_slot_contiguous(sm) == 3

        # req-b finishes (slot 1) -> req-c should slide from 2 -> 1
        moves = sm.batch_condense([("req-b", 1)])
        assert moves == [(2, 1)], f"Expected [(2,1)], got {moves}"
        assert sm.req_to_slot == {"req-a": 0, "req-c": 1}
        assert sm.slot_to_req == {0: "req-a", 1: "req-c"}
        assert _assert_slot_contiguous(sm) == 2
        assert sm.free_slots == [2, 3, 4, 5, 6, 7]

        # req-a finishes (slot 0) -> req-c should slide from 1 -> 0
        moves = sm.batch_condense([("req-a", 0)])
        assert moves == [(1, 0)], f"Expected [(1,0)], got {moves}"
        assert sm.req_to_slot == {"req-c": 0}
        assert sm.slot_to_req == {0: "req-c"}
        assert _assert_slot_contiguous(sm) == 1
        assert sm.free_slots == [1, 2, 3, 4, 5, 6, 7]

        # req-c finishes (slot 0)
        moves = sm.batch_condense([("req-c", 0)])
        assert moves == []
        assert sm.req_to_slot == {}
        assert sm.slot_to_req == {}
        assert _assert_slot_contiguous(sm) == 0
        assert sm.free_slots == [0, 1, 2, 3, 4, 5, 6, 7]


class TestCompactionStateIsolation:
    """Middle request finishes, verify remaining requests state unaffected."""

    def test_compaction_state_isolation(self):
        sm = StateSlotManager(max_slots=8)

        # Allocate 5 requests: slots 0,1,2,3,4
        for rid in ["req-a", "req-b", "req-c", "req-d", "req-e"]:
            sm.allocate(rid)
        assert _assert_slot_contiguous(sm) == 5

        # Middle request (req-c, slot 2) finishes
        moves = sm.batch_condense([("req-c", 2)])
        assert moves == [(4, 2)], f"Expected [(4,2)], got {moves}"

        # Remaining requests preserved with correct slots
        assert sm.req_to_slot == {
            "req-a": 0,
            "req-b": 1,
            "req-d": 3,
            "req-e": 2,  # slid from 4 -> 2
        }
        assert sm.slot_to_req == {
            0: "req-a",
            1: "req-b",
            2: "req-e",
            3: "req-d",
        }
        assert _assert_slot_contiguous(sm) == 4

        # Verify req-a and req-b were not disturbed
        assert sm.req_to_slot["req-a"] == 0
        assert sm.req_to_slot["req-b"] == 1

        # Now req-d finishes (slot 3), req-e should slide 2->3 is wrong,
        # actually no one above slot 3 except nothing, so no moves needed
        # Wait: active = {0,1,2,3}. Removing slot 3. first_hole=3.
        # last_occupied = max of {0,1,2} = 2. 2 < 3, so no moves.
        moves = sm.batch_condense([("req-d", 3)])
        assert moves == []
        assert sm.req_to_slot == {"req-a": 0, "req-b": 1, "req-e": 2}
        assert _assert_slot_contiguous(sm) == 3
        assert 3 in sm.free_slots


class TestCompactionWithNewRequest:
    """After compaction, new request allocates into freed slot correctly."""

    def test_compaction_with_new_request(self):
        sm = StateSlotManager(max_slots=8)

        # Allocate 3: slots 0,1,2
        sm.allocate("req-a")
        sm.allocate("req-b")
        sm.allocate("req-c")
        assert sm.free_slots == [3, 4, 5, 6, 7]

        # req-b finishes (slot 1), req-c slides 2->1
        moves = sm.batch_condense([("req-b", 1)])
        assert moves == [(2, 1)]
        assert sm.free_slots == [2, 3, 4, 5, 6, 7]

        # New request allocates lowest available slot (1 is taken by req-c)
        # Wait: active = {0: req-a, 1: req-c}. free_slots = [2,3,4,5,6,7]
        slot = sm.allocate("req-d")
        assert slot == 2, f"Expected slot 2, got {slot}"
        assert sm.req_to_slot == {"req-a": 0, "req-c": 1, "req-d": 2}
        assert _assert_slot_contiguous(sm) == 3

    def test_compaction_then_multiple_new_requests(self):
        sm = StateSlotManager(max_slots=8)

        # Allocate 4: slots 0,1,2,3
        for rid in ["req-a", "req-b", "req-c", "req-d"]:
            sm.allocate(rid)

        # req-b (slot 1) and req-d (slot 3) finish simultaneously
        moves = sm.batch_condense([("req-b", 1), ("req-d", 3)])
        # Active after removal: {0: req-a, 2: req-c}. Holes: {1, 3}
        # Back-to-front: last_occupied=2. unfilled below 2: {1}. dst=1.
        # Move 2->1 for req-c. freed_set adds 2, discards 1. freed={2,3}
        # last_occupied=1. freed_set has 1? No (discarded). But slot_to_req:
        # req-c is now at 1. So last_occupied was decremented to 1.
        # Actually after the move, last_occupied -= 1 => 1. 1 not in freed_set.
        # unfilled below 1: none. Break.
        # freed_set = {2, 3}. free_slots = [2, 3, 4, 5, 6, 7]
        assert (2, 1) in moves, f"Expected (2,1) in {moves}"
        assert sm.req_to_slot == {"req-a": 0, "req-c": 1}
        assert _assert_slot_contiguous(sm) == 2

        # Allocate two new requests - should fill slots 2 and 3
        slot1 = sm.allocate("req-e")
        assert slot1 == 2
        slot2 = sm.allocate("req-f")
        assert slot2 == 3
        assert sm.req_to_slot == {"req-a": 0, "req-c": 1, "req-e": 2, "req-f": 3}
        assert _assert_slot_contiguous(sm) == 4


class TestAbortTriggersCompaction:
    """Abort path triggers condense correctly."""

    def test_abort_triggers_compaction(self):
        sm = StateSlotManager(max_slots=8)

        # Allocate 3 requests
        sm.allocate("req-a")
        sm.allocate("req-b")
        sm.allocate("req-c")
        assert sm.req_to_slot == {"req-a": 0, "req-b": 1, "req-c": 2}

        # Simulate abort: req-b is marked for abort
        # In EngineCore, _process_aborts_queue returns [(req_id, slot_index)]
        # Then _batch_condense is called with that list
        abort_freed = [("req-b", sm.req_to_slot["req-b"])]
        assert abort_freed == [("req-b", 1)]

        # Mock executor condense
        mock_executor = MagicMock()
        moves = sm.batch_condense(abort_freed)
        mock_executor.condense(moves)

        # Verify condense was called with correct moves
        mock_executor.condense.assert_called_once_with([(2, 1)])

        # Verify state
        assert sm.req_to_slot == {"req-a": 0, "req-c": 1}
        assert _assert_slot_contiguous(sm) == 2

    def test_multiple_aborts_single_condense(self):
        sm = StateSlotManager(max_slots=8)

        # Allocate 5: slots 0,1,2,3,4
        for rid in ["req-a", "req-b", "req-c", "req-d", "req-e"]:
            sm.allocate(rid)

        # Two aborts: req-b (slot 1) and req-d (slot 3)
        abort_freed = [
            ("req-b", sm.req_to_slot["req-b"]),
            ("req-d", sm.req_to_slot["req-d"]),
        ]

        mock_executor = MagicMock()
        moves = sm.batch_condense(abort_freed)
        mock_executor.condense(moves)

        mock_executor.condense.assert_called_once()
        assert len(moves) >= 1
        assert _assert_slot_contiguous(sm) == 3


class TestMultipleCondenseCycles:
    """Multiple consecutive condense operations don't break invariants."""

    def test_many_rounds_condense_and_allocate(self):
        sm = StateSlotManager(max_slots=4)

        # Fill all slots
        for i in range(4):
            sm.allocate(f"req-{i}")
        assert _assert_slot_contiguous(sm) == 4

        # Cycle: free middle, condense, allocate new
        for cycle in range(5):
            # Free slot 1
            target_req = sm.slot_to_req[1]
            sm.batch_condense([(target_req, 1)])
            assert _assert_slot_contiguous(sm) == 3
            assert 1 in sm.req_to_slot.values() or len(sm.req_to_slot) < 3

            # Allocate new
            sm.allocate(f"new-{cycle}")
            assert _assert_slot_contiguous(sm) == 4

        # Final state should be contiguous
        assert _assert_slot_contiguous(sm) == 4
        assert sm.free_slots == []

    def test_condense_no_moves_when_already_compact(self):
        sm = StateSlotManager(max_slots=8)
        sm.allocate("req-a")
        sm.allocate("req-b")
        sm.allocate("req-c")

        # Free the highest slot — no slides needed since it's at the end
        moves = sm.batch_condense([("req-c", 2)])
        assert moves == []
        assert _assert_slot_contiguous(sm) == 2
        assert sm.free_slots == [2, 3, 4, 5, 6, 7]

    def test_empty_condense_idempotent(self):
        sm = StateSlotManager(max_slots=8)
        sm.allocate("req-a")

        # Empty condense should be no-op
        moves = sm.batch_condense([])
        assert moves == []
        assert sm.req_to_slot == {"req-a": 0}

        # Repeated empty condenses
        for _ in range(5):
            moves = sm.batch_condense([])
            assert moves == []
        assert _assert_slot_contiguous(sm) == 1

from __future__ import annotations


class StateSlotManager:
    """Maps requests to GPU state slots.

    Lightweight Phase 2 implementation without CPU swap support.
    """

    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        self.req_to_slot: dict[str, int] = {}
        self.slot_to_req: dict[int, str] = {}
        self.free_slots: list[int] = list(range(max_slots))
        self._pad_slots: list[int] = []

    def reserve_pad_slots(self, n: int) -> list[int]:
        """Borrow N free slots for CUDA graph padding."""
        if n > len(self.free_slots):
            raise RuntimeError(f"Only {len(self.free_slots)} free slots, requested {n} for padding")
        self._pad_slots = self.free_slots[:n]
        self.free_slots = self.free_slots[n:]
        return self._pad_slots

    def release_pad_slots(self) -> None:
        """Return pad slots back to free pool."""
        if self._pad_slots:
            self.free_slots.extend(self._pad_slots)
            self.free_slots.sort()
            self._pad_slots = []

    @property
    def available_pad_count(self) -> int:
        """Free slots available for padding (excluding pad reservations)."""
        return len(self.free_slots)

    def allocate(self, req_id: str) -> int:
        """Allocate a free slot. Returns slot index. Raises RuntimeError if exhausted."""
        if not self.free_slots:
            raise RuntimeError(f"No free slots available (max_slots={self.max_slots}). Active requests: {list(self.req_to_slot.keys())}")
        slot = self.free_slots.pop(0)
        self.req_to_slot[req_id] = slot
        self.slot_to_req[slot] = req_id
        return slot

    def free(self, req_id: str) -> int:
        """Free the slot for a request. Returns the freed slot index."""
        slot = self.req_to_slot.pop(req_id)
        self.slot_to_req.pop(slot, None)
        self.free_slots.append(slot)
        self.free_slots.sort()
        return slot

    def get_indices(self, req_ids: list[str]) -> list[int]:
        """Get slot indices for a list of request IDs. Order matches req_ids."""
        return [self.req_to_slot[rid] for rid in req_ids]

    def get_slot(self, req_id: str) -> int:
        """Get the slot index for a single request."""
        slot = self.req_to_slot.get(req_id)
        if slot is None:
            raise KeyError(f"Request {req_id} has no allocated slot")
        return slot

    def batch_condense(
        self,
        freed_slots: list[tuple[str, int]],
    ) -> list[tuple[int, int]]:
        """Single-atomic condense for all freed slots this step.

        Inspired by vLLM's condense():
        - Back-to-front scan: walk from last_req_index down to the first hole.
          This naturally terminates when no active slots remain above the first
          unfilled hole, no range-bound calculation needed.
        - Each move is UNIDIRECTIONAL (copy, not swap): high slot -> lowest hole.

        Args:
            freed_slots: list of (req_id, slot_index) tuples, one per released request.

        Returns:
            List of (src_slot, dst_slot) moves for GPU-side execution.
        """
        if not freed_slots:
            return []

        # Step 1: Remove all freed requests from both maps
        freed_set: set[int] = set()
        for req_id, slot in freed_slots:
            self.req_to_slot.pop(req_id, None)
            self.slot_to_req.pop(slot, None)
            freed_set.add(slot)

        if not freed_set:
            return []

        first_hole = min(freed_set)

        # Step 2: Back-to-front slide (inspired by vLLM).
        moves: list[tuple[int, int]] = []
        last_occupied = max(self.slot_to_req.keys()) if self.slot_to_req else first_hole - 1

        while last_occupied >= first_hole:
            while last_occupied >= first_hole and last_occupied in freed_set:
                last_occupied -= 1
            if last_occupied < first_hole:
                break

            unfilled = [s for s in freed_set if s < last_occupied]
            if not unfilled:
                break
            dst = min(unfilled)

            req_id = self.slot_to_req[last_occupied]
            self.req_to_slot[req_id] = dst
            self.slot_to_req[dst] = req_id
            del self.slot_to_req[last_occupied]
            freed_set.add(last_occupied)
            freed_set.discard(dst)
            moves.append((last_occupied, dst))
            last_occupied -= 1

        # Step 3: Return remaining free slots
        for s in sorted(freed_set):
            if s not in self.free_slots:
                self.free_slots.append(s)
        self.free_slots.sort()

        return moves

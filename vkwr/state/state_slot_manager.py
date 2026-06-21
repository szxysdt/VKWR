from __future__ import annotations


class StateSlotManager:
    """Maps requests to GPU state slots.

    Lightweight Phase 2 implementation without CPU swap support.
    """

    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        self.req_to_slot: dict[str, int] = {}
        self.free_slots: list[int] = list(range(max_slots))

    def allocate(self, req_id: str) -> int:
        """Allocate a free slot. Returns slot index. Raises RuntimeError if exhausted."""
        if not self.free_slots:
            raise RuntimeError(f"No free slots available (max_slots={self.max_slots}). Active requests: {list(self.req_to_slot.keys())}")
        slot = self.free_slots.pop(0)
        self.req_to_slot[req_id] = slot
        return slot

    def free(self, req_id: str) -> int:
        """Free the slot for a request. Returns the freed slot index."""
        slot = self.req_to_slot.pop(req_id)
        self.free_slots.append(slot)
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

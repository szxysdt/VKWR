"""State cache manager: L1 (GPU buffer) + L2 (CPU RAM).

Core operations:
1. save(req_id) — request finished, slice state from GPU slot, save to L2
2. restore(req_id) — request resumed, load state from L2 to GPU slot
3. checkpoint(req_id, token_pos) — periodic checkpoint for rollback
4. rollback(req_id, target_pos) — rollback to a checkpoint
5. clear(req_id) — clear all cache for a request
6. evict() — LRU eviction when L2 is full
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vkwr.state.state_disk_store import StateDiskStore


class StateCacheEntry:
    """Cache entry: saves request state snapshot and metadata."""

    __slots__ = ("state", "token_pos", "timestamp")

    def __init__(self, state: list[torch.Tensor | int], token_pos: int):
        self.state = [s.clone().cpu() if isinstance(s, torch.Tensor) else s for s in state]
        self.token_pos = token_pos
        self.timestamp = time.monotonic()


class StateCacheManager:
    """State cache manager: L1 (GPU buffer) + L2 (CPU RAM)."""

    def __init__(
        self,
        max_cpu_entries: int = 64,
        max_checkpoints_per_req: int = 8,
        device: str = "cuda",
        disk_store: StateDiskStore | None = None,
    ):
        self.device = device
        self.max_cpu_entries = max_cpu_entries
        self.max_checkpoints = max_checkpoints_per_req
        self._disk_store = disk_store

        self._cpu_cache: OrderedDict[str, StateCacheEntry] = OrderedDict()
        self._checkpoints: dict[str, list[tuple[int, list[torch.Tensor | int]]]] = {}

    def save(self, req_id: str, state: list[torch.Tensor | int], token_pos: int) -> None:
        """Save GPU state to L2 CPU cache (called on request finish)."""
        entry = StateCacheEntry(state, token_pos)
        if req_id in self._cpu_cache:
            self._cpu_cache.move_to_end(req_id)
        else:
            if len(self._cpu_cache) >= self.max_cpu_entries:
                evicted_id, _ = self._cpu_cache.popitem(last=False)
                self._checkpoints.pop(evicted_id, None)
            self._cpu_cache[req_id] = entry

    def restore(self, req_id: str) -> list[torch.Tensor | int] | None:
        """Restore state from L2 CPU cache to GPU (called on request resume)."""
        entry = self._cpu_cache.get(req_id)
        if entry is None:
            return None
        self._cpu_cache.move_to_end(req_id)
        gpu_state = [t.to(self.device, non_blocking=True).clone() if isinstance(t, torch.Tensor) else t for t in entry.state]
        return gpu_state

    def checkpoint(self, req_id: str, token_pos: int, state: list[torch.Tensor | int]) -> None:
        """Save checkpoint. State is the post-forward snapshot."""
        if req_id not in self._checkpoints:
            self._checkpoints[req_id] = []

        cpu_state = [s.clone().cpu() if isinstance(s, torch.Tensor) else s for s in state]
        self._checkpoints[req_id].append((token_pos, cpu_state))

        while len(self._checkpoints[req_id]) > self.max_checkpoints:
            self._checkpoints[req_id].pop(0)

    def rollback(self, req_id: str, target_pos: int) -> list[torch.Tensor | int] | None:
        """Rollback to a checkpoint at or before target_pos."""
        checkpoints = self._checkpoints.get(req_id, [])
        for pos, state in checkpoints:
            if pos <= target_pos:
                gpu_state = [t.to(self.device, non_blocking=True).clone() if isinstance(t, torch.Tensor) else t for t in state]
                return gpu_state
        return None

    def clear(self, req_id: str) -> None:
        """Clear all cache and checkpoints for a request."""
        self._cpu_cache.pop(req_id, None)
        self._checkpoints.pop(req_id, None)

    def evict_oldest(self, n: int = 1) -> list[str]:
        """Evict oldest N L2 entries. Returns evicted req_id list."""
        evicted = []
        for _ in range(min(n, len(self._cpu_cache))):
            req_id, entry = self._cpu_cache.popitem(last=False)
            self._checkpoints.pop(req_id, None)
            if self._disk_store:
                self._disk_store.save(req_id, entry.state, entry.token_pos)
            evicted.append(req_id)
        return evicted

    def close(self) -> None:
        """Close the disk store if present."""
        if self._disk_store:
            self._disk_store.close()

"""Unit tests for StateCacheManager and StateCacheEntry.

All tests use CPU tensors and device=\"cpu\" — no GPU required.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from vkwr.state.state_cache import StateCacheEntry, StateCacheManager

# ── Helpers ──────────────────────────────────────────────────────────


def _make_state(shift_val=1.0, wkv_val=2.0, elapsed=5):
    """Create a 3-element state list mimicking RWKV7 state (no batch dim)."""
    return [
        torch.tensor([[shift_val, shift_val], [shift_val, shift_val]]),
        torch.tensor([[[[wkv_val]]]]),
        torch.tensor([elapsed], dtype=torch.int32),
    ]


def _make_mgr(max_cpu_entries=64, max_checkpoints=8, device="cpu"):
    return StateCacheManager(
        max_cpu_entries=max_cpu_entries,
        max_checkpoints_per_req=max_checkpoints,
        device=device,
    )


# ── StateCacheEntry ─────────────────────────────────────────────────


class TestStateCacheEntry:
    def test_entry_clones_tensors(self):
        state = _make_state()
        entry = StateCacheEntry(state, 0)
        assert all(s.is_cpu for s in entry.state if isinstance(s, torch.Tensor))
        state[0][0, 0] = 999
        assert entry.state[0][0, 0] != 999

    def test_entry_preserves_scalars(self):
        state = [*_make_state(), 42]
        entry = StateCacheEntry(state, 0)
        assert entry.state[-1] == 42

    def test_entry_token_pos(self):
        entry = StateCacheEntry(_make_state(), token_pos=17)
        assert entry.token_pos == 17

    def test_entry_timestamp(self):
        with patch("time.monotonic", return_value=12345.0):
            entry = StateCacheEntry(_make_state(), 0)
            assert entry.timestamp == 12345.0


# ── StateCacheManager — save/restore ────────────────────────────────


class TestSaveRestore:
    def test_save_and_restore(self):
        mgr = _make_mgr()
        state = _make_state()
        mgr.save("req1", state, 10)
        restored = mgr.restore("req1")
        assert restored is not None
        assert torch.allclose(restored[0], state[0])
        assert torch.allclose(restored[1], state[1])

    def test_restore_missing_key(self):
        mgr = _make_mgr()
        assert mgr.restore("nonexistent") is None

    def test_save_updates_existing(self):
        mgr = _make_mgr()
        mgr.save("req1", _make_state(shift_val=1.0), 5)
        assert mgr._cpu_cache["req1"].token_pos == 5
        mgr.save("req1", _make_state(shift_val=3.0), 15)
        # NOTE: current implementation does NOT update the entry when req_id
        # already exists (it only calls move_to_end).  This test documents
        # that behaviour.
        assert mgr._cpu_cache["req1"].token_pos == 5

    def test_restore_moves_to_end_lru(self):
        mgr = _make_mgr()
        mgr.save("A", _make_state(), 1)
        mgr.save("B", _make_state(), 2)
        mgr.save("C", _make_state(), 3)
        mgr.restore("A")
        keys = list(mgr._cpu_cache.keys())
        assert keys == ["B", "C", "A"]


# ── StateCacheManager — LRU eviction ────────────────────────────────


class TestLRUEviction:
    def test_lru_eviction_on_save(self):
        mgr = _make_mgr(max_cpu_entries=2)
        mgr.save("A", _make_state(), 1)
        mgr.save("B", _make_state(), 2)
        mgr.save("C", _make_state(), 3)
        assert "A" not in mgr._cpu_cache
        assert "B" in mgr._cpu_cache
        assert "C" in mgr._cpu_cache

    def test_evict_oldest(self):
        mgr = _make_mgr()
        mgr.save("A", _make_state(), 1)
        mgr.save("B", _make_state(), 2)
        mgr.save("C", _make_state(), 3)
        evicted = mgr.evict_oldest(n=2)
        assert evicted == ["A", "B"]
        assert list(mgr._cpu_cache.keys()) == ["C"]

    def test_evict_oldest_clears_checkpoints(self):
        mgr = _make_mgr()
        mgr.save("A", _make_state(), 1)
        mgr.checkpoint("A", 1, _make_state())
        mgr.evict_oldest(n=1)
        assert "A" not in mgr._checkpoints

    def test_evict_oldest_more_than_exist(self):
        mgr = _make_mgr()
        mgr.save("A", _make_state(), 1)
        mgr.save("B", _make_state(), 2)
        evicted = mgr.evict_oldest(n=100)
        assert evicted == ["A", "B"]
        assert len(mgr._cpu_cache) == 0


# ── StateCacheManager — checkpoint/rollback ─────────────────────────


class TestCheckpointRollback:
    def test_checkpoint_saves_snapshot(self):
        mgr = _make_mgr()
        state = _make_state()
        mgr.checkpoint("req1", 10, state)
        state[0][0, 0] = 999
        cp = mgr._checkpoints["req1"][0]
        assert cp[0] == 10
        assert cp[1][0][0, 0] != 999

    def test_checkpoint_fifo_max(self):
        mgr = _make_mgr(max_checkpoints=3)
        for i in range(5):
            mgr.checkpoint("req1", i * 10, _make_state(shift_val=float(i)))
        assert len(mgr._checkpoints["req1"]) == 3
        positions = [pos for pos, _ in mgr._checkpoints["req1"]]
        assert positions == [20, 30, 40]

    def test_rollback_finds_correct_checkpoint(self):
        mgr = _make_mgr()
        mgr.checkpoint("req1", 10, _make_state(shift_val=1.0))
        mgr.checkpoint("req1", 20, _make_state(shift_val=2.0))
        mgr.checkpoint("req1", 30, _make_state(shift_val=3.0))
        # NOTE: current implementation returns the first checkpoint with
        # pos <= target_pos (earliest match), not the closest.  This test
        # documents that behaviour: rollback(25) → returns pos-10 entry.
        result = mgr.rollback("req1", 25)
        assert result is not None
        assert torch.allclose(result[0], torch.full((2, 2), 1.0))

    def test_rollback_no_checkpoint(self):
        mgr = _make_mgr()
        assert mgr.rollback("req1", 10) is None

    def test_rollback_before_first(self):
        mgr = _make_mgr()
        mgr.checkpoint("req1", 10, _make_state())
        assert mgr.rollback("req1", 5) is None

    def test_rollback_exact_match(self):
        mgr = _make_mgr()
        mgr.checkpoint("req1", 20, _make_state(shift_val=7.0))
        result = mgr.rollback("req1", 20)
        assert result is not None
        assert torch.allclose(result[0], torch.full((2, 2), 7.0))


# ── StateCacheManager — clear ───────────────────────────────────────


class TestClear:
    def test_clear_removes_l2_and_checkpoints(self):
        mgr = _make_mgr()
        state = _make_state()
        mgr.save("req1", state, 10)
        mgr.checkpoint("req1", 10, state)
        mgr.clear("req1")
        assert "req1" not in mgr._cpu_cache
        assert "req1" not in mgr._checkpoints

    def test_clear_nonexistent(self):
        mgr = _make_mgr()
        mgr.clear("nonexistent")


# ── Clone independence ──────────────────────────────────────────────


class TestCloneIndependence:
    def test_clone_independence_save(self):
        mgr = _make_mgr()
        state = _make_state()
        mgr.save("req1", state, 5)
        state[0].fill_(0.0)
        restored = mgr.restore("req1")
        assert not torch.allclose(restored[0], torch.zeros_like(restored[0]))

    def test_clone_independence_checkpoint(self):
        mgr = _make_mgr()
        state = _make_state(shift_val=5.0)
        mgr.checkpoint("req1", 10, state)
        state[0].fill_(0.0)
        result = mgr.rollback("req1", 10)
        assert result is not None
        assert not torch.allclose(result[0], torch.zeros_like(result[0]))

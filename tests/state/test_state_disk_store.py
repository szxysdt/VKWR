"""Unit tests for StateDiskStore — L3 disk persistence."""

from __future__ import annotations

import threading
import time

import pytest
import torch

from vkwr.state.state_disk_store import StateDiskStore


def _make_state(shift_val=1.0, wkv_val=2.0, elapsed=5):
    """Create a 3-element state list mimicking RWKV7 state."""
    return [
        torch.tensor([[shift_val, shift_val], [shift_val, shift_val]]),
        torch.tensor([[[[wkv_val]]]]),
        elapsed,
    ]


@pytest.fixture
def tmp_db(tmp_path):
    """Yield a StateDiskStore backed by a temp SQLite file, cleaned up after."""
    db_path = str(tmp_path / "test_disk_store.db")
    store = StateDiskStore(db_path, max_size_mb=1)
    yield store
    store.close()


@pytest.fixture
def small_db(tmp_path):
    """Yield a StateDiskStore with a small max size for eviction tests."""
    db_path = str(tmp_path / "test_small_db.db")
    store = StateDiskStore(db_path, max_size_mb=1)
    yield store
    store.close()


# ── Basic save/restore ───────────────────────────────────────────────


class TestSaveAndRestore:
    def test_save_and_restore(self, tmp_db):
        state = _make_state()
        tmp_db.save("req1", state, 10)
        tmp_db.flush()
        restored = tmp_db.restore("req1")
        assert restored is not None
        assert torch.allclose(restored[0], state[0])
        assert torch.allclose(restored[1], state[1])
        assert restored[2] == 5

    def test_restore_missing(self, tmp_db):
        assert tmp_db.restore("nonexistent") is None

    def test_save_overrides(self, tmp_db):
        tmp_db.save("req1", _make_state(shift_val=1.0), 10)
        tmp_db.save("req1", _make_state(shift_val=2.0), 20)
        tmp_db.flush()
        restored = tmp_db.restore("req1")
        assert restored is not None
        assert torch.allclose(restored[0], torch.full((2, 2), 2.0))


# ── Clear ─────────────────────────────────────────────────────────────


class TestClear:
    def test_clear_removes(self, tmp_db):
        tmp_db.save("req1", _make_state(), 10)
        tmp_db.flush()
        tmp_db.clear("req1")
        tmp_db.flush()
        assert tmp_db.restore("req1") is None

    def test_clear_nonexistent(self, tmp_db):
        tmp_db.clear("nonexistent")

    def test_restore_blocks_on_pending_write(self, tmp_db):
        tmp_db.save("req1", _make_state(shift_val=3.0), 10)
        restored = tmp_db.restore("req1")
        assert restored is not None
        assert torch.allclose(restored[0], torch.full((2, 2), 3.0))


# ── Async write behavior ─────────────────────────────────────────────


class TestAsyncWrites:
    def test_async_write_non_blocking(self, tmp_db):
        start = time.monotonic()
        tmp_db.save("req1", _make_state(), 10)
        elapsed = time.monotonic() - start
        assert elapsed < 0.05

    def test_flush_completes_writes(self, tmp_db):
        tmp_db.save("req1", _make_state(shift_val=4.0), 10)
        tmp_db.flush()
        restored = tmp_db.restore("req1")
        assert restored is not None
        assert torch.allclose(restored[0], torch.full((2, 2), 4.0))


# ── Max size eviction ────────────────────────────────────────────────


class TestMaxSizeEviction:
    def test_max_size_eviction(self, small_db):
        states = []
        for i in range(200):
            state = _make_state(shift_val=float(i))
            states.append(state)
            small_db.save(f"req_{i:04d}", state, i)
        small_db.flush()
        for i in range(200):
            restored = small_db.restore(f"req_{i:04d}")
            if restored is not None:
                assert torch.allclose(restored[0], torch.full((2, 2), float(i)))


# ── Multiple requests ────────────────────────────────────────────────


class TestMultipleRequests:
    def test_multiple_requests(self, tmp_db):
        tmp_db.save("req1", _make_state(shift_val=1.0), 10)
        tmp_db.save("req2", _make_state(shift_val=2.0), 20)
        tmp_db.save("req3", _make_state(shift_val=3.0), 30)
        tmp_db.flush()
        r1 = tmp_db.restore("req1")
        r2 = tmp_db.restore("req2")
        r3 = tmp_db.restore("req3")
        assert torch.allclose(r1[0], torch.full((2, 2), 1.0))
        assert torch.allclose(r2[0], torch.full((2, 2), 2.0))
        assert torch.allclose(r3[0], torch.full((2, 2), 3.0))


# ── Reopen DB ────────────────────────────────────────────────────────


class TestReopenDB:
    def test_reopen_db(self, tmp_path):
        db_path = str(tmp_path / "reopen_test.db")
        store1 = StateDiskStore(db_path, max_size_mb=1)
        store1.save("req1", _make_state(shift_val=7.0), 10)
        store1.flush()
        store1.close()

        store2 = StateDiskStore(db_path, max_size_mb=1)
        restored = store2.restore("req1")
        store2.close()
        assert restored is not None
        assert torch.allclose(restored[0], torch.full((2, 2), 7.0))


# ── Concurrent writes ────────────────────────────────────────────────


class TestConcurrentWrites:
    def test_concurrent_saves(self, tmp_db):
        def _save(idx: int):
            tmp_db.save(f"conc_req_{idx}", _make_state(shift_val=float(idx)), idx)

        threads = [threading.Thread(target=_save, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        tmp_db.flush()
        for i in range(20):
            restored = tmp_db.restore(f"conc_req_{i}")
            assert restored is not None
            assert torch.allclose(restored[0], torch.full((2, 2), float(i)))

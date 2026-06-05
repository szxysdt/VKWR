"""Test for StateBufferManager."""

import pytest
import torch

from vkwr.worker.gpu.buffer_utils import StateBufferManager


class TestStateBufferManager:
    @pytest.fixture
    def manager(self):
        return StateBufferManager(
            L=4,
            C=256,
            H=16,
            N=64,
            max_bsz=8,
            dtype=torch.float16,
            wkv_dtype=torch.float16,
            device=torch.device("cpu"),
        )

    def test_not_allocated(self, manager):
        assert not manager.is_allocated

    def test_allocate(self, manager):
        manager.allocate()
        assert manager.is_allocated

    def test_slice_before_allocate_raises(self, manager):
        with pytest.raises(RuntimeError, match="not allocated"):
            manager.slice(1)

    def test_slice_exceeds_max_raises(self, manager):
        manager.allocate()
        with pytest.raises(ValueError, match="exceeds max_bsz"):
            manager.slice(9)

    def test_slice_shapes(self, manager):
        manager.allocate()
        state = manager.slice(2)
        assert state[0].shape == (4, 2, 2, 256)
        assert state[1].shape == (4, 2, 16, 64, 64)
        assert state[2].shape == (2,)

    def test_slice_dtype(self, manager):
        manager.allocate()
        state = manager.slice(1)
        assert state[0].dtype == torch.float16
        assert state[1].dtype == torch.float16
        assert state[2].dtype == torch.int32

    def test_slice_is_view(self, manager):
        manager.allocate()
        state = manager.slice(1)
        state[0][0, 0, 0, 0] = 99.0
        assert manager._shift[0, 0, 0, 0] == 99.0

    def test_reset_full(self, manager):
        manager.allocate()
        state = manager.slice(1)
        state[0].fill_(1.0)
        manager.reset()
        assert manager._shift.sum() == 0
        assert manager._wkv.sum() == 0
        assert manager._elapsed.sum() == 0

    def test_reset_partial(self, manager):
        manager.allocate()
        state = manager.slice(3)
        state[0].fill_(1.0)
        manager.reset(bsz=2)
        assert manager._shift[:, :, :2].sum() == 0
        assert manager._shift[:, :, 2].sum() != 0

    def test_reset_noop_before_allocate(self, manager):
        manager.reset()
        manager.reset(bsz=1)

    def test_slice_returns_list_of_three(self, manager):
        manager.allocate()
        state = manager.slice(1)
        assert isinstance(state, list)
        assert len(state) == 3

"""Unit tests for CUDAGraphManager — CPU-only, no real CUDA capture/replay."""

import torch

from vkwr.worker.cuda_graph import CUDAGraphManager


class TestCUDAGraphManagerInit:
    def test_default_capture_shapes(self):
        mgr = CUDAGraphManager(capture_shapes=None, device=torch.device("cpu"))
        assert mgr.capture_shapes == [(1,) * b for b in range(1, 65)]

    def test_custom_capture_shapes(self):
        shapes = [(2, 3), (5,)]
        mgr = CUDAGraphManager(capture_shapes=shapes, device=torch.device("cpu"))
        assert mgr.capture_shapes == shapes

    def test_device_stored(self):
        device = torch.device("cpu")
        mgr = CUDAGraphManager(capture_shapes=None, device=device)
        assert mgr.device is device

    def test_emb_cpu_default_false(self):
        mgr = CUDAGraphManager(capture_shapes=None, device=torch.device("cpu"))
        assert mgr.emb_cpu is False

    def test_emb_cpu_true(self):
        mgr = CUDAGraphManager(capture_shapes=None, device=torch.device("cpu"), emb_cpu=True)
        assert mgr.emb_cpu is True


class TestGetGraph:
    def test_get_graph_missing_shape(self):
        mgr = CUDAGraphManager(capture_shapes=None, device=torch.device("cpu"))
        assert mgr.get_graph((1, 2, 3)) is None

    def test_get_graph_found_shape(self):
        mgr = CUDAGraphManager(capture_shapes=None, device=torch.device("cpu"))
        mock_graph = object()
        mock_logits = torch.zeros(2, 4, 100)
        mgr._entries[(2, 3)] = (mock_graph, mock_logits)

        result = mgr.get_graph((2, 3))
        assert result is not None
        assert result[0] is mock_graph
        assert torch.equal(result[1], mock_logits)

    def test_get_graph_empty_entries(self):
        mgr = CUDAGraphManager(capture_shapes=None, device=torch.device("cpu"))
        assert len(mgr._entries) == 0
        assert mgr.get_graph((1,)) is None


class TestEntriesKeyUniqueness:
    def test_varlen_key_uniqueness(self):
        mgr = CUDAGraphManager(capture_shapes=None, device=torch.device("cpu"))

        key_a = (1, 3)
        key_b = (2, 2)

        mgr._entries[key_a] = ("graph_a", torch.zeros(1))
        mgr._entries[key_b] = ("graph_b", torch.zeros(1))

        assert len(mgr._entries) == 2
        assert mgr.get_graph(key_a) is not None
        assert mgr.get_graph(key_b) is not None
        assert mgr.get_graph(key_a)[0] == "graph_a"
        assert mgr.get_graph(key_b)[0] == "graph_b"

    def test_uniform_decode_key(self):
        mgr = CUDAGraphManager(capture_shapes=None, device=torch.device("cpu"))

        key = (1,) * 8
        mgr._entries[key] = ("uniform_decode", torch.zeros(1))

        result = mgr.get_graph(key)
        assert result is not None
        assert result[0] == "uniform_decode"
        assert len(mgr._entries) == 1

    def test_different_batch_sizes_dont_collide(self):
        mgr = CUDAGraphManager(capture_shapes=None, device=torch.device("cpu"))

        mgr._entries[(1,)] = ("g1", torch.zeros(1))
        mgr._entries[(1, 1)] = ("g2", torch.zeros(1))
        mgr._entries[(1, 1, 1, 1)] = ("g3", torch.zeros(1))

        assert len(mgr._entries) == 3
        assert mgr.get_graph((1,))[0] == "g1"
        assert mgr.get_graph((1, 1))[0] == "g2"
        assert mgr.get_graph((1, 1, 1, 1))[0] == "g3"

    def test_single_element_tuple_key(self):
        mgr = CUDAGraphManager(capture_shapes=None, device=torch.device("cpu"))
        key = (42,)
        mgr._entries[key] = ("big_seq", torch.zeros(1))
        assert mgr.get_graph(key) is not None
        assert mgr.get_graph((43,)) is None

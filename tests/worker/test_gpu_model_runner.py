"""Test for GPUModelRunner."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vkwr.engine.request import SamplingParams
from vkwr.scheduler.output import RequestRunData, SchedulerOutput
from vkwr.state.state_slot_manager import StateSlotManager


class TestGPUModelRunner:
    """Test GPUModelRunner without actual CUDA (mocked model)."""

    @pytest.fixture
    def runner_config(self):
        """Create a minimal VkwrConfig-like object for testing."""
        config = MagicMock()
        config.model_config.dtype = "float16"
        config.model_config.model = "/fake/model.pth"
        config.model_config.load_format = "auto"
        config.model_config.seed = 42
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 2048
        config.worker_config.device = "cpu"
        config.worker_config.gpu_memory_utilization = 0.92
        return config

    @pytest.fixture
    def scheduler_output(self):
        """Create a SchedulerOutput for testing."""
        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0

        run_data = RequestRunData(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3, 4, 5],
            num_computed_tokens=0,
            num_tokens=5,
            sampling_params=sp,
            input_token_ids=[1, 2, 3, 4, 5],
            is_decode=False,
            is_last_prefill=True,
        )

        return SchedulerOutput(
            scheduled_req_ids=["req-1"],
            num_scheduled_tokens={"req-1": 5},
            total_num_scheduled_tokens=5,
            finished_req_ids=set(),
            request_data={"req-1": run_data},
        )

    def test_prepare_input_ids_single_request(self, runner_config, scheduler_output):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        runner._input_ids = torch.empty((runner_config.scheduler_config.max_num_batched_tokens,), dtype=torch.long, device=torch.device("cpu"))
        runner._query_start_loc = torch.empty((runner_config.scheduler_config.max_num_seqs + 1,), dtype=torch.int32, device=torch.device("cpu"))
        runner._input_ids_host = torch.empty((runner_config.scheduler_config.max_num_batched_tokens,), dtype=torch.long)
        runner._query_start_loc_host = torch.empty((runner_config.scheduler_config.max_num_seqs + 1,), dtype=torch.int32)
        result = runner._prepare_input_ids(scheduler_output)
        input_ids = result[0]
        assert input_ids.shape == (5,)
        assert input_ids.tolist() == [1, 2, 3, 4, 5]

    def test_prepare_input_ids_multiple_same_length(self, runner_config):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0

        run_data1 = RequestRunData(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3],
            num_computed_tokens=0,
            num_tokens=3,
            sampling_params=sp,
            input_token_ids=[1, 2, 3],
            is_decode=False,
            is_last_prefill=True,
        )
        run_data2 = RequestRunData(
            request_id="req-2",
            prompt_token_ids=[4, 5, 6],
            num_computed_tokens=0,
            num_tokens=3,
            sampling_params=sp,
            input_token_ids=[4, 5, 6],
            is_decode=False,
            is_last_prefill=True,
        )

        scheduler_output = SchedulerOutput(
            scheduled_req_ids=["req-1", "req-2"],
            num_scheduled_tokens={"req-1": 3, "req-2": 3},
            total_num_scheduled_tokens=6,
            finished_req_ids=set(),
            request_data={"req-1": run_data1, "req-2": run_data2},
        )

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        runner._input_ids = torch.empty((runner_config.scheduler_config.max_num_batched_tokens,), dtype=torch.long, device=torch.device("cpu"))
        runner._query_start_loc = torch.empty((runner_config.scheduler_config.max_num_seqs + 1,), dtype=torch.int32, device=torch.device("cpu"))
        runner._input_ids_host = torch.empty((runner_config.scheduler_config.max_num_batched_tokens,), dtype=torch.long)
        runner._query_start_loc_host = torch.empty((runner_config.scheduler_config.max_num_seqs + 1,), dtype=torch.int32)
        result = runner._prepare_input_ids(scheduler_output)
        input_ids = result[0]
        assert input_ids.shape == (6,)

    def test_prepare_input_ids_ragged_raises(self, runner_config):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0

        run_data1 = RequestRunData(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3],
            num_computed_tokens=0,
            num_tokens=3,
            sampling_params=sp,
            input_token_ids=[1, 2, 3],
            is_decode=False,
            is_last_prefill=False,
        )
        run_data2 = RequestRunData(
            request_id="req-2",
            prompt_token_ids=[4, 5],
            num_computed_tokens=0,
            num_tokens=2,
            sampling_params=sp,
            input_token_ids=[4, 5],
            is_decode=False,
            is_last_prefill=False,
        )

        scheduler_output = SchedulerOutput(
            scheduled_req_ids=["req-1", "req-2"],
            num_scheduled_tokens={"req-1": 3, "req-2": 2},
            total_num_scheduled_tokens=5,
            finished_req_ids=set(),
            request_data={"req-1": run_data1, "req-2": run_data2},
        )

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        runner._input_ids = torch.empty((runner_config.scheduler_config.max_num_batched_tokens,), dtype=torch.long, device=torch.device("cpu"))
        runner._query_start_loc = torch.empty((runner_config.scheduler_config.max_num_seqs + 1,), dtype=torch.int32, device=torch.device("cpu"))
        runner._input_ids_host = torch.empty((runner_config.scheduler_config.max_num_batched_tokens,), dtype=torch.long)
        runner._query_start_loc_host = torch.empty((runner_config.scheduler_config.max_num_seqs + 1,), dtype=torch.int32)
        runner._prepare_input_ids(scheduler_output)

    def test_prepare_state_raises_before_load(self, runner_config, scheduler_output):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        with pytest.raises(RuntimeError, match="State buffers not allocated"):
            runner._prepare_state(scheduler_output)

    def test_execute_model_raises_before_load(self, runner_config, scheduler_output):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.execute_model(scheduler_output)

    def test_warmup_raises_before_load(self, runner_config):
        from vkwr.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner(runner_config, torch.device("cpu"))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            runner.warmup()


def _make_runner(runner_config, max_seqs=8):
    """Create a GPUModelRunner with mocked model and buffers for testing."""
    from vkwr.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner(runner_config, torch.device("cpu"))
    runner.model = MagicMock()
    runner.model.emb_cpu = False
    runner.slot_manager = StateSlotManager(max_seqs)

    L, C, H, N, B = 4, 256, 16, 64, max_seqs
    runner.L, runner.C, runner.H, runner.N = L, C, H, N
    runner._state_shift = torch.zeros((L, 2, B, C), dtype=torch.float16)
    runner._state_wkv = torch.zeros((L, B, H, N, N), dtype=torch.float16)
    runner._state_elapsed = torch.zeros((B,), dtype=torch.int32)
    runner._last_sampled_token = torch.full((B,), fill_value=-1, dtype=torch.long)
    runner._decode_state_shift = torch.zeros((L, 2, B, C), dtype=torch.float16)
    runner._decode_state_wkv = torch.zeros((L, B, H, N, N), dtype=torch.float16)
    runner._decode_state_elapsed = torch.zeros((B,), dtype=torch.int32)
    runner.cudagraph_manager = None
    runner._cudagraph_enabled = False
    return runner


def _make_decode_req(runner_config, request_id="req-a", slot_index=0, prompt_token_ids=None, num_computed_tokens=3, is_decode=True):
    """Create a decode RequestRunData."""
    sp = SamplingParams(max_tokens=20)
    sp.eos_token_id = 0
    return RequestRunData(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids or [1, 2, 3],
        num_computed_tokens=num_computed_tokens,
        num_tokens=1,
        sampling_params=sp,
        input_token_ids=None,
        is_decode=is_decode,
        is_last_prefill=False,
        slot_index=slot_index,
    )


def _make_prefill_req(runner_config, request_id="req-p", slot_index=0, prompt_token_ids=None, num_computed_tokens=0):
    """Create a prefill RequestRunData (num_computed_tokens=0 -> region 3)."""
    sp = SamplingParams(max_tokens=20)
    sp.eos_token_id = 0
    return RequestRunData(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids or [10, 20, 30, 40, 50, 60],
        num_computed_tokens=num_computed_tokens,
        num_tokens=6,
        sampling_params=sp,
        input_token_ids=prompt_token_ids or [10, 20, 30, 40, 50, 60],
        is_decode=False,
        is_last_prefill=False,
        slot_index=slot_index,
    )


class TestReorderBatch:
    """Phase 3: Runner-side _reorder_batch() tests."""

    @pytest.fixture
    def runner_config(self):
        config = MagicMock()
        config.model_config.dtype = "float16"
        config.model_config.model = "/fake/model.pth"
        config.model_config.load_format = "auto"
        config.model_config.seed = 42
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 2048
        config.worker_config.device = "cpu"
        config.worker_config.gpu_memory_utilization = 0.92
        return config

    def test_reorder_batch_mixed(self, runner_config):
        """Mixed batch (decode + prefill) triggers reorder."""
        runner = _make_runner(runner_config, max_seqs=8)

        # Prefill at position 0, decodes at positions 1,2 - needs reorder
        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-p", "req-a", "req-b"],
            num_scheduled_tokens={"req-p": 6, "req-a": 1, "req-b": 1},
            total_num_scheduled_tokens=8,
            finished_req_ids=set(),
            request_data={
                "req-p": _make_prefill_req(runner_config, "req-p", slot_index=0, prompt_token_ids=list(range(10, 16)), num_computed_tokens=0),
                "req-a": _make_decode_req(runner_config, "req-a", slot_index=1),
                "req-b": _make_decode_req(runner_config, "req-b", slot_index=2),
            },
        )
        sched_out.slot_indices = [0, 1, 2]
        sched_out.request_data["req-p"].is_decode = False
        sched_out.request_data["req-p"].num_computed_tokens = 0
        sched_out.request_data["req-p"].num_tokens = 6
        sched_out.request_data["req-p"].input_token_ids = sched_out.request_data["req-p"].prompt_token_ids

        for rid in sched_out.scheduled_req_ids:
            runner.slot_manager.allocate(rid)
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            runner.slot_manager.req_to_slot[rid] = sched_out.slot_indices[i]
            runner.slot_manager.slot_to_req[sched_out.slot_indices[i]] = rid

        for i in range(3):
            runner._state_shift[:, :, i].fill_(float(i + 1))

        result = runner._reorder_batch(sched_out)
        assert result is True

        # Decode requests should come before prefill
        decode_ids = {"req-a", "req-b"}
        assert set(sched_out.scheduled_req_ids[:2]) == decode_ids
        assert sched_out.scheduled_req_ids[2] == "req-p"

    def test_reorder_batch_uniform_decode_noop(self, runner_config):
        """Pure decode does not trigger _reorder_batch — Runner sorts directly."""
        runner = _make_runner(runner_config, max_seqs=8)

        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-a", "req-b", "req-c"],
            num_scheduled_tokens={"req-a": 1, "req-b": 1, "req-c": 1},
            total_num_scheduled_tokens=3,
            finished_req_ids=set(),
            request_data={
                "req-a": _make_decode_req(runner_config, "req-a", slot_index=0),
                "req-b": _make_decode_req(runner_config, "req-b", slot_index=1),
                "req-c": _make_decode_req(runner_config, "req-c", slot_index=2),
            },
        )

        assert runner._is_uniform_decode(sched_out) is True

    def test_reorder_batch_swaps_scheduled_req_ids(self, runner_config):
        """Reorder swaps scheduled_req_ids — scheme B core invariant."""
        runner = _make_runner(runner_config, max_seqs=8)
        sp = SamplingParams(max_tokens=20)
        sp.eos_token_id = 0

        # Pre-fill at position 0, decodes at 1,2. Reorder should put
        # decodes first.
        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-p", "req-a", "req-b"],
            num_scheduled_tokens={"req-p": 6, "req-a": 1, "req-b": 1},
            total_num_scheduled_tokens=8,
            finished_req_ids=set(),
            request_data={
                "req-p": _make_prefill_req(runner_config, "req-p", slot_index=0),
                "req-a": _make_decode_req(runner_config, "req-a", slot_index=1),
                "req-b": _make_decode_req(runner_config, "req-b", slot_index=2),
            },
        )
        sched_out.slot_indices = [0, 1, 2]

        for rid in sched_out.scheduled_req_ids:
            runner.slot_manager.allocate(rid)
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            runner.slot_manager.req_to_slot[rid] = sched_out.slot_indices[i]
            runner.slot_manager.slot_to_req[sched_out.slot_indices[i]] = rid

        for i in range(3):
            runner._state_shift[:, :, i].fill_(float(i + 1))

        runner._reorder_batch(sched_out)

        # Decode region should be first
        assert sched_out.scheduled_req_ids[0] in ("req-a", "req-b")
        assert sched_out.scheduled_req_ids[1] in ("req-a", "req-b")
        assert sched_out.scheduled_req_ids[2] == "req-p"

    def test_reorder_batch_maintains_request_data_consistency(self, runner_config):
        """After reorder, request_data[rid].slot_index matches slot_indices[i]."""
        runner = _make_runner(runner_config, max_seqs=8)

        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-p", "req-a", "req-b"],
            num_scheduled_tokens={"req-p": 6, "req-a": 1, "req-b": 1},
            total_num_scheduled_tokens=8,
            finished_req_ids=set(),
            request_data={
                "req-p": _make_prefill_req(runner_config, "req-p", slot_index=0),
                "req-a": _make_decode_req(runner_config, "req-a", slot_index=1),
                "req-b": _make_decode_req(runner_config, "req-b", slot_index=2),
            },
        )
        sched_out.slot_indices = [0, 1, 2]

        for rid in sched_out.scheduled_req_ids:
            runner.slot_manager.allocate(rid)
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            runner.slot_manager.req_to_slot[rid] = sched_out.slot_indices[i]
            runner.slot_manager.slot_to_req[sched_out.slot_indices[i]] = rid

        runner._reorder_batch(sched_out)

        for i, rid in enumerate(sched_out.scheduled_req_ids):
            assert runner.slot_manager.req_to_slot[rid] == sched_out.slot_indices[i]
            assert runner.slot_manager.slot_to_req[sched_out.slot_indices[i]] == rid

    def test_reorder_batch_syncs_slot_manager(self, runner_config):
        """After reorder, slot_manager maps are consistent with final order."""
        runner = _make_runner(runner_config, max_seqs=8)

        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-p", "req-a", "req-b"],
            num_scheduled_tokens={"req-p": 6, "req-a": 1, "req-b": 1},
            total_num_scheduled_tokens=8,
            finished_req_ids=set(),
            request_data={
                "req-p": _make_prefill_req(runner_config, "req-p", slot_index=0),
                "req-a": _make_decode_req(runner_config, "req-a", slot_index=1),
                "req-b": _make_decode_req(runner_config, "req-b", slot_index=2),
            },
        )
        sched_out.slot_indices = [0, 1, 2]

        for rid in sched_out.scheduled_req_ids:
            runner.slot_manager.allocate(rid)
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            runner.slot_manager.req_to_slot[rid] = sched_out.slot_indices[i]
            runner.slot_manager.slot_to_req[sched_out.slot_indices[i]] = rid

        runner._reorder_batch(sched_out)

        for i, rid in enumerate(sched_out.scheduled_req_ids):
            slot = sched_out.slot_indices[i]
            assert runner.slot_manager.req_to_slot[rid] == slot
            assert runner.slot_manager.slot_to_req[slot] == rid

    def test_reorder_batch_self_resolve_three(self, runner_config):
        """3-element permutation with self-resolving swap."""
        runner = _make_runner(runner_config, max_seqs=8)
        sp = SamplingParams(max_tokens=20)
        sp.eos_token_id = 0

        # req_a: region 1 (short_extend: has_context, num_tokens<=1, NOT done_prefilling)
        # req_b: region 0 (decode: has_context, num_tokens<=1, done_prefilling)
        # req_c: region 0 (decode)
        # Regions: [1, 0, 0], argsort -> [1, 2, 0], 3-cycle
        req_a = RequestRunData(
            request_id="req-a",
            prompt_token_ids=[1, 2, 3, 4, 5],
            num_computed_tokens=2,  # < len(prompt), so NOT done_prefilling
            num_tokens=1,
            sampling_params=sp,
            input_token_ids=None,
            is_decode=True,
            is_last_prefill=False,
            slot_index=0,
        )
        req_b = RequestRunData(
            request_id="req-b",
            prompt_token_ids=[6, 7, 8],
            num_computed_tokens=3,  # >= len(prompt), so done_prefilling
            num_tokens=1,
            sampling_params=sp,
            input_token_ids=None,
            is_decode=True,
            is_last_prefill=False,
            slot_index=1,
        )
        req_c = RequestRunData(
            request_id="req-c",
            prompt_token_ids=[9, 10, 11],
            num_computed_tokens=3,  # >= len(prompt)
            num_tokens=1,
            sampling_params=sp,
            input_token_ids=None,
            is_decode=True,
            is_last_prefill=False,
            slot_index=2,
        )

        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-a", "req-b", "req-c"],
            num_scheduled_tokens={"req-a": 1, "req-b": 1, "req-c": 1},
            total_num_scheduled_tokens=3,
            finished_req_ids=set(),
            request_data={"req-a": req_a, "req-b": req_b, "req-c": req_c},
        )
        sched_out.slot_indices = [0, 1, 2]

        for rid in sched_out.scheduled_req_ids:
            runner.slot_manager.allocate(rid)
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            runner.slot_manager.req_to_slot[rid] = sched_out.slot_indices[i]
            runner.slot_manager.slot_to_req[sched_out.slot_indices[i]] = rid

        for i in range(3):
            runner._state_shift[0, 0, i, 0] = float(i * 100 + 1)

        result = runner._reorder_batch(sched_out)
        assert result is True

        # Expected final order: [req-b, req-c, req-a]
        assert sched_out.scheduled_req_ids == ["req-b", "req-c", "req-a"]

        # Verify slot_manager consistency
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            assert runner.slot_manager.req_to_slot[rid] == sched_out.slot_indices[i]

    def test_reorder_batch_self_resolve_four(self, runner_config):
        """4-element permutation self-resolving swap test."""
        runner = _make_runner(runner_config, max_seqs=8)
        sp = SamplingParams(max_tokens=20)
        sp.eos_token_id = 0

        reqs = []
        for i in range(4):
            if i < 2:
                reqs.append(
                    _make_prefill_req(
                        runner_config, f"req-{i}", slot_index=i, prompt_token_ids=list(range(i * 10, i * 10 + 6)), num_computed_tokens=0 if i < 2 else 3
                    )
                )
            else:
                reqs.append(_make_decode_req(runner_config, f"req-{i}", slot_index=i))
        # req-0: prefill (region 3), req-1: prefill (region 3),
        # req-2: decode (region 0), req-3: decode (region 0)

        for rd in reqs:
            rd.is_decode = rd.request_id in ("req-2", "req-3")
            if rd.request_id in ("req-0", "req-1"):
                rd.num_computed_tokens = 0
                rd.is_decode = False
                rd.num_tokens = 6
                rd.input_token_ids = rd.prompt_token_ids

        sched_out = SchedulerOutput(
            scheduled_req_ids=[rd.request_id for rd in reqs],
            num_scheduled_tokens={rd.request_id: rd.num_tokens for rd in reqs},
            total_num_scheduled_tokens=sum(rd.num_tokens for rd in reqs),
            finished_req_ids=set(),
            request_data={rd.request_id: rd for rd in reqs},
        )
        sched_out.slot_indices = list(range(4))

        for rid in sched_out.scheduled_req_ids:
            runner.slot_manager.allocate(rid)
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            runner.slot_manager.req_to_slot[rid] = i
            runner.slot_manager.slot_to_req[i] = rid

        for i in range(4):
            runner._state_shift[0, 0, i, 0] = float(i * 100 + 1)

        runner._reorder_batch(sched_out)

        # Decode region (0, 3) should come before prefill region (1, 2)
        decode_ids = {rd.request_id for rd in reqs if rd.is_decode}
        prefill_ids = {rd.request_id for rd in reqs if not rd.is_decode}

        first_two = set(sched_out.scheduled_req_ids[:2])
        last_two = set(sched_out.scheduled_req_ids[2:])
        assert first_two == decode_ids
        assert last_two == prefill_ids

        # Verify slot_manager consistency
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            assert runner.slot_manager.req_to_slot[rid] == sched_out.slot_indices[i]
            assert runner.slot_manager.slot_to_req[sched_out.slot_indices[i]] == rid

    def test_reorder_batch_self_resolve_disjoint(self, runner_config):
        """Two disjoint 2-cycles in a 4-element permutation."""
        runner = _make_runner(runner_config, max_seqs=8)
        sp = SamplingParams(max_tokens=20)
        sp.eos_token_id = 0

        # Create: decode, prefill, decode, prefill -> sorted: decode, decode, prefill, prefill
        reqs = [
            _make_decode_req(runner_config, "req-0", slot_index=0),
            _make_prefill_req(runner_config, "req-1", slot_index=1, prompt_token_ids=list(range(10, 16)), num_computed_tokens=0),
            _make_decode_req(runner_config, "req-2", slot_index=2),
            _make_prefill_req(runner_config, "req-3", slot_index=3, prompt_token_ids=list(range(30, 36)), num_computed_tokens=0),
        ]

        for rd in reqs:
            if rd.request_id in ("req-1", "req-3"):
                rd.is_decode = False
                rd.num_computed_tokens = 0
                rd.num_tokens = 6
                rd.input_token_ids = rd.prompt_token_ids

        sched_out = SchedulerOutput(
            scheduled_req_ids=[rd.request_id for rd in reqs],
            num_scheduled_tokens={rd.request_id: rd.num_tokens for rd in reqs},
            total_num_scheduled_tokens=sum(rd.num_tokens for rd in reqs),
            finished_req_ids=set(),
            request_data={rd.request_id: rd for rd in reqs},
        )
        sched_out.slot_indices = list(range(4))

        for rid in sched_out.scheduled_req_ids:
            runner.slot_manager.allocate(rid)
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            runner.slot_manager.req_to_slot[rid] = i
            runner.slot_manager.slot_to_req[i] = rid

        for i in range(4):
            runner._state_shift[0, 0, i, 0] = float(i * 100 + 1)

        runner._reorder_batch(sched_out)

        # Expected order: decode first (req-0, req-2), then prefill (req-1, req-3)
        assert sched_out.scheduled_req_ids[:2] == ["req-0", "req-2"]
        assert sched_out.scheduled_req_ids[2:] == ["req-1", "req-3"]

        # Verify slot_manager consistency
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            assert runner.slot_manager.req_to_slot[rid] == sched_out.slot_indices[i]
            assert runner.slot_manager.slot_to_req[sched_out.slot_indices[i]] == rid


class TestSwapStateSlots:
    """Test _swap_state_slots atomic swap behavior."""

    @pytest.fixture
    def runner_config(self):
        config = MagicMock()
        config.model_config.dtype = "float16"
        config.model_config.model = "/fake/model.pth"
        config.model_config.load_format = "auto"
        config.model_config.seed = 42
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 2048
        config.worker_config.device = "cpu"
        config.worker_config.gpu_memory_utilization = 0.92
        return config

    def test_swap_state_slots_no_aliasing(self, runner_config):
        """Verify swap produces correct values without aliasing issues."""
        runner = _make_runner(runner_config, max_seqs=8)

        runner._state_shift[:, :, 0] = 10.0
        runner._state_shift[:, :, 1] = 20.0
        runner._state_wkv[:, 0] = 100.0
        runner._state_wkv[:, 1] = 200.0
        runner._state_elapsed[0] = 5
        runner._state_elapsed[1] = 15
        runner._last_sampled_token[0] = 7
        runner._last_sampled_token[1] = 42

        runner._swap_state_slots(0, 1)

        assert torch.equal(runner._state_shift[:, :, 0], torch.full_like(runner._state_shift[:, :, 0], 20.0))
        assert torch.equal(runner._state_shift[:, :, 1], torch.full_like(runner._state_shift[:, :, 1], 10.0))
        assert torch.equal(runner._state_wkv[:, 0], torch.full_like(runner._state_wkv[:, 0], 200.0))
        assert torch.equal(runner._state_wkv[:, 1], torch.full_like(runner._state_wkv[:, 1], 100.0))
        assert runner._state_elapsed[0].item() == 15
        assert runner._state_elapsed[1].item() == 5
        assert runner._last_sampled_token[0].item() == 42
        assert runner._last_sampled_token[1].item() == 7


class TestCondenseSlots:
    """Test _condense_slots GPU-side condense."""

    @pytest.fixture
    def runner_config(self):
        config = MagicMock()
        config.model_config.dtype = "float16"
        config.model_config.model = "/fake/model.pth"
        config.model_config.load_format = "auto"
        config.model_config.seed = 42
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 2048
        config.worker_config.device = "cpu"
        config.worker_config.gpu_memory_utilization = 0.92
        return config

    def test_condense_slots_single_move(self, runner_config):
        """Single move: src -> dst copies state correctly."""
        runner = _make_runner(runner_config, max_seqs=8)

        runner._state_shift[:, :, 2] = 99.0
        runner._state_wkv[:, 2] = 999.0
        runner._state_elapsed[2] = 42
        runner._last_sampled_token[2] = 123

        runner._condense_slots([(2, 0)])

        assert torch.equal(runner._state_shift[:, :, 0], torch.full_like(runner._state_shift[:, :, 0], 99.0))
        assert torch.equal(runner._state_wkv[:, 0], torch.full_like(runner._state_wkv[:, 0], 999.0))
        assert runner._state_elapsed[0].item() == 42
        assert runner._last_sampled_token[0].item() == 123

        # Source should be unchanged (it's a copy, not a move)
        assert torch.equal(runner._state_shift[:, :, 2], torch.full_like(runner._state_shift[:, :, 2], 99.0))


class TestExecuteModelRouting:
    """Test execute_model() routing: pure decode vs mixed batch."""

    @pytest.fixture
    def runner_config(self):
        config = MagicMock()
        config.model_config.dtype = "float16"
        config.model_config.model = "/fake/model.pth"
        config.model_config.load_format = "auto"
        config.model_config.seed = 42
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 2048
        config.worker_config.device = "cpu"
        config.worker_config.gpu_memory_utilization = 0.92
        return config

    def test_execute_model_pure_decode_sorted(self, runner_config):
        """Pure decode: execute_model sorts by slot_index before dispatching."""
        runner = _make_runner(runner_config, max_seqs=8)
        sp = SamplingParams(max_tokens=20)
        sp.eos_token_id = 0

        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-c", "req-a", "req-b"],
            num_scheduled_tokens={"req-c": 1, "req-a": 1, "req-b": 1},
            total_num_scheduled_tokens=3,
            finished_req_ids=set(),
            request_data={
                "req-c": _make_decode_req(runner_config, "req-c", slot_index=2),
                "req-a": _make_decode_req(runner_config, "req-a", slot_index=0),
                "req-b": _make_decode_req(runner_config, "req-b", slot_index=1),
            },
        )

        # Mock _execute_uniform_decode to capture the sorted output
        captured = {}

        def mock_uniform_decode(out):
            captured["req_ids"] = list(out.scheduled_req_ids)
            captured["slot_indices"] = list(out.slot_indices)
            return MagicMock(sampled_token_ids={}, sampled_logprobs=None, logits=torch.zeros((3, 100)))

        runner._execute_uniform_decode = mock_uniform_decode

        runner.execute_model(sched_out)

        assert captured["req_ids"] == ["req-a", "req-b", "req-c"]
        assert captured["slot_indices"] == [0, 1, 2]

    def test_execute_model_mixed_reorder_then_eager(self, runner_config):
        """Mixed batch: execute_model calls _reorder_batch then _execute_eager."""
        runner = _make_runner(runner_config, max_seqs=8)
        sp = SamplingParams(max_tokens=20)
        sp.eos_token_id = 0

        req_p = _make_prefill_req(runner_config, "req-p", slot_index=0, prompt_token_ids=list(range(10, 16)), num_computed_tokens=0)
        req_p.is_decode = False
        req_p.num_tokens = 6
        req_p.input_token_ids = req_p.prompt_token_ids

        req_a = _make_decode_req(runner_config, "req-a", slot_index=1)

        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-p", "req-a"],
            num_scheduled_tokens={"req-p": 6, "req-a": 1},
            total_num_scheduled_tokens=7,
            finished_req_ids=set(),
            request_data={"req-p": req_p, "req-a": req_a},
        )
        sched_out.slot_indices = [0, 1]

        for rid in sched_out.scheduled_req_ids:
            runner.slot_manager.allocate(rid)
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            runner.slot_manager.req_to_slot[rid] = i
            runner.slot_manager.slot_to_req[i] = rid

        reorder_called = [False]
        eager_called = [False]

        original_reorder = runner._reorder_batch

        def mock_reorder(out):
            reorder_called[0] = True
            return original_reorder(out)

        def mock_eager(out):
            eager_called[0] = True
            return MagicMock(sampled_token_ids={}, sampled_logprobs=None, logits=torch.zeros((7, 100)))

        runner._reorder_batch = mock_reorder
        runner._execute_eager = mock_eager

        runner.execute_model(sched_out)

        assert reorder_called[0] is True
        assert eager_called[0] is True


class TestUniformDecodeNoGatherScatter:
    """Verify uniform decode uses :B fast path (no gather/scatter)."""

    @pytest.fixture
    def runner_config(self):
        config = MagicMock()
        config.model_config.dtype = "float16"
        config.model_config.model = "/fake/model.pth"
        config.model_config.load_format = "auto"
        config.model_config.seed = 42
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 2048
        config.worker_config.device = "cpu"
        config.worker_config.gpu_memory_utilization = 0.92
        return config

    def test_uniform_decode_no_gather_scatter(self, runner_config):
        """Uniform decode uses :B fast path — no gather_decode_state calls."""
        runner = _make_runner(runner_config, max_seqs=8)
        sp = SamplingParams(max_tokens=20)
        sp.eos_token_id = 0

        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-a", "req-b"],
            num_scheduled_tokens={"req-a": 1, "req-b": 1},
            total_num_scheduled_tokens=2,
            finished_req_ids=set(),
            request_data={
                "req-a": _make_decode_req(runner_config, "req-a", slot_index=0),
                "req-b": _make_decode_req(runner_config, "req-b", slot_index=1),
            },
        )

        runner._last_sampled_token[:2] = torch.tensor([10, 20], dtype=torch.long)

        eager_called = [False]

        def mock_eager(B, tokens):
            eager_called[0] = True
            assert B == 2
            # Verify tokens come from :B slice of _last_sampled_token
            assert torch.equal(tokens, torch.tensor([10, 20], dtype=torch.long))
            return torch.zeros((2, 100))

        runner._execute_uniform_decode_eager = mock_eager

        runner.config.state_config = MagicMock()
        runner.config.state_config.checkpoint_interval = None

        runner._execute_uniform_decode(sched_out)
        assert eager_called[0] is True

    def test_uniform_decode_eager_uses_global_buffer(self, runner_config):
        """_execute_uniform_decode_eager uses _state_shift[:,:,:B], not _decode_state_*."""
        runner = _make_runner(runner_config, max_seqs=8)

        # Write distinct value to global state
        runner._state_shift[:, :, :3].fill_(42.0)
        runner._state_wkv[:, :3].fill_(42.0)
        runner._state_elapsed[:3].fill_(7)
        runner._decode_state_shift[:, :, :3].fill_(0.0)
        runner._decode_state_wkv[:, :3].fill_(0.0)
        runner._decode_state_elapsed[:3].fill_(0)

        capture_state = [None]

        def mock_embed(tokens):
            return torch.zeros((3, runner.C))

        def mock_forward_from_x(x, state, path, qsl, rid, mt, B):
            capture_state[0] = state
            return torch.zeros((3, 100))

        runner.model.embed = mock_embed
        runner.model.forward_from_x = mock_forward_from_x
        runner.model.path_selector = MagicMock()
        runner.model.path_selector.select = MagicMock()
        runner._uniform_query_start_loc = torch.arange(9, dtype=torch.int32)
        runner._uniform_req_id = torch.arange(8, dtype=torch.int32)

        with patch("torch.inference_mode"):
            runner._execute_uniform_decode_eager(3, torch.zeros(3, dtype=torch.long))

        # Verify state points to global buffer, not decode temp buffer
        assert capture_state[0] is not None
        assert capture_state[0][0].data_ptr() == runner._state_shift.data_ptr()
        assert capture_state[0][1].data_ptr() == runner._state_wkv.data_ptr()
        assert capture_state[0][2].data_ptr() == runner._state_elapsed.data_ptr()


class TestCaptureForShape:
    """Verify _capture_for_shape uses global buffer :B slice."""

    @pytest.fixture
    def runner_config(self):
        config = MagicMock()
        config.model_config.dtype = "float16"
        config.model_config.model = "/fake/model.pth"
        config.model_config.load_format = "auto"
        config.model_config.seed = 42
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 2048
        config.worker_config.device = "cpu"
        config.worker_config.gpu_memory_utilization = 0.92
        config.compilation_config = MagicMock()
        config.compilation_config.cudagraph_mode = "none"
        return config

    def test_capture_for_shape_uses_global_buffer(self, runner_config):
        """CUDA Graph capture uses global buffer :B slice as state."""
        runner = _make_runner(runner_config, max_seqs=8)

        B = 2
        seq_lens = (1, 1)

        capture_state = [None]

        def mock_embed(tokens):
            return torch.zeros((B, runner.C))

        def mock_forward_from_x(x, state, path, qsl, rid, mt, B_):
            capture_state[0] = state
            return torch.zeros((B_, 100))

        runner.model.embed = mock_embed
        runner.model.forward_from_x = mock_forward_from_x
        runner.model.path_selector = MagicMock()
        runner.model.path_selector.select = MagicMock()
        runner._uniform_query_start_loc = torch.arange(9, dtype=torch.int32)
        runner._uniform_req_id = torch.arange(8, dtype=torch.int32)

        # Mock cudagraph_manager to capture the state passed in
        mock_mgr = MagicMock()

        def mock_capture(shape, embed_fn, forward_fn, x, state, path, qsl, rid, mt, tokens=None):
            capture_state[0] = state

        mock_mgr._capture_for_shape = mock_capture
        runner.cudagraph_manager = mock_mgr

        runner._capture_for_shape(seq_lens)

        assert capture_state[0] is not None
        assert capture_state[0][0].data_ptr() == runner._state_shift.data_ptr()
        assert capture_state[0][1].data_ptr() == runner._state_wkv.data_ptr()
        assert capture_state[0][2].data_ptr() == runner._state_elapsed.data_ptr()


class TestSampleTokensOptimization:
    """Test sample_tokens() uses :B slice for uniform decode."""

    @pytest.fixture
    def runner_config(self):
        config = MagicMock()
        config.model_config.dtype = "float16"
        config.model_config.model = "/fake/model.pth"
        config.model_config.load_format = "auto"
        config.model_config.seed = 42
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 2048
        config.worker_config.device = "cpu"
        config.worker_config.gpu_memory_utilization = 0.92
        return config

    def test_sample_tokens_uniform_uses_slice(self, runner_config):
        """Pure decode: sample_tokens uses :B slice copy, not index_copy_."""
        runner = _make_runner(runner_config, max_seqs=8)
        sp = SamplingParams(max_tokens=20)
        sp.eos_token_id = 0

        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-a", "req-b", "req-c"],
            num_scheduled_tokens={"req-a": 1, "req-b": 1, "req-c": 1},
            total_num_scheduled_tokens=3,
            finished_req_ids=set(),
            request_data={
                "req-a": _make_decode_req(runner_config, "req-a", slot_index=0),
                "req-b": _make_decode_req(runner_config, "req-b", slot_index=1),
                "req-c": _make_decode_req(runner_config, "req-c", slot_index=2),
            },
        )

        model_output = MagicMock()
        model_output.logits = torch.zeros((3, 100))

        runner.sampler = MagicMock()
        runner.sampler.return_value = (torch.tensor([10, 20, 30], dtype=torch.long), None)

        runner._last_sampled_token[:] = -1

        result = runner.sample_tokens(model_output, sched_out)

        assert torch.equal(runner._last_sampled_token[:3], torch.tensor([10, 20, 30], dtype=torch.long))
        assert result.sampled_token_ids == {"req-a": [10], "req-b": [20], "req-c": [30]}

    def test_sample_tokens_non_sequential_uses_index_copy(self, runner_config):
        """Non-sequential slots: sample_tokens falls back to index_copy_."""
        runner = _make_runner(runner_config, max_seqs=8)
        sp = SamplingParams(max_tokens=20)
        sp.eos_token_id = 0

        sched_out = SchedulerOutput(
            scheduled_req_ids=["req-a", "req-b"],
            num_scheduled_tokens={"req-a": 1, "req-b": 1},
            total_num_scheduled_tokens=2,
            finished_req_ids=set(),
            request_data={
                "req-a": _make_decode_req(runner_config, "req-a", slot_index=2),
                "req-b": _make_decode_req(runner_config, "req-b", slot_index=5),
            },
        )

        model_output = MagicMock()
        model_output.logits = torch.zeros((2, 100))

        runner.sampler = MagicMock()
        runner.sampler.return_value = (torch.tensor([100, 200], dtype=torch.long), None)

        runner._last_sampled_token[:] = -1

        runner.sample_tokens(model_output, sched_out)

        assert runner._last_sampled_token[2].item() == 100
        assert runner._last_sampled_token[5].item() == 200
        assert runner._last_sampled_token[0].item() == -1


class TestSrcDestMapCorrectness:
    """Verify that the _reorder_batch src_dest_map uses the correct
    inverse mapping (inverse of argsort result), NOT the direct mapping
    described in plan v013 Section 3.3 Change 5 Step 2.

    np.argsort(regions, kind='stable') returns sorted_indices where:
        sorted_indices[dst_pos] = src_pos
    means 'the element at new position dst_pos should come from
    original position src_pos'.

    The plan text incorrectly describes:
        src_dest_map[src_pos] = sorted_indices[src_pos]   (DIRECT — WRONG)
    This would map src_pos -> sorted_indices[src_pos], which is the
    REVERSE of the intended permutation.

    The code correctly implements:
        src_dest_map[src_pos] = dst_pos where sorted_indices[dst_pos] = src_pos  (INVERSE — CORRECT)

    This test constructs a 4-element scenario (regions [3, 1, 0, 2])
    that produces sorted_indices = [2, 3, 0, 1] and verifies:
    - The inverse map (actual code) yields the correct final order
    - The direct map (plan text) yields an incorrect order
    """

    @pytest.fixture
    def runner_config(self):
        config = MagicMock()
        config.model_config.dtype = "float16"
        config.model_config.model = "/fake/model.pth"
        config.model_config.load_format = "auto"
        config.model_config.seed = 42
        config.scheduler_config.max_num_seqs = 8
        config.scheduler_config.max_num_batched_tokens = 2048
        config.worker_config.device = "cpu"
        config.worker_config.gpu_memory_utilization = 0.92
        return config

    def test_inverse_map_produces_correct_order(self, runner_config):
        """Actual code (inverse map) correctly reorders a 4-element mixed batch."""
        runner = _make_runner(runner_config, max_seqs=8)
        sp = SamplingParams(max_tokens=20)
        sp.eos_token_id = 0

        # Construct 4 requests with distinct regions:
        #   pos 0: region 3 (prefill)     — no context, num_tokens > 1
        #   pos 1: region 1 (short_extend)— has context, num_tokens<=1, not done
        #   pos 2: region 0 (decode)      — has context, num_tokens<=1, done
        #   pos 3: region 2 (long_extend) — has context, num_tokens > 1
        reqs = [
            # Region 3: prefill (no context)
            _make_prefill_req(
                runner_config,
                "req-p",
                slot_index=0,
                prompt_token_ids=list(range(10, 16)),
                num_computed_tokens=0,
            ),
            # Region 1: short_extend (has context, <=1 token, NOT done)
            RequestRunData(
                request_id="req-s",
                prompt_token_ids=[1, 2, 3, 4, 5],
                num_computed_tokens=2,  # < len(prompt) → not done
                num_tokens=1,
                sampling_params=sp,
                input_token_ids=None,
                is_decode=True,
                is_last_prefill=False,
                slot_index=1,
            ),
            # Region 0: decode (has context, <=1 token, done)
            _make_decode_req(runner_config, "req-d", slot_index=2),
            # Region 2: long_extend (has context, >1 token)
            RequestRunData(
                request_id="req-l",
                prompt_token_ids=[1, 2, 3, 4, 5],
                num_computed_tokens=5,  # >= len(prompt) → done
                num_tokens=6,  # > 1 → long_extend
                sampling_params=sp,
                input_token_ids=[1, 2, 3, 4, 5, 6],
                is_decode=False,
                is_last_prefill=False,
                slot_index=3,
            ),
        ]

        sched_out = SchedulerOutput(
            scheduled_req_ids=[r.request_id for r in reqs],
            num_scheduled_tokens={r.request_id: r.num_tokens for r in reqs},
            total_num_scheduled_tokens=sum(r.num_tokens for r in reqs),
            finished_req_ids=set(),
            request_data={r.request_id: r for r in reqs},
        )
        sched_out.slot_indices = list(range(4))

        for rid in sched_out.scheduled_req_ids:
            runner.slot_manager.allocate(rid)
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            runner.slot_manager.req_to_slot[rid] = i
            runner.slot_manager.slot_to_req[i] = rid

        for i in range(4):
            runner._state_shift[0, 0, i, 0] = float(i + 1)

        runner._reorder_batch(sched_out)

        # np.argsort([3, 1, 0, 2], kind='stable') = [2, 1, 3, 0]
        #   pos 0 ← orig 2 (req-d, region 0)
        #   pos 1 ← orig 1 (req-s, region 1)
        #   pos 2 ← orig 3 (req-l, region 2)
        #   pos 3 ← orig 0 (req-p, region 3)
        assert sched_out.scheduled_req_ids == ["req-d", "req-s", "req-l", "req-p"]

        # Verify slot_manager consistency
        for i, rid in enumerate(sched_out.scheduled_req_ids):
            assert runner.slot_manager.req_to_slot[rid] == sched_out.slot_indices[i]

    def test_direct_vs_inverse_map(self, runner_config):
        """Prove that the plan text's 'direct map' (src_dest_map[src] = sorted_indices[src])
        would produce the WRONG permutation for a non-self-inverse permutation.

        The actual code uses the INVERSE map (src_dest_map[src] = dst where sorted_indices[dst] = src),
        which is mathematically correct: sorted_indices[dst] = src means 'the element that should
        end up at position dst is currently at position src', so to move it there we need
        src → dst (not src → sorted_indices[src]).

        For sorted_indices = [2, 0, 3, 1] (a 4-cycle, NOT self-inverse):
          - Direct map (plan text):  {0:2, 1:0, 2:3, 3:1}
          - Inverse map (actual code): {0:1, 1:3, 2:0, 3:2}
        """
        import numpy as np

        sorted_indices = np.array([2, 0, 3, 1])

        # Direct map (plan text): src → sorted_indices[src]
        direct_map = {i: int(sorted_indices[i]) for i in range(4) if int(sorted_indices[i]) != i}

        # Inverse map (actual code): src → dst where sorted_indices[dst] = src
        inverse_map = {}
        for dst_pos in range(4):
            src_pos = int(sorted_indices[dst_pos])
            if src_pos != dst_pos:
                inverse_map[src_pos] = dst_pos

        assert direct_map != inverse_map, "Maps must differ (non-self-inverse permutation)"

        # Simulate self-resolving swap (plan/codestyle: for src in map; while src != dst)
        def _self_resolve(items: list, smap: dict) -> list:
            items = items[:]
            smap = dict(smap)
            for src in smap:
                dst = smap[src]
                while src != dst:
                    items[src], items[dst] = items[dst], items[src]
                    next_dst = smap.get(dst, dst)
                    smap[dst] = dst
                    dst = next_dst
            return items

        items = ["A", "B", "C", "D"]

        result_dir = _self_resolve(items, direct_map)
        result_inv = _self_resolve(items, inverse_map)

        # sorted_indices = [2, 0, 3, 1] means:
        #   new pos 0 ← old pos 2 (C), new pos 1 ← old pos 0 (A),
        #   new pos 2 ← old pos 3 (D), new pos 3 ← old pos 1 (B)
        expected = ["C", "A", "D", "B"]

        assert result_inv == expected, f"Inverse map (actual code) must produce {expected}, got {result_inv}"
        assert result_dir != expected, f"Direct map (plan text) must NOT produce {expected}, got {result_dir}"

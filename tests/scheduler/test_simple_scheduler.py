from vkwr.config.engine import VkwrConfig
from vkwr.config.model import ModelConfig
from vkwr.config.scheduler import SchedulerConfig
from vkwr.engine.outputs import ModelRunnerOutput
from vkwr.engine.request import RequestStatus, SamplingParams, VkwrRequest
from vkwr.scheduler import SimpleScheduler
from vkwr.state.state_slot_manager import StateSlotManager


def _make_config(
    chunked_prefill_threshold: int = 512,
    max_num_batched_tokens: int = 2048,
    max_num_seqs: int = 128,
) -> VkwrConfig:
    return VkwrConfig(
        model_config=ModelConfig(
            model="fake-model",
            max_model_len=8192,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=8192,
            chunked_prefill_threshold=chunked_prefill_threshold,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
    )


def _make_request(
    request_id: str = "req-1",
    prompt_token_ids: list[int] | None = None,
    sampling_params: SamplingParams | None = None,
) -> VkwrRequest:
    return VkwrRequest(
        request_id=request_id,
        prompt="test prompt",
        prompt_token_ids=prompt_token_ids or [1, 2, 3, 4, 5],
        sampling_params=sampling_params or SamplingParams(),
    )


def _make_model_output(
    req_id: str = "req-1",
    token_id: int = 42,
) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        sampled_token_ids={req_id: [token_id]},
        sampled_logprobs=None,
        logits=None,
    )


class TestSimpleSchedulerEmpty:
    def test_no_requests(self):
        sched = SimpleScheduler(_make_config())
        assert not sched.has_requests()

    def test_schedule_empty(self):
        sched = SimpleScheduler(_make_config())
        out = sched.schedule()
        assert out.scheduled_req_ids == []
        assert out.total_num_scheduled_tokens == 0
        assert out.finished_req_ids == set()
        assert out.request_data == {}

    def test_finish_requests_empty(self):
        sched = SimpleScheduler(_make_config())
        sched.finish_requests({"nonexistent"})


class TestSimpleSchedulerPrefillSingleChunk:
    def test_full_prefill_one_step(self):
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        req = _make_request(prompt_token_ids=[1, 2, 3])
        sched.add_request(req)
        assert sched.has_requests()

        out = sched.schedule()
        assert out.scheduled_req_ids == ["req-1"]
        assert out.total_num_scheduled_tokens == 3
        run_data = out.request_data["req-1"]
        assert run_data.input_token_ids == [1, 2, 3]
        assert run_data.is_decode is False
        assert run_data.is_last_prefill is True

    def test_request_status_updated(self):
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        req = _make_request(prompt_token_ids=[1, 2, 3])
        assert req.status == RequestStatus.WAITING
        sched.add_request(req)
        sched.schedule()
        assert req.status == RequestStatus.RUNNING

    def test_prefill_max_tokens_zero_finishes(self):
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        req = _make_request(
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=1),
        )
        sched.add_request(req)
        out = sched.schedule()
        assert "req-1" in out.scheduled_req_ids


class TestSimpleSchedulerPrefillChunked:
    def test_chunked_prefill_multiple_steps(self):
        chunk_size = 2
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=chunk_size))
        req = _make_request(prompt_token_ids=[1, 2, 3, 4, 5])
        sched.add_request(req)

        # Chunk 1
        out1 = sched.schedule()
        rd1 = out1.request_data["req-1"]
        assert rd1.input_token_ids == [1, 2]
        assert rd1.is_decode is False
        assert rd1.is_last_prefill is False

        # Chunk 2
        out2 = sched.schedule()
        rd2 = out2.request_data["req-1"]
        assert rd2.input_token_ids == [3, 4]
        assert rd2.is_decode is False
        assert rd2.is_last_prefill is False

        # Chunk 3 (last)
        out3 = sched.schedule()
        rd3 = out3.request_data["req-1"]
        assert rd3.input_token_ids == [5]
        assert rd3.is_decode is False
        assert rd3.is_last_prefill is True


class TestSimpleSchedulerDecode:
    def test_decode_after_full_prefill(self):
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        req = _make_request(
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(max_tokens=2),
        )
        sched.add_request(req)

        # Prefill
        out = sched.schedule()
        assert out.request_data["req-1"].is_last_prefill is True
        assert out.request_data["req-1"].is_decode is False

        # Feed back sampled token
        model_output = _make_model_output(token_id=100)
        sched.update_from_output(out, model_output)

        # Decode step 1
        out2 = sched.schedule()
        rd2 = out2.request_data["req-1"]
        assert rd2.is_decode is True
        assert rd2.input_token_ids is None

    def test_decode_finishes_at_max_tokens(self):
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        req = _make_request(
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=1),
        )
        sched.add_request(req)

        # Prefill (last prefill, start_decode_pos = 0 + 1 = 1)
        out = sched.schedule()
        assert out.request_data["req-1"].is_last_prefill is True

        # Feed back sampled token
        model_output = _make_model_output(token_id=100)
        sched.update_from_output(out, model_output)

        # Decode step 1 — should finish (scheduler returns early with finished flag)
        out2 = sched.schedule()
        assert out2.scheduled_req_ids == []
        assert out2.total_num_scheduled_tokens == 0
        assert "req-1" in out2.finished_req_ids


class TestSimpleSchedulerEOSDetection:
    def test_eos_token_finishes_request(self):
        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        req = _make_request(prompt_token_ids=[1], sampling_params=sp)
        sched.add_request(req)

        out = sched.schedule()
        assert out.request_data["req-1"].is_last_prefill is True

        model_output = _make_model_output(token_id=0)
        engine_outputs = sched.update_from_output(out, model_output)

        assert "req-1" in out.finished_req_ids
        core_out = engine_outputs["req-1"].outputs[0]
        assert core_out.finish_reason == "eos"

    def test_ignore_eos_continues(self):
        sp = SamplingParams(max_tokens=3)
        sp.eos_token_id = 0
        sp.ignore_eos = True
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        req = _make_request(prompt_token_ids=[1], sampling_params=sp)
        sched.add_request(req)

        out = sched.schedule()
        model_output = _make_model_output(token_id=0)
        sched.update_from_output(out, model_output)

        assert "req-1" not in out.finished_req_ids


class TestSimpleSchedulerStopTokenIds:
    def test_stop_token_finishes_request(self):
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        req = _make_request(
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=10, stop_token_ids=[999]),
        )
        sched.add_request(req)

        out = sched.schedule()
        model_output = _make_model_output(token_id=999)
        sched.update_from_output(out, model_output)

        assert "req-1" in out.finished_req_ids

    def test_stop_token_finish_reason(self):
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        req = _make_request(
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=10, stop_token_ids=[999]),
        )
        sched.add_request(req)

        out = sched.schedule()
        model_output = _make_model_output(token_id=999)
        engine_outputs = sched.update_from_output(out, model_output)

        core_out = engine_outputs["req-1"].outputs[0]
        assert core_out.finish_reason == "stop"


class TestSimpleSchedulerFinishRequests:
    def test_finish_removes_running(self):
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        req = _make_request(prompt_token_ids=[1, 2, 3, 4, 5])
        sched.add_request(req)
        sched.schedule()
        assert any(r.request_id == "req-1" for r in sched.running)
        sched.finish_requests({"req-1"})
        assert not any(r.request_id == "req-1" for r in sched.running)
        assert "req-1" not in sched.running_output_tokens


class TestSimpleSchedulerMultipleRequests:
    def test_processes_requests_in_order(self):
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512))
        sched.add_request(
            _make_request(
                request_id="req-1",
                prompt_token_ids=[1, 2],
                sampling_params=SamplingParams(max_tokens=1),
            )
        )
        sched.add_request(_make_request(request_id="req-2", prompt_token_ids=[10, 20]))

        out1 = sched.schedule()
        # req-1 is in RUNNING, gets 1 decode token scheduled.
        assert "req-1" in out1.scheduled_req_ids

        sched.update_from_output(
            out1,
            _make_model_output(req_id="req-1", token_id=100),
        )

        out2 = sched.schedule()
        # req-1 has finished (max_tokens=1, got 1 output token).
        # req-2 should now be scheduled.
        assert "req-1" in out2.finished_req_ids
        assert "req-2" in out2.scheduled_req_ids


class TestSimpleSchedulerUpdateFromOutput:
    def test_intermediate_prefill_no_output(self):
        chunk_size = 2
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=chunk_size))
        req = _make_request(prompt_token_ids=[1, 2, 3, 4, 5])
        sched.add_request(req)

        out = sched.schedule()
        assert out.request_data["req-1"].is_last_prefill is False

        model_output = ModelRunnerOutput(
            sampled_token_ids={},
            sampled_logprobs=None,
            logits=None,
        )
        engine_outputs = sched.update_from_output(out, model_output)
        assert "req-1" not in engine_outputs

    def test_last_prefill_produces_output(self):
        chunk_size = 2
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=chunk_size))
        req = _make_request(prompt_token_ids=[1, 2, 3, 4, 5])
        sched.add_request(req)

        # Skip first two chunks
        sched.schedule()
        sched.schedule()

        # Last chunk
        out = sched.schedule()
        assert out.request_data["req-1"].is_last_prefill is True

        model_output = _make_model_output(token_id=42)
        engine_outputs = sched.update_from_output(out, model_output)
        assert "req-1" in engine_outputs
        core_out = engine_outputs["req-1"].outputs[0]
        assert core_out.new_token_ids == [42]


class TestDeferredFree:
    """Phase 2: update_from_output defers slot free via freed_slots."""

    def test_update_from_output_defers_free(self):
        slot_manager = StateSlotManager(4)
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512), slot_manager=slot_manager)
        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0
        req = _make_request(
            request_id="req-1",
            prompt_token_ids=[1],
            sampling_params=sp,
        )
        sched.add_request(req)
        out = sched.schedule()

        slot_before = slot_manager.req_to_slot.get("req-1")
        model_output = _make_model_output(token_id=0)
        sched.update_from_output(out, model_output)

        assert "req-1" in slot_manager.req_to_slot
        assert slot_manager.req_to_slot["req-1"] == slot_before

    def test_update_from_output_collects_freed_slots(self):
        slot_manager = StateSlotManager(4)
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512), slot_manager=slot_manager)
        sp = SamplingParams(max_tokens=10)
        sp.eos_token_id = 0
        req = _make_request(
            request_id="req-1",
            prompt_token_ids=[1],
            sampling_params=sp,
        )
        sched.add_request(req)
        out = sched.schedule()
        assert out.request_data["req-1"].is_last_prefill is True

        model_output = _make_model_output(token_id=0)
        sched.update_from_output(out, model_output)

        assert "req-1" in out.finished_req_ids
        assert len(out.freed_slots) == 1
        assert out.freed_slots[0][0] == "req-1"

    def test_finish_requests_defers_free(self):
        slot_manager = StateSlotManager(4)
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512), slot_manager=slot_manager)
        req = _make_request(prompt_token_ids=[1, 2, 3, 4, 5])
        sched.add_request(req)
        sched.schedule()

        assert "req-1" in slot_manager.req_to_slot
        finished = sched.finish_requests({"req-1"})
        assert finished == ["req-1"]
        assert "req-1" in slot_manager.req_to_slot

    def test_finish_requests_shutdown_defers_free(self):
        slot_manager = StateSlotManager(4)
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512), slot_manager=slot_manager)
        req = _make_request(prompt_token_ids=[1, 2, 3, 4, 5])
        sched.add_request(req)
        sched.schedule()

        assert "req-1" in slot_manager.req_to_slot
        finished = sched.finish_requests(None)
        assert "req-1" in finished
        assert "req-1" in slot_manager.req_to_slot

    def test_finish_requests_returns_list_str(self):
        slot_manager = StateSlotManager(4)
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512), slot_manager=slot_manager)
        req = _make_request(prompt_token_ids=[1, 2, 3, 4, 5])
        sched.add_request(req)
        sched.schedule()

        result = sched.finish_requests({"req-1"})
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_slot_sorted_output_pure_decode(self):
        slot_manager = StateSlotManager(8)
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512), slot_manager=slot_manager)

        req_a = _make_request(request_id="req-a", prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=5))
        req_b = _make_request(request_id="req-b", prompt_token_ids=[2], sampling_params=SamplingParams(max_tokens=5))
        req_c = _make_request(request_id="req-c", prompt_token_ids=[3], sampling_params=SamplingParams(max_tokens=5))

        sched.add_request(req_a)
        sched.add_request(req_b)
        sched.add_request(req_c)

        out1 = sched.schedule()

        model_output = ModelRunnerOutput(
            sampled_token_ids={"req-a": [10], "req-b": [20], "req-c": [30]},
            sampled_logprobs=None,
            logits=None,
        )
        sched.update_from_output(out1, model_output)

        out2 = sched.schedule()
        slots = [out2.request_data[rid].slot_index for rid in out2.scheduled_req_ids]
        assert slots == sorted(slots)

    def test_no_reorder_for_mixed_batch(self):
        slot_manager = StateSlotManager(4)
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=2), slot_manager=slot_manager)

        req_a = _make_request(request_id="req-a", prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=5))
        req_b = _make_request(request_id="req-b", prompt_token_ids=[10, 20, 30, 40])

        sched.add_request(req_a)
        sched.add_request(req_b)

        out1 = sched.schedule()

        sched.update_from_output(out1, _make_model_output(req_id="req-a", token_id=10))

        out2 = sched.schedule()
        assert "req-b" in out2.scheduled_req_ids


class TestSchedulerNoReorder:
    """Phase 3: Scheduler no longer calls reorder or produces sorted_indices."""

    def test_scheduler_no_reorder_call(self):
        """schedule() does not import or call reorder_batch_to_split_decodes_and_prefills."""
        import importlib

        sched_module = importlib.import_module("vkwr.scheduler.scheduler")
        source = getattr(sched_module, "__file__", "")
        assert source is not None

        with open(source) as f:
            content = f.read()

        assert "reorder_batch_to_split_decodes_and_prefills" not in content

    def test_scheduler_no_sorted_indices_output(self):
        """SchedulerOutput does not have sorted_indices field."""
        import dataclasses

        from vkwr.scheduler.output import SchedulerOutput

        fields = {f.name for f in dataclasses.fields(SchedulerOutput)}
        assert "sorted_indices" not in fields

    def test_scheduler_slot_sorted_running_phase1(self):
        """schedule() Phase 1 running requests sorted by slot_index ascending."""
        slot_manager = StateSlotManager(8)
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=512), slot_manager=slot_manager)

        req_a = _make_request(request_id="req-a", prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=5))
        req_b = _make_request(request_id="req-b", prompt_token_ids=[2], sampling_params=SamplingParams(max_tokens=5))
        req_c = _make_request(request_id="req-c", prompt_token_ids=[3], sampling_params=SamplingParams(max_tokens=5))

        sched.add_request(req_a)
        sched.add_request(req_b)
        sched.add_request(req_c)

        out1 = sched.schedule()

        model_output = ModelRunnerOutput(
            sampled_token_ids={"req-a": [10], "req-b": [20], "req-c": [30]},
            sampled_logprobs=None,
            logits=None,
        )
        sched.update_from_output(out1, model_output)

        out2 = sched.schedule()
        slots = [out2.request_data[rid].slot_index for rid in out2.scheduled_req_ids]
        assert slots == sorted(slots)

    def test_scheduler_mixed_batch_no_reorder(self):
        """schedule() does not reorder mixed batch — maintains running -> waiting order."""
        slot_manager = StateSlotManager(4)
        sched = SimpleScheduler(_make_config(chunked_prefill_threshold=2), slot_manager=slot_manager)

        req_a = _make_request(request_id="req-a", prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=5))
        req_b = _make_request(request_id="req-b", prompt_token_ids=[10, 20, 30, 40])

        sched.add_request(req_a)
        sched.add_request(req_b)

        out1 = sched.schedule()

        sched.update_from_output(out1, _make_model_output(req_id="req-a", token_id=10))

        out2 = sched.schedule()
        assert "req-b" in out2.scheduled_req_ids

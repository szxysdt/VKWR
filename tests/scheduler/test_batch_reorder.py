"""Unit tests for vkwr.scheduler.batch_reorder.reorder_batch_to_split_decodes_and_prefills."""

from unittest.mock import MagicMock

from vkwr.scheduler.batch_reorder import reorder_batch_to_split_decodes_and_prefills
from vkwr.scheduler.output import RequestRunData, SchedulerOutput


def _make_request_run_data(req_id: str, prompt_len: int, computed: int, num_tokens: int) -> RequestRunData:
    return RequestRunData(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        num_computed_tokens=computed,
        num_tokens=num_tokens,
        sampling_params=MagicMock(),
        input_token_ids=[],
    )


def _make_output(requests: list[tuple[str, int, int, int]]) -> SchedulerOutput:
    req_data = {rid: _make_request_run_data(rid, pl, comp, nt) for rid, pl, comp, nt in requests}
    return SchedulerOutput(
        scheduled_req_ids=[rid for rid, _, _, _ in requests],
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        finished_req_ids=set(),
        request_data=req_data,
    )


class TestNoReorderCases:
    def test_no_reorder_single_request(self):
        out = _make_output([("r0", 10, 0, 1)])
        result = reorder_batch_to_split_decodes_and_prefills(out)
        assert result is None

    def test_no_reorder_all_decode(self):
        out = _make_output(
            [
                ("r0", 10, 10, 1),
                ("r1", 20, 20, 1),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out)
        assert result is None

    def test_no_reorder_all_prefill(self):
        out = _make_output(
            [
                ("r0", 10, 0, 1),
                ("r1", 20, 0, 1),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out)
        assert result is None


class TestReorderBasic:
    def test_reorder_decode_before_prefill(self):
        out = _make_output(
            [
                ("prefill", 10, 0, 1),
                ("decode", 10, 10, 1),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out)
        assert result is not None
        assert list(result) == [1, 0]
        assert out.scheduled_req_ids == ["decode", "prefill"]

    def test_reorder_four_regions(self):
        out = _make_output(
            [
                ("prefill", 10, 0, 1),
                ("short_extend", 10, 5, 1),
                ("long_extend", 10, 5, 10),
                ("decode", 10, 10, 1),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out)
        assert result is not None
        assert list(result) == [3, 1, 2, 0]
        assert out.scheduled_req_ids == ["decode", "short_extend", "long_extend", "prefill"]


class TestStableSort:
    def test_stable_sort_preserves_order(self):
        out = _make_output(
            [
                ("p1", 10, 0, 1),
                ("d1", 10, 10, 1),
                ("p2", 20, 0, 1),
                ("d2", 20, 20, 1),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out)
        assert result is not None
        assert list(result) == [1, 3, 0, 2]
        assert out.scheduled_req_ids == ["d1", "d2", "p1", "p2"]


class TestClassification:
    def test_decode_classification(self):
        out = _make_output(
            [
                ("prefill", 10, 0, 1),
                ("decode", 10, 10, 1),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out)
        assert result is not None
        assert list(result) == [1, 0]

    def test_short_extend_classification(self):
        out = _make_output(
            [
                ("prefill", 10, 0, 1),
                ("short_ext", 10, 5, 1),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out)
        assert result is not None
        assert list(result) == [1, 0]

    def test_long_extend_classification(self):
        out = _make_output(
            [
                ("prefill", 10, 0, 1),
                ("long_ext", 10, 5, 10),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out)
        assert result is not None
        assert list(result) == [1, 0]

    def test_prefill_classification(self):
        out = _make_output(
            [
                ("prefill", 10, 0, 1),
                ("short_extend", 10, 5, 1),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out)
        assert result is not None
        assert list(result) == [1, 0]


class TestScheduledReqIdsUpdated:
    def test_scheduled_req_ids_updated(self):
        out = _make_output(
            [
                ("a", 10, 0, 1),
                ("b", 10, 10, 1),
                ("c", 10, 5, 1),
            ]
        )
        reorder_batch_to_split_decodes_and_prefills(out)
        assert out.scheduled_req_ids == ["b", "c", "a"]


class TestCustomThreshold:
    def test_custom_threshold(self):
        out = _make_output(
            [
                ("prefill", 10, 0, 1),
                ("r1", 10, 5, 2),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out, decode_threshold=2)
        assert result is not None
        assert list(result) == [1, 0]

    def test_default_threshold_makes_num_tokens_2_long_extend(self):
        out = _make_output(
            [
                ("prefill", 10, 0, 1),
                ("r1", 10, 5, 2),
            ]
        )
        result = reorder_batch_to_split_decodes_and_prefills(out, decode_threshold=1)
        assert result is not None
        assert list(result) == [1, 0]

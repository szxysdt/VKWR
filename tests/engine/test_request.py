import pytest

from vkwr.engine.request import RequestStatus, SamplingParams, VkwrRequest


class TestRequestStatus:
    def test_status_values(self):
        assert RequestStatus.WAITING == 0
        assert RequestStatus.RUNNING == 1
        assert RequestStatus.FINISHED_STOPPED == 2
        assert RequestStatus.FINISHED_ABORTED == 3


class TestSamplingParams:
    def test_default_values(self):
        sp = SamplingParams()
        assert sp.n == 1
        assert sp.temperature == 1.0
        assert sp.top_p == 1.0
        assert sp.top_k == -1
        assert sp.max_tokens is None
        assert sp.eos_token_id is None
        assert sp.ignore_eos is False

    def test_custom_values(self):
        sp = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=128,
        )
        assert sp.temperature == 0.7
        assert sp.top_p == 0.9
        assert sp.top_k == 50
        assert sp.max_tokens == 128

    def test_temperature_negative(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=-0.1)

    def test_top_p_out_of_range_low(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=-0.1)

    def test_top_p_out_of_range_high(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.1)

    def test_top_k_invalid(self):
        with pytest.raises(ValueError, match="top_k"):
            SamplingParams(top_k=-2)

    def test_max_tokens_zero_allowed(self):
        sp = SamplingParams(max_tokens=0)
        assert sp.max_tokens == 0

    def test_max_tokens_negative(self):
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams(max_tokens=-1)

    def test_eos_token_id_property(self):
        sp = SamplingParams()
        assert sp.eos_token_id is None
        sp.eos_token_id = 0
        assert sp.eos_token_id == 0

    def test_stop_token_ids(self):
        sp = SamplingParams(stop_token_ids=[0, 1])
        assert sp.stop_token_ids == [0, 1]

    def test_stop_strings(self):
        sp = SamplingParams(stop=["\n", "</s>"])
        assert sp.stop == ["\n", "</s>"]

    def test_seed(self):
        sp = SamplingParams(seed=42)
        assert sp.seed == 42


class TestVkwrRequest:
    def test_basic_request(self):
        sp = SamplingParams()
        req = VkwrRequest(
            request_id="req-1",
            prompt="Hello",
            prompt_token_ids=[1, 2, 3],
            sampling_params=sp,
        )
        assert req.request_id == "req-1"
        assert req.prompt == "Hello"
        assert req.prompt_token_ids == [1, 2, 3]
        assert req.status == RequestStatus.WAITING
        assert req.priority == 0

    def test_request_with_token_ids(self):
        sp = SamplingParams()
        req = VkwrRequest(
            request_id="req-2",
            prompt=[1001, 1002, 1003],
            prompt_token_ids=[1001, 1002, 1003],
            sampling_params=sp,
        )
        assert isinstance(req.prompt, list)
        assert req.prompt == [1001, 1002, 1003]

    def test_request_arrival_time(self):

        sp = SamplingParams()
        req = VkwrRequest(
            request_id="req-3",
            prompt="test",
            prompt_token_ids=[1],
            sampling_params=sp,
        )
        assert req.arrival_time > 0

    def test_request_custom_arrival_time(self):
        sp = SamplingParams()
        req = VkwrRequest(
            request_id="req-4",
            prompt="test",
            prompt_token_ids=[1],
            sampling_params=sp,
            arrival_time=12345.0,
        )
        assert req.arrival_time == 12345.0

    def test_request_custom_priority(self):
        sp = SamplingParams()
        req = VkwrRequest(
            request_id="req-5",
            prompt="test",
            prompt_token_ids=[1],
            sampling_params=sp,
            priority=10,
        )
        assert req.priority == 10

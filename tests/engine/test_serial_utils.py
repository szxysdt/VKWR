import pickle
import time

import pytest

from vkwr.engine.core_request import EngineCoreRequest, UtilityOutput, UtilityResult
from vkwr.engine.outputs import EngineCoreOutput, EngineCoreOutputs
from vkwr.engine.request import SamplingParams
from vkwr.engine.serial_utils import MsgpackDecoder, MsgpackEncoder


class TestMsgpackEncoderDecoder:
    def test_roundtrip_dict(self):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()
        obj = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
        bufs = encoder.encode(obj)
        result = decoder.decode(bufs)
        assert result == obj

    def test_roundtrip_list(self):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()
        obj = [1, "two", 3.0, True, None]
        bufs = encoder.encode(obj)
        result = decoder.decode(bufs)
        assert result == obj

    def test_roundtrip_intenum(self):
        from vkwr.engine.request import RequestOutputKind

        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()
        obj = {"kind": RequestOutputKind.DELTA}
        bufs = encoder.encode(obj)
        result = decoder.decode(bufs)
        assert result["kind"] == RequestOutputKind.DELTA.value

    def test_cloudpickle_fallback(self):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()

        def my_func():
            return 42

        obj = {"func": my_func}
        bufs = encoder.encode(obj)
        result = decoder.decode(bufs, t=dict)
        assert result["func"]() == 42


class TestSamplingParamsSerialization:
    def test_roundtrip(self):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()
        sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
        bufs = encoder.encode(sp)
        result = decoder.decode(bufs, t=SamplingParams)
        assert result.temperature == 0.7
        assert result.top_p == 0.9
        assert result.max_tokens == 128
        assert result.output_kind == 0

    def test_output_kind_is_int(self):
        sp = SamplingParams()
        assert isinstance(sp.output_kind, int)
        assert sp.output_kind == 0

    def test_stop_normalization(self):
        sp_str = SamplingParams(stop="</s>")
        assert sp_str.stop == ["</s>"]
        sp_list = SamplingParams(stop=["\n", "</s>"])
        assert sp_list.stop == ["\n", "</s>"]
        sp_none = SamplingParams()
        assert sp_none.stop == []

    def test_zero_temperature_fallback(self):
        sp = SamplingParams(temperature=0)
        assert sp.top_p == 1.0
        assert sp.top_k == 0
        assert sp.min_p == 0.0

    def test_max_tokens_none_allowed(self):
        sp = SamplingParams(max_tokens=None)
        assert sp.max_tokens is None

    def test_max_tokens_negative_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams(max_tokens=-1)

    def test_max_tokens_zero_rejected(self):
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams(max_tokens=0)

    def test_top_p_zero_rejected(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0)


class TestEngineCoreOutputSerialization:
    def test_roundtrip(self):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()
        output = EngineCoreOutput(
            request_id="req-1",
            new_token_ids=[1, 2, 3],
            new_logprobs=[{1: 0.5}],
            finish_reason="stop",
            stop_reason=0,
        )
        bufs = encoder.encode(output)
        result = decoder.decode(bufs, t=EngineCoreOutput)
        assert result.request_id == "req-1"
        assert result.new_token_ids == [1, 2, 3]
        assert result.finish_reason == "stop"
        assert result.stop_reason == 0


class TestEngineCoreOutputsSerialization:
    def test_roundtrip(self):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()
        outputs = EngineCoreOutputs(
            outputs=[
                EngineCoreOutput(request_id="req-1", new_token_ids=[1]),
                EngineCoreOutput(request_id="req-2", new_token_ids=[2, 3]),
            ],
            timestamp=12345.0,
        )
        bufs = encoder.encode(outputs)
        result = decoder.decode(bufs, t=EngineCoreOutputs)
        assert len(result.outputs) == 2
        assert result.outputs[0].request_id == "req-1"
        assert result.outputs[1].new_token_ids == [2, 3]
        assert result.timestamp == 12345.0

    def test_timestamp_auto_set(self):
        outputs = EngineCoreOutputs()
        assert outputs.timestamp > 0

    def test_timestamp_not_overridden(self):
        t = 99999.0
        outputs = EngineCoreOutputs(timestamp=t)
        assert outputs.timestamp == t


class TestEngineCoreRequestSerialization:
    def test_roundtrip(self):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()
        sp = SamplingParams(temperature=0.7, max_tokens=64)
        req = EngineCoreRequest(
            request_id="req-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=sp,
            arrival_time=100.0,
            priority=5,
        )
        bufs = encoder.encode(req)
        result = decoder.decode(bufs, t=EngineCoreRequest)
        assert result.request_id == "req-1"
        assert result.prompt_token_ids == [1, 2, 3]
        assert result.sampling_params.temperature == 0.7
        assert result.sampling_params.max_tokens == 64
        assert result.arrival_time == 100.0
        assert result.priority == 5


class TestUtilityTypesSerialization:
    def test_utility_result_roundtrip(self):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()
        ur = UtilityResult(result={"key": "value"})
        bufs = encoder.encode(ur)
        result = decoder.decode(bufs, t=UtilityResult)
        assert result.result == {"key": "value"}

    def test_utility_output_roundtrip(self):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()
        uo = UtilityOutput(
            call_id=1,
            failure_message=None,
            result=UtilityResult(result=42),
        )
        bufs = encoder.encode(uo)
        result = decoder.decode(bufs, t=UtilityOutput)
        assert result.call_id == 1
        assert result.failure_message is None
        assert result.result.result == 42


class TestSerializationPerformance:
    def test_msgpack_faster_than_pickle(self):
        encoder = MsgpackEncoder()
        decoder = MsgpackDecoder()
        sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128)
        req = EngineCoreRequest(
            request_id="req-1",
            prompt_token_ids=list(range(500)),
            sampling_params=sp,
            arrival_time=100.0,
        )

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            bufs = encoder.encode(req)
            decoder.decode(bufs, t=EngineCoreRequest)
        msgpack_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(iterations):
            data = pickle.dumps(req)
            pickle.loads(data)
        pickle_time = time.perf_counter() - start

        assert msgpack_time < pickle_time, f"msgpack ({msgpack_time:.4f}s) should be faster than pickle ({pickle_time:.4f}s)"

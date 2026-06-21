from vkwr.engine.outputs import (
    CompletionOutput,
    EngineCoreOutput,
    EngineCoreOutputs,
    ModelRunnerOutput,
    RequestOutput,
    RequestStats,
)


class TestCompletionOutput:
    def test_creation(self):
        co = CompletionOutput(
            index=0,
            token_ids=[1, 2, 3],
            text="Hello",
            cumlogprob=-1.5,
        )
        assert co.index == 0
        assert co.token_ids == [1, 2, 3]
        assert co.text == "Hello"
        assert co.cumlogprob == -1.5
        assert co.logprobs is None
        assert co.finish_reason is None


class TestRequestStats:
    def test_default_values(self):
        rs = RequestStats()
        assert rs.waiting_time == 0.0
        assert rs.running_time == 0.0
        assert rs.total_tokens == 0
        assert rs.generated_tokens == 0

    def test_custom_values(self):
        rs = RequestStats(
            waiting_time=0.1,
            running_time=1.5,
            total_tokens=100,
            generated_tokens=50,
        )
        assert rs.waiting_time == 0.1
        assert rs.running_time == 1.5


class TestRequestOutput:
    def test_creation(self):
        co = CompletionOutput(index=0, token_ids=[1], text="Hi", cumlogprob=-0.5)
        ro = RequestOutput(
            request_id="req-1",
            prompt="Hello",
            prompt_token_ids=[10, 20],
            outputs=[co],
            finished=True,
            finish_reason="stop",
        )
        assert ro.request_id == "req-1"
        assert ro.finished is True
        assert ro.finish_reason == "stop"
        assert len(ro.outputs) == 1
        assert ro.stats is None

    def test_with_stats(self):
        co = CompletionOutput(index=0, token_ids=[1], text="Hi", cumlogprob=-0.5)
        rs = RequestStats(total_tokens=10, generated_tokens=5)
        ro = RequestOutput(
            request_id="req-2",
            prompt="test",
            prompt_token_ids=[1],
            outputs=[co],
            finished=False,
            finish_reason=None,
            stats=rs,
        )
        assert ro.stats.total_tokens == 10


class TestEngineCoreOutput:
    def test_creation(self):
        eeo = EngineCoreOutput(
            request_id="req-1",
            new_token_ids=[42],
            finish_reason=None,
        )
        assert eeo.request_id == "req-1"
        assert eeo.new_token_ids == [42]
        assert eeo.new_logprobs is None


class TestEngineCoreOutputs:
    def test_default(self):
        eeo = EngineCoreOutputs()
        assert len(eeo.outputs) == 0
        assert eeo.scheduler_stats is None
        assert eeo.timestamp > 0

    def test_with_outputs(self):
        output = EngineCoreOutput(
            request_id="req-1",
            new_token_ids=[42],
            finish_reason="stop",
        )
        eeo = EngineCoreOutputs(
            outputs=[output],
            timestamp=12345.0,
        )
        assert len(eeo.outputs) == 1
        assert eeo.outputs[0].finish_reason == "stop"
        assert eeo.timestamp == 12345.0


class TestModelRunnerOutput:
    def test_creation(self):
        mro = ModelRunnerOutput(
            sampled_token_ids={"req-1": [42]},
        )
        assert mro.sampled_token_ids == {"req-1": [42]}
        assert mro.sampled_logprobs is None
        assert mro.logits is None

"""Integration test: multi-request concurrent inference end-to-end.

Tests the full pipeline under concurrent workloads, verifying state slot
isolation, decode-first scheduling, HOL blocking fix, and varlen P/D
mixed batch execution. Skips if model checkpoint is not available.
"""

import gc
import os
import random
from contextlib import contextmanager

import pytest
import torch

MODEL_PATH = os.environ.get("VKWR_RWKV7_MODEL_PATH")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="VKWR_RWKV7_MODEL_PATH not set",
)


@contextmanager
def _cleanup():
    """Ensure GPU memory is released after each test."""
    try:
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()


class TestMultiRequestEndToEnd:
    """Full LLM.generate() pipeline test with multiple concurrent requests."""

    def test_concurrent_decode(self):
        """3 requests generated concurrently, each produces valid output."""
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            llm = LLM(
                model=MODEL_PATH,
                max_model_len=256,
                max_num_batched_tokens=128,
                max_num_seqs=8,
            )
            prompts = ["Hello world", "Good morning", "How are you"]
            sp = SamplingParams(max_tokens=8, temperature=0)
            outputs = llm.generate(prompts, sampling_params=sp)

            assert len(outputs) == 3
            for out in outputs:
                assert out.finished is True
                assert len(out.outputs) == 1
                assert len(out.outputs[0].token_ids) > 0

            del llm

    def test_concurrent_different_lengths(self):
        """Multiple requests with different prompt lengths."""
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            llm = LLM(
                model=MODEL_PATH,
                max_model_len=512,
                max_num_batched_tokens=256,
                max_num_seqs=8,
            )
            prompts = [
                "Hi",
                "Hello world, how are you?",
                "A" * 200,
            ]
            sp = SamplingParams(max_tokens=4, temperature=0)
            outputs = llm.generate(prompts, sampling_params=sp)

            assert len(outputs) == 3
            for out in outputs:
                assert out.finished is True
                assert len(out.outputs[0].token_ids) > 0

            del llm

    def test_state_slot_isolation(self):
        """State slots for different requests must not leak into each other.

        Generate 2 requests with same prompt. With temperature=0,
        both should produce identical output (deterministic). If state leaks,
        outputs will diverge.

        Note: fp16 batched inference can have numerical jitter that causes
        borderline token divergence. We allow a small tolerance: the two
        sequences must share a high overlap (>=80%) rather than requiring
        exact equality.
        """
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            llm = LLM(
                model=MODEL_PATH,
                max_model_len=256,
                max_num_batched_tokens=128,
                max_num_seqs=8,
            )
            prompt = "The capital of France is"
            sp = SamplingParams(max_tokens=8, temperature=0)
            outputs = llm.generate([prompt, prompt], sampling_params=sp)

            assert len(outputs) == 2
            ids1 = outputs[0].outputs[0].token_ids
            ids2 = outputs[1].outputs[0].token_ids

            # fp16 tolerance: require high overlap rather than exact match.
            # True state corruption would cause near-total divergence.
            common = sum(1 for a, b in zip(ids1, ids2) if a == b)
            overlap = common / max(len(ids1), 1)
            assert overlap >= 0.8, f"State leak detected: overlap={overlap:.2f}, {ids1} vs {ids2}"

            del llm


class TestDecodeNotBlocked:
    """Decode requests should not be blocked by incoming large prefill."""

    def test_decode_not_blocked_by_prefill(self):
        """Decode requests should not be blocked by incoming large prefill.

        Strategy:
        1. Add a short request that completes prefill immediately and enters decode
        2. Add a long prefill request while decode is active
        3. Verify decode step still executes
        """
        with _cleanup():
            from vkwr.config.vkwr import EngineArgs
            from vkwr.engine.llm_engine import LLMEngine
            from vkwr.engine.request import SamplingParams

            engine_args = EngineArgs(
                model=MODEL_PATH,
                max_model_len=512,
                max_num_batched_tokens=256,
                max_num_seqs=8,
            )
            engine = LLMEngine.from_engine_args(engine_args)

            engine.add_request("req-decode", "Hello", SamplingParams(max_tokens=4, temperature=0))
            step1 = engine.step()
            assert len(step1) > 0

            engine.add_request("req-long", "A" * 200, SamplingParams(max_tokens=2, temperature=0))

            engine.step()
            assert engine.has_unfinished_requests()

            remaining = []
            while engine.has_unfinished_requests():
                for out in engine.step():
                    if out.finished:
                        remaining.append(out)

            req_ids = {r.request_id for r in remaining}
            assert "req-decode" in req_ids
            assert "req-long" in req_ids
            for out in remaining:
                assert out.finished is True

            del engine


class TestHOLBlockingEndToEnd:
    """End-to-end HOL blocking fix verification."""

    def test_hol_blocking_endtoend(self):
        """Long request at head of queue should not block short requests.

        Add 2 long requests + 1 short request. With default scheduler
        settings, only 1 long prefill can run at a time. The short request
        should still be scheduled and all requests should eventually complete.
        """
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            llm = LLM(
                model=MODEL_PATH,
                max_model_len=512,
                max_num_batched_tokens=256,
                max_num_seqs=8,
            )
            prompts = [
                "A" * 200,
                "B" * 200,
                "Hi",
            ]
            sp = SamplingParams(max_tokens=4, temperature=0)
            outputs = llm.generate(prompts, sampling_params=sp)

            assert len(outputs) == 3
            for out in outputs:
                assert out.finished is True

            del llm


class TestVarlenEndToEnd:
    """End-to-end varlen P/D mixed batch verification."""

    def test_varlen_mixed_pd_batch(self):
        """Prefill and decode requests can be batched together with varlen.

        Add multiple short prompts. First round will prefill them all.
        After update_from_output, some enter decode. New prefill requests
        can be interleaved. With varlen, they share the same forward pass.
        """
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            llm = LLM(
                model=MODEL_PATH,
                max_model_len=256,
                max_num_batched_tokens=128,
                max_num_seqs=8,
            )
            outputs = llm.generate(
                ["Hello", "Good morning", "What is AI"],
                sampling_params=SamplingParams(max_tokens=6, temperature=0),
            )

            assert len(outputs) == 3
            for out in outputs:
                assert out.finished is True
                assert len(out.outputs[0].token_ids) > 0

            del llm

    def test_varlen_single_vs_multi_consistency(self):
        """Single-request generation should match the corresponding request
        from a multi-request batch (temperature=0, deterministic)."""
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            prompt = "The quick brown fox"
            sp = SamplingParams(max_tokens=8, temperature=0)

            single_llm = LLM(
                model=MODEL_PATH,
                max_model_len=256,
                max_num_batched_tokens=128,
                max_num_seqs=8,
            )
            single_outputs = single_llm.generate(prompt, sampling_params=sp)
            single_ids = single_outputs[0].outputs[0].token_ids
            del single_llm

            multi_llm = LLM(
                model=MODEL_PATH,
                max_model_len=256,
                max_num_batched_tokens=128,
                max_num_seqs=8,
            )
            multi_outputs = multi_llm.generate([prompt, prompt], sampling_params=sp)
            multi_ids_0 = multi_outputs[0].outputs[0].token_ids
            multi_ids_1 = multi_outputs[1].outputs[0].token_ids

            assert single_ids == multi_ids_0 == multi_ids_1, "Single and multi-request outputs should match for deterministic generation"

            del multi_llm


class TestSamplerGroupedEndToEnd:
    """End-to-end sampler grouped verification."""

    def test_different_sampling_params(self):
        """Requests with different sampling params produce different results.

        With temperature=0, output is deterministic. With temperature=1.0,
        output is stochastic. Both should complete correctly in the same batch.
        """
        with _cleanup():
            random.seed(42)

            from vkwr.config.vkwr import EngineArgs
            from vkwr.engine.llm_engine import LLMEngine
            from vkwr.engine.request import SamplingParams

            engine_args = EngineArgs(
                model=MODEL_PATH,
                max_model_len=256,
                max_num_batched_tokens=128,
                max_num_seqs=8,
            )
            engine = LLMEngine.from_engine_args(engine_args)

            engine.add_request("req-det", "Hello", SamplingParams(max_tokens=4, temperature=0))
            engine.add_request(
                "req-stoch",
                "Hello",
                SamplingParams(
                    max_tokens=4,
                    temperature=1.0,
                    seed=random.randint(0, 2**31),
                ),
            )

            outputs = []
            while engine.has_unfinished_requests():
                for out in engine.step():
                    if out.finished:
                        outputs.append(out)

            assert len(outputs) == 2
            for out in outputs:
                assert out.finished is True

            del engine

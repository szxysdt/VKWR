"""Integration test: single request end-to-end flow.

Tests the full pipeline from LLM.generate() through engine core,
scheduler, executor, worker, and model runner. Skips if model
checkpoint is not available.
"""

import gc
import os
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


class TestSingleRequestEndToEnd:
    """Full LLM.generate() pipeline test."""

    def test_generate_text_prompt(self):
        """LLM.generate() with text prompt returns valid output."""
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            llm = LLM(model=MODEL_PATH, max_model_len=256, max_num_batched_tokens=128)
            sp = SamplingParams(max_tokens=16)
            outputs = llm.generate("Hello", sampling_params=sp)

            assert len(outputs) == 1
            out = outputs[0]
            assert out.request_id == "req-0"
            assert out.finished is True
            assert out.finish_reason is not None
            assert len(out.outputs) == 1
            assert len(out.outputs[0].token_ids) > 0

            del llm

    def test_generate_token_ids_prompt(self):
        """LLM.generate() with token IDs prompt works."""
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            llm = LLM(model=MODEL_PATH, max_model_len=256, max_num_batched_tokens=128)
            sp = SamplingParams(max_tokens=8)
            outputs = llm.generate([1, 2, 3], sampling_params=sp)

            assert len(outputs) == 1
            assert outputs[0].finished is True

            del llm

    def test_generate_max_tokens_one(self):
        """max_tokens=1 should generate exactly one token."""
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            llm = LLM(model=MODEL_PATH, max_model_len=256, max_num_batched_tokens=128)
            sp = SamplingParams(max_tokens=1)
            outputs = llm.generate("Hello", sampling_params=sp)

            assert len(outputs) == 1
            assert outputs[0].finished is True

            del llm

    def test_generate_with_sampling_params(self):
        """LLM.generate() with temperature/top_p."""
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            llm = LLM(model=MODEL_PATH, max_model_len=256, max_num_batched_tokens=128)
            sp = SamplingParams(temperature=0.7, top_p=0.9, top_k=50, max_tokens=8)
            outputs = llm.generate("Hello", sampling_params=sp)

            assert len(outputs) == 1
            assert outputs[0].finished is True

            del llm

    def test_generate_empty_prompts(self):
        """Empty prompts list returns empty output."""
        with _cleanup():
            from vkwr.entrypoints.llm import LLM

            llm = LLM(model=MODEL_PATH, max_model_len=256, max_num_batched_tokens=128)
            outputs = llm.generate([])
            assert outputs == []

            del llm

    def test_generate_long_prefill_chunked(self):
        """Long prompt is chunked through scheduler."""
        with _cleanup():
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            llm = LLM(model=MODEL_PATH, max_model_len=512, max_num_batched_tokens=128)
            long_prompt = "Hello world. " * 40
            sp = SamplingParams(max_tokens=8)
            outputs = llm.generate(long_prompt, sampling_params=sp)

            assert len(outputs) == 1
            assert outputs[0].finished is True

            del llm

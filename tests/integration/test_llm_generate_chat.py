"""Integration test: LLM generate with chat template.

Tests the full pipeline with system+user messages formatted via
the default RWKV chat template.  Requires a GPU and model
checkpoint.  Run with:

    VKWR_RWKV7_MODEL_PATH=/path/to/model.pth uv run pytest tests/integration/test_llm_generate_chat.py -v -s
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


class TestLLMGenerateChat:
    """End-to-end chat generation test."""

    def test_generate_chat_system_user(self):
        """Generate a reply given a system prompt and a user greeting."""
        with _cleanup():
            from vkwr.engine.prompts import apply_chat_template, get_default_stop_tokens
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            messages = [
                {"role": "system", "content": "你是一个AI助手"},
                {"role": "user", "content": "基于cpp给我写一个冒泡排序"},
            ]

            prompt = apply_chat_template(messages, add_generation_prompt=True)
            stop_tokens = get_default_stop_tokens()

            print(f"\n--- Chat prompt:\n{prompt}\n---")
            print(f"Stop tokens: {stop_tokens}")

            llm = LLM(
                model=MODEL_PATH,
                max_model_len=256,
                max_num_batched_tokens=128,
            )
            sp = SamplingParams(
                temperature=1.0,
                top_p=0.95,
                max_tokens=64,
                stop=stop_tokens,
                seed=42,
            )
            outputs = llm.generate(prompt, sampling_params=sp)

            assert len(outputs) == 1
            out = outputs[0]
            assert out.finished is True
            assert out.finish_reason is not None
            assert len(out.outputs) == 1

            generated_text = out.outputs[0].text
            print(f"\n--- Generated ({len(generated_text)} chars):\n{generated_text}\n---")
            print(f"Token IDs: {out.outputs[0].token_ids}")

            assert len(generated_text) > 0, "Model produced empty output"

            del llm

    def test_generate_chat_user_only(self):
        """Generate a reply with only a user message (no system prompt)."""
        with _cleanup():
            from vkwr.engine.prompts import apply_chat_template, get_default_stop_tokens
            from vkwr.engine.request import SamplingParams
            from vkwr.entrypoints.llm import LLM

            messages = [{"role": "user", "content": "你好"}]

            prompt = apply_chat_template(messages, add_generation_prompt=True)
            stop_tokens = get_default_stop_tokens()

            print(f"\n--- Chat prompt:\n{prompt}\n---")

            llm = LLM(
                model=MODEL_PATH,
                max_model_len=256,
                max_num_batched_tokens=128,
            )
            sp = SamplingParams(
                temperature=0.8,
                top_p=0.9,
                max_tokens=32,
                stop=stop_tokens,
                seed=123,
            )
            outputs = llm.generate(prompt, sampling_params=sp)

            assert len(outputs) == 1
            assert outputs[0].finished is True
            assert len(outputs[0].outputs[0].text) > 0

            print(f"\n--- Generated:\n{outputs[0].outputs[0].text}\n---")

            del llm

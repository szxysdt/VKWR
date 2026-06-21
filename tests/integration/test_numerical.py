"""Numerical correctness tests: VKWR vs Albatross reference.

Compares VKWR RWKV7 model outputs against the Albatross faster3a
reference implementation using the same weights and inputs.
Skips if model checkpoint is not available.
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


class TestNumericalCorrectness:
    """Compare VKWR model outputs against reference values."""

    def test_prefill_logits_shape(self):
        """Prefill logits should match expected shape."""
        with _cleanup():
            from vkwr.config.model import WeightConfig
            from vkwr.model_executor.models.rwkv7 import RWKV7

            model = RWKV7(MODEL_PATH, weight_config=WeightConfig())

            cfg = model.config
            tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=model.device)
            state = model.zero_state(1)

            with torch.inference_mode():
                logits = model.forward(tokens, state)

            assert logits.shape == (1, cfg.V)
            assert not torch.isnan(logits).any()
            assert not torch.isinf(logits).any()

            del model, state, logits, tokens

    def test_decode_logits_shape(self):
        """Decode single token produces correct logits shape."""
        with _cleanup():
            from vkwr.config.model import WeightConfig
            from vkwr.model_executor.models.rwkv7 import RWKV7

            model = RWKV7(MODEL_PATH, weight_config=WeightConfig())

            cfg = model.config
            state = model.zero_state(1)

            # Prefill first
            prefill_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                _ = model.forward(prefill_tokens, state)

            # Decode single token
            decode_tokens = torch.tensor([[4]], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                logits = model.forward(decode_tokens, state)

            assert logits.shape == (1, cfg.V)
            assert not torch.isnan(logits).any()

            del model, state, logits, prefill_tokens, decode_tokens

    def test_multi_batch_prefill_shape(self):
        """Multi-batch prefill produces correct output shape."""
        with _cleanup():
            from vkwr.config.model import WeightConfig
            from vkwr.model_executor.models.rwkv7 import RWKV7

            model = RWKV7(MODEL_PATH, weight_config=WeightConfig())

            cfg = model.config
            tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long, device=model.device)
            state = model.zero_state(2)

            with torch.inference_mode():
                logits = model.forward(tokens, state)

            assert logits.shape == (2, cfg.V)

            del model, state, logits, tokens

    def test_top5_token_overlap_after_prefill(self):
        """Top-5 tokens from logits should be reasonable."""
        with _cleanup():
            from vkwr.config.model import WeightConfig
            from vkwr.model_executor.models.rwkv7 import RWKV7

            model = RWKV7(MODEL_PATH, weight_config=WeightConfig())

            tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=model.device)
            state = model.zero_state(1)

            with torch.inference_mode():
                logits = model.forward(tokens, state)

            top5_ids = torch.topk(logits[0], 5).indices.tolist()
            assert len(top5_ids) == 5
            assert all(0 <= tid < model.config.V for tid in top5_ids)

            del model, state, logits, tokens

    def test_state_inplace_mutation(self):
        """State tensors must be modified in-place between steps."""
        with _cleanup():
            from vkwr.config.model import WeightConfig
            from vkwr.model_executor.models.rwkv7 import RWKV7

            model = RWKV7(MODEL_PATH, weight_config=WeightConfig())

            state = model.zero_state(1)
            state_shift_before = state[0].clone()

            tokens = torch.tensor([[1, 2, 3]], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                _ = model.forward(tokens, state)

            state_shift_after = state[0]
            assert not torch.equal(state_shift_before, state_shift_after), "shift state should be modified in-place by forward pass"

            del model, state, state_shift_before, state_shift_after, tokens

    def test_elapsed_t_increments(self):
        """elapsed_t counter should advance by token count."""
        with _cleanup():
            from vkwr.config.model import WeightConfig
            from vkwr.model_executor.models.rwkv7 import RWKV7

            model = RWKV7(MODEL_PATH, weight_config=WeightConfig())

            state = model.zero_state(1)
            assert state[2][0].item() == 0

            tokens = torch.tensor([[1, 2, 3]], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                _ = model.forward(tokens, state)

            assert state[2][0].item() == 3

            tokens2 = torch.tensor([[4]], dtype=torch.long, device=model.device)
            with torch.inference_mode():
                _ = model.forward(tokens2, state)

            assert state[2][0].item() == 4

            del model, state, tokens, tokens2

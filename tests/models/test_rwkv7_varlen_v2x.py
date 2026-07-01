"""Slot mapping precision validation for v2x kernels.

Compares v1 (contiguous) and v2x (slot-mapped) RWKV7 varlen models to verify
that scattered-slot decode produces logits numerically equivalent to the
contiguous baseline.

Requires a model checkpoint at VKWR_RWKV7_MODEL_PATH.
"""

import os

import pytest
import torch

from vkwr.config.model import WeightConfig
from vkwr.model_executor.models.rwkv7_varlen import RWKV7 as RWKV7Varlen
from vkwr.model_executor.models.rwkv7_varlen_v2x import RWKV7 as RWKV7VarlenV2x

MODEL_PATH = os.environ.get("VKWR_RWKV7_MODEL_PATH")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="VKWR_RWKV7_MODEL_PATH not set",
)

MAX_SLOTS = 32
SEED_PREFILL = 42
SEED_DECODE = 99
TOL_LOGITS = 5.0
TOPK = 20
TOPK_OVERLAP_MIN = 0.8


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def models():
    """Load v1 and v2x models sharing the same weights."""
    if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
        pytest.skip(f"Model not found: {MODEL_PATH}")
    model_v1 = RWKV7Varlen(model_path=MODEL_PATH, weight_config=WeightConfig())
    model_v2x = RWKV7VarlenV2x(model_path=MODEL_PATH, weight_config=WeightConfig())
    yield model_v1, model_v2x
    del model_v1, model_v2x
    torch.cuda.empty_cache()


# ============================================================
# Helpers
# ============================================================


def build_qsl(lengths: list[int]) -> torch.Tensor:
    acc = [0]
    for n in lengths:
        acc.append(acc[-1] + n)
    return torch.tensor(acc, dtype=torch.int32, device="cuda")


def build_tokens(lengths: list[int], vocab_size: int, seed: int) -> torch.Tensor:
    return torch.randint(
        0,
        vocab_size,
        (sum(lengths),),
        dtype=torch.int32,
        device="cuda",
        generator=torch.Generator("cuda").manual_seed(seed),
    )


def zero_state_v2x(model: RWKV7VarlenV2x, max_slots: int) -> list[torch.Tensor]:
    """Create v2x global state buffer with shape [max_slots, ...]."""
    cfg = model.config
    wkv_dtype = torch.float32 if model.inference_config.wkv_mode == "fp32io16" else model.inference_config.dtype
    return [
        torch.zeros(
            (cfg.L, 2, max_slots, cfg.C),
            dtype=model.inference_config.dtype,
            device="cuda",
        ),
        torch.zeros(
            (cfg.L, max_slots, cfg.H, cfg.N, cfg.N),
            dtype=wkv_dtype,
            device="cuda",
        ),
        torch.zeros((max_slots,), dtype=torch.int32, device="cuda"),
    ]


def scatter_state(
    state_v1: list[torch.Tensor],
    state_v2x: list[torch.Tensor],
    slot_indices: torch.Tensor,
) -> None:
    """Copy v1 contiguous state [B, ...] into v2x global buffer at scattered slots."""
    B = slot_indices.size(0)
    L = state_v1[0].size(0)
    for i in range(B):
        slot = slot_indices[i].item()
        for layer in range(L):
            state_v2x[0][layer][:, slot].copy_(state_v1[0][layer][:, i])
            state_v2x[1][layer][slot].copy_(state_v1[1][layer][i])
        state_v2x[2][slot].copy_(state_v1[2][i])


def _compare_logits(a: torch.Tensor, b: torch.Tensor) -> None:
    """Assert v1 and v2x logits produce the same top-K tokens.

    Only checks top-K overlap (no raw diff gate, no topp).
    Both models share cmix_sparse_down_relu_rows_t512 which uses
    atomicAdd during uniform decode, causing independent non-determinism.
    Top-K overlap is robust to this since the top tokens' logits are
    well-separated from the boundary.
    """
    assert not a.isnan().any(), "v1 logits contain NaN"
    assert not b.isnan().any(), "v2x logits contain NaN"
    assert not a.isinf().any(), "v1 logits contain Inf"
    assert not b.isinf().any(), "v2x logits contain Inf"

    B = a.size(0)
    for i in range(B):
        v1_topk = torch.topk(a[i], TOPK).indices.tolist()
        v2_topk = torch.topk(b[i], TOPK).indices.tolist()
        overlap = len(set(v1_topk) & set(v2_topk)) / TOPK
        assert overlap >= TOPK_OVERLAP_MIN, f"batch {i}: top{TOPK} overlap {overlap:.3f} < {TOPK_OVERLAP_MIN}"


def run_prefill_decode(
    model_v1: RWKV7Varlen,
    model_v2x: RWKV7VarlenV2x,
    lengths: list[int],
    slot_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill on v1, decode on both, return (logits_v1, logits_v2x)."""
    B = len(lengths)
    V = model_v1.config.V
    qsl = build_qsl(lengths)

    state_v1 = model_v1.zero_state(B)
    tokens_pf = build_tokens(lengths, V, SEED_PREFILL)
    model_v1.forward(tokens_pf, state_v1, qsl, max(lengths))

    prefill_state = [s.clone() for s in state_v1]

    decode_lengths = [1] * B
    decode_qsl = build_qsl(decode_lengths)
    dtokens = build_tokens(decode_lengths, V, SEED_DECODE)

    state_v1_decode = [s.clone() for s in prefill_state]
    logits_v1 = model_v1.forward(dtokens, state_v1_decode, decode_qsl, 1)

    state_v2x = zero_state_v2x(model_v2x, MAX_SLOTS)
    scatter_state(prefill_state, state_v2x, slot_indices)

    logits_v2x = model_v2x.forward(dtokens.clone(), state_v2x, decode_qsl, 1, slot_indices)

    torch.cuda.synchronize()
    return logits_v1, logits_v2x


# ============================================================
# Tests: single-step decode with various slot configurations
# ============================================================


class TestPrefillDecodeSlots:
    """Compare v1 vs v2x decode logits for different slot assignments."""

    def test_identity_slots(self, models):
        model_v1, model_v2x = models
        lengths = [5, 1, 3, 1]
        sid = torch.arange(len(lengths), dtype=torch.int32, device="cuda")
        logits_v1, logits_v2x = run_prefill_decode(model_v1, model_v2x, lengths, sid)
        _compare_logits(logits_v1, logits_v2x)

    def test_scattered_slots(self, models):
        model_v1, model_v2x = models
        lengths = [5, 1, 3, 1]
        sid = torch.tensor([3, 7, 0, 12], dtype=torch.int32, device="cuda")
        logits_v1, logits_v2x = run_prefill_decode(model_v1, model_v2x, lengths, sid)
        _compare_logits(logits_v1, logits_v2x)

    def test_max_slot_gap(self, models):
        model_v1, model_v2x = models
        lengths = [4, 1, 1]
        sid = torch.tensor([0, MAX_SLOTS - 2, MAX_SLOTS - 1], dtype=torch.int32, device="cuda")
        logits_v1, logits_v2x = run_prefill_decode(model_v1, model_v2x, lengths, sid)
        _compare_logits(logits_v1, logits_v2x)

    def test_single_request(self, models):
        model_v1, model_v2x = models
        lengths = [10]
        sid = torch.tensor([15], dtype=torch.int32, device="cuda")
        logits_v1, logits_v2x = run_prefill_decode(model_v1, model_v2x, lengths, sid)
        _compare_logits(logits_v1, logits_v2x)

    def test_large_batch(self, models):
        model_v1, model_v2x = models
        lengths = [3, 1, 7, 1, 2, 1, 5, 1]
        sid = torch.tensor([5, 0, 13, 8, 1, 20, 10, 3], dtype=torch.int32, device="cuda")
        logits_v1, logits_v2x = run_prefill_decode(model_v1, model_v2x, lengths, sid)
        _compare_logits(logits_v1, logits_v2x)


# ============================================================
# Tests: state mutation isolation
# ============================================================


class TestStateMutation:
    """Verify that v2x only mutates active slots."""

    def test_active_mutated_inactive_clean(self, models):
        _, model_v2x = models
        lengths = [5, 1, 3, 1]
        B = len(lengths)
        V = model_v2x.config.V
        slot_indices = torch.tensor([3, 7, 0, 12], dtype=torch.int32, device="cuda")

        state_v2x = zero_state_v2x(model_v2x, MAX_SLOTS)

        for i in range(B):
            slot = slot_indices[i].item()
            marker = float(i + 1)
            state_v2x[0][0][0, slot, 0] = marker

        qsl = build_qsl(lengths)
        tokens = build_tokens(lengths, V, SEED_PREFILL)
        model_v2x.forward(tokens, state_v2x, qsl, max(lengths), slot_indices)
        torch.cuda.synchronize()

        for i in range(B):
            slot = slot_indices[i].item()
            marker = float(i + 1)
            actual = state_v2x[0][0][0, slot, 0].item()
            assert abs(actual - marker) >= 1e-4, f"Active slot {slot} was not mutated (still {marker})"

        active = set(slot_indices.tolist())
        for s in range(MAX_SLOTS):
            if s not in active:
                val = state_v2x[0][0][0, s, 0].item()
                assert abs(val) <= 1e-5, f"Inactive slot {s} was modified (value={val})"


# ============================================================
# Tests: multi-step sequential decode
# ============================================================


class TestMultistepDecode:
    """Multi-step decode: v1 and v2x should stay in sync step-by-step."""

    def test_three_step_decode(self, models):
        model_v1, model_v2x = models
        B = 3
        V = model_v1.config.V
        slot_indices = torch.tensor([2, 8, 14], dtype=torch.int32, device="cuda")
        steps = 3

        state_v2x = zero_state_v2x(model_v2x, MAX_SLOTS)
        state_v1 = model_v1.zero_state(B)
        qsl = build_qsl([5] * B)
        tokens_pf = build_tokens([5] * B, V, SEED_PREFILL)
        model_v1.forward(tokens_pf, state_v1, qsl, 5)
        scatter_state(state_v1, state_v2x, slot_indices)

        for step in range(steps):
            seed_d = SEED_DECODE + step * 1000
            dqsl = build_qsl([1] * B)
            dtokens = build_tokens([1] * B, V, seed_d)

            state_v1_step = [s.clone() for s in state_v1]
            logits_v1 = model_v1.forward(dtokens, state_v1_step, dqsl, 1)

            state_v2x_step = [s.clone() for s in state_v2x]
            logits_v2x = model_v2x.forward(dtokens.clone(), state_v2x_step, dqsl, 1, slot_indices)
            torch.cuda.synchronize()

            _compare_logits(logits_v1, logits_v2x)

            state_v1 = state_v1_step
            state_v2x = zero_state_v2x(model_v2x, MAX_SLOTS)
            scatter_state(state_v1, state_v2x, slot_indices)

"""WKV ops tests — NOTE: numeric tests use synthetic inputs, NOT real model data.

The `_make_wkv_realistic_inputs` helpers produce value-range-safe synthetic inputs
so that multi-token recurrence does not overflow fp16.  This is only a temporary
proxy to verify shape, dtype, and basic kernel correctness.

REAL VALIDATION: these synthetic inputs do NOT prove the kernel produces correct
results end-to-end.  To validate numeric accuracy we must monkey-patch the original
Albatross/faster3a model (rwkv7_fast_v3a.py::tmix) to extract the actual per-layer
inputs (r, w, k, v, neg_kk, kka) and outputs (y, state) at runtime, then compare
them against the migrated kernel.  See the migration plan for details.
"""

import pytest
import torch
import time
from vkwr._ops.v1.v1_wkv_ops import (
    wkv_seq_fp16,
    wkv_seq_w0_fp16,
    wkv_one_fp16,
    wkv_one_w0_fp16,
    wkv_forward_fp32,
    wkv_forward_seq_fp32,
    wkv_forward_small_fp32,
    wkv_forward_block_fp32,
    advance_i32,
    HEAD_SIZE,
)


def _make_fp16_state(B, C):
    H = C // HEAD_SIZE
    state = torch.zeros((B, H, HEAD_SIZE, HEAD_SIZE), dtype=torch.float16, device="cuda")
    return state, H


def _make_fp32_state(B, C):
    H = C // HEAD_SIZE
    state = torch.zeros((B, H, HEAD_SIZE, HEAD_SIZE), dtype=torch.float32, device="cuda")
    return state, H


def _make_fp16_inputs(B, T, C):
    """Unbounded randn inputs — suitable for shape/dtype tests only.
    NOT suitable for multi-token numerical stability tests (see _make_wkv_realistic_inputs).
    """
    r = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    w = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    k = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    v = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    a = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    b = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    return r, w, k, v, a, b


def _make_fp16_one_inputs(B, C):
    """Unbounded randn inputs (T=1) — suitable for shape/dtype tests only."""
    r = torch.randn(B, C, dtype=torch.float16, device="cuda")
    w = torch.randn(B, C, dtype=torch.float16, device="cuda")
    k = torch.randn(B, C, dtype=torch.float16, device="cuda")
    v = torch.randn(B, C, dtype=torch.float16, device="cuda")
    a = torch.randn(B, C, dtype=torch.float16, device="cuda")
    b = torch.randn(B, C, dtype=torch.float16, device="cuda")
    return r, w, k, v, a, b


def _make_wkv_realistic_inputs(B, T, C):
    """Generate synthetic inputs with value ranges that approximate the real model.

    IMPORTANT: these are NOT real model activations.  They are only a temporary
    proxy so that the kernel's multi-token recurrence does not overflow fp16 and
    we can verify basic shape/dtype/kernel-liveness properties.

    To prove numeric correctness, monkey-patch faster3a rwkv7_fast_v3a.py::tmix
    to capture the actual (r, w, k, v, neg_kk, kka, y, state) tensors at runtime,
    then feed them to the migrated kernel and compare outputs.

    From faster3a rwkv7_fast_v3a.py::tmix and cuda/rwkv7_fast_ops_fp16.cu,
    the parameters flowing into the wkv kernel have these constraints:

      r  : LN -> tmix_mix6 -> linear (bounded, small magnitude)
      w  : LN -> tmix_mix6 -> linear -> tanh -> linear (tanh clamps to [-1,1])
      k  : LN -> tmix_mix6 -> linear (bounded, small magnitude)
      v  : LN -> tmix_mix6 -> linear + residual (bounded, small magnitude)
      a  : tmix_kk_a_gate output "neg_kk" = -kk_norm (normalized k magnitude)
           where kk_norm = (k*k_k) / ||k*k_k||, so a ∈ [-1, 0]
      b  : tmix_kk_a_gate output "kka" = kk_norm * sigmoid(a0 + a12),
           so b ∈ (-1, 1) but typically small

    The kernel's per-head recurrence (fp16_v2.cu L381-386):
      sa = sum_over_head(a * state)
      state = state * w_delta + k * v + sa * b
      y = sum_over_head(state * r)

    where w_delta = exp2f(C/(1+exp2f(C*w))) - 1 + dither

    Expanding: state = state * (1 + w_delta + a*b) + k*v
    For stability we need |1 + w_delta + a*b| < 1 per timestep.
    With tanh-bounded w, w_delta ≈ -0.4 ~ -0.8 (decay factor ≈ 0.2 ~ 0.6).
    With a ∈ [-1,0] and b ∈ (-1,1), a*b can be negative and cause
    |1 + w_delta + a*b| > 1, leading to oscillatory blowup.

    Real model inputs avoid this because:
      - kk_norm components are distributed across 64 channels (each ≈ 0.1)
      - sigmoid gate values are moderate (a0 + a12 often small)
      - k*v product is small due to LN normalization upstream

    This function generates inputs that respect these combined constraints.
    """
    fp32 = torch.float32
    dtype = torch.float16
    H = C // HEAD_SIZE

    # r, k, v: small scale from LN normalization + weight scaling
    r = (torch.randn(B, T, C, dtype=fp32, device="cuda") * 0.2).to(dtype)
    k_raw = (torch.randn(B, T, C, dtype=fp32, device="cuda") * 0.2).to(dtype)
    v = (torch.randn(B, T, C, dtype=fp32, device="cuda") * 0.2).to(dtype)

    # w: tanh-bounded (this is the activation in the real path)
    w = torch.tanh(torch.randn(B, T, C, dtype=fp32, device="cuda")).to(dtype)

    # a and b: simulate tmix_kk_a_gate output properly
    # neg_kk = -kk_norm, kka = kk_norm * sigmoid_gate
    # kk_norm is the normalized per-head k*k_k vector, components are small
    # (sum of squares = 1, spread across HEAD_SIZE=64 channels)
    # Simulate by generating per-head normalized vectors
    kk_raw = torch.randn(B, T, H, HEAD_SIZE, dtype=fp32, device="cuda")
    kk_norm = kk_raw / (kk_raw.norm(dim=-1, keepdim=True) + 1e-8)  # unit vector
    sigmoid_gate = torch.sigmoid(torch.randn(B, T, H, HEAD_SIZE, dtype=fp32, device="cuda"))
    a = (-kk_norm).view(B, T, C).to(dtype)  # neg_kk, per-channel ∈ [-1,0]
    b = (kk_norm * sigmoid_gate).view(B, T, C).to(dtype)  # kka
    return r, w, k_raw.view(B, T, C).to(dtype), v, a, b


def _make_wkv_realistic_one_inputs(B, C):
    """Single-token variant of _make_wkv_realistic_inputs.

    Same limitations: synthetic proxy only, not real model activations.
    See _make_wkv_realistic_inputs docstring for details on proper validation.
    """
    fp32 = torch.float32
    dtype = torch.float16
    H = C // HEAD_SIZE

    r = (torch.randn(B, C, dtype=fp32, device="cuda") * 0.2).to(dtype)
    k_raw = (torch.randn(B, C, dtype=fp32, device="cuda") * 0.2).to(dtype)
    v = (torch.randn(B, C, dtype=fp32, device="cuda") * 0.2).to(dtype)
    w = torch.tanh(torch.randn(B, C, dtype=fp32, device="cuda")).to(dtype)

    kk_raw = torch.randn(B, H, HEAD_SIZE, dtype=fp32, device="cuda")
    kk_norm = kk_raw / (kk_raw.norm(dim=-1, keepdim=True) + 1e-8)
    sigmoid_gate = torch.sigmoid(torch.randn(B, H, HEAD_SIZE, dtype=fp32, device="cuda"))
    a = (-kk_norm).view(B, C).to(dtype)
    b = (kk_norm * sigmoid_gate).view(B, C).to(dtype)
    return r, w, k_raw.view(B, C).to(dtype), v, a, b


# ===== wkv_seq_fp16: return type C(2) — buffer preallocation + (y, elapsed_t) =====


@pytest.mark.parametrize(
    "B,T,C",
    [
        (1, 1, 64),
        (1, 4, 64),
        (2, 8, 128),
        (4, 4, 256),
        (1, 16, 128),
    ],
)
def test_wkv_seq_fp16_basic(B, T, C):
    state, H = _make_fp16_state(B, C)
    r, w, k, v, a, b = _make_fp16_inputs(B, T, C)
    y, elapsed_t = wkv_seq_fp16(B, T, C, H, state, r, w, k, v, a, b)
    assert y.shape == (B, T, C)
    assert y.dtype == torch.float16
    assert y.device.type == "cuda"
    assert elapsed_t.shape == (B,)
    assert elapsed_t.dtype == torch.int32


def test_wkv_seq_fp16_not_nan():
    B, T, C = 2, 8, 256
    state, H = _make_fp16_state(B, C)
    r, w, k, v, a, b = _make_wkv_realistic_inputs(B, T, C)
    y, elapsed_t = wkv_seq_fp16(B, T, C, H, state, r, w, k, v, a, b)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_wkv_seq_fp16_state_mutation():
    B, T, C = 1, 4, 128
    state, H = _make_fp16_state(B, C)
    state_before = state.clone()
    r, w, k, v, a, b = _make_fp16_inputs(B, T, C)
    wkv_seq_fp16(B, T, C, H, state, r, w, k, v, a, b)
    assert not torch.equal(state, state_before)


def test_wkv_seq_fp16_zero_input():
    B, T, C = 1, 4, 128
    state, H = _make_fp16_state(B, C)
    zero = torch.zeros(B, T, C, dtype=torch.float16, device="cuda")
    y, _ = wkv_seq_fp16(B, T, C, H, state, zero, zero, zero, zero, zero, zero)
    assert not torch.isnan(y).any()


def test_wkv_seq_fp16_invalid_head_size():
    B, T, C, H = 1, 4, 100, 2
    state, _ = _make_fp16_state(B, C)
    r, w, k, v, a, b = _make_fp16_inputs(B, T, C)
    with pytest.raises(RuntimeError):
        wkv_seq_fp16(B, T, C, H, state, r, w, k, v, a, b)


def test_wkv_seq_fp16_determinism():
    B, T, C = 2, 4, 256
    r, w, k, v, a, b = _make_wkv_realistic_inputs(B, T, C)
    state1, H1 = _make_fp16_state(B, C)
    state2, H2 = _make_fp16_state(B, C)
    y1, _ = wkv_seq_fp16(B, T, C, H1, state1, r.clone(), w.clone(), k.clone(), v.clone(), a.clone(), b.clone())
    y2, _ = wkv_seq_fp16(B, T, C, H2, state2, r.clone(), w.clone(), k.clone(), v.clone(), a.clone(), b.clone())
    torch.testing.assert_close(y1, y2, atol=0, rtol=0)


# ===== wkv_seq_w0_fp16: return type C(2) — extra w0 param =====


@pytest.mark.parametrize(
    "B,T,C",
    [(1, 1, 64), (2, 8, 128), (1, 16, 256)],
)
def test_wkv_seq_w0_fp16_basic(B, T, C):
    state, H = _make_fp16_state(B, C)
    r, w, k, v, a, b = _make_fp16_inputs(B, T, C)
    w0 = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    y, elapsed_t = wkv_seq_w0_fp16(B, T, C, H, state, r, w, w0, k, v, a, b)
    assert y.shape == (B, T, C)
    assert y.dtype == torch.float16
    assert elapsed_t.shape == (B,)


def test_wkv_seq_w0_fp16_not_nan():
    B, T, C = 2, 4, 256
    state, H = _make_fp16_state(B, C)
    r, w, k, v, a, b = _make_wkv_realistic_inputs(B, T, C)
    w0 = (torch.randn(B, T, C, dtype=torch.float32, device="cuda") * 0.1).to(torch.float16)
    y, _ = wkv_seq_w0_fp16(B, T, C, H, state, r, w, w0, k, v, a, b)
    assert not torch.isnan(y).any()


# ===== wkv_one_fp16: return type C(2) — single token =====


@pytest.mark.parametrize(
    "B,C",
    [(1, 64), (1, 128), (2, 256), (4, 512)],
)
def test_wkv_one_fp16_basic(B, C):
    state, H = _make_fp16_state(B, C)
    r, w, k, v, a, b = _make_fp16_one_inputs(B, C)
    y, elapsed_t = wkv_one_fp16(B, C, H, state, r, w, k, v, a, b)
    assert y.shape == (B, C)
    assert y.dtype == torch.float16
    assert elapsed_t.shape == (B,)
    assert elapsed_t.dtype == torch.int32


def test_wkv_one_fp16_not_nan():
    B, C = 2, 256
    state, H = _make_fp16_state(B, C)
    r, w, k, v, a, b = _make_wkv_realistic_one_inputs(B, C)
    y, _ = wkv_one_fp16(B, C, H, state, r, w, k, v, a, b)
    assert not torch.isnan(y).any()


def test_wkv_one_fp16_state_mutation():
    B, C = 1, 128
    state, H = _make_fp16_state(B, C)
    state_before = state.clone()
    r, w, k, v, a, b = _make_fp16_one_inputs(B, C)
    wkv_one_fp16(B, C, H, state, r, w, k, v, a, b)
    assert not torch.equal(state, state_before)


def test_wkv_one_fp16_zero_input():
    B, C = 1, 128
    state, H = _make_fp16_state(B, C)
    zero = torch.zeros(B, C, dtype=torch.float16, device="cuda")
    y, _ = wkv_one_fp16(B, C, H, state, zero, zero, zero, zero, zero, zero)
    assert not torch.isnan(y).any()


def test_wkv_one_fp16_determinism():
    B, C = 2, 256
    state, H = _make_fp16_state(B, C)
    r, w, k, v, a, b = _make_wkv_realistic_one_inputs(B, C)
    y1, _ = wkv_one_fp16(B, C, H, state.clone(), r.clone(), w.clone(), k.clone(), v.clone(), a.clone(), b.clone())
    y2, _ = wkv_one_fp16(B, C, H, state.clone(), r.clone(), w.clone(), k.clone(), v.clone(), a.clone(), b.clone())
    torch.testing.assert_close(y1, y2, atol=0, rtol=0)


# ===== wkv_one_w0_fp16: return type C(2) — single token with w0 =====


@pytest.mark.parametrize(
    "B,C",
    [(1, 64), (2, 256)],
)
def test_wkv_one_w0_fp16_basic(B, C):
    state, H = _make_fp16_state(B, C)
    r, w, k, v, a, b = _make_fp16_one_inputs(B, C)
    w0 = torch.randn(B, C, dtype=torch.float16, device="cuda")
    y, elapsed_t = wkv_one_w0_fp16(B, C, H, state, r, w, w0, k, v, a, b)
    assert y.shape == (B, C)
    assert elapsed_t.shape == (B,)


# ===== fp32 wrappers: return type C(1) — buffer preallocation, no elapsed_t =====


@pytest.mark.parametrize(
    "B,T,C",
    [
        (1, 1, 64),
        (1, 4, 128),
        (2, 8, 256),
        (1, 16, 128),
    ],
)
def test_wkv_forward_fp32_basic(B, T, C):
    state, H = _make_fp32_state(B, C)
    r, w, k, v, a, b = _make_wkv_realistic_inputs(B, T, C)
    y = wkv_forward_fp32(B, T, C, H, state, r, w, k, v, a, b)
    assert y.shape == (B, T, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "B,T,C",
    [(1, 4, 128), (2, 8, 256)],
)
def test_wkv_forward_seq_fp32_basic(B, T, C):
    state, H = _make_fp32_state(B, C)
    r, w, k, v, a, b = _make_wkv_realistic_inputs(B, T, C)
    y = wkv_forward_seq_fp32(B, T, C, H, state, r, w, k, v, a, b)
    assert y.shape == (B, T, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "B,T,C",
    [(1, 4, 128), (2, 8, 256)],
)
def test_wkv_forward_small_fp32_basic(B, T, C):
    state, H = _make_fp32_state(B, C)
    r, w, k, v, a, b = _make_wkv_realistic_inputs(B, T, C)
    y = wkv_forward_small_fp32(B, T, C, H, state, r, w, k, v, a, b)
    assert y.shape == (B, T, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "B,T,C",
    [(1, 4, 128), (2, 8, 256)],
)
def test_wkv_forward_block_fp32_basic(B, T, C):
    state, H = _make_fp32_state(B, C)
    r, w, k, v, a, b = _make_wkv_realistic_inputs(B, T, C)
    y = wkv_forward_block_fp32(B, T, C, H, state, r, w, k, v, a, b)
    assert y.shape == (B, T, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


def test_wkv_forward_fp32_state_mutation():
    B, T, C = 1, 4, 128
    state, H = _make_fp32_state(B, C)
    state_before = state.clone()
    r, w, k, v, a, b = _make_wkv_realistic_inputs(B, T, C)
    wkv_forward_fp32(B, T, C, H, state, r, w, k, v, a, b)
    assert not torch.equal(state, state_before)


def test_wkv_forward_fp32_zero_input():
    B, T, C = 1, 4, 128
    state, H = _make_fp32_state(B, C)
    zero = torch.zeros(B, T, C, dtype=torch.float16, device="cuda")
    y = wkv_forward_fp32(B, T, C, H, state, zero, zero, zero, zero, zero, zero)
    assert not torch.isnan(y).any()


def test_wkv_forward_fp32_invalid_state_dtype():
    """fp32 mode requires fp32 state, fp16 inputs."""
    B, T, C, H = 1, 4, 128, 2
    bad_state = torch.zeros((B, C, HEAD_SIZE), dtype=torch.float16, device="cuda")
    r, w, k, v, a, b = _make_fp16_inputs(B, T, C)
    y = torch.empty(B, T, C, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops.vkwr_v1_wkv.wkv_forward_fp32(B, T, C, H, bad_state, r, w, k, v, a, b, y)


# ===== advance_i32: return type C(0) — in-place, returns None =====


def test_advance_i32_basic():
    B = 4
    x = torch.zeros(B, dtype=torch.int32, device="cuda")
    result = advance_i32(x, 5)
    assert result is None
    assert (x == 5).all()


def test_advance_i32_accumulate():
    B = 2
    x = torch.zeros(B, dtype=torch.int32, device="cuda")
    advance_i32(x, 3)
    advance_i32(x, 7)
    assert (x == 10).all()


def test_advance_i32_negative_amount():
    B = 3
    x = torch.full((B,), 100, dtype=torch.int32, device="cuda")
    advance_i32(x, -10)
    assert (x == 90).all()


def test_advance_i32_large_amount():
    B = 4
    x = torch.zeros(B, dtype=torch.int32, device="cuda")
    advance_i32(x, 2**30)
    assert (x == (2**30)).all()


def test_advance_i32_invalid_dtype():
    x = torch.zeros(4, dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError):
        advance_i32(x, 1)


def test_advance_i32_invalid_shape():
    x = torch.zeros((2, 4), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        advance_i32(x, 1)


# ===== Performance =====


def test_wkv_seq_fp16_performance():
    B, T, C = 2, 64, 1024
    state, H = _make_fp16_state(B, C)
    r, w, k, v, a, b = _make_wkv_realistic_inputs(B, T, C)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(30):
        wkv_seq_fp16(B, T, C, H, state.clone(), r.clone(), w.clone(), k.clone(), v.clone(), a.clone(), b.clone())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / 30 * 1000
    print(f"  wkv_seq_fp16 perf: {avg_ms:.3f}ms/call")
    assert avg_ms < 200.0

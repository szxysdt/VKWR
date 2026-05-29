import pytest
import torch

from vkwr._ops.v1.v1_norm_ops import (
    add_f16,
    add_last_layer_norm_f16,
    add_layer_norm_cmix_mix_f16,
    add_layer_norm_cmix_mix_f16_cfg,
    add_layer_norm_cmix_mix_f16_scalar_stats,
    add_layer_norm_f16,
    add_layer_norm_tmix_mix6_f16,
    add_layer_norm_tmix_mix6_f16_cfg,
    add_layer_norm_tmix_mix6_f16_scalar_stats,
    emb_ln0_bf16_to_f16,
    layer_norm_f16,
    layer_norm_f16_small,
    layer_norm_f16_small512,
)


def _make_ln_inputs(B, T, C):
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    return x, weight, bias


# ===== layer_norm_f16: return type A =====


@pytest.mark.parametrize(
    "B,T,C",
    [
        (1, 1, 64),
        (2, 8, 256),
        (1, 16, 1024),
        (4, 4, 4096),
    ],
)
def test_layer_norm_f16_basic(B, T, C):
    x, weight, bias = _make_ln_inputs(B, T, C)
    y = layer_norm_f16(x, weight, bias)
    assert y.shape == x.shape
    assert y.dtype == torch.float16


def test_layer_norm_f16_not_nan():
    x, weight, bias = _make_ln_inputs(2, 8, 512)
    y = layer_norm_f16(x, weight, bias)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_layer_norm_f16_1d_input():
    C = 256
    x = torch.randn(C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    y = layer_norm_f16(x, weight, bias)
    assert y.shape == (C,)


def test_layer_norm_f16_invalid_C():
    C = 10000
    x = torch.randn(1, 1, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        layer_norm_f16(x, weight, bias)


# ===== layer_norm_f16_small: return type A (C=4096 only) =====


def test_layer_norm_f16_small_basic():
    C = 4096
    x = torch.randn(2, 8, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    y = layer_norm_f16_small(x, weight, bias)
    assert y.shape == x.shape
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


def test_layer_norm_f16_small_invalid_C():
    C = 2048
    x = torch.randn(1, 1, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        layer_norm_f16_small(x, weight, bias)


# ===== layer_norm_f16_small512: return type A (C=4096 only) =====


def test_layer_norm_f16_small512_basic():
    C = 4096
    x = torch.randn(2, 8, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    y = layer_norm_f16_small512(x, weight, bias)
    assert y.shape == x.shape
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


def test_layer_norm_f16_small512_invalid_C():
    C = 2048
    x = torch.randn(1, 1, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        layer_norm_f16_small512(x, weight, bias)


# ===== emb_ln0_bf16_to_f16: return type A (bf16 input -> fp16 output) =====


@pytest.mark.parametrize(
    "V,C",
    [(1024, 256), (4096, 1024)],
)
def test_emb_ln0_bf16_to_f16_basic(V, C):
    emb = torch.randn(V, C, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(C, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros(C, dtype=torch.bfloat16, device="cuda")
    y = emb_ln0_bf16_to_f16(emb, weight, bias)
    assert y.shape == (V, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


def test_emb_ln0_bf16_to_f16_invalid_dtype():
    V, C = 1024, 256
    emb = torch.randn(V, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        emb_ln0_bf16_to_f16(emb, weight, bias)


# ===== add_f16: return type A =====


@pytest.mark.parametrize(
    "shape",
    [(128,), (4, 256), (2, 8, 512)],
)
def test_add_f16_basic(shape):
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    y_in = torch.randn(shape, dtype=torch.float16, device="cuda")
    y = add_f16(x, y_in)
    assert y.shape == x.shape
    assert y.dtype == torch.float16
    expected = (x + y_in).to(torch.float16)
    torch.testing.assert_close(y, expected, atol=1e-2, rtol=1e-2)


def test_add_f16_zero():
    shape = (4, 256)
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    zero = torch.zeros(shape, dtype=torch.float16, device="cuda")
    y = add_f16(x, zero)
    torch.testing.assert_close(y, x, atol=0, rtol=0)


# ===== add_layer_norm_f16: return type B(2) — [x_out, normed] =====


@pytest.mark.parametrize(
    "B,T,C",
    [(1, 1, 64), (2, 8, 256), (1, 16, 1024)],
)
def test_add_layer_norm_f16_basic(B, T, C):
    x, weight, bias = _make_ln_inputs(B, T, C)
    residual = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    x_out, normed = add_layer_norm_f16(x, residual, weight, bias)
    assert len((x_out, normed)) == 2
    assert x_out.shape == x.shape
    assert normed.shape == x.shape
    assert x_out.dtype == torch.float16
    assert normed.dtype == torch.float16


def test_add_layer_norm_f16_not_nan():
    B, T, C = 2, 8, 512
    x, weight, bias = _make_ln_inputs(B, T, C)
    residual = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    x_out, normed = add_layer_norm_f16(x, residual, weight, bias)
    assert not torch.isnan(x_out).any()
    assert not torch.isnan(normed).any()


def test_add_layer_norm_f16_invalid_C():
    C = 10000
    x = torch.randn(1, 1, C, dtype=torch.float16, device="cuda")
    residual = torch.randn(1, 1, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        add_layer_norm_f16(x, residual, weight, bias)


# ===== add_last_layer_norm_f16: return type A (squeezes T dim) =====


def test_add_last_layer_norm_f16_basic():
    B, T, C = 2, 8, 256
    x, weight, bias = _make_ln_inputs(B, T, C)
    residual = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    y = add_last_layer_norm_f16(x, residual, weight, bias)
    assert y.shape == (B, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


# ===== add_layer_norm_cmix_mix_f16: return type B(2) — [x_out, mixed] =====


@pytest.mark.parametrize("B", [1, 2, 4])
def test_add_layer_norm_cmix_mix_f16_basic(B):
    C = 256
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    residual = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(B, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    x_out, mixed = add_layer_norm_cmix_mix_f16(x, residual, shift_state, weight, bias, x_k)
    assert len((x_out, mixed)) == 2
    assert x_out.shape == x.shape
    assert mixed.shape == x.shape
    assert x_out.dtype == torch.float16
    assert mixed.dtype == torch.float16


def test_add_layer_norm_cmix_mix_f16_not_nan():
    B, C = 2, 512
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    residual = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(B, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    x_out, mixed = add_layer_norm_cmix_mix_f16(x, residual, shift_state, weight, bias, x_k)
    assert not torch.isnan(x_out).any()
    assert not torch.isnan(mixed).any()


# ===== add_layer_norm_cmix_mix_f16_cfg: return type B(2), C=4096 only =====


@pytest.mark.parametrize("threads", [256, 512, 1024])
def test_add_layer_norm_cmix_mix_f16_cfg_basic(threads):
    B, C = 2, 4096
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    residual = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(B, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    x_out, mixed = add_layer_norm_cmix_mix_f16_cfg(x, residual, shift_state, weight, bias, x_k, 1e-5, threads)
    assert len((x_out, mixed)) == 2
    assert x_out.shape == x.shape
    assert mixed.shape == x.shape


def test_add_layer_norm_cmix_mix_f16_cfg_invalid_threads():
    B, C = 1, 4096
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    residual = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(B, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        add_layer_norm_cmix_mix_f16_cfg(x, residual, shift_state, weight, bias, x_k, 1e-5, 64)


# ===== add_layer_norm_cmix_mix_f16_scalar_stats: return type B(2), C=4096 only =====


def test_add_layer_norm_cmix_mix_f16_scalar_stats_basic():
    B, C = 2, 4096
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    residual = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(B, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    x_out, mixed = add_layer_norm_cmix_mix_f16_scalar_stats(x, residual, shift_state, weight, bias, x_k)
    assert len((x_out, mixed)) == 2
    assert x_out.shape == x.shape
    assert mixed.shape == x.shape
    assert not torch.isnan(x_out).any()
    assert not torch.isnan(mixed).any()


# ===== add_layer_norm_tmix_mix6_f16: return type B(7) — [x_out, r, w, k, v, a, g] =====


@pytest.mark.parametrize("B", [1, 2])
def test_add_layer_norm_tmix_mix6_f16_basic(B):
    C = 256
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    residual = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(B, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    x_r = torch.randn(C, dtype=torch.float16, device="cuda")
    x_w = torch.randn(C, dtype=torch.float16, device="cuda")
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    x_v = torch.randn(C, dtype=torch.float16, device="cuda")
    x_a = torch.randn(C, dtype=torch.float16, device="cuda")
    x_g = torch.randn(C, dtype=torch.float16, device="cuda")
    results = add_layer_norm_tmix_mix6_f16(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g)
    assert len(results) == 7
    for r in results:
        assert r.shape == x.shape
        assert r.dtype == torch.float16
        assert not torch.isnan(r).any()


def test_add_layer_norm_tmix_mix6_f16_return_count():
    B, C = 1, 128
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    residual = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(B, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    results = add_layer_norm_tmix_mix6_f16(x, residual, shift_state, weight, bias, *vecs)
    assert len(results) == 7, "add_layer_norm_tmix_mix6_f16 must return 7 tensors"


# ===== add_layer_norm_tmix_mix6_f16_cfg: return type B(7), C=4096 only =====


@pytest.mark.parametrize("threads", [256, 512, 1024])
def test_add_layer_norm_tmix_mix6_f16_cfg_basic(threads):
    B, C = 1, 4096
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    residual = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(B, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    results = add_layer_norm_tmix_mix6_f16_cfg(x, residual, shift_state, weight, bias, *vecs, 1e-5, threads)
    assert len(results) == 7
    for r in results:
        assert r.shape == x.shape
        assert r.dtype == torch.float16


# ===== add_layer_norm_tmix_mix6_f16_scalar_stats: return type B(7), C=4096 only =====


def test_add_layer_norm_tmix_mix6_f16_scalar_stats_basic():
    B, C = 1, 4096
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    residual = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(B, C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    results = add_layer_norm_tmix_mix6_f16_scalar_stats(x, residual, shift_state, weight, bias, *vecs)
    assert len(results) == 7
    for r in results:
        assert r.shape == x.shape
        assert r.dtype == torch.float16
        assert not torch.isnan(r).any()

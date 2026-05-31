import pytest
import torch

from vkwr._ops.v1.v1_rank_ops import (
    linear_wag_rank_in_f16,
    linear_wag_rank_out_f16,
    linear_wagv_rank_in_f16,
    linear_wagv_rank_out_f16,
)

# ===== linear_wag_rank_in_f16: return type B(3) — [w1, a1, g1] =====


@pytest.mark.parametrize(
    "shape,K,Rw,Ra,Rg",
    [
        ((2, 128), 128, 64, 64, 64),
        ((4, 256), 256, 128, 64, 128),
        ((2, 3, 256), 256, 128, 64, 128),
    ],
)
def test_linear_wag_rank_in_f16_basic(shape, K, Rw, Ra, Rg):
    xw = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    w1, a1, g1 = linear_wag_rank_in_f16(xw, xa, xg, w1_t, a1_t, g1_t)
    expected_w = list(shape) + [Rw]
    expected_a = list(shape) + [Ra]
    expected_g = list(shape) + [Rg]
    assert w1.shape == tuple(expected_w)
    assert a1.shape == tuple(expected_a)
    assert g1.shape == tuple(expected_g)
    assert w1.dtype == torch.float16
    assert a1.dtype == torch.float16
    assert g1.dtype == torch.float16


def test_linear_wag_rank_in_f16_not_nan():
    shape = (2, 256)
    K = shape[-1]
    Rw, Ra, Rg = 128, 128, 128
    xw = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    w1, a1, g1 = linear_wag_rank_in_f16(xw, xa, xg, w1_t, a1_t, g1_t)
    assert not torch.isnan(w1).any()
    assert not torch.isnan(a1).any()
    assert not torch.isnan(g1).any()


def test_linear_wag_rank_in_f16_return_count():
    shape = (1, 64)
    K = shape[-1]
    Rw, Ra, Rg = 32, 32, 32
    xw = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    results = linear_wag_rank_in_f16(xw, xa, xg, w1_t, a1_t, g1_t)
    assert len(results) == 3, "linear_wag_rank_in_f16 must return 3 tensors"


def test_linear_wag_rank_in_f16_invalid_shape():
    K = 64
    Rw, Ra, Rg = 32, 32, 32
    xw = torch.randn(3, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(2, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(2, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        linear_wag_rank_in_f16(xw, xa, xg, w1_t, a1_t, g1_t)


# ===== linear_wagv_rank_in_f16: return type B(4) — [w1, a1, g1, v_out] =====


@pytest.mark.parametrize(
    "shape,K,Rw,Ra,Rg,Rv",
    [
        ((2, 128), 128, 64, 64, 64, 64),
        ((2, 3, 256), 256, 128, 64, 128, 96),
    ],
)
def test_linear_wagv_rank_in_f16_basic(shape, K, Rw, Ra, Rg, Rv):
    xw = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xv = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    v1_t = torch.randn(Rv, K, dtype=torch.float16, device="cuda")
    w1, a1, g1, v_out = linear_wagv_rank_in_f16(xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t)
    expected_w = list(shape) + [Rw]
    expected_a = list(shape) + [Ra]
    expected_g = list(shape) + [Rg]
    expected_v = list(shape) + [Rv]
    assert w1.shape == tuple(expected_w)
    assert a1.shape == tuple(expected_a)
    assert g1.shape == tuple(expected_g)
    assert v_out.shape == tuple(expected_v)
    for t in (w1, a1, g1, v_out):
        assert t.dtype == torch.float16


def test_linear_wagv_rank_in_f16_not_nan():
    shape = (2, 128)
    K = shape[-1]
    Rw, Ra, Rg, Rv = 64, 64, 64, 64
    tensors = [torch.randn(*shape, K, dtype=torch.float16, device="cuda") for _ in range(4)]
    weights = [torch.randn(r, K, dtype=torch.float16, device="cuda") for r in (Rw, Ra, Rg, Rv)]
    w1, a1, g1, v_out = linear_wagv_rank_in_f16(*tensors, *weights)
    assert not torch.isnan(w1).any()
    assert not torch.isnan(a1).any()
    assert not torch.isnan(g1).any()
    assert not torch.isnan(v_out).any()


def test_linear_wagv_rank_in_f16_return_count():
    shape = (1, 64)
    K = shape[-1]
    Rw, Ra, Rg, Rv = 32, 32, 32, 32
    xw = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    xv = torch.randn(*shape, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    v1_t = torch.randn(Rv, K, dtype=torch.float16, device="cuda")
    results = linear_wagv_rank_in_f16(xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t)
    assert len(results) == 4, "linear_wagv_rank_in_f16 must return 4 tensors"


# ===== linear_wag_rank_out_f16: return type B(3) — [w2, a2, g2] =====


@pytest.mark.parametrize(
    "shape,C,Kw,Ka,Kg",
    [
        ((2, 128), 256, 128, 128, 128),
        ((4, 256), 512, 256, 128, 256),
        ((2, 3, 128), 256, 128, 128, 128),
    ],
)
def test_linear_wag_rank_out_f16_basic(shape, C, Kw, Ka, Kg):
    w1 = torch.randn(*shape, Kw, dtype=torch.float16, device="cuda")
    a1 = torch.randn(*shape, Ka, dtype=torch.float16, device="cuda")
    g1 = torch.randn(*shape, Kg, dtype=torch.float16, device="cuda")
    w2_t = torch.randn(C, Kw, dtype=torch.float16, device="cuda")
    a2_t = torch.randn(C, Ka, dtype=torch.float16, device="cuda")
    g2_t = torch.randn(C, Kg, dtype=torch.float16, device="cuda")
    w2, a2, g2 = linear_wag_rank_out_f16(w1, a1, g1, w2_t, a2_t, g2_t)
    expected = list(shape) + [C]
    assert w2.shape == tuple(expected)
    assert a2.shape == tuple(expected)
    assert g2.shape == tuple(expected)
    for t in (w2, a2, g2):
        assert t.dtype == torch.float16


def test_linear_wag_rank_out_f16_not_nan():
    shape = (2, 128)
    C = 256
    Kw, Ka, Kg = 128, 128, 128
    inputs = [torch.randn(*shape, k, dtype=torch.float16, device="cuda") for k in (Kw, Ka, Kg)]
    weights = [torch.randn(C, k, dtype=torch.float16, device="cuda") for k in (Kw, Ka, Kg)]
    w2, a2, g2 = linear_wag_rank_out_f16(*inputs, *weights)
    assert not torch.isnan(w2).any()
    assert not torch.isnan(a2).any()
    assert not torch.isnan(g2).any()


def test_linear_wag_rank_out_f16_return_count():
    shape = (1, 32)
    C = 64
    Kw, Ka, Kg = 32, 32, 32
    w1 = torch.randn(*shape, Kw, dtype=torch.float16, device="cuda")
    a1 = torch.randn(*shape, Ka, dtype=torch.float16, device="cuda")
    g1 = torch.randn(*shape, Kg, dtype=torch.float16, device="cuda")
    w2_t = torch.randn(C, Kw, dtype=torch.float16, device="cuda")
    a2_t = torch.randn(C, Ka, dtype=torch.float16, device="cuda")
    g2_t = torch.randn(C, Kg, dtype=torch.float16, device="cuda")
    results = linear_wag_rank_out_f16(w1, a1, g1, w2_t, a2_t, g2_t)
    assert len(results) == 3, "linear_wag_rank_out_f16 must return 3 tensors"


# ===== linear_wagv_rank_out_f16: return type B(4) — [w2, a2, g2, v_out] =====


@pytest.mark.parametrize(
    "shape,C,Kw,Ka,Kg,Kv",
    [
        ((2, 128), 256, 128, 128, 128, 64),
        ((2, 3, 64), 128, 64, 64, 64, 64),
    ],
)
def test_linear_wagv_rank_out_f16_basic(shape, C, Kw, Ka, Kg, Kv):
    w1 = torch.randn(*shape, Kw, dtype=torch.float16, device="cuda")
    a1 = torch.randn(*shape, Ka, dtype=torch.float16, device="cuda")
    g1 = torch.randn(*shape, Kg, dtype=torch.float16, device="cuda")
    v1 = torch.randn(*shape, Kv, dtype=torch.float16, device="cuda")
    w2_t = torch.randn(C, Kw, dtype=torch.float16, device="cuda")
    a2_t = torch.randn(C, Ka, dtype=torch.float16, device="cuda")
    g2_t = torch.randn(C, Kg, dtype=torch.float16, device="cuda")
    v2_t = torch.randn(C, Kv, dtype=torch.float16, device="cuda")
    v = torch.randn(*shape[:-1], C, dtype=torch.float16, device="cuda")
    v_first = torch.randn(*shape[:-1], C, dtype=torch.float16, device="cuda")
    v0 = torch.randn(C, dtype=torch.float16, device="cuda")
    w2, a2, g2, v_out = linear_wagv_rank_out_f16(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0)
    expected = list(shape) + [C]
    assert w2.shape == tuple(expected)
    assert a2.shape == tuple(expected)
    assert g2.shape == tuple(expected)
    assert v_out.shape == tuple(expected)
    for t in (w2, a2, g2, v_out):
        assert t.dtype == torch.float16


def test_linear_wagv_rank_out_f16_not_nan():
    shape = (2, 64)
    C = 128
    Kw, Ka, Kg, Kv = 64, 64, 64, 64
    w1 = torch.randn(*shape, Kw, dtype=torch.float16, device="cuda")
    a1 = torch.randn(*shape, Ka, dtype=torch.float16, device="cuda")
    g1 = torch.randn(*shape, Kg, dtype=torch.float16, device="cuda")
    v1 = torch.randn(*shape, Kv, dtype=torch.float16, device="cuda")
    w2_t = torch.randn(C, Kw, dtype=torch.float16, device="cuda")
    a2_t = torch.randn(C, Ka, dtype=torch.float16, device="cuda")
    g2_t = torch.randn(C, Kg, dtype=torch.float16, device="cuda")
    v2_t = torch.randn(C, Kv, dtype=torch.float16, device="cuda")
    v = torch.randn(*shape[:-1], C, dtype=torch.float16, device="cuda")
    v_first = torch.randn(*shape[:-1], C, dtype=torch.float16, device="cuda")
    v0 = torch.randn(C, dtype=torch.float16, device="cuda")
    w2, a2, g2, v_out = linear_wagv_rank_out_f16(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0)
    assert not torch.isnan(w2).any()
    assert not torch.isnan(a2).any()
    assert not torch.isnan(g2).any()
    assert not torch.isnan(v_out).any()


def test_linear_wagv_rank_out_f16_return_count():
    shape = (1, 32)
    C = 64
    Kw, Ka, Kg, Kv = 32, 32, 32, 32
    w1 = torch.randn(*shape, Kw, dtype=torch.float16, device="cuda")
    a1 = torch.randn(*shape, Ka, dtype=torch.float16, device="cuda")
    g1 = torch.randn(*shape, Kg, dtype=torch.float16, device="cuda")
    v1 = torch.randn(*shape, Kv, dtype=torch.float16, device="cuda")
    w2_t = torch.randn(C, Kw, dtype=torch.float16, device="cuda")
    a2_t = torch.randn(C, Ka, dtype=torch.float16, device="cuda")
    g2_t = torch.randn(C, Kg, dtype=torch.float16, device="cuda")
    v2_t = torch.randn(C, Kv, dtype=torch.float16, device="cuda")
    v = torch.randn(*shape[:-1], C, dtype=torch.float16, device="cuda")
    v_first = torch.randn(*shape[:-1], C, dtype=torch.float16, device="cuda")
    v0 = torch.randn(C, dtype=torch.float16, device="cuda")
    results = linear_wagv_rank_out_f16(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0)
    assert len(results) == 4, "linear_wagv_rank_out_f16 must return 4 tensors"

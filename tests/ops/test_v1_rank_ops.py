import pytest
import torch
from vkwr._ops.v1.v1_rank_ops import (
    linear_wag_rank_in_f16,
    linear_wagv_rank_in_f16,
    linear_wag_rank_out_f16,
    linear_wagv_rank_out_f16,
)


# ===== linear_wag_rank_in_f16: return type B(3) — [w1, a1, g1] =====


@pytest.mark.parametrize(
    "M,K,Rw,Ra,Rg",
    [
        (1, 64, 32, 32, 32),
        (2, 128, 64, 64, 64),
        (4, 256, 128, 64, 128),
    ],
)
def test_linear_wag_rank_in_f16_basic(M, K, Rw, Ra, Rg):
    xw = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    w1, a1, g1 = linear_wag_rank_in_f16(M, K, Rw, Ra, Rg, xw, xa, xg, w1_t, a1_t, g1_t)
    assert w1.shape == (M, Rw)
    assert a1.shape == (M, Ra)
    assert g1.shape == (M, Rg)
    assert w1.dtype == torch.float16
    assert a1.dtype == torch.float16
    assert g1.dtype == torch.float16


def test_linear_wag_rank_in_f16_not_nan():
    M, K = 2, 256
    Rw, Ra, Rg = 128, 128, 128
    xw = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    w1, a1, g1 = linear_wag_rank_in_f16(M, K, Rw, Ra, Rg, xw, xa, xg, w1_t, a1_t, g1_t)
    assert not torch.isnan(w1).any()
    assert not torch.isnan(a1).any()
    assert not torch.isnan(g1).any()


def test_linear_wag_rank_in_f16_return_count():
    M, K = 1, 64
    Rw, Ra, Rg = 32, 32, 32
    xw = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    results = linear_wag_rank_in_f16(M, K, Rw, Ra, Rg, xw, xa, xg, w1_t, a1_t, g1_t)
    assert len(results) == 3, "linear_wag_rank_in_f16 must return 3 tensors"


def test_linear_wag_rank_in_f16_invalid_shape():
    M, K = 2, 64
    Rw, Ra, Rg = 32, 32, 32
    xw = torch.randn(3, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        linear_wag_rank_in_f16(M, K, Rw, Ra, Rg, xw, xa, xg, w1_t, a1_t, g1_t)


# ===== linear_wagv_rank_in_f16: return type B(4) — [w1, a1, g1, v_out] =====


@pytest.mark.parametrize(
    "M,K,Rw,Ra,Rg,Rv",
    [
        (1, 64, 32, 32, 32, 32),
        (2, 128, 64, 64, 64, 64),
    ],
)
def test_linear_wagv_rank_in_f16_basic(M, K, Rw, Ra, Rg, Rv):
    xw = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xv = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    v1_t = torch.randn(Rv, K, dtype=torch.float16, device="cuda")
    w1, a1, g1, v_out = linear_wagv_rank_in_f16(M, K, Rw, Ra, Rg, Rv, xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t)
    assert w1.shape == (M, Rw)
    assert a1.shape == (M, Ra)
    assert g1.shape == (M, Rg)
    assert v_out.shape == (M, Rv)
    for t in (w1, a1, g1, v_out):
        assert t.dtype == torch.float16


def test_linear_wagv_rank_in_f16_not_nan():
    M, K = 2, 128
    Rw, Ra, Rg, Rv = 64, 64, 64, 64
    tensors = [torch.randn(M, K, dtype=torch.float16, device="cuda") for _ in range(4)]
    weights = [torch.randn(r, K, dtype=torch.float16, device="cuda") for r in (Rw, Ra, Rg, Rv)]
    w1, a1, g1, v_out = linear_wagv_rank_in_f16(M, K, Rw, Ra, Rg, Rv, *tensors, *weights)
    assert not torch.isnan(w1).any()
    assert not torch.isnan(a1).any()
    assert not torch.isnan(g1).any()
    assert not torch.isnan(v_out).any()


def test_linear_wagv_rank_in_f16_return_count():
    M, K = 1, 64
    Rw, Ra, Rg, Rv = 32, 32, 32, 32
    xw = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xa = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xg = torch.randn(M, K, dtype=torch.float16, device="cuda")
    xv = torch.randn(M, K, dtype=torch.float16, device="cuda")
    w1_t = torch.randn(Rw, K, dtype=torch.float16, device="cuda")
    a1_t = torch.randn(Ra, K, dtype=torch.float16, device="cuda")
    g1_t = torch.randn(Rg, K, dtype=torch.float16, device="cuda")
    v1_t = torch.randn(Rv, K, dtype=torch.float16, device="cuda")
    results = linear_wagv_rank_in_f16(M, K, Rw, Ra, Rg, Rv, xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t)
    assert len(results) == 4, "linear_wagv_rank_in_f16 must return 4 tensors"


# ===== linear_wag_rank_out_f16: return type B(3) — [w2, a2, g2] =====


@pytest.mark.parametrize(
    "M,C,Kw,Ka,Kg",
    [
        (1, 64, 32, 32, 32),
        (2, 256, 128, 128, 128),
        (4, 512, 256, 128, 256),
    ],
)
def test_linear_wag_rank_out_f16_basic(M, C, Kw, Ka, Kg):
    w1 = torch.randn(M, Kw, dtype=torch.float16, device="cuda")
    a1 = torch.randn(M, Ka, dtype=torch.float16, device="cuda")
    g1 = torch.randn(M, Kg, dtype=torch.float16, device="cuda")
    w2_t = torch.randn(C, Kw, dtype=torch.float16, device="cuda")
    a2_t = torch.randn(C, Ka, dtype=torch.float16, device="cuda")
    g2_t = torch.randn(C, Kg, dtype=torch.float16, device="cuda")
    w2, a2, g2 = linear_wag_rank_out_f16(M, C, Kw, Ka, Kg, w1, a1, g1, w2_t, a2_t, g2_t)
    assert w2.shape == (M, C)
    assert a2.shape == (M, C)
    assert g2.shape == (M, C)
    for t in (w2, a2, g2):
        assert t.dtype == torch.float16


def test_linear_wag_rank_out_f16_not_nan():
    M, C = 2, 256
    Kw, Ka, Kg = 128, 128, 128
    inputs = [torch.randn(M, k, dtype=torch.float16, device="cuda") for k in (Kw, Ka, Kg)]
    weights = [torch.randn(C, k, dtype=torch.float16, device="cuda") for k in (Kw, Ka, Kg)]
    w2, a2, g2 = linear_wag_rank_out_f16(M, C, Kw, Ka, Kg, *inputs, *weights)
    assert not torch.isnan(w2).any()
    assert not torch.isnan(a2).any()
    assert not torch.isnan(g2).any()


def test_linear_wag_rank_out_f16_return_count():
    M, C = 1, 64
    Kw, Ka, Kg = 32, 32, 32
    w1 = torch.randn(M, Kw, dtype=torch.float16, device="cuda")
    a1 = torch.randn(M, Ka, dtype=torch.float16, device="cuda")
    g1 = torch.randn(M, Kg, dtype=torch.float16, device="cuda")
    w2_t = torch.randn(C, Kw, dtype=torch.float16, device="cuda")
    a2_t = torch.randn(C, Ka, dtype=torch.float16, device="cuda")
    g2_t = torch.randn(C, Kg, dtype=torch.float16, device="cuda")
    results = linear_wag_rank_out_f16(M, C, Kw, Ka, Kg, w1, a1, g1, w2_t, a2_t, g2_t)
    assert len(results) == 3, "linear_wag_rank_out_f16 must return 3 tensors"


# ===== linear_wagv_rank_out_f16: return type B(4) — [w2, a2, g2, v_out] =====


@pytest.mark.parametrize(
    "M,C,Kw,Ka,Kg,Kv",
    [
        (1, 64, 32, 32, 32, 32),
        (2, 256, 128, 128, 128, 64),
    ],
)
def test_linear_wagv_rank_out_f16_basic(M, C, Kw, Ka, Kg, Kv):
    w1 = torch.randn(M, Kw, dtype=torch.float16, device="cuda")
    a1 = torch.randn(M, Ka, dtype=torch.float16, device="cuda")
    g1 = torch.randn(M, Kg, dtype=torch.float16, device="cuda")
    v1 = torch.randn(M, Kv, dtype=torch.float16, device="cuda")
    w2_t = torch.randn(C, Kw, dtype=torch.float16, device="cuda")
    a2_t = torch.randn(C, Ka, dtype=torch.float16, device="cuda")
    g2_t = torch.randn(C, Kg, dtype=torch.float16, device="cuda")
    v2_t = torch.randn(C, Kv, dtype=torch.float16, device="cuda")
    v = torch.randn(M, C, dtype=torch.float16, device="cuda")
    v_first = torch.randn(M, C, dtype=torch.float16, device="cuda")
    v0 = torch.randn(C, dtype=torch.float16, device="cuda")
    w2, a2, g2, v_out = linear_wagv_rank_out_f16(M, C, Kw, Ka, Kg, Kv, w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0)
    assert w2.shape == (M, C)
    assert a2.shape == (M, C)
    assert g2.shape == (M, C)
    assert v_out.shape == (M, C)
    for t in (w2, a2, g2, v_out):
        assert t.dtype == torch.float16


def test_linear_wagv_rank_out_f16_not_nan():
    M, C = 2, 128
    Kw, Ka, Kg, Kv = 64, 64, 64, 64
    w1 = torch.randn(M, Kw, dtype=torch.float16, device="cuda")
    a1 = torch.randn(M, Ka, dtype=torch.float16, device="cuda")
    g1 = torch.randn(M, Kg, dtype=torch.float16, device="cuda")
    v1 = torch.randn(M, Kv, dtype=torch.float16, device="cuda")
    w2_t = torch.randn(C, Kw, dtype=torch.float16, device="cuda")
    a2_t = torch.randn(C, Ka, dtype=torch.float16, device="cuda")
    g2_t = torch.randn(C, Kg, dtype=torch.float16, device="cuda")
    v2_t = torch.randn(C, Kv, dtype=torch.float16, device="cuda")
    v = torch.randn(M, C, dtype=torch.float16, device="cuda")
    v_first = torch.randn(M, C, dtype=torch.float16, device="cuda")
    v0 = torch.randn(C, dtype=torch.float16, device="cuda")
    w2, a2, g2, v_out = linear_wagv_rank_out_f16(M, C, Kw, Ka, Kg, Kv, w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0)
    assert not torch.isnan(w2).any()
    assert not torch.isnan(a2).any()
    assert not torch.isnan(g2).any()
    assert not torch.isnan(v_out).any()


def test_linear_wagv_rank_out_f16_return_count():
    M, C = 1, 64
    Kw, Ka, Kg, Kv = 32, 32, 32, 32
    w1 = torch.randn(M, Kw, dtype=torch.float16, device="cuda")
    a1 = torch.randn(M, Ka, dtype=torch.float16, device="cuda")
    g1 = torch.randn(M, Kg, dtype=torch.float16, device="cuda")
    v1 = torch.randn(M, Kv, dtype=torch.float16, device="cuda")
    w2_t = torch.randn(C, Kw, dtype=torch.float16, device="cuda")
    a2_t = torch.randn(C, Ka, dtype=torch.float16, device="cuda")
    g2_t = torch.randn(C, Kg, dtype=torch.float16, device="cuda")
    v2_t = torch.randn(C, Kv, dtype=torch.float16, device="cuda")
    v = torch.randn(M, C, dtype=torch.float16, device="cuda")
    v_first = torch.randn(M, C, dtype=torch.float16, device="cuda")
    v0 = torch.randn(C, dtype=torch.float16, device="cuda")
    results = linear_wagv_rank_out_f16(M, C, Kw, Ka, Kg, Kv, w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0)
    assert len(results) == 4, "linear_wagv_rank_out_f16 must return 4 tensors"

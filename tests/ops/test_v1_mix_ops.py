import pytest
import torch

from vkwr._ops.v1.v1_mix_ops import (
    act_sigmoid,
    act_tanh,
    add_vec,
    cmix_mix,
    cmix_mix_cfg,
    cmix_sparse_down_one,
    cmix_sparse_down_relu_one,
    cmix_sparse_down_relu_rows,
    cmix_sparse_down_relu_rows_t512,
    cmix_sparse_down_rows,
    cmix_sparse_one,
    cmix_sparse_rows,
    relu_square,
    tmix_kk_a_gate,
    tmix_kk_a_gate_update_shift,
    tmix_lnx_rkvres_xg,
    tmix_mix6,
    tmix_mix6_cfg,
    tmix_mix6_t1_c4096,
    tmix_vres_gate,
)


def _make_shift_state(B, C):
    return torch.randn(B, C, dtype=torch.float16, device="cuda")


# ===== tmix_mix6: return type B(6) — [x_r, x_w, x_k, x_v, x_a, x_g] =====


@pytest.mark.parametrize(
    "B,T,C",
    [(1, 1, 64), (2, 4, 256), (1, 16, 512)],
)
def test_tmix_mix6_basic(B, T, C):
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    results = tmix_mix6(B, T, C, x, shift_state, *vecs)
    assert len(results) == 6
    for r in results:
        assert r.shape == x.shape
        assert r.dtype == torch.float16


def test_tmix_mix6_not_nan():
    B, T, C = 2, 4, 256
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    results = tmix_mix6(B, T, C, x, shift_state, *vecs)
    for r in results:
        assert not torch.isnan(r).any()


def test_tmix_mix6_return_count():
    B, T, C = 1, 1, 128
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    results = tmix_mix6(B, T, C, x, shift_state, *vecs)
    assert len(results) == 6, "tmix_mix6 must return 6 tensors"


def test_tmix_mix6_invalid_odd_C():
    x = torch.randn(1, 1, 65, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(1, 65, dtype=torch.float16, device="cuda")
    vecs = [torch.randn(65, dtype=torch.float16, device="cuda") for _ in range(6)]
    with pytest.raises(RuntimeError):
        tmix_mix6(1, 1, 65, x, shift_state, *vecs)


# ===== tmix_mix6_cfg: return type B(6), threads param =====


@pytest.mark.parametrize("threads", [128, 256, 512, 1024])
def test_tmix_mix6_cfg_basic(threads):
    B, T, C = 2, 4, 256
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    results = tmix_mix6_cfg(B, T, C, x, shift_state, *vecs, threads)
    assert len(results) == 6
    for r in results:
        assert r.shape == x.shape


def test_tmix_mix6_cfg_invalid_threads():
    B, T, C = 1, 1, 128
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    with pytest.raises(RuntimeError):
        tmix_mix6_cfg(B, T, C, x, shift_state, *vecs, 64)


# ===== tmix_mix6_t1_c4096: return type B(6), C=4096 hardcoded =====


def test_tmix_mix6_t1_c4096_basic():
    B, C = 2, 4096
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    results = tmix_mix6_t1_c4096(B, x, shift_state, *vecs)
    assert len(results) == 6
    for r in results:
        assert r.shape == x.shape
        assert not torch.isnan(r).any()


def test_tmix_mix6_t1_c4096_invalid_C():
    B, C = 1, 2048
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    with pytest.raises(ValueError):
        tmix_mix6_t1_c4096(B, x, shift_state, *vecs)


def test_tmix_mix6_t1_c4096_param_options():
    B, C = 1, 4096
    x = torch.randn(B, 1, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    vecs = [torch.randn(C, dtype=torch.float16, device="cuda") for _ in range(6)]
    for threads in [256, 512]:
        for vec in [1, 2, 4]:
            results = tmix_mix6_t1_c4096(B, x, shift_state, *vecs, threads=threads, vec=vec, half_math=False)
            assert len(results) == 6


# ===== tmix_kk_a_gate: return type B(3) — [new_k, neg_kk, kka] =====


@pytest.mark.parametrize(
    "B,T,C,H",
    [(1, 1, 64, 1), (2, 4, 256, 4), (1, 16, 512, 8)],
)
def test_tmix_kk_a_gate_basic(B, T, C, H):
    k = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    k_k = torch.randn(C, dtype=torch.float16, device="cuda")
    a0 = torch.randn(C, dtype=torch.float16, device="cuda")
    a12 = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    k_a = torch.randn(C, dtype=torch.float16, device="cuda")
    results = tmix_kk_a_gate(B, T, C, H, k, k_k, a0, a12, k_a)
    assert len(results) == 3
    for r in results:
        assert r.shape == k.shape
        assert r.dtype == torch.float16


def test_tmix_kk_a_gate_not_nan():
    B, T, C, H = 2, 4, 256, 4
    k = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    k_k = torch.randn(C, dtype=torch.float16, device="cuda")
    a0 = torch.randn(C, dtype=torch.float16, device="cuda")
    a12 = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    k_a = torch.randn(C, dtype=torch.float16, device="cuda")
    results = tmix_kk_a_gate(B, T, C, H, k, k_k, a0, a12, k_a)
    for r in results:
        assert not torch.isnan(r).any()


# ===== tmix_kk_a_gate_update_shift: return type B(3) =====


@pytest.mark.parametrize(
    "B,C,H",
    [(1, 64, 1), (2, 256, 4)],
)
def test_tmix_kk_a_gate_update_shift_basic(B, C, H):
    T = 1
    k = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    k_k = torch.randn(C, dtype=torch.float16, device="cuda")
    a0 = torch.randn(C, dtype=torch.float16, device="cuda")
    a12 = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    k_a = torch.randn(C, dtype=torch.float16, device="cuda")
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    results = tmix_kk_a_gate_update_shift(B, T, C, H, k, k_k, a0, a12, k_a, x, shift_state)
    assert len(results) == 3
    for r in results:
        assert r.shape == k.shape


def test_tmix_kk_a_gate_update_shift_shift_state_mutation():
    B, T, C, H = 1, 1, 128, 2
    k = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    k_k = torch.randn(C, dtype=torch.float16, device="cuda")
    a0 = torch.randn(C, dtype=torch.float16, device="cuda")
    a12 = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    k_a = torch.randn(C, dtype=torch.float16, device="cuda")
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    shift_state_before = shift_state.clone()
    tmix_kk_a_gate_update_shift(B, T, C, H, k, k_k, a0, a12, k_a, x, shift_state)
    assert not torch.equal(shift_state, shift_state_before)


# ===== tmix_lnx_rkvres_xg: return type A =====


@pytest.mark.parametrize(
    "B,T,C,H",
    [(1, 1, 64, 1), (2, 4, 256, 4)],
)
def test_tmix_lnx_rkvres_xg_basic(B, T, C, H):
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    r = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    k = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    v = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    r_k = torch.randn(C, dtype=torch.float16, device="cuda")
    weight = torch.ones(C, dtype=torch.float16, device="cuda")
    bias = torch.zeros(C, dtype=torch.float16, device="cuda")
    g = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    y = tmix_lnx_rkvres_xg(B, T, C, H, x, r, k, v, r_k, weight, bias, g)
    assert y.shape == x.shape
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


# ===== tmix_vres_gate: return type A =====


@pytest.mark.parametrize(
    "B,T,C",
    [(1, 1, 64), (2, 8, 256)],
)
def test_tmix_vres_gate_basic(B, T, C):
    v = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    v_first = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    v0 = torch.randn(C, dtype=torch.float16, device="cuda")
    v12 = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    y = tmix_vres_gate(B, T, C, v, v_first, v0, v12)
    assert y.shape == v.shape
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


# ===== cmix_sparse_one: return type A =====


@pytest.mark.parametrize(
    "C,F",
    [(256, 256), (512, 512)],
)
def test_cmix_sparse_one_basic(C, F):
    x = torch.randn(1, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(1, C, dtype=torch.float16, device="cuda")
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    key_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    value_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    y = cmix_sparse_one(C, F, x, shift_state, x_k, key_fc, value_fc)
    assert y.shape == (1, 1, C)
    assert y.dtype == torch.float16


def test_cmix_sparse_one_not_nan():
    C, F = 256, 256
    x = torch.randn(1, 1, C, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(1, C, dtype=torch.float16, device="cuda")
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    key_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    value_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    y = cmix_sparse_one(C, F, x, shift_state, x_k, key_fc, value_fc)
    assert not torch.isnan(y).any()


def test_cmix_sparse_one_invalid_C():
    x = torch.randn(1, 1, 100, dtype=torch.float16, device="cuda")
    shift_state = torch.randn(1, 100, dtype=torch.float16, device="cuda")
    x_k = torch.randn(100, dtype=torch.float16, device="cuda")
    key_fc = torch.randn(128, 100, dtype=torch.float16, device="cuda")
    value_fc = torch.randn(128, 100, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        cmix_sparse_one(100, 128, x, shift_state, x_k, key_fc, value_fc)


# ===== cmix_sparse_rows: return type A =====


@pytest.mark.parametrize(
    "B,T,C,F",
    [(2, 8, 256, 512)],
)
def test_cmix_sparse_rows_basic(B, T, C, F):
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    key_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    value_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    y = cmix_sparse_rows(B, T, C, F, x, shift_state, x_k, key_fc, value_fc)
    assert y.shape == (B, T, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


# ===== cmix_sparse_down_one: return type A =====


@pytest.mark.parametrize(
    "C,F",
    [(256, 512)],
)
def test_cmix_sparse_down_one_basic(C, F):
    act = torch.randn(F, dtype=torch.float16, device="cuda")
    value_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    y = cmix_sparse_down_one(C, F, act, value_fc)
    assert y.shape == (1, 1, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


# ===== cmix_sparse_down_rows: return type A =====


@pytest.mark.parametrize(
    "B,T,C,F",
    [(2, 8, 256, 512)],
)
def test_cmix_sparse_down_rows_basic(B, T, C, F):
    act = torch.randn(B, T, F, dtype=torch.float16, device="cuda")
    value_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    y = cmix_sparse_down_rows(B, T, C, F, act, value_fc)
    assert y.shape == (B, T, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


# ===== cmix_sparse_down_relu_one: return type A =====


@pytest.mark.parametrize(
    "C,F",
    [(256, 512)],
)
def test_cmix_sparse_down_relu_one_basic(C, F):
    preact = torch.randn(F, dtype=torch.float16, device="cuda")
    value_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    y = cmix_sparse_down_relu_one(C, F, preact, value_fc)
    assert y.shape == (1, 1, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


# ===== cmix_sparse_down_relu_rows: return type A =====


@pytest.mark.parametrize(
    "B,T,C,F",
    [(2, 8, 256, 512)],
)
def test_cmix_sparse_down_relu_rows_basic(B, T, C, F):
    preact = torch.randn(B, T, F, dtype=torch.float16, device="cuda")
    value_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    y = cmix_sparse_down_relu_rows(B, T, C, F, preact, value_fc)
    assert y.shape == (B, T, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


# ===== cmix_sparse_down_relu_rows_t512: return type A =====


@pytest.mark.parametrize(
    "B,T,C,F",
    [(1, 4, 512, 512), (2, 8, 1024, 1024)],
)
def test_cmix_sparse_down_relu_rows_t512_basic(B, T, C, F):
    preact = torch.randn(B, T, F, dtype=torch.float16, device="cuda")
    value_fc = torch.randn(F, C, dtype=torch.float16, device="cuda")
    y = cmix_sparse_down_relu_rows_t512(B, T, C, F, preact, value_fc)
    assert y.shape == (B, T, C)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


# ===== cmix_mix: return type A =====


@pytest.mark.parametrize(
    "B,T,C",
    [(1, 1, 64), (2, 4, 256)],
)
def test_cmix_mix_basic(B, T, C):
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    y = cmix_mix(B, T, C, x, shift_state, x_k)
    assert y.shape == x.shape
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


def test_cmix_mix_shift_state_mutation():
    B, T, C = 1, 1, 128
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    shift_state_before = shift_state.clone()
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    cmix_mix(B, T, C, x, shift_state, x_k)
    assert not torch.equal(shift_state, shift_state_before)


# ===== cmix_mix_cfg: return type A =====


@pytest.mark.parametrize("threads", [128, 256, 512, 1024])
def test_cmix_mix_cfg_basic(threads):
    B, T, C = 2, 4, 256
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    y = cmix_mix_cfg(B, T, C, x, shift_state, x_k, threads)
    assert y.shape == x.shape
    assert y.dtype == torch.float16


def test_cmix_mix_cfg_invalid_threads():
    B, T, C = 1, 1, 128
    x = torch.randn(B, T, C, dtype=torch.float16, device="cuda")
    shift_state = _make_shift_state(B, C)
    x_k = torch.randn(C, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        cmix_mix_cfg(B, T, C, x, shift_state, x_k, 64)


# ===== relu_square: return type A =====


@pytest.mark.parametrize(
    "shape",
    [(128,), (4, 256), (2, 8, 512)],
)
def test_relu_square_basic(shape):
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    y = relu_square(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float16
    expected = torch.relu(x) ** 2
    torch.testing.assert_close(y, expected.to(torch.float16), atol=5e-3, rtol=5e-3)


def test_relu_square_negative_input():
    x = -torch.abs(torch.randn(256, dtype=torch.float16, device="cuda"))
    y = relu_square(x)
    assert (y == 0).all()


# ===== act_tanh: return type A =====


@pytest.mark.parametrize(
    "shape",
    [(128,), (4, 256), (2, 8, 512)],
)
def test_act_tanh_basic(shape):
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    y = act_tanh(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float16
    expected = torch.tanh(x)
    torch.testing.assert_close(y, expected.to(torch.float16), atol=5e-3, rtol=5e-3)


def test_act_tanh_range():
    x = torch.randn(256, dtype=torch.float16, device="cuda")
    y = act_tanh(x)
    assert (y >= -1.0).all() and (y <= 1.0).all()


# ===== act_sigmoid: return type A =====


@pytest.mark.parametrize(
    "shape",
    [(128,), (4, 256), (2, 8, 512)],
)
def test_act_sigmoid_basic(shape):
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    y = act_sigmoid(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float16
    expected = torch.sigmoid(x)
    torch.testing.assert_close(y, expected.to(torch.float16), atol=5e-3, rtol=5e-3)


def test_act_sigmoid_range():
    x = torch.randn(256, dtype=torch.float16, device="cuda")
    y = act_sigmoid(x)
    assert (y >= 0.0).all() and (y <= 1.0).all()


# ===== add_vec: return type A =====


@pytest.mark.parametrize(
    "shape,C",
    [((128,), 128), ((4, 256), 256), ((2, 8, 512), 512)],
)
def test_add_vec_basic(shape, C):
    x = torch.randn(shape, dtype=torch.float16, device="cuda")
    vec = torch.randn(C, dtype=torch.float16, device="cuda")
    y = add_vec(C, x, vec)
    assert y.shape == x.shape
    assert y.dtype == torch.float16
    not_nan = not torch.isnan(y).any()
    assert not_nan


def test_add_vec_zero():
    C = 256
    x = torch.randn(4, C, dtype=torch.float16, device="cuda")
    vec = torch.zeros(C, dtype=torch.float16, device="cuda")
    y = add_vec(C, x, vec)
    torch.testing.assert_close(y, x, atol=0, rtol=0)

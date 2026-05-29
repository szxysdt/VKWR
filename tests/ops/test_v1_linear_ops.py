import pytest
import torch

from vkwr._ops.v1.v1_linear_ops import (
    linear_f16,
    linear_f16_lt,
    linear_f16_m1_splitk,
    linear_f16_m1_splitk_cfg,
    linear_f16_m1_splitk_tile,
    linear_f16_m1_splitk_warpred_tile,
    linear_f16_orig,
    linear_f16_orig_lt,
    linear_f16_orig_lt_cfg,
    linear_f16_rows_splitk,
    linear_orig_rows_cfg_f16,
    linear_orig_rows_exact_f16,
    linear_orig_rows_f16,
    linear_orig_wmma16_f16,
    linear_t_act_f16,
    linear_t_f16,
    linear_t_vres_f16,
)

# ===== Helper functions =====


def _make_input_m1(K):
    """M=1 scenario: 2D input [1, K] (minimum 2D required by ops)."""
    return torch.randn(1, K, dtype=torch.float16, device="cuda")


def _make_input_3d(B, T, C):
    """Standard scenario: 3D input [B,T,C]."""
    return torch.randn(B, T, C, dtype=torch.float16, device="cuda")


# ===== Group A: Transposed weight [K,N] ops =====
# weight [K, N], x.size(-1) == K, output appends N dimension


@pytest.mark.parametrize(
    "B,T,K,N",
    [
        (1, 1, 64, 128),
        (2, 4, 128, 256),
        (1, 16, 256, 512),
        (4, 8, 512, 1024),
    ],
)
def test_linear_f16_basic(B, T, K, N):
    x = _make_input_3d(B, T, K)
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16(x, weight)
    assert y.shape == (B, T, N)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


def test_linear_f16_1d_squeezed_input():
    """M=1 scenario: 2D input [1, K], output [1, N]."""
    K, N = 128, 256
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16(x, weight)
    assert y.shape == (1, N)


def test_linear_f16_shape_mismatch():
    K, N = 128, 256
    x = torch.randn(2, 4, 64, dtype=torch.float16, device="cuda")
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        linear_f16(x, weight)


def test_linear_f16_wrong_dtype():
    x = torch.randn(2, 128, dtype=torch.float32, device="cuda")
    weight = torch.randn(128, 256, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        linear_f16(x, weight)


def test_linear_f16_non_contiguous():
    x = torch.randn(2, 256, 128, dtype=torch.float16, device="cuda").transpose(1, 2)
    weight = torch.randn(128, 64, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        linear_f16(x, weight)


# cuBLASLt variant


@pytest.mark.parametrize(
    "B,T,K,N",
    [
        (2, 4, 128, 256),
        (1, 16, 256, 512),
    ],
)
def test_linear_f16_lt_basic(B, T, K, N):
    x = _make_input_3d(B, T, K)
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16_lt(x, weight)
    assert y.shape == (B, T, N)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


# M=1 splitk series — input is [1, K], output is [1, N]


@pytest.mark.parametrize("K,N", [(128, 256), (256, 512), (512, 1024)])
def test_linear_f16_m1_splitk_basic(K, N):
    x = _make_input_m1(K)
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16_m1_splitk(x, weight)
    assert y.shape == (1, N)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


@pytest.mark.parametrize("K,N", [(128, 256), (256, 512)])
def test_linear_f16_m1_splitk_cfg_basic(K, N):
    x = _make_input_m1(K)
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16_m1_splitk_cfg(x, weight, chunk_k=64)
    assert y.shape == (1, N)
    assert not torch.isnan(y).any()


@pytest.mark.parametrize("K,N", [(128, 256), (256, 512)])
def test_linear_f16_m1_splitk_tile_basic(K, N):
    x = _make_input_m1(K)
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16_m1_splitk_tile(x, weight, chunk_k=64, tile_cols=64)
    assert y.shape == (1, N)
    assert not torch.isnan(y).any()


@pytest.mark.parametrize("K,N", [(128, 256), (256, 512)])
def test_linear_f16_m1_splitk_warpred_tile_basic(K, N):
    x = _make_input_m1(K)
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16_m1_splitk_warpred_tile(x, weight, chunk_k=64, tile_cols=64)
    assert y.shape == (1, N)
    assert not torch.isnan(y).any()


# rows_splitk — supports 3D input


@pytest.mark.parametrize(
    "B,T,K,N",
    [(2, 4, 128, 256), (1, 16, 256, 512)],
)
def test_linear_f16_rows_splitk_basic(B, T, K, N):
    x = _make_input_3d(B, T, K)
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16_rows_splitk(x, weight, chunk_k=128)
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()


# ===== Group B: Original weight [N,K] ops =====
# weight_orig [N, K], x.size(-1) == K, output appends N dimension


@pytest.mark.parametrize(
    "B,T,N,K",
    [
        (1, 1, 128, 64),
        (2, 4, 256, 128),
        (1, 16, 512, 256),
    ],
)
def test_linear_f16_orig_basic(B, T, N, K):
    x = _make_input_3d(B, T, K)
    weight_orig = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_f16_orig(x, weight_orig)
    assert y.shape == (B, T, N)
    assert y.dtype == torch.float16
    assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "B,T,N,K",
    [(2, 4, 256, 128), (1, 16, 512, 256)],
)
def test_linear_orig_rows_f16_basic(B, T, N, K):
    x = _make_input_3d(B, T, K)
    weight_orig = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_orig_rows_f16(x, weight_orig, row_tile=4, out_tile=8)
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "B,T,N,K",
    [(2, 4, 256, 128), (1, 16, 512, 256)],
)
def test_linear_orig_rows_cfg_f16_basic(B, T, N, K):
    x = _make_input_3d(B, T, K)
    weight_orig = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_orig_rows_cfg_f16(x, weight_orig, threads=64, row_tile=4, out_tile=8)
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "B,T,N,K",
    [(1, 1, 256, 128), (1, 2, 256, 128)],
)
def test_linear_orig_rows_exact_f16_basic(B, T, N, K):
    x = _make_input_3d(B, T, K)
    weight_orig = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_orig_rows_exact_f16(x, weight_orig, threads=128, out_tile=2, use4=False)
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "B,T,N,K",
    [(2, 4, 256, 128), (1, 16, 512, 256)],
)
def test_linear_orig_wmma16_f16_basic(B, T, N, K):
    x = _make_input_3d(B, T, K)
    weight_orig = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_orig_wmma16_f16(x, weight_orig)
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "B,T,N,K",
    [(2, 4, 256, 128), (1, 16, 512, 256)],
)
def test_linear_f16_orig_lt_basic(B, T, N, K):
    x = _make_input_3d(B, T, K)
    weight_orig = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_f16_orig_lt(x, weight_orig)
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()


# cuBLASLt cfg — additional parameter boundary tests


@pytest.mark.parametrize(
    "workspace_mb,algo_index",
    [(0, 0), (4, 1), (32, 5), (1024, 63)],
)
def test_linear_f16_orig_lt_cfg_basic(workspace_mb, algo_index):
    B, T, N, K = 2, 4, 256, 128
    x = _make_input_3d(B, T, K)
    weight_orig = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_f16_orig_lt_cfg(x, weight_orig, workspace_mb, algo_index)
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()


def test_linear_f16_orig_lt_cfg_workspace_mb_out_of_range():
    B, T, N, K = 2, 4, 256, 128
    x = _make_input_3d(B, T, K)
    weight_orig = torch.randn(N, K, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        linear_f16_orig_lt_cfg(x, weight_orig, workspace_mb=-1, algo_index=0)
    with pytest.raises(RuntimeError):
        linear_f16_orig_lt_cfg(x, weight_orig, workspace_mb=1025, algo_index=0)


def test_linear_f16_orig_lt_cfg_algo_index_out_of_range():
    B, T, N, K = 2, 4, 256, 128
    x = _make_input_3d(B, T, K)
    weight_orig = torch.randn(N, K, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        linear_f16_orig_lt_cfg(x, weight_orig, workspace_mb=4, algo_index=-1)
    with pytest.raises(RuntimeError):
        linear_f16_orig_lt_cfg(x, weight_orig, workspace_mb=4, algo_index=64)


# ===== Group C: [N,K] transposed weight + fused ops =====
# weight_t [N, K], x.size(-1) == K


@pytest.mark.parametrize(
    "B,T,N,K",
    [(2, 4, 256, 128), (1, 16, 512, 256)],
)
def test_linear_t_f16_basic(B, T, N, K):
    x = _make_input_3d(B, T, K)
    weight_t = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_t_f16(x, weight_t)
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "B,T,N,K",
    # Constraint: K <= 512 && N >= 1024 && M(B*T) <= 4
    [(1, 1, 1024, 64), (1, 1, 4096, 256), (2, 1, 4096, 512)],
)
def test_linear_t_act_f16_tanh(B, T, N, K):
    x = _make_input_3d(B, T, K)
    weight_t = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_t_act_f16(x, weight_t, act=1)  # tanh
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()
    # Kernel computes y = weight_t @ tanh(x), NOT tanh(weight_t @ x).
    expected = torch.matmul(torch.tanh(x), weight_t.transpose(0, 1))
    torch.testing.assert_close(y, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "B,T,N,K",
    # Constraint: K <= 512 && N >= 1024 && M(B*T) <= 4
    [(1, 1, 1024, 64), (1, 1, 4096, 256), (2, 1, 4096, 512)],
)
def test_linear_t_act_f16_sigmoid(B, T, N, K):
    x = _make_input_3d(B, T, K)
    weight_t = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_t_act_f16(x, weight_t, act=2)  # sigmoid
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()
    # Kernel computes y = weight_t @ sigmoid(x), NOT sigmoid(weight_t @ x).
    expected = torch.matmul(torch.sigmoid(x), weight_t.transpose(0, 1))
    torch.testing.assert_close(y, expected, atol=1e-2, rtol=1e-2)


def test_linear_t_act_f16_invalid_act():
    # Use dims that pass the K<=512 && N>=1024 && M<=4 constraint so the
    # act validation fires (not the dimension check).
    x = _make_input_3d(1, 1, 256)
    weight_t = torch.randn(4096, 256, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        linear_t_act_f16(x, weight_t, act=0)
    with pytest.raises(RuntimeError):
        linear_t_act_f16(x, weight_t, act=3)


@pytest.mark.parametrize(
    "B,T,N,K",
    # Constraint: K <= 512 && N >= 1024 && M(B*T) <= 4
    [(1, 1, 1024, 64), (1, 1, 4096, 256), (2, 1, 4096, 512)],
)
def test_linear_t_vres_f16_basic(B, T, N, K):
    x = _make_input_3d(B, T, K)
    weight_t = torch.randn(N, K, dtype=torch.float16, device="cuda")
    v = torch.randn(B, T, N, dtype=torch.float16, device="cuda")
    v_first = torch.randn(B, T, N, dtype=torch.float16, device="cuda")
    v0 = torch.randn(N, dtype=torch.float16, device="cuda")
    y = linear_t_vres_f16(x, weight_t, v, v_first, v0)
    assert y.shape == (B, T, N)
    assert not torch.isnan(y).any()


def test_linear_t_vres_f16_v0_shape_mismatch():
    # Use dims that pass K<=512 && N>=1024 && M<=4 so the v0 shape check fires.
    x = _make_input_3d(1, 1, 256)
    weight_t = torch.randn(4096, 256, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 1, 4096, dtype=torch.float16, device="cuda")
    v_first = torch.randn(1, 1, 4096, dtype=torch.float16, device="cuda")
    v0 = torch.randn(128, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        linear_t_vres_f16(x, weight_t, v, v_first, v0)


# ===== Numerical correctness cross-validation =====


def test_linear_f16_vs_torch():
    """linear_f16 result should match torch.matmul (within fp16 tolerance)."""
    B, T, K, N = 2, 4, 256, 512
    x = _make_input_3d(B, T, K)
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16(x, weight)
    expected = torch.matmul(x, weight)
    torch.testing.assert_close(y, expected, atol=5e-3, rtol=5e-3)


def test_linear_f16_orig_vs_torch():
    """linear_f16_orig result should match torch.matmul with transposed weight."""
    B, T, N, K = 2, 4, 256, 128
    x = _make_input_3d(B, T, K)
    weight_orig = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_f16_orig(x, weight_orig)
    expected = torch.matmul(x, weight_orig.transpose(0, 1))
    torch.testing.assert_close(y, expected, atol=5e-3, rtol=5e-3)


def test_linear_t_f16_vs_torch():
    """linear_t_f16 result should match torch.matmul with transposed weight."""
    B, T, N, K = 2, 4, 256, 128
    x = _make_input_3d(B, T, K)
    weight_t = torch.randn(N, K, dtype=torch.float16, device="cuda")
    y = linear_t_f16(x, weight_t)
    expected = torch.matmul(x, weight_t.transpose(0, 1))
    torch.testing.assert_close(y, expected, atol=5e-3, rtol=5e-3)


# ===== Boundary conditions =====


def test_linear_f16_zero_input():
    K, N = 128, 256
    x = torch.zeros(2, 4, K, dtype=torch.float16, device="cuda")
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16(x, weight)
    assert (y == 0.0).all()


def test_linear_f16_small_dims():
    K, N = 64, 64
    x = torch.randn(1, 1, K, dtype=torch.float16, device="cuda")
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16(x, weight)
    assert y.shape == (1, 1, N)
    assert not torch.isnan(y).any()


def test_linear_f16_large_dims():
    K, N = 4096, 8192
    x = torch.randn(1, 1, K, dtype=torch.float16, device="cuda")
    weight = torch.randn(K, N, dtype=torch.float16, device="cuda")
    y = linear_f16(x, weight)
    assert y.shape == (1, 1, N)
    assert not torch.isnan(y).any()

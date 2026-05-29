import torch

from vkwr import _v1_linear_C  # noqa: F401


@torch.library.register_fake("vkwr_v1_linear::linear_f16")
def _(x, weight):
    N = weight.shape[1]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_f16(x, weight):
    return torch.ops.vkwr_v1_linear.linear_f16(x, weight)


@torch.library.register_fake("vkwr_v1_linear::linear_f16_orig")
def _(x, weight_orig):
    N = weight_orig.shape[0]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_f16_orig(x, weight_orig):
    return torch.ops.vkwr_v1_linear.linear_f16_orig(x, weight_orig)


@torch.library.register_fake("vkwr_v1_linear::linear_orig_rows_f16")
def _(x, weight_orig, row_tile, out_tile):
    N = weight_orig.shape[0]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_orig_rows_f16(x, weight_orig, row_tile, out_tile):
    return torch.ops.vkwr_v1_linear.linear_orig_rows_f16(x, weight_orig, row_tile, out_tile)


@torch.library.register_fake("vkwr_v1_linear::linear_orig_rows_cfg_f16")
def _(x, weight_orig, threads, row_tile, out_tile):
    N = weight_orig.shape[0]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_orig_rows_cfg_f16(x, weight_orig, threads, row_tile, out_tile):
    return torch.ops.vkwr_v1_linear.linear_orig_rows_cfg_f16(x, weight_orig, threads, row_tile, out_tile)


@torch.library.register_fake("vkwr_v1_linear::linear_orig_rows_exact_f16")
def _(x, weight_orig, threads, out_tile, use4):
    N = weight_orig.shape[0]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_orig_rows_exact_f16(x, weight_orig, threads, out_tile, use4):
    return torch.ops.vkwr_v1_linear.linear_orig_rows_exact_f16(x, weight_orig, threads, out_tile, use4)


@torch.library.register_fake("vkwr_v1_linear::linear_orig_wmma16_f16")
def _(x, weight_orig):
    N = weight_orig.shape[0]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_orig_wmma16_f16(x, weight_orig):
    return torch.ops.vkwr_v1_linear.linear_orig_wmma16_f16(x, weight_orig)


@torch.library.register_fake("vkwr_v1_linear::linear_f16_orig_lt")
def _(x, weight_orig):
    N = weight_orig.shape[0]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_f16_orig_lt(x, weight_orig):
    return torch.ops.vkwr_v1_linear.linear_f16_orig_lt(x, weight_orig)


@torch.library.register_fake("vkwr_v1_linear::linear_f16_orig_lt_cfg")
def _(x, weight_orig, workspace_mb, algo_index):
    N = weight_orig.shape[0]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_f16_orig_lt_cfg(x, weight_orig, workspace_mb, algo_index):
    return torch.ops.vkwr_v1_linear.linear_f16_orig_lt_cfg(x, weight_orig, workspace_mb, algo_index)


@torch.library.register_fake("vkwr_v1_linear::linear_f16_lt")
def _(x, weight):
    N = weight.shape[1]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_f16_lt(x, weight):
    return torch.ops.vkwr_v1_linear.linear_f16_lt(x, weight)


@torch.library.register_fake("vkwr_v1_linear::linear_f16_m1_splitk")
def _(x, weight):
    N = weight.shape[1]
    return torch.empty(N, dtype=x.dtype, device=x.device)


def linear_f16_m1_splitk(x, weight):
    return torch.ops.vkwr_v1_linear.linear_f16_m1_splitk(x, weight)


@torch.library.register_fake("vkwr_v1_linear::linear_f16_m1_splitk_cfg")
def _(x, weight, chunk_k):
    N = weight.shape[1]
    return torch.empty(N, dtype=x.dtype, device=x.device)


def linear_f16_m1_splitk_cfg(x, weight, chunk_k):
    return torch.ops.vkwr_v1_linear.linear_f16_m1_splitk_cfg(x, weight, chunk_k)


@torch.library.register_fake("vkwr_v1_linear::linear_f16_m1_splitk_tile")
def _(x, weight, chunk_k, tile_cols):
    N = weight.shape[1]
    return torch.empty(N, dtype=x.dtype, device=x.device)


def linear_f16_m1_splitk_tile(x, weight, chunk_k, tile_cols):
    return torch.ops.vkwr_v1_linear.linear_f16_m1_splitk_tile(x, weight, chunk_k, tile_cols)


@torch.library.register_fake("vkwr_v1_linear::linear_f16_m1_splitk_warpred_tile")
def _(x, weight, chunk_k, tile_cols):
    N = weight.shape[1]
    return torch.empty(N, dtype=x.dtype, device=x.device)


def linear_f16_m1_splitk_warpred_tile(x, weight, chunk_k, tile_cols):
    return torch.ops.vkwr_v1_linear.linear_f16_m1_splitk_warpred_tile(x, weight, chunk_k, tile_cols)


@torch.library.register_fake("vkwr_v1_linear::linear_f16_rows_splitk")
def _(x, weight, chunk_k):
    N = weight.shape[1]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_f16_rows_splitk(x, weight, chunk_k):
    return torch.ops.vkwr_v1_linear.linear_f16_rows_splitk(x, weight, chunk_k)


@torch.library.register_fake("vkwr_v1_linear::linear_t_f16")
def _(x, weight_t):
    N = weight_t.shape[0]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_t_f16(x, weight_t):
    return torch.ops.vkwr_v1_linear.linear_t_f16(x, weight_t)


@torch.library.register_fake("vkwr_v1_linear::linear_t_act_f16")
def _(x, weight_t, act):
    N = weight_t.shape[0]
    out_sizes = list(x.shape[:-1]) + [N]
    return torch.empty(out_sizes, dtype=x.dtype, device=x.device)


def linear_t_act_f16(x, weight_t, act):
    return torch.ops.vkwr_v1_linear.linear_t_act_f16(x, weight_t, act)


@torch.library.register_fake("vkwr_v1_linear::linear_t_vres_f16")
def _(x, weight_t, v, v_first, v0):
    return torch.empty_like(v)


def linear_t_vres_f16(x, weight_t, v, v_first, v0):
    return torch.ops.vkwr_v1_linear.linear_t_vres_f16(x, weight_t, v, v_first, v0)

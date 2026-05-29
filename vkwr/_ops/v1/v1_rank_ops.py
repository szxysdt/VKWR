import torch

from vkwr import _v1_rank_C  # noqa: F401


@torch.library.register_fake("vkwr_v1_rank::linear_wag_rank_in_f16")
def _(M, K, Rw, Ra, Rg, xw, xa, xg, w1_t, a1_t, g1_t):
    return [
        torch.empty(M, Rw, dtype=xw.dtype, device=xw.device),
        torch.empty(M, Ra, dtype=xa.dtype, device=xa.device),
        torch.empty(M, Rg, dtype=xg.dtype, device=xg.device),
    ]


def linear_wag_rank_in_f16(M, K, Rw, Ra, Rg, xw, xa, xg, w1_t, a1_t, g1_t):
    return torch.ops.vkwr_v1_rank.linear_wag_rank_in_f16(M, K, Rw, Ra, Rg, xw, xa, xg, w1_t, a1_t, g1_t)


@torch.library.register_fake("vkwr_v1_rank::linear_wagv_rank_in_f16")
def _(M, K, Rw, Ra, Rg, Rv, xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t):
    return [
        torch.empty(M, Rw, dtype=xw.dtype, device=xw.device),
        torch.empty(M, Ra, dtype=xa.dtype, device=xa.device),
        torch.empty(M, Rg, dtype=xg.dtype, device=xg.device),
        torch.empty(M, Rv, dtype=xv.dtype, device=xv.device),
    ]


def linear_wagv_rank_in_f16(M, K, Rw, Ra, Rg, Rv, xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t):
    return torch.ops.vkwr_v1_rank.linear_wagv_rank_in_f16(M, K, Rw, Ra, Rg, Rv, xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t)


@torch.library.register_fake("vkwr_v1_rank::linear_wag_rank_out_f16")
def _(M, C, Kw, Ka, Kg, w1, a1, g1, w2_t, a2_t, g2_t):
    return [
        torch.empty(M, C, dtype=w1.dtype, device=w1.device),
        torch.empty(M, C, dtype=a1.dtype, device=a1.device),
        torch.empty(M, C, dtype=g1.dtype, device=g1.device),
    ]


def linear_wag_rank_out_f16(M, C, Kw, Ka, Kg, w1, a1, g1, w2_t, a2_t, g2_t):
    return torch.ops.vkwr_v1_rank.linear_wag_rank_out_f16(M, C, Kw, Ka, Kg, w1, a1, g1, w2_t, a2_t, g2_t)


@torch.library.register_fake("vkwr_v1_rank::linear_wagv_rank_out_f16")
def _(M, C, Kw, Ka, Kg, Kv, w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0):
    return [
        torch.empty(M, C, dtype=w1.dtype, device=w1.device),
        torch.empty(M, C, dtype=a1.dtype, device=a1.device),
        torch.empty(M, C, dtype=g1.dtype, device=g1.device),
        torch.empty(M, C, dtype=v1.dtype, device=v1.device),
    ]


def linear_wagv_rank_out_f16(M, C, Kw, Ka, Kg, Kv, w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0):
    return torch.ops.vkwr_v1_rank.linear_wagv_rank_out_f16(M, C, Kw, Ka, Kg, Kv, w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0)

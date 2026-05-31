import torch

from vkwr import _v1_rank_C  # noqa: F401


@torch.library.register_fake("vkwr_v1_rank::linear_wag_rank_in_f16")
def _(xw, xa, xg, w1_t, a1_t, g1_t):
    out_shape = list(xw.shape[:-1])
    return [
        torch.empty(out_shape + [w1_t.size(0)], dtype=xw.dtype, device=xw.device),
        torch.empty(out_shape + [a1_t.size(0)], dtype=xa.dtype, device=xa.device),
        torch.empty(out_shape + [g1_t.size(0)], dtype=xg.dtype, device=xg.device),
    ]


def linear_wag_rank_in_f16(xw, xa, xg, w1_t, a1_t, g1_t):
    return torch.ops.vkwr_v1_rank.linear_wag_rank_in_f16(xw, xa, xg, w1_t, a1_t, g1_t)


@torch.library.register_fake("vkwr_v1_rank::linear_wagv_rank_in_f16")
def _(xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t):
    out_shape = list(xw.shape[:-1])
    return [
        torch.empty(out_shape + [w1_t.size(0)], dtype=xw.dtype, device=xw.device),
        torch.empty(out_shape + [a1_t.size(0)], dtype=xa.dtype, device=xa.device),
        torch.empty(out_shape + [g1_t.size(0)], dtype=xg.dtype, device=xg.device),
        torch.empty(out_shape + [v1_t.size(0)], dtype=xv.dtype, device=xv.device),
    ]


def linear_wagv_rank_in_f16(xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t):
    return torch.ops.vkwr_v1_rank.linear_wagv_rank_in_f16(xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t)


@torch.library.register_fake("vkwr_v1_rank::linear_wag_rank_out_f16")
def _(w1, a1, g1, w2_t, a2_t, g2_t):
    out_shape = list(w1.shape[:-1])
    return [
        torch.empty(out_shape + [w2_t.size(0)], dtype=w1.dtype, device=w1.device),
        torch.empty(out_shape + [a2_t.size(0)], dtype=a1.dtype, device=a1.device),
        torch.empty(out_shape + [g2_t.size(0)], dtype=g1.dtype, device=g1.device),
    ]


def linear_wag_rank_out_f16(w1, a1, g1, w2_t, a2_t, g2_t):
    return torch.ops.vkwr_v1_rank.linear_wag_rank_out_f16(w1, a1, g1, w2_t, a2_t, g2_t)


@torch.library.register_fake("vkwr_v1_rank::linear_wagv_rank_out_f16")
def _(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0):
    out_shape = list(w1.shape[:-1])
    return [
        torch.empty(out_shape + [w2_t.size(0)], dtype=w1.dtype, device=w1.device),
        torch.empty(out_shape + [a2_t.size(0)], dtype=a1.dtype, device=a1.device),
        torch.empty(out_shape + [g2_t.size(0)], dtype=g1.dtype, device=g1.device),
        torch.empty(out_shape + [v2_t.size(0)], dtype=v1.dtype, device=v1.device),
    ]


def linear_wagv_rank_out_f16(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0):
    return torch.ops.vkwr_v1_rank.linear_wagv_rank_out_f16(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0)

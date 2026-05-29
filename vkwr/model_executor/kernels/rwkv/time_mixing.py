import torch

from vkwr._ops.rwkv7_ops import rwkv7_fwd_one, rwkv7_fwd_seq
from vkwr._ops.v1.v1_linear_ops import linear_t_act_f16, linear_t_f16
from vkwr._ops.v1.v1_mix_ops import (
    add_vec,
    tmix_kk_a_gate,
    tmix_lnx_rkvres_xg,
    tmix_mix6,
    tmix_vres_gate,
)


def _lowrank_linear(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w1t: torch.Tensor, w2t: torch.Tensor, act=None) -> torch.Tensor:
    hid = linear_t_f16(x, w1t)
    if act is None:
        return linear_t_f16(hid, w2t)
    if act == 1:
        return linear_t_act_f16(hid, w2t, act=1)
    return linear_t_act_f16(hid, w2t, act=2)


def rwkv7_tmix(
    x: torch.Tensor,
    shift_state,
    wkv_state,
    elapsed_t,
    v_first,
    weights: dict,
    path: str = "dense",
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, C = x.shape
    H = weights["att.r_k"].shape[0]

    result = tmix_mix6(
        B,
        T,
        C,
        x,
        shift_state,
        weights["att.x_r"],
        weights["att.x_w"],
        weights["att.x_k"],
        weights["att.x_v"],
        weights["att.x_a"],
        weights["att.x_g"],
    )
    xr, xw, xk, xv, xa, xg = result

    r = linear_t_f16(xr, weights["att.receptance.weight"])
    k = linear_t_f16(xk, weights["att.key.weight"])
    v = linear_t_f16(xv, weights["att.value.weight"])

    w = _lowrank_linear(
        xw,
        weights["att.w1"],
        weights["att.w2"],
        weights["att.w1t"],
        weights["att.w2t"],
        act=1,
    )
    a = _lowrank_linear(
        xa,
        weights["att.a1"],
        weights["att.a2"],
        weights["att.a1t"],
        weights["att.a2t"],
        act=None,
    )
    g = _lowrank_linear(
        xg,
        weights["att.g1"],
        weights["att.g2"],
        weights["att.g1t"],
        weights["att.g2t"],
        act=2,
    )

    if v_first is None:
        v_first = v
    else:
        v12 = _lowrank_linear(
            xv,
            weights["att.v1"],
            weights["att.v2"],
            weights["att.v1t"],
            weights["att.v2t"],
        )
        v = tmix_vres_gate(B, T, C, v, v_first, weights["att.v0"], v12)

    kk_result = tmix_kk_a_gate(
        B,
        T,
        C,
        H,
        k,
        weights["att.k_k"],
        weights["att.a0"],
        a,
        weights["att.k_a"],
    )
    k, neg_kk, kka = kk_result

    w_raw = add_vec(C, w, weights["att.w0"])

    if T == 1:
        y = rwkv7_fwd_one(B, C, wkv_state, r, w_raw, k, v, neg_kk, kka, elapsed_t)
    else:
        y = rwkv7_fwd_seq(B, T, C, wkv_state, r, w_raw, k, v, neg_kk, kka, elapsed_t)

    y = tmix_lnx_rkvres_xg(
        B,
        T,
        C,
        H,
        y,
        r,
        k,
        v,
        weights["att.r_k"],
        weights["att.ln_x.weight"],
        weights["att.ln_x.bias"],
        g,
    )

    out = linear_t_f16(y, weights["att.output.weight"])
    return out, v_first

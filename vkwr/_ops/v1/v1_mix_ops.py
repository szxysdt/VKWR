import torch

from vkwr import _v1_mix_C  # noqa: F401


@torch.library.register_fake("vkwr_v1_mix::tmix_mix6")
def _(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g):
    return [
        torch.empty_like(x),  # x_r
        torch.empty_like(x),  # x_w
        torch.empty_like(x),  # x_k
        torch.empty_like(x),  # x_v
        torch.empty_like(x),  # x_a
        torch.empty_like(x),  # x_g
    ]


def tmix_mix6(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g):
    return torch.ops.vkwr_v1_mix.tmix_mix6(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g)


@torch.library.register_fake("vkwr_v1_mix::tmix_mix6_cfg")
def _(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads):
    return [
        torch.empty_like(x),  # x_r
        torch.empty_like(x),  # x_w
        torch.empty_like(x),  # x_k
        torch.empty_like(x),  # x_v
        torch.empty_like(x),  # x_a
        torch.empty_like(x),  # x_g
    ]


def tmix_mix6_cfg(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads):
    return torch.ops.vkwr_v1_mix.tmix_mix6_cfg(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads)


@torch.library.register_fake("vkwr_v1_mix::tmix_mix6_t1_c4096")
def _(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads=256, vec=4, half_math=False):
    return [
        torch.empty_like(x),  # x_r
        torch.empty_like(x),  # x_w
        torch.empty_like(x),  # x_k
        torch.empty_like(x),  # x_v
        torch.empty_like(x),  # x_a
        torch.empty_like(x),  # x_g
    ]


def tmix_mix6_t1_c4096(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads=256, vec=4, half_math=False):
    C = x.size(2)
    if C != 4096:
        raise ValueError(f"tmix_mix6_t1_c4096 requires C=4096, got C={C}. Use tmix_mix6 for other channel sizes.")
    return torch.ops.vkwr_v1_mix.tmix_mix6_t1_c4096(B, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, threads, vec, half_math)


@torch.library.register_fake("vkwr_v1_mix::tmix_kk_a_gate")
def _(B, T, C, H, k, k_k, a0, a12, k_a):
    return [
        torch.empty_like(k),  # new_k
        torch.empty_like(k),  # neg_kk
        torch.empty_like(k),  # kka
    ]


def tmix_kk_a_gate(B, T, C, H, k, k_k, a0, a12, k_a):
    return torch.ops.vkwr_v1_mix.tmix_kk_a_gate(B, T, C, H, k, k_k, a0, a12, k_a)


@torch.library.register_fake("vkwr_v1_mix::tmix_kk_a_gate_update_shift")
def _(B, T, C, H, k, k_k, a0, a12, k_a, x, shift_state):
    return [
        torch.empty_like(k),  # new_k
        torch.empty_like(k),  # neg_kk
        torch.empty_like(k),  # kka
    ]


def tmix_kk_a_gate_update_shift(B, T, C, H, k, k_k, a0, a12, k_a, x, shift_state):
    return torch.ops.vkwr_v1_mix.tmix_kk_a_gate_update_shift(B, T, C, H, k, k_k, a0, a12, k_a, x, shift_state)


@torch.library.register_fake("vkwr_v1_mix::tmix_lnx_rkvres_xg")
def _(B, T, C, H, x, r, k, v, r_k, weight, bias, g):
    return torch.empty_like(x)


def tmix_lnx_rkvres_xg(B, T, C, H, x, r, k, v, r_k, weight, bias, g):
    return torch.ops.vkwr_v1_mix.tmix_lnx_rkvres_xg(B, T, C, H, x, r, k, v, r_k, weight, bias, g)


@torch.library.register_fake("vkwr_v1_mix::tmix_vres_gate")
def _(B, T, C, v, v_first, v0, v12):
    return torch.empty_like(v)


def tmix_vres_gate(B, T, C, v, v_first, v0, v12):
    return torch.ops.vkwr_v1_mix.tmix_vres_gate(B, T, C, v, v_first, v0, v12)


@torch.library.register_fake("vkwr_v1_mix::cmix_sparse_one")
def _(C, F, x, shift_state, x_k, key_fc, value_fc):
    return torch.empty(1, 1, C, dtype=x.dtype, device=x.device)


def cmix_sparse_one(C, F, x, shift_state, x_k, key_fc, value_fc):
    return torch.ops.vkwr_v1_mix.cmix_sparse_one(C, F, x, shift_state, x_k, key_fc, value_fc)


@torch.library.register_fake("vkwr_v1_mix::cmix_sparse_rows")
def _(B, T, C, F, x, shift_state, x_k, key_fc, value_fc):
    return torch.empty(B, T, C, dtype=x.dtype, device=x.device)


def cmix_sparse_rows(B, T, C, F, x, shift_state, x_k, key_fc, value_fc):
    return torch.ops.vkwr_v1_mix.cmix_sparse_rows(B, T, C, F, x, shift_state, x_k, key_fc, value_fc)


@torch.library.register_fake("vkwr_v1_mix::cmix_sparse_down_one")
def _(C, F, act, value_fc):
    return torch.empty(1, 1, C, dtype=act.dtype, device=act.device)


def cmix_sparse_down_one(C, F, act, value_fc):
    return torch.ops.vkwr_v1_mix.cmix_sparse_down_one(C, F, act, value_fc)


@torch.library.register_fake("vkwr_v1_mix::cmix_sparse_down_rows")
def _(B, T, C, F, act, value_fc):
    return torch.empty(B, T, C, dtype=act.dtype, device=act.device)


def cmix_sparse_down_rows(B, T, C, F, act, value_fc):
    return torch.ops.vkwr_v1_mix.cmix_sparse_down_rows(B, T, C, F, act, value_fc)


@torch.library.register_fake("vkwr_v1_mix::cmix_sparse_down_relu_one")
def _(C, F, preact, value_fc):
    return torch.empty(1, 1, C, dtype=preact.dtype, device=preact.device)


def cmix_sparse_down_relu_one(C, F, preact, value_fc):
    return torch.ops.vkwr_v1_mix.cmix_sparse_down_relu_one(C, F, preact, value_fc)


@torch.library.register_fake("vkwr_v1_mix::cmix_sparse_down_relu_rows")
def _(B, T, C, F, preact, value_fc):
    return torch.empty(B, T, C, dtype=preact.dtype, device=preact.device)


def cmix_sparse_down_relu_rows(B, T, C, F, preact, value_fc):
    return torch.ops.vkwr_v1_mix.cmix_sparse_down_relu_rows(B, T, C, F, preact, value_fc)


@torch.library.register_fake("vkwr_v1_mix::cmix_sparse_down_relu_rows_t512")
def _(B, T, C, F, preact, value_fc):
    return torch.empty(B, T, C, dtype=preact.dtype, device=preact.device)


def cmix_sparse_down_relu_rows_t512(B, T, C, F, preact, value_fc):
    return torch.ops.vkwr_v1_mix.cmix_sparse_down_relu_rows_t512(B, T, C, F, preact, value_fc)


@torch.library.register_fake("vkwr_v1_mix::cmix_mix")
def _(B, T, C, x, shift_state, x_k):
    return torch.empty_like(x)


def cmix_mix(B, T, C, x, shift_state, x_k):
    return torch.ops.vkwr_v1_mix.cmix_mix(B, T, C, x, shift_state, x_k)


@torch.library.register_fake("vkwr_v1_mix::cmix_mix_cfg")
def _(B, T, C, x, shift_state, x_k, threads):
    return torch.empty_like(x)


def cmix_mix_cfg(B, T, C, x, shift_state, x_k, threads):
    return torch.ops.vkwr_v1_mix.cmix_mix_cfg(B, T, C, x, shift_state, x_k, threads)


@torch.library.register_fake("vkwr_v1_mix::relu_square")
def _(x):
    return torch.empty_like(x)


def relu_square(x):
    return torch.ops.vkwr_v1_mix.relu_square(x)


@torch.library.register_fake("vkwr_v1_mix::act_tanh")
def _(x):
    return torch.empty_like(x)


def act_tanh(x):
    return torch.ops.vkwr_v1_mix.act_tanh(x)


@torch.library.register_fake("vkwr_v1_mix::act_sigmoid")
def _(x):
    return torch.empty_like(x)


def act_sigmoid(x):
    return torch.ops.vkwr_v1_mix.act_sigmoid(x)


@torch.library.register_fake("vkwr_v1_mix::add_vec")
def _(C, x, vec):
    return torch.empty_like(x)


def add_vec(C, x, vec):
    return torch.ops.vkwr_v1_mix.add_vec(C, x, vec)

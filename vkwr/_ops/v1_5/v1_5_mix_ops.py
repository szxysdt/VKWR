import torch

from vkwr import _v1_5_mix_C  # noqa: F401


@torch.library.register_fake("vkwr_v1_5_mix::tmix_mix6_varlen")
def _(B, total_tokens, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc, req_id):
    return [
        torch.empty_like(x),
        torch.empty_like(x),
        torch.empty_like(x),
        torch.empty_like(x),
        torch.empty_like(x),
        torch.empty_like(x),
    ]


def tmix_mix6_varlen(B, total_tokens, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc, req_id):
    return torch.ops.vkwr_v1_5_mix.tmix_mix6_varlen(B, total_tokens, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc, req_id)


@torch.library.register_fake("vkwr_v1_5_mix::tmix_mix6_t1_c4096_varlen")
def _(B, total_tokens, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc, req_id, threads=256, vec=4, half_math=False):
    return [
        torch.empty_like(x),
        torch.empty_like(x),
        torch.empty_like(x),
        torch.empty_like(x),
        torch.empty_like(x),
        torch.empty_like(x),
    ]


def tmix_mix6_t1_c4096_varlen(B, total_tokens, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc, req_id, threads=256, vec=4, half_math=False):
    C = x.size(1)
    if C != 4096:
        raise ValueError(f"tmix_mix6_t1_c4096_varlen requires C=4096, got C={C}. Use tmix_mix6_varlen for other channel sizes.")
    return torch.ops.vkwr_v1_5_mix.tmix_mix6_t1_c4096_varlen(
        B, total_tokens, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc, req_id, threads, vec, half_math
    )


@torch.library.register_fake("vkwr_v1_5_mix::cmix_mix_varlen")
def _(B, total_tokens, C, x, shift_state, x_k, query_start_loc, req_id):
    return torch.empty_like(x)


def cmix_mix_varlen(B, total_tokens, C, x, shift_state, x_k, query_start_loc, req_id):
    return torch.ops.vkwr_v1_5_mix.cmix_mix_varlen(B, total_tokens, C, x, shift_state, x_k, query_start_loc, req_id)


@torch.library.register_fake("vkwr_v1_5_mix::cmix_sparse_rows_varlen")
def _(B, total_tokens, C, F, x, shift_state, x_k, key_fc, value_fc, query_start_loc, req_id):
    return torch.empty(total_tokens, C, dtype=x.dtype, device=x.device)


def cmix_sparse_rows_varlen(B, total_tokens, C, F, x, shift_state, x_k, key_fc, value_fc, query_start_loc, req_id):
    return torch.ops.vkwr_v1_5_mix.cmix_sparse_rows_varlen(B, total_tokens, C, F, x, shift_state, x_k, key_fc, value_fc, query_start_loc, req_id)


@torch.library.register_fake("vkwr_v1_5_mix::cmix_sparse_down_relu_rows_varlen")
def _(rows, C, F, preact, value_fc):
    return torch.empty(rows, C, dtype=preact.dtype, device=preact.device)


def cmix_sparse_down_relu_rows_varlen(rows, C, F, preact, value_fc):
    return torch.ops.vkwr_v1_5_mix.cmix_sparse_down_relu_rows_varlen(rows, C, F, preact, value_fc)


@torch.library.register_fake("vkwr_v1_5_mix::cmix_sparse_down_relu_rows_t512_varlen")
def _(rows, C, F, preact, value_fc):
    return torch.empty(rows, C, dtype=preact.dtype, device=preact.device)


def cmix_sparse_down_relu_rows_t512_varlen(rows, C, F, preact, value_fc):
    return torch.ops.vkwr_v1_5_mix.cmix_sparse_down_relu_rows_t512_varlen(rows, C, F, preact, value_fc)

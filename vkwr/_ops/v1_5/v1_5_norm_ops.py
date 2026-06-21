import torch

from vkwr import _v1_5_norm_C  # noqa: F401


@torch.library.register_fake("vkwr_v1_5_norm::add_layer_norm_tmix_mix6_f16_varlen")
def _(total_tokens, x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc, req_id, eps=1e-5):
    return [torch.empty_like(x)] * 7


@torch.library.register_fake("vkwr_v1_5_norm::add_layer_norm_cmix_mix_f16_varlen")
def _(total_tokens, x, residual, shift_state, weight, bias, x_k, query_start_loc, req_id, eps=1e-5):
    return [torch.empty_like(x)] * 2


@torch.library.register_fake("vkwr_v1_5_norm::add_last_layer_norm_f16_varlen")
def _(total_tokens, x, residual, weight, bias, query_start_loc, eps=1e-5):
    B = query_start_loc.size(0) - 1
    C = x.size(-1)
    return torch.empty((B, C), dtype=x.dtype, device=x.device)


def add_layer_norm_tmix_mix6_f16_varlen(total_tokens, x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc, req_id, eps=1e-5):
    """Fused add+LN+tmix_mix6 varlen forward (fp16).

    x, residual: [total_tokens, C] (fp16)
    shift_state: [B, C] (fp16), updated in-place for last token per request
    weight, bias: [C] (fp16)
    x_r, x_w, x_k, x_v, x_a, x_g: [C] (fp16)
    query_start_loc: [B+1] (int32, CUDA)
    req_id: [total_tokens] (int32, CUDA)

    Returns: (x_out, out_r, out_w, out_k, out_v, out_a, out_g)
    """
    return torch.ops.vkwr_v1_5_norm.add_layer_norm_tmix_mix6_f16_varlen(
        total_tokens, x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, query_start_loc, req_id, eps
    )


def add_layer_norm_cmix_mix_f16_varlen(total_tokens, x, residual, shift_state, weight, bias, x_k, query_start_loc, req_id, eps=1e-5):
    """Fused add+LN+cmix_mix varlen forward (fp16).

    x, residual: [total_tokens, C] (fp16)
    shift_state: [B, C] (fp16), updated in-place for last token per request
    weight, bias: [C] (fp16)
    x_k: [C] (fp16)
    query_start_loc: [B+1] (int32, CUDA)
    req_id: [total_tokens] (int32, CUDA)

    Returns: (x_out, mixed)
    """
    return torch.ops.vkwr_v1_5_norm.add_layer_norm_cmix_mix_f16_varlen(total_tokens, x, residual, shift_state, weight, bias, x_k, query_start_loc, req_id, eps)


def add_last_layer_norm_f16_varlen(total_tokens, x, residual, weight, bias, query_start_loc, eps=1e-5):
    """Final layer add+LN varlen forward (fp16).

    Reads last token per request from x/residual using query_start_loc.

    x, residual: [total_tokens, C] (fp16)
    weight, bias: [C] (fp16)
    query_start_loc: [B+1] (int32, CUDA)

    Returns: [B, C] (fp16)
    """
    return torch.ops.vkwr_v1_5_norm.add_last_layer_norm_f16_varlen(total_tokens, x, residual, weight, bias, query_start_loc, eps)

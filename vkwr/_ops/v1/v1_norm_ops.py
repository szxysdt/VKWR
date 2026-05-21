import torch

from vkwr import _v1_norm_C  # noqa: F401


@torch.library.register_fake("vkwr_v1_norm::layer_norm_f16")
def _(x, weight, bias, eps=1e-5):
    return torch.empty_like(x)


def layer_norm_f16(x, weight, bias, eps=1e-5):
    return torch.ops.vkwr_v1_norm.layer_norm_f16(x, weight, bias, eps)


@torch.library.register_fake("vkwr_v1_norm::layer_norm_f16_small")
def _(x, weight, bias, eps=1e-5):
    return torch.empty_like(x)


def layer_norm_f16_small(x, weight, bias, eps=1e-5):
    return torch.ops.vkwr_v1_norm.layer_norm_f16_small(x, weight, bias, eps)


@torch.library.register_fake("vkwr_v1_norm::layer_norm_f16_small512")
def _(x, weight, bias, eps=1e-5):
    return torch.empty_like(x)


def layer_norm_f16_small512(x, weight, bias, eps=1e-5):
    return torch.ops.vkwr_v1_norm.layer_norm_f16_small512(x, weight, bias, eps)


@torch.library.register_fake("vkwr_v1_norm::emb_ln0_bf16_to_f16")
def _(emb, weight, bias, eps=1e-5):
    return torch.empty(emb.shape, dtype=torch.float16, device=emb.device)


def emb_ln0_bf16_to_f16(emb, weight, bias, eps=1e-5):
    return torch.ops.vkwr_v1_norm.emb_ln0_bf16_to_f16(emb, weight, bias, eps)


@torch.library.register_fake("vkwr_v1_norm::add_f16")
def _(x, y):
    return torch.empty_like(x)


def add_f16(x, y):
    return torch.ops.vkwr_v1_norm.add_f16(x, y)


@torch.library.register_fake("vkwr_v1_norm::add_layer_norm_f16")
def _(x, residual, weight, bias, eps=1e-5):
    return [torch.empty_like(x), torch.empty_like(x)]


def add_layer_norm_f16(x, residual, weight, bias, eps=1e-5):
    return torch.ops.vkwr_v1_norm.add_layer_norm_f16(x, residual, weight, bias, eps)


@torch.library.register_fake("vkwr_v1_norm::add_last_layer_norm_f16")
def _(x, residual, weight, bias, eps=1e-5):
    B, T, C = x.shape
    return torch.empty(B, C, dtype=x.dtype, device=x.device)


def add_last_layer_norm_f16(x, residual, weight, bias, eps=1e-5):
    return torch.ops.vkwr_v1_norm.add_last_layer_norm_f16(x, residual, weight, bias, eps)


@torch.library.register_fake("vkwr_v1_norm::add_layer_norm_cmix_mix_f16")
def _(x, residual, shift_state, weight, bias, x_k, eps=1e-5):
    return [torch.empty_like(x), torch.empty_like(x)]


def add_layer_norm_cmix_mix_f16(x, residual, shift_state, weight, bias, x_k, eps=1e-5):
    return torch.ops.vkwr_v1_norm.add_layer_norm_cmix_mix_f16(x, residual, shift_state, weight, bias, x_k, eps)


@torch.library.register_fake("vkwr_v1_norm::add_layer_norm_cmix_mix_f16_cfg")
def _(x, residual, shift_state, weight, bias, x_k, eps, threads):
    return [torch.empty_like(x), torch.empty_like(x)]


def add_layer_norm_cmix_mix_f16_cfg(x, residual, shift_state, weight, bias, x_k, eps, threads):
    return torch.ops.vkwr_v1_norm.add_layer_norm_cmix_mix_f16_cfg(x, residual, shift_state, weight, bias, x_k, eps, threads)


@torch.library.register_fake("vkwr_v1_norm::add_layer_norm_cmix_mix_f16_scalar_stats")
def _(x, residual, shift_state, weight, bias, x_k, eps=1e-5):
    return [torch.empty_like(x), torch.empty_like(x)]


def add_layer_norm_cmix_mix_f16_scalar_stats(x, residual, shift_state, weight, bias, x_k, eps=1e-5):
    return torch.ops.vkwr_v1_norm.add_layer_norm_cmix_mix_f16_scalar_stats(x, residual, shift_state, weight, bias, x_k, eps)


@torch.library.register_fake("vkwr_v1_norm::add_layer_norm_tmix_mix6_f16")
def _(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps=1e-5):
    return [
        torch.empty_like(x),  # x_out
        torch.empty_like(x),  # r
        torch.empty_like(x),  # w
        torch.empty_like(x),  # k
        torch.empty_like(x),  # v
        torch.empty_like(x),  # a
        torch.empty_like(x),  # g
    ]


def add_layer_norm_tmix_mix6_f16(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps=1e-5):
    return torch.ops.vkwr_v1_norm.add_layer_norm_tmix_mix6_f16(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps)


@torch.library.register_fake("vkwr_v1_norm::add_layer_norm_tmix_mix6_f16_cfg")
def _(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps, threads):
    return [
        torch.empty_like(x),  # x_out
        torch.empty_like(x),  # r
        torch.empty_like(x),  # w
        torch.empty_like(x),  # k
        torch.empty_like(x),  # v
        torch.empty_like(x),  # a
        torch.empty_like(x),  # g
    ]


def add_layer_norm_tmix_mix6_f16_cfg(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps, threads):
    return torch.ops.vkwr_v1_norm.add_layer_norm_tmix_mix6_f16_cfg(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps, threads)


@torch.library.register_fake("vkwr_v1_norm::add_layer_norm_tmix_mix6_f16_scalar_stats")
def _(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps=1e-5):
    return [
        torch.empty_like(x),  # x_out
        torch.empty_like(x),  # r
        torch.empty_like(x),  # w
        torch.empty_like(x),  # k
        torch.empty_like(x),  # v
        torch.empty_like(x),  # a
        torch.empty_like(x),  # g
    ]


def add_layer_norm_tmix_mix6_f16_scalar_stats(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps=1e-5):
    return torch.ops.vkwr_v1_norm.add_layer_norm_tmix_mix6_f16_scalar_stats(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps)

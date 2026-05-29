import torch
import torch.library

from vkwr import _v1_wkv_C  # noqa: F401

HEAD_SIZE = 64


@torch.library.register_fake("vkwr_v1_wkv::wkv_seq_fp16")
def _(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t):
    return [y, elapsed_t]


@torch.library.register_fake("vkwr_v1_wkv::wkv_seq_w0_fp16")
def _(B, T, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t):
    return [y, elapsed_t]


@torch.library.register_fake("vkwr_v1_wkv::wkv_one_fp16")
def _(B, C, H, state, r, w, k, v, a, b, y, elapsed_t):
    return [y, elapsed_t]


@torch.library.register_fake("vkwr_v1_wkv::wkv_one_w0_fp16")
def _(B, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t):
    return [y, elapsed_t]


@torch.library.register_fake("vkwr_v1_wkv::wkv_forward_fp32")
def _(B, T, C, H, state, r, w, k, v, a, b, y):
    return [y]


@torch.library.register_fake("vkwr_v1_wkv::wkv_forward_seq_fp32")
def _(B, T, C, H, state, r, w, k, v, a, b, y):
    return [y]


@torch.library.register_fake("vkwr_v1_wkv::wkv_forward_small_fp32")
def _(B, T, C, H, state, r, w, k, v, a, b, y):
    return [y]


@torch.library.register_fake("vkwr_v1_wkv::wkv_forward_block_fp32")
def _(B, T, C, H, state, r, w, k, v, a, b, y):
    return [y]


@torch.library.register_fake("vkwr_v1_wkv::advance_i32")
def _(x, amount):
    return []


# ===== fp16 wrappers (return type C: output buffer) =====
def wkv_seq_fp16(B, T, C, H, state, r, w, k, v, a, b):
    y = torch.empty(B, T, C, dtype=r.dtype, device=r.device)
    elapsed_t = torch.zeros(B, dtype=torch.int32, device=r.device)
    torch.ops.vkwr_v1_wkv.wkv_seq_fp16(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t)
    return y, elapsed_t


def wkv_seq_w0_fp16(B, T, C, H, state, r, w, w0, k, v, a, b):
    y = torch.empty(B, T, C, dtype=r.dtype, device=r.device)
    elapsed_t = torch.zeros(B, dtype=torch.int32, device=r.device)
    torch.ops.vkwr_v1_wkv.wkv_seq_w0_fp16(B, T, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t)
    return y, elapsed_t


def wkv_one_fp16(B, C, H, state, r, w, k, v, a, b):
    y = torch.empty(B, C, dtype=r.dtype, device=r.device)
    elapsed_t = torch.zeros(B, dtype=torch.int32, device=r.device)
    torch.ops.vkwr_v1_wkv.wkv_one_fp16(B, C, H, state, r, w, k, v, a, b, y, elapsed_t)
    return y, elapsed_t


def wkv_one_w0_fp16(B, C, H, state, r, w, w0, k, v, a, b):
    y = torch.empty(B, C, dtype=r.dtype, device=r.device)
    elapsed_t = torch.zeros(B, dtype=torch.int32, device=r.device)
    torch.ops.vkwr_v1_wkv.wkv_one_w0_fp16(B, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t)
    return y, elapsed_t


# ===== fp32 wrappers (return type C: output buffer, no elapsed_t) =====
def wkv_forward_fp32(B, T, C, H, state, r, w, k, v, a, b):
    y = torch.empty(B, T, C, dtype=r.dtype, device=r.device)
    torch.ops.vkwr_v1_wkv.wkv_forward_fp32(B, T, C, H, state, r, w, k, v, a, b, y)
    return y


def wkv_forward_seq_fp32(B, T, C, H, state, r, w, k, v, a, b):
    y = torch.empty(B, T, C, dtype=r.dtype, device=r.device)
    torch.ops.vkwr_v1_wkv.wkv_forward_seq_fp32(B, T, C, H, state, r, w, k, v, a, b, y)
    return y


def wkv_forward_small_fp32(B, T, C, H, state, r, w, k, v, a, b):
    y = torch.empty(B, T, C, dtype=r.dtype, device=r.device)
    torch.ops.vkwr_v1_wkv.wkv_forward_small_fp32(B, T, C, H, state, r, w, k, v, a, b, y)
    return y


def wkv_forward_block_fp32(B, T, C, H, state, r, w, k, v, a, b):
    y = torch.empty(B, T, C, dtype=r.dtype, device=r.device)
    torch.ops.vkwr_v1_wkv.wkv_forward_block_fp32(B, T, C, H, state, r, w, k, v, a, b, y)
    return y


# ===== advance_i32 wrapper (return type C: in-place, returns None) =====
def advance_i32(x, amount):
    torch.ops.vkwr_v1_wkv.advance_i32(x, amount)

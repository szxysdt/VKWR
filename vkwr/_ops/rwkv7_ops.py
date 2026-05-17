import torch
import torch.library
from vkwr import _rwkv7_C

HEAD_SIZE = 64


@torch.library.register_fake("vkwr_rwkv7::forward_seq")
def _(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t):
    pass


@torch.library.register_fake("vkwr_rwkv7::forward_one")
def _(B, C, H, state, r, w, k, v, a, b, y, elapsed_t):
    pass


@torch.library.register_fake("vkwr_rwkv7::spmv_forward")
def _(D, C, vec, mat, out):
    pass


def rwkv7_fwd_seq(B, T, C, state, r, w, k, v, a, b, elapsed_t):
    H = C // HEAD_SIZE
    y = torch.empty((B, T, C), device=k.device, dtype=torch.float16, memory_format=torch.contiguous_format)
    torch.ops.vkwr_rwkv7.forward_seq(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t)
    return y


def rwkv7_fwd_one(B, C, state, r, w, k, v, a, b, elapsed_t):
    H = C // HEAD_SIZE
    y = torch.empty((B, C), device=k.device, dtype=torch.float16, memory_format=torch.contiguous_format)
    torch.ops.vkwr_rwkv7.forward_one(B, C, H, state, r, w, k, v, a, b, y, elapsed_t)
    return y


def spmv_forward(vec, mat):
    D, C = mat.size()
    out = torch.zeros((C,), device=vec.device, dtype=torch.float16)
    torch.ops.vkwr_rwkv7.spmv_forward(D, C, vec, mat, out)
    return out

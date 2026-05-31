import torch
import torch.library

from vkwr import _v1_wkv_C  # noqa: F401

HEAD_SIZE = 64


@torch.library.register_fake("vkwr_v1_wkv::wkv_seq_fp16")
def _(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t):
    pass


@torch.library.register_fake("vkwr_v1_wkv::wkv_seq_w0_fp16")
def _(B, T, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t):
    pass


@torch.library.register_fake("vkwr_v1_wkv::wkv_one_fp16")
def _(B, C, H, state, r, w, k, v, a, b, y, elapsed_t):
    pass


@torch.library.register_fake("vkwr_v1_wkv::wkv_one_w0_fp16")
def _(B, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t):
    pass


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


# ===== fp16 wrappers (elapsed_t managed externally by caller) =====
# Common parameters:
#   B       - batch size
#   C       - hidden dimension
#   H       - number of heads (C = H * HEAD_SIZE)
#   state   - running state buffer [B, H, 64, 64] (fp16), updated in-place
#   w0      - time_decay base value [C] (fp16), added inside kernel before
#             computing decay; only for w0 variants
#   y       - output buffer, written in-place (this is the final output)
#   elapsed_t - int32 counter [B], managed externally by caller
def wkv_seq_fp16(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t):
    """WKV sequence forward (fp16). Writes result to y in-place. y is the final output.

    r, w, k, v, a, b: [B, T, C] (fp16)
    y: [B, T, C] (fp16), output
    """
    torch.ops.vkwr_v1_wkv.wkv_seq_fp16(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t)


def wkv_seq_w0_fp16(B, T, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t):
    """WKV sequence forward with time_decay base w0 (fp16). Writes result to y in-place. y is the final output.

    r, w, k, v, a, b: [B, T, C] (fp16)
    w0: [C] (fp16), time_decay base value
    y: [B, T, C] (fp16), output
    """
    torch.ops.vkwr_v1_wkv.wkv_seq_w0_fp16(B, T, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t)


def wkv_one_fp16(B, C, H, state, r, w, k, v, a, b, y, elapsed_t):
    """WKV single-token forward (fp16). Writes result to y in-place. y is the final output.

    r, w, k, v, a, b: [B, C] (fp16)
    y: [B, C] (fp16), output
    """
    torch.ops.vkwr_v1_wkv.wkv_one_fp16(B, C, H, state, r, w, k, v, a, b, y, elapsed_t)


def wkv_one_w0_fp16(B, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t):
    """WKV single-token forward with time_decay base w0 (fp16). Writes result to y in-place. y is the final output.

    r, w, k, v, a, b: [B, C] (fp16)
    w0: [C] (fp16), time_decay base value
    y: [B, C] (fp16), output
    """
    torch.ops.vkwr_v1_wkv.wkv_one_w0_fp16(B, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t)


# ===== fp32 wrappers (fp32 state accumulation, fp16 I/O, no elapsed_t) =====
# Parameters (shared across all fp32 variants):
#   B       - batch size
#   T       - sequence length
#   C       - hidden dimension
#   H       - number of heads (C = H * HEAD_SIZE)
#   state   - running state buffer [B, H, 64, 64] (fp32), updated in-place
#   r, w, k, v, a, b: [B, T, C] (fp16)
#   y       - output buffer [B, T, C] (fp16), written in-place (this is the final output)
def wkv_forward_fp32(B, T, C, H, state, r, w, k, v, a, b, y):
    """WKV forward with auto kernel selection (mode 0). Dispatches to small or block kernel based on B,T. y is the final output."""
    torch.ops.vkwr_v1_wkv.wkv_forward_fp32(B, T, C, H, state, r, w, k, v, a, b, y)


def wkv_forward_seq_fp32(B, T, C, H, state, r, w, k, v, a, b, y):
    """WKV forward with block kernel (mode 1). Always uses full block kernel regardless of B,T. y is the final output."""
    torch.ops.vkwr_v1_wkv.wkv_forward_seq_fp32(B, T, C, H, state, r, w, k, v, a, b, y)


def wkv_forward_small_fp32(B, T, C, H, state, r, w, k, v, a, b, y):
    """WKV forward with small warp kernel (mode 2). Optimized for short sequences / small batches. y is the final output."""
    torch.ops.vkwr_v1_wkv.wkv_forward_small_fp32(B, T, C, H, state, r, w, k, v, a, b, y)


def wkv_forward_block_fp32(B, T, C, H, state, r, w, k, v, a, b, y):
    """WKV forward with short block kernel (mode 3). Always uses short block kernel. y is the final output."""
    torch.ops.vkwr_v1_wkv.wkv_forward_block_fp32(B, T, C, H, state, r, w, k, v, a, b, y)


# ===== advance_i32 wrapper (return type C: in-place, returns None) =====
def advance_i32(x, amount):
    torch.ops.vkwr_v1_wkv.advance_i32(x, amount)

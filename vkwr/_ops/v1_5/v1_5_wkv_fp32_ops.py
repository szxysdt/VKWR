import torch

from vkwr import _v1_5_wkv_fp32_C  # noqa: F401


@torch.library.register_fake("vkwr_v1_5_wkv::wkv_forward_fp32_varlen")
def _(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, k, v, a, b, y):
    pass


@torch.library.register_fake("vkwr_v1_5_wkv::wkv_forward_seq_fp32_varlen")
def _(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, k, v, a, b, y):
    pass


@torch.library.register_fake("vkwr_v1_5_wkv::wkv_forward_small_fp32_varlen")
def _(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, k, v, a, b, y):
    pass


@torch.library.register_fake("vkwr_v1_5_wkv::wkv_forward_block_fp32_varlen")
def _(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, k, v, a, b, y):
    pass


def wkv_forward_fp32_varlen(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, k, v, a, b, y):
    """WKV fp32 varlen forward (auto mode).

    r, w, k, v, a, b: [total_tokens, C] (fp16)
    y: [total_tokens, C] (fp16), output, written in-place
    max_t: int, max sequence length across all requests (for kernel selection)
    query_start_loc: [B+1] (int32, CUDA)
    state: [B, H, 64, 64] (fp32), updated in-place
    """
    torch.ops.vkwr_v1_5_wkv.wkv_forward_fp32_varlen(
        B,
        total_tokens,
        max_t,
        C,
        H,
        query_start_loc,
        state,
        r,
        w,
        k,
        v,
        a,
        b,
        y,
    )


def wkv_forward_seq_fp32_varlen(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, k, v, a, b, y):
    """WKV fp32 varlen forward (seq mode)."""
    torch.ops.vkwr_v1_5_wkv.wkv_forward_seq_fp32_varlen(
        B,
        total_tokens,
        max_t,
        C,
        H,
        query_start_loc,
        state,
        r,
        w,
        k,
        v,
        a,
        b,
        y,
    )


def wkv_forward_small_fp32_varlen(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, k, v, a, b, y):
    """WKV fp32 varlen forward (small/warp mode)."""
    torch.ops.vkwr_v1_5_wkv.wkv_forward_small_fp32_varlen(
        B,
        total_tokens,
        max_t,
        C,
        H,
        query_start_loc,
        state,
        r,
        w,
        k,
        v,
        a,
        b,
        y,
    )


def wkv_forward_block_fp32_varlen(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, k, v, a, b, y):
    """WKV fp32 varlen forward (block mode)."""
    torch.ops.vkwr_v1_5_wkv.wkv_forward_block_fp32_varlen(
        B,
        total_tokens,
        max_t,
        C,
        H,
        query_start_loc,
        state,
        r,
        w,
        k,
        v,
        a,
        b,
        y,
    )

import torch

from vkwr import _v1_5_wkv_fp16_C  # noqa: F401


@torch.library.register_fake("vkwr_v1_5_wkv::wkv_seq_fp16_varlen")
def _(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, k, v, a, b, y, elapsed_t):
    pass


@torch.library.register_fake("vkwr_v1_5_wkv::wkv_seq_w0_fp16_varlen")
def _(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, w0, k, v, a, b, y, elapsed_t):
    pass


def wkv_seq_fp16_varlen(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, k, v, a, b, y, elapsed_t):
    """WKV varlen forward (fp16).

    r, w, k, v, a, b: [total_tokens, C] (fp16)
    y: [total_tokens, C] (fp16), output, written in-place
    max_t: int, max sequence length across all requests (for kernel selection)
    query_start_loc: [B+1] (int32, CUDA)
    state: [B, H, 64, 64] (fp16), updated in-place
    elapsed_t: [B] (int32)
    """
    torch.ops.vkwr_v1_5_wkv.wkv_seq_fp16_varlen(
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
        elapsed_t,
    )


def wkv_seq_w0_fp16_varlen(B, total_tokens, max_t, C, H, query_start_loc, state, r, w, w0, k, v, a, b, y, elapsed_t):
    """WKV varlen forward with time_decay base w0 (fp16).

    r, w, k, v, a, b: [total_tokens, C] (fp16)
    w0: [C] (fp16), time_decay base value
    y: [total_tokens, C] (fp16), output, written in-place
    max_t: int, max sequence length across all requests (for kernel selection)
    query_start_loc: [B+1] (int32, CUDA)
    state: [B, H, 64, 64] (fp16), updated in-place
    elapsed_t: [B] (int32)
    """
    torch.ops.vkwr_v1_5_wkv.wkv_seq_w0_fp16_varlen(
        B,
        total_tokens,
        max_t,
        C,
        H,
        query_start_loc,
        state,
        r,
        w,
        w0,
        k,
        v,
        a,
        b,
        y,
        elapsed_t,
    )


@torch.library.register_fake("vkwr_v1_5_wkv::advance_i32_varlen")
def _(elapsed, query_start_loc):
    pass


def advance_i32_varlen(elapsed, query_start_loc):
    """Advance int32 elapsed counter in-place by per-batch seq length from query_start_loc.

    elapsed: [B] (int32), updated in-place
    query_start_loc: [B+1] (int32, CUDA)
    """
    torch.ops.vkwr_v1_5_wkv.advance_i32_varlen(elapsed, query_start_loc)

import torch
import torch.library

from vkwr import _common_C  # noqa: F401 — triggers TORCH_LIBRARY registration


@torch.library.register_fake("vkwr_common::gather_decode_state")
def _(L, C, H, N, B, global_shift, global_wkv, global_elapsed, decode_shift, decode_wkv, decode_elapsed, slot_indices):
    pass


@torch.library.register_fake("vkwr_common::scatter_decode_state")
def _(L, C, H, N, B, global_shift, global_wkv, global_elapsed, decode_shift, decode_wkv, decode_elapsed, slot_indices):
    pass


def gather_decode_state(
    L: int,
    C: int,
    H: int,
    N: int,
    B: int,
    global_shift: torch.Tensor,  # [L, 2, max_bsz, C] fp16
    global_wkv: torch.Tensor,  # [L, max_bsz, H, N, N] fp16
    global_elapsed: torch.Tensor,  # [max_bsz] int32
    decode_shift: torch.Tensor,  # [L, 2, B, C] fp16
    decode_wkv: torch.Tensor,  # [L, B, H, N, N] fp16
    decode_elapsed: torch.Tensor,  # [B] int32
    slot_indices: torch.Tensor,  # [B] int64
):
    """Gather state from global buffers into decode temp buffers.

    Uses slot_indices to map each decode position i -> global slot s,
    copying shift, wkv, and elapsed state in a single kernel launch pair.

    Args:
        L: number of layers
        C: hidden dimension
        H: number of heads
        N: head size (state matrix dimension)
        B: current batch size
        global_shift: global shift state [L, 2, max_bsz, C]
        global_wkv: global wkv state [L, max_bsz, H, N, N]
        global_elapsed: global elapsed counter [max_bsz]
        decode_shift: decode temp shift buffer [L, 2, B, C]
        decode_wkv: decode temp wkv buffer [L, B, H, N, N]
        decode_elapsed: decode temp elapsed buffer [B]
        slot_indices: mapping [B] where slot_indices[i] is the global slot for decode position i
    """
    torch.ops.vkwr_common.gather_decode_state(
        L,
        C,
        H,
        N,
        B,
        global_shift,
        global_wkv,
        global_elapsed,
        decode_shift,
        decode_wkv,
        decode_elapsed,
        slot_indices,
    )


def scatter_decode_state(
    L: int,
    C: int,
    H: int,
    N: int,
    B: int,
    global_shift: torch.Tensor,  # [L, 2, max_bsz, C] fp16
    global_wkv: torch.Tensor,  # [L, max_bsz, H, N, N] fp16
    global_elapsed: torch.Tensor,  # [max_bsz] int32
    decode_shift: torch.Tensor,  # [L, 2, B, C] fp16
    decode_wkv: torch.Tensor,  # [L, B, H, N, N] fp16
    decode_elapsed: torch.Tensor,  # [B] int32
    slot_indices: torch.Tensor,  # [B] int64
):
    """Scatter state from decode temp buffers back into global buffers.

    Inverse of gather_decode_state. Uses slot_indices to map each decode
    position i -> global slot s, writing updated shift, wkv, and elapsed
    state back.

    Args:
        L: number of layers
        C: hidden dimension
        H: number of heads
        N: head size (state matrix dimension)
        B: current batch size
        global_shift: global shift state [L, 2, max_bsz, C]
        global_wkv: global wkv state [L, max_bsz, H, N, N]
        global_elapsed: global elapsed counter [max_bsz]
        decode_shift: decode temp shift buffer [L, 2, B, C]
        decode_wkv: decode temp wkv buffer [L, B, H, N, N]
        decode_elapsed: decode temp elapsed buffer [B]
        slot_indices: mapping [B] where slot_indices[i] is the global slot for decode position i
    """
    torch.ops.vkwr_common.scatter_decode_state(
        L,
        C,
        H,
        N,
        B,
        global_shift,
        global_wkv,
        global_elapsed,
        decode_shift,
        decode_wkv,
        decode_elapsed,
        slot_indices,
    )

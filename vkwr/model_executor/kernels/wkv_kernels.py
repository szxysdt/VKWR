from vkwr._ops.rwkv7_ops import rwkv7_fwd_one, rwkv7_fwd_seq

__all__ = ["rwkv7_fwd_seq", "rwkv7_fwd_one"]


def wkv_forward_seq(B, T, C, wkv_state, r, w, k, v, neg_kk, kka, elapsed_t):
    return rwkv7_fwd_seq(B, T, C, wkv_state, r, w, k, v, neg_kk, kka, elapsed_t)


def wkv_forward_one(B, C, wkv_state, r, w, k, v, neg_kk, kka, elapsed_t):
    return rwkv7_fwd_one(B, C, wkv_state, r, w, k, v, neg_kk, kka, elapsed_t)

import torch

from vkwr._ops.v1.v1_linear_ops import linear_t_f16
from vkwr._ops.v1.v1_mix_ops import cmix_mix, relu_square


def rwkv7_cmix(
    x: torch.Tensor,
    shift_state,
    weights: dict,
    path: str = "dense",
) -> torch.Tensor:
    B, T, C = x.shape
    mixed = cmix_mix(B, T, C, x, shift_state, weights["ffn.x_k"])
    hid = linear_t_f16(mixed, weights["ffn.key.weight"])
    k = relu_square(hid)
    return linear_t_f16(k, weights["ffn.value.weight"])

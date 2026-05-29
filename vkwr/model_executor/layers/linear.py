import torch
import torch.nn as nn

from vkwr._ops.v1.v1_linear_ops import linear_f16, linear_t_act_f16, linear_t_f16


class RWKV7Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, is_transposed: bool = True):
        super().__init__()
        self.register_buffer("weight", weight)
        self.is_transposed = is_transposed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_transposed:
            return linear_t_f16(x, self.weight)
        return linear_f16(x, self.weight)


class RWKV7LowRankLinear(nn.Module):
    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, w1t: torch.Tensor, w2t: torch.Tensor, act: int = 0):
        super().__init__()
        self.register_buffer("w1", w1)
        self.register_buffer("w2", w2)
        self.register_buffer("w1t", w1t)
        self.register_buffer("w2t", w2t)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        intermediate = linear_t_f16(x, self.w1t)
        if self.act == 0:
            return linear_t_f16(intermediate, self.w2t)
        if self.act == 1:
            return linear_t_act_f16(intermediate, self.w2t, act=1)
        return linear_t_act_f16(intermediate, self.w2t, act=2)

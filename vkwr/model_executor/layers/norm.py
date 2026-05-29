import torch
import torch.nn as nn

from vkwr._ops.v1.v1_norm_ops import add_layer_norm_f16, layer_norm_f16


class RWKV7LayerNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm_f16(x, self.weight, self.bias)


class RWKV7AddLayerNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        return add_layer_norm_f16(x, residual, self.weight, self.bias)

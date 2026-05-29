import torch
import torch.nn as nn

from vkwr._ops.v1.v1_norm_ops import emb_ln0_bf16_to_f16


class RWKV7Embedding(nn.Module):
    def __init__(self, z: dict, device: torch.device):
        super().__init__()
        emb_w = z["emb.weight"]
        ln0_w = z["blocks.0.ln0.weight"]
        ln0_b = z["blocks.0.ln0.bias"]
        self.register_buffer("weight", emb_ln0_bf16_to_f16(emb_w, ln0_w, ln0_b))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]

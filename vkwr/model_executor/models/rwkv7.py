import torch
import torch.nn as nn

from vkwr._ops.rwkv7_ops import HEAD_SIZE
from vkwr._ops.v1.v1_linear_ops import linear_t_f16
from vkwr._ops.v1.v1_norm_ops import add_last_layer_norm_f16, add_layer_norm_f16, layer_norm_f16
from vkwr._ops.v1.v1_wkv_ops import advance_i32
from vkwr.model_executor.kernels.rwkv.channel_mixing import rwkv7_cmix
from vkwr.model_executor.kernels.rwkv.time_mixing import rwkv7_tmix
from vkwr.model_executor.layers.embedding import RWKV7Embedding
from vkwr.model_executor.layers.norm import RWKV7LayerNorm


class RWKV7Block(nn.Module):
    def __init__(self, layer_id: int, z: dict, device: torch.device):
        super().__init__()
        p = f"blocks.{layer_id}."
        self.register_buffer("ln1_weight", z[f"{p}ln1.weight"].squeeze())
        self.register_buffer("ln1_bias", z[f"{p}ln1.bias"].squeeze())
        self.ln2 = RWKV7LayerNorm(z[f"{p}ln2.weight"], z[f"{p}ln2.bias"])
        self.tmix_weights = self._collect_tmix_weights(z, p)
        self.cmix_weights = self._collect_cmix_weights(z, p)

    def _collect_tmix_weights(self, z: dict, p: str) -> dict:
        return {
            "att.r_k": z[f"{p}att.r_k"].contiguous(),
            "att.x_r": z[f"{p}att.x_r"],
            "att.x_w": z[f"{p}att.x_w"],
            "att.x_k": z[f"{p}att.x_k"],
            "att.x_v": z[f"{p}att.x_v"],
            "att.x_a": z[f"{p}att.x_a"],
            "att.x_g": z[f"{p}att.x_g"],
            "att.receptance.weight": z[f"{p}att.receptance.weight"],
            "att.key.weight": z[f"{p}att.key.weight"],
            "att.value.weight": z[f"{p}att.value.weight"],
            "att.output.weight": z[f"{p}att.output.weight"],
            "att.w1": z[f"{p}att.w1"],
            "att.w2": z[f"{p}att.w2"],
            "att.w1t": z[f"{p}att.w1.t"],
            "att.w2t": z[f"{p}att.w2.t"],
            "att.a1": z[f"{p}att.a1"],
            "att.a2": z[f"{p}att.a2"],
            "att.a1t": z[f"{p}att.a1.t"],
            "att.a2t": z[f"{p}att.a2.t"],
            "att.g1": z[f"{p}att.g1"],
            "att.g2": z[f"{p}att.g2"],
            "att.g1t": z[f"{p}att.g1.t"],
            "att.g2t": z[f"{p}att.g2.t"],
            "att.v0": z[f"{p}att.v0"],
            "att.v1": z[f"{p}att.v1"],
            "att.v2": z[f"{p}att.v2"],
            "att.v1t": z[f"{p}att.v1.t"],
            "att.v2t": z[f"{p}att.v2.t"],
            "att.k_k": z[f"{p}att.k_k"],
            "att.a0": z[f"{p}att.a0"],
            "att.k_a": z[f"{p}att.k_a"],
            "att.w0": z[f"{p}att.w0"],
            "att.ln_x.weight": z[f"{p}att.ln_x.weight"],
            "att.ln_x.bias": z[f"{p}att.ln_x.bias"],
        }

    def _collect_cmix_weights(self, z: dict, p: str) -> dict:
        return {
            "ffn.x_k": z[f"{p}ffn.x_k"],
            "ffn.key.weight": z[f"{p}ffn.key.weight"],
            "ffn.value.weight": z[f"{p}ffn.value.weight"],
        }

    def forward(
        self,
        x: torch.Tensor,
        xx: torch.Tensor,
        shift_state,
        wkv_state,
        elapsed_t,
        v_first,
        path: str,
    ):
        xx_tmix, v_first = rwkv7_tmix(
            xx,
            shift_state[0],
            wkv_state,
            elapsed_t,
            v_first,
            self.tmix_weights,
            path,
        )
        x, xx_ln2 = add_layer_norm_f16(x, xx_tmix, self.ln2.weight, self.ln2.bias)
        xx_cmix = rwkv7_cmix(xx_ln2, shift_state[1], self.cmix_weights, path)
        return x, xx_cmix, v_first


class RWKV7Model(nn.Module):
    def __init__(self, z: dict, device: torch.device, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = hidden_size // HEAD_SIZE
        self.head_size = HEAD_SIZE
        self.device = device
        self.embedding = RWKV7Embedding(z, device)
        self.blocks = nn.ModuleList([RWKV7Block(i, z, device) for i in range(num_layers)])
        self.register_buffer("ln_out_weight", z["ln_out.weight"])
        self.register_buffer("ln_out_bias", z["ln_out.bias"])
        self.register_buffer("head_weight", z["head.weight"])

    def zero_state(self, B: int) -> list:
        """创建零初始化推理状态，对齐 Albatross RWKV7.zero_state。

        State 结构:
          state[0] = shift: [L, 2, B, C] — shift[0] 为 tmix, shift[1] 为 cmix
          state[1] = wkv:   [L, B, H, N, N] — WKV state matrix per layer
          state[2] = elapsed_t: [B] int32 — 全局时间步计数器
        """
        L = self.num_layers
        B = B
        H = self.num_heads
        N = self.head_size
        C = self.hidden_size
        return [
            torch.zeros((L, 2, B, C), dtype=torch.float16, device=self.device),
            torch.zeros((L, B, H, N, N), dtype=torch.float16, device=self.device),
            torch.zeros((B,), dtype=torch.int32, device=self.device),
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        state,
        path: str = "dense",
    ) -> torch.Tensor:
        B, T = input_ids.shape
        L = self.num_layers
        x = self.embedding(input_ids)
        xx = layer_norm_f16(x, self.blocks[0].ln1_weight, self.blocks[0].ln1_bias)
        v_first = None
        for i in range(L):
            shift_state = state[0][i]
            wkv_state = state[1][i]
            elapsed_t = state[2]
            x, xx_cmix, v_first = self.blocks[i](x, xx, shift_state, wkv_state, elapsed_t, v_first, path)
            if i + 1 < L:
                x, xx = add_layer_norm_f16(
                    x,
                    xx_cmix,
                    self.blocks[i + 1].ln1_weight,
                    self.blocks[i + 1].ln1_bias,
                )
            else:
                x = add_last_layer_norm_f16(
                    x,
                    xx_cmix,
                    self.ln_out_weight,
                    self.ln_out_bias,
                )
        advance_i32(state[2], T)
        logits = linear_t_f16(x, self.head_weight)
        return logits

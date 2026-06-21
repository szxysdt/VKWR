import torch
import torch.nn as nn

from vkwr._ops.v1.v1_linear_ops import (
    linear_f16,
    linear_f16_orig,
    linear_f16_orig_lt_cfg,
    linear_orig_rows_cfg_f16,
    linear_orig_rows_exact_f16,
    linear_orig_rows_f16,
    linear_t_act_f16,
    linear_t_f16,
)
from vkwr._ops.v1.v1_mix_ops import act_sigmoid, act_tanh
from vkwr.config.model import WeightConfig
from vkwr.model_executor.layers.path_dispatcher_config import PathConfig, lorank_cfg


class RWKV7LinearDispatcher(nn.Module):
    def __init__(self, hidden_size: int, weight_config: WeightConfig | None = None):
        super().__init__()
        self.C = hidden_size
        self.weight_config = weight_config if weight_config is not None else WeightConfig()

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.numel() == x.size(-1) and weight.size(1) % 64 == 0:
            return torch.ops.vkwr_v1_linear.linear_f16_m1_splitk(x.contiguous(), weight)
        return linear_f16(x.contiguous(), weight)

    def linear_orig_layout(self, x: torch.Tensor, weight: torch.Tensor, path: PathConfig, group: str) -> torch.Tensor:
        if not self.weight_config.use_orig_linear(group):
            return self.linear(x, weight)
        if path.rows == 1:
            if group == "ffn_key":
                if self.C == 2560:
                    return linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, True)
                return linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, self.C <= 1024)
            return linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, group != "att_c2c" or self.C < 2048)
        if path.rows == 2:
            if group == "att_c2c":
                return linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
            if group == "ffn_key":
                if self.C == 2560:
                    return linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
                if self.C < 4096:
                    return linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
                return linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
            if group == "head" and self.C == 2560:
                return linear_orig_rows_exact_f16(x.contiguous(), weight, 128, 2, False)
            return linear_orig_rows_exact_f16(x.contiguous(), weight, 64, 2, True)
        if path.rows == 3:
            if group == "head":
                if self.C <= 2048:
                    return linear_f16_orig(x.contiguous(), weight)
                if self.C == 2560:
                    return linear_f16_orig(x.contiguous(), weight)
                return linear_orig_rows_f16(x.contiguous(), weight, 3, 2)
            if group == "ffn_key":
                if self.C <= 1024:
                    return linear_orig_rows_cfg_f16(x.contiguous(), weight, 64, 3, 4)
                if self.C == 2048:
                    return linear_f16_orig(x.contiguous(), weight)
                if self.C == 2560:
                    return linear_f16_orig(x.contiguous(), weight)
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if group == "att_c2c":
                if self.C == 768:
                    return linear_orig_rows_f16(x.contiguous(), weight, 1, 2)
                if self.C == 1024:
                    return linear_orig_rows_f16(x.contiguous(), weight, 2, 2)
                if self.C == 2048:
                    return linear_orig_rows_f16(x.contiguous(), weight, 3, 4)
                if self.C == 2560:
                    return linear_orig_rows_f16(x.contiguous(), weight, 3, 2)
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
            return linear_orig_rows_cfg_f16(x.contiguous(), weight, 64, 3, 4)
        if path.rows == 4:
            if group == "ffn_key":
                if self.C <= 1024:
                    return linear_orig_rows_cfg_f16(x.contiguous(), weight, 64, 2, 4)
                if self.C == 2048:
                    return linear_f16_orig(x.contiguous(), weight)
                if self.C == 2560:
                    return linear_f16_orig(x.contiguous(), weight)
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if group == "att_c2c":
                if self.C <= 1024:
                    return linear_orig_rows_f16(x.contiguous(), weight, 2, 2)
                if self.C == 2048:
                    return linear_orig_rows_f16(x.contiguous(), weight, 4, 2)
                if self.C == 2560:
                    return linear_orig_rows_f16(x.contiguous(), weight, 4, 2)
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
        if group == "head":
            if self.C == 768:
                if 192 <= path.rows < 256:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 3)
                if 96 <= path.rows < 160:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if self.C == 1024:
                if 256 <= path.rows < 384:
                    return linear_f16_orig(x.contiguous(), weight)
                if 192 <= path.rows < 256:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
                if 96 <= path.rows < 160:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if self.C == 2048:
                if 256 <= path.rows < 384:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
                if 192 <= path.rows < 256:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 6)
                if 128 <= path.rows < 160:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
                if 96 <= path.rows < 112:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if self.C == 2560:
                if path.rows >= 256:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
                if path.rows >= 192:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 5)
                if path.rows >= 160:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 5)
                if path.rows >= 128:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
                if path.rows >= 96:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
                if path.rows >= 80:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
                if path.rows >= 72:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 1024:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
            if path.rows >= 512:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
            if path.rows >= 384:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
            if path.rows >= 256:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if path.rows >= 192:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
            if path.rows >= 160:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 128:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
            if path.rows >= 112:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 96:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 80:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
            if path.rows >= 72:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
        if group == "att_c2c":
            if self.C == 2560 and 17 <= path.rows <= 20:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if self.C == 768:
                if 256 <= path.rows < 384:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
                if 96 <= path.rows < 112:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 3)
            if self.C == 1024:
                if 256 <= path.rows < 384:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
                if 96 <= path.rows < 112:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 6)
            if self.C == 2048:
                if 256 <= path.rows < 384:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 3)
                if 192 <= path.rows < 256:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 0)
                if 96 <= path.rows < 112:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
            if self.C == 2560:
                if path.rows >= 256:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
                if path.rows >= 160:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
                if path.rows >= 128:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
                if path.rows >= 112:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 3)
                if path.rows >= 96:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
                if path.rows >= 72:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
                if path.rows >= 5:
                    return linear_f16_orig(x.contiguous(), weight)
            if path.rows >= 1024:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
            if path.rows >= 768:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 512:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 384:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 2)
            if path.rows >= 256:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
            if path.rows >= 192:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 160:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
            if path.rows >= 112:
                return linear_f16_orig(x.contiguous(), weight)
            if path.rows >= 96:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 5)
            if path.rows >= 72:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 48:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 6)
            if path.rows >= 32:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 24:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 6)
            if path.rows >= 12:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 5:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
        if group == "ffn_key":
            if self.C == 2560 and 17 <= path.rows <= 20:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if self.C == 768:
                if 256 <= path.rows < 384:
                    return linear_f16_orig(x.contiguous(), weight)
                if 96 <= path.rows < 112:
                    return linear_f16_orig(x.contiguous(), weight)
            if self.C == 1024:
                if 256 <= path.rows < 384:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
                if 192 <= path.rows < 256:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
                if 96 <= path.rows < 160:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 2)
            if self.C == 2048 and 128 <= path.rows < 160:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 3)
            if self.C == 2560:
                if path.rows >= 192:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 5)
                if path.rows >= 160:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 4)
                if path.rows >= 128:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 5)
                if path.rows >= 112:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 4)
                if path.rows >= 96:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 4)
                if path.rows >= 80:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 3)
                if path.rows >= 72:
                    return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 4)
                if path.rows >= 3:
                    return linear_f16_orig(x.contiguous(), weight)
            if path.rows >= 1024:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows >= 768:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 512:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 3)
            if path.rows >= 384:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 256:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 4)
            if path.rows >= 192:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if path.rows >= 160:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 2)
            if path.rows >= 128:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 0)
            if path.rows >= 112:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 3)
            if path.rows >= 96:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 32, 1)
            if path.rows >= 72:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 128, 1)
            if path.rows >= 48:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
            if path.rows >= 12:
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 0)
            if path.rows in (5, 6):
                return linear_f16_orig_lt_cfg(x.contiguous(), weight, 0, 1)
        return linear_f16_orig(x.contiguous(), weight)

    def linear_lowrank_orig(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return linear_f16(x.contiguous(), weight)

    def linear_t_orig(self, x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        return linear_f16_orig(x.contiguous(), weight_t)

    def linear_rank_in(self, x, weight, weight_t, rows):
        if weight_t is not None and rows <= lorank_cfg.IN_ROWS_T:
            return linear_t_f16(x.contiguous(), weight_t)
        return self.linear_lowrank_orig(x, weight) if weight is not None else self.linear_t_orig(x, weight_t)

    def linear_rank_out(self, x, weight, weight_t, rows):
        if weight_t is not None and lorank_cfg.can_use_lowrank_out_fused(rows, self.C):
            return linear_t_f16(x.contiguous(), weight_t)
        return self.linear_lowrank_orig(x, weight) if weight is not None else self.linear_t_orig(x, weight_t)

    def linear_rank_out_act(self, x, weight, weight_t, rows, act):
        if weight_t is not None and lorank_cfg.can_use_lowrank_out_fused(rows, self.C):
            return linear_t_act_f16(x.contiguous(), weight_t, act)
        x = act_tanh(x.contiguous()) if act == 1 else act_sigmoid(x.contiguous())
        return self.linear_lowrank_orig(x.contiguous(), weight) if weight is not None else self.linear_t_orig(x, weight_t)

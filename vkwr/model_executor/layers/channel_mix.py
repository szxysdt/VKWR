import torch
from torch import nn

from vkwr._ops.v1.v1_mix_ops import (
    cmix_mix,
    cmix_sparse_down_relu_one,
    cmix_sparse_down_relu_rows,
    cmix_sparse_down_relu_rows_t512,
    cmix_sparse_one,
    cmix_sparse_rows,
    relu_square,
)
from vkwr.model_executor.layers.linear import RWKV7LinearDispatcher
from vkwr.model_executor.layers.path_dispatcher_config import CmixConfig, PathConfig


class RWKV7ChannelMixDispatcher(nn.Module):
    def __init__(
        self,
        C: int,
        linear_dispatcher: RWKV7LinearDispatcher,
        cmix_nofc_t512_min_rows: int = 8,
        cmix_config: CmixConfig | None = None,
        model_dict: dict | None = None,
    ):
        super().__init__()
        self.C = C
        self.linear_dispatcher = linear_dispatcher
        self.cmix_nofc_t512_min_rows = cmix_nofc_t512_min_rows
        self.cmix_config = cmix_config or CmixConfig()
        self.model_dict = model_dict

    def forward(
        self,
        x: torch.Tensor,
        shift_state: torch.Tensor,
        param_prefix: str,
        path: PathConfig,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if weights is None:
            weights = self.model_dict
        B, T, _ = x.shape
        cmix_mode = path.cmix_mode

        if cmix_mode == self.cmix_config.CMIX_B1T1_SPARSE:
            return cmix_sparse_one(
                self.C,
                weights[param_prefix + "ffn.key.weight.fc"].size(0),
                x.contiguous(),
                shift_state[1],
                weights[param_prefix + "ffn.x_k"],
                weights[param_prefix + "ffn.key.weight.fc"],
                weights[param_prefix + "ffn.value.weight"],
            )
        if cmix_mode == self.cmix_config.CMIX_ROWS2_SPARSE:
            # currently dead code — select_path never assigns CMIX_ROWS2_SPARSE (rows==2 goes to CMIX_ROWS2_NOFC)
            return cmix_sparse_rows(
                B,
                T,
                self.C,
                weights[param_prefix + "ffn.key.weight.fc"].size(0),
                x.contiguous(),
                shift_state[1],
                weights[param_prefix + "ffn.x_k"],
                weights[param_prefix + "ffn.key.weight.fc"],
                weights[param_prefix + "ffn.value.weight"],
            )

        mixed = cmix_mix(B, T, self.C, x.contiguous(), shift_state[1], weights[param_prefix + "ffn.x_k"])
        return self.forward_from_mixed(mixed, param_prefix, path, weights)

    def forward_from_mixed(
        self,
        mixed: torch.Tensor,
        param_prefix: str,
        path: PathConfig,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if weights is None:
            weights = self.model_dict
        B, T, _ = mixed.shape
        hid = self.linear_dispatcher.linear_orig_layout(mixed, weights[param_prefix + "ffn.key.weight"], path, "ffn_key")
        cmix_mode = path.cmix_mode

        if cmix_mode == self.cmix_config.CMIX_B1T1_NOFC:
            return cmix_sparse_down_relu_one(
                self.C,
                weights[param_prefix + "ffn.value.weight"].size(0),
                hid.view(-1).contiguous(),
                weights[param_prefix + "ffn.value.weight"],
            )
        if cmix_mode == self.cmix_config.CMIX_ROWS2_NOFC:
            F = weights[param_prefix + "ffn.value.weight"].size(0)
            if path.rows >= self.cmix_nofc_t512_min_rows and self.C % 512 == 0 and F % 512 == 0:
                return cmix_sparse_down_relu_rows_t512(
                    B,
                    T,
                    self.C,
                    F,
                    hid.contiguous(),
                    weights[param_prefix + "ffn.value.weight"],
                )
            return cmix_sparse_down_relu_rows(
                B,
                T,
                self.C,
                F,
                hid.contiguous(),
                weights[param_prefix + "ffn.value.weight"],
            )
        k = relu_square(hid.contiguous())
        return self.linear_dispatcher.linear(k, weights[param_prefix + "ffn.value.weight"])

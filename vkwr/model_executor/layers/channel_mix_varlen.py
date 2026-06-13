import torch
from torch import nn

from vkwr._ops.v1.v1_mix_ops import cmix_mix, relu_square
from vkwr._ops.v1_5.v1_5_mix_ops import cmix_mix_varlen
from vkwr.model_executor.layers.linear import RWKV7LinearDispatcher
from vkwr.model_executor.layers.path_dispatcher_config_varlen import CmixConfig, PathConfig


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
        query_start_loc: torch.Tensor,
        req_id: torch.Tensor,
        B: int,
        is_uniform: bool,
        max_t: int,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if weights is None:
            weights = self.model_dict

        if is_uniform:
            mixed = cmix_mix(
                B,
                max_t,
                self.C,
                x.view(B, max_t, self.C).contiguous(),
                shift_state[1],
                weights[param_prefix + "ffn.x_k"],
            )
            out = self.forward_from_mixed(mixed, param_prefix, path, weights)
            return out.view(-1, self.C)
        else:
            mixed = cmix_mix_varlen(
                B,
                x.size(0),
                self.C,
                x.contiguous(),
                shift_state[1],
                weights[param_prefix + "ffn.x_k"],
                query_start_loc,
                req_id,
            )
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
        # Linear + relu_square + Linear — all per-token, walk v1
        hid = self.linear_dispatcher.linear_orig_layout(mixed, weights[param_prefix + "ffn.key.weight"], path, "ffn_key")
        k = relu_square(hid.contiguous())
        return self.linear_dispatcher.linear(k, weights[param_prefix + "ffn.value.weight"])

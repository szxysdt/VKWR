import torch
from torch import nn

from vkwr._ops.v2.v2_mix_ops import (
    cmix_mix,
    cmix_sparse_down_relu_one,
    cmix_sparse_down_relu_rows,
    cmix_sparse_down_relu_rows_t512,
    cmix_sparse_one,
    cmix_sparse_rows,
    relu_square,
)
from vkwr._ops.v2_5.v2_5_mix_ops import cmix_mix_varlen
from vkwr.config.model import RWKV7InferenceConfig
from vkwr.model_executor.layers.linear import RWKV7LinearDispatcher
from vkwr.model_executor.layers.path_dispatcher_config_varlen import CmixConfig, CmixThresholds, PathConfig


class RWKV7ChannelMixDispatcher(nn.Module):
    def __init__(
        self,
        C: int,
        linear_dispatcher: RWKV7LinearDispatcher,
        cmix_nofc_t512_min_rows: int = 8,
        cmix_config: CmixConfig | None = None,
        model_dict: dict | None = None,
        inference_config: RWKV7InferenceConfig | None = None,
    ):
        super().__init__()
        self.C = C
        self.linear_dispatcher = linear_dispatcher
        self.cmix_nofc_t512_min_rows = cmix_nofc_t512_min_rows
        self.cmix_config = cmix_config or CmixConfig()
        self.model_dict = model_dict
        self.inference_config = inference_config if inference_config is not None else RWKV7InferenceConfig()

    def _compute_cmix_mode(self, B: int, T: int) -> str:
        """Compute cmix_mode using the original fixed PathSelector logic."""
        rows = B * T
        cc = self.cmix_config
        th = CmixThresholds()

        if self.inference_config.cmix_sparse == "off":
            return cc.CMIX_DENSE
        if self.inference_config.cmix_sparse == "no-fc":
            use_nofc = rows <= th.nofc_max_rows or (rows == 20 and T <= th.nofc_row20_max_t)
            if rows == 1:
                return cc.CMIX_B1T1_NOFC
            if use_nofc:
                return cc.CMIX_ROWS2_NOFC
            return cc.CMIX_DENSE
        if rows == 1:
            return cc.CMIX_B1T1_SPARSE
        if rows == 2:
            return cc.CMIX_ROWS2_NOFC
        use_nofc = rows <= th.nofc_max_rows or (rows == 20 and T <= th.nofc_row20_max_t)
        if use_nofc:
            return cc.CMIX_ROWS2_NOFC
        return cc.CMIX_DENSE

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
        slot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if weights is None:
            weights = self.model_dict

        if is_uniform:
            x3d = x.view(B, max_t, self.C).contiguous()
            cmix_mode = self._compute_cmix_mode(B, max_t)

            if cmix_mode == self.cmix_config.CMIX_B1T1_SPARSE:
                return cmix_sparse_one(
                    self.C,
                    weights[param_prefix + "ffn.key.weight.fc"].size(0),
                    x3d.contiguous(),
                    slot_indices,
                    shift_state[1],
                    weights[param_prefix + "ffn.x_k"],
                    weights[param_prefix + "ffn.key.weight.fc"],
                    weights[param_prefix + "ffn.value.weight"],
                ).view(-1, self.C)
            if cmix_mode == self.cmix_config.CMIX_ROWS2_SPARSE:
                return cmix_sparse_rows(
                    B,
                    max_t,
                    self.C,
                    weights[param_prefix + "ffn.key.weight.fc"].size(0),
                    x3d.contiguous(),
                    slot_indices,
                    shift_state[1],
                    weights[param_prefix + "ffn.x_k"],
                    weights[param_prefix + "ffn.key.weight.fc"],
                    weights[param_prefix + "ffn.value.weight"],
                ).view(-1, self.C)

            mixed = cmix_mix(B, max_t, self.C, x3d, slot_indices, shift_state[1], weights[param_prefix + "ffn.x_k"])
            return self.forward_from_mixed(mixed, param_prefix, path, weights, cmix_mode).view(-1, self.C)
        else:
            mixed = cmix_mix_varlen(
                B,
                x.size(0),
                self.C,
                x.contiguous(),
                slot_indices,
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
        cmix_mode: str | None = None,
    ) -> torch.Tensor:
        if weights is None:
            weights = self.model_dict

        if cmix_mode is None:
            cmix_mode = path.cmix_mode

        if cmix_mode == self.cmix_config.CMIX_B1T1_NOFC:
            B, T, _ = mixed.shape
            hid = self.linear_dispatcher.linear_orig_layout(mixed, weights[param_prefix + "ffn.key.weight"], path, "ffn_key")
            return cmix_sparse_down_relu_one(
                self.C,
                weights[param_prefix + "ffn.value.weight"].size(0),
                hid.view(-1).contiguous(),
                weights[param_prefix + "ffn.value.weight"],
            )
        if cmix_mode == self.cmix_config.CMIX_ROWS2_NOFC:
            B, T, _ = mixed.shape
            hid = self.linear_dispatcher.linear_orig_layout(mixed, weights[param_prefix + "ffn.key.weight"], path, "ffn_key")
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

        hid = self.linear_dispatcher.linear_orig_layout(mixed, weights[param_prefix + "ffn.key.weight"], path, "ffn_key")
        k = relu_square(hid.contiguous())
        return self.linear_dispatcher.linear(k, weights[param_prefix + "ffn.value.weight"])

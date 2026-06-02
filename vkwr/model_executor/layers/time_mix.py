import warnings

import torch
from torch import nn

from vkwr._ops.v1.v1_mix_ops import (
    add_vec,
    tmix_kk_a_gate,
    tmix_lnx_rkvres_xg,
    tmix_mix6,
    tmix_vres_gate,
)
from vkwr._ops.v1.v1_rank_ops import (
    linear_wag_rank_in_f16,
    linear_wag_rank_out_f16,
    linear_wagv_rank_in_f16,
    linear_wagv_rank_out_f16,
)
from vkwr._ops.v1.v1_wkv_ops import wkv_forward_fp32, wkv_seq_fp16, wkv_seq_w0_fp16
from vkwr.model_executor.layers.linear import RWKV7LinearDispatcher
from vkwr.model_executor.layers.path_dispatcher_config import PathConfig

LOWRANK_IN_ROWS_T = 7
LOWRANK_OUT_ROWS_T = 4
LOWRANK_FUSED_MIN_C = 1024


def _can_use_lowrank_fused(rows: int, C: int) -> bool:
    return C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_IN_ROWS_T


def _can_use_lowrank_out_fused(rows: int, C: int) -> bool:
    return C >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T


class RWKV7TimeMixDispatcher(nn.Module):
    """1:1 对齐 Albatross tmix 方法的无状态类。

    权重通过 weights dict 传入，dispatcher 外部注入，
    所有全局常量（LOWRANK_WEIGHT, WKV_MODE）通过构造参数传入。
    """

    def __init__(
        self,
        C: int,
        H: int,
        linear_dispatcher: RWKV7LinearDispatcher,
        lowrank_weight: str = "both",
        wkv_mode: str = "fp16",
        model_dict: dict | None = None,
    ):
        super().__init__()
        self.C = C
        self.H = H
        self.linear_dispatcher = linear_dispatcher
        self.lowrank_weight = lowrank_weight
        self.wkv_mode = wkv_mode
        self.model_dict = model_dict

    def forward(
        self,
        layer: int,
        x: torch.Tensor,
        shift_state: torch.Tensor,
        wkv_state: torch.Tensor,
        elapsed_t: torch.Tensor,
        v_first: torch.Tensor,
        param_prefix: str,
        path: PathConfig,
        weights: dict | None = None,
        pre_mix=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weights is None:
            weights = self.model_dict
        B, T, _ = x.shape
        H = self.H
        C = self.C

        if pre_mix is not None:
            xr, xw, xk, xv, xa, xg = pre_mix
        else:
            xr, xw, xk, xv, xa, xg = tmix_mix6(
                B,
                T,
                C,
                x.contiguous(),
                shift_state[0],
                weights[param_prefix + "x_r"],
                weights[param_prefix + "x_w"],
                weights[param_prefix + "x_k"],
                weights[param_prefix + "x_v"],
                weights[param_prefix + "x_a"],
                weights[param_prefix + "x_g"],
            )

        if path.use_batched_rkv:
            flat = torch.stack((xr.reshape(-1, C), xk.reshape(-1, C), xv.reshape(-1, C)))
            rkv = torch.bmm(flat, weights[param_prefix + "rkv.weight"])
            r, k, v = [t.view(B, T, C) for t in rkv.unbind(0)]
        else:
            r = self.linear_dispatcher.linear_orig_layout(xr, weights[param_prefix + "receptance.weight"], path, "att_c2c")
            k = self.linear_dispatcher.linear_orig_layout(xk, weights[param_prefix + "key.weight"], path, "att_c2c")
            v = self.linear_dispatcher.linear_orig_layout(xv, weights[param_prefix + "value.weight"], path, "att_c2c")

        v1 = None
        if self.lowrank_weight != "orig" and _can_use_lowrank_fused(path.rows, C) and _can_use_lowrank_out_fused(path.rows, C) and layer != 0:
            w1, a1, g1, v1 = linear_wagv_rank_in_f16(
                xw.contiguous(),
                xa.contiguous(),
                xg.contiguous(),
                xv.contiguous(),
                weights[param_prefix + "w1.t"],
                weights[param_prefix + "a1.t"],
                weights[param_prefix + "g1.t"],
                weights[param_prefix + "v1.t"],
            )
        elif self.lowrank_weight != "orig" and _can_use_lowrank_fused(path.rows, C):
            w1, a1, g1 = linear_wag_rank_in_f16(
                xw.contiguous(),
                xa.contiguous(),
                xg.contiguous(),
                weights[param_prefix + "w1.t"],
                weights[param_prefix + "a1.t"],
                weights[param_prefix + "g1.t"],
            )
        else:
            w1 = self.linear_dispatcher.linear_rank_in(xw, weights.get(param_prefix + "w1"), weights.get(param_prefix + "w1.t"), path.rows)
            a1 = self.linear_dispatcher.linear_rank_in(xa, weights.get(param_prefix + "a1"), weights.get(param_prefix + "a1.t"), path.rows)
            g1 = self.linear_dispatcher.linear_rank_in(xg, weights.get(param_prefix + "g1"), weights.get(param_prefix + "g1.t"), path.rows)

        v_done = False
        if self.lowrank_weight != "orig" and _can_use_lowrank_out_fused(path.rows, C) and layer != 0 and v1 is not None:
            w, a, g, v = linear_wagv_rank_out_f16(
                w1.contiguous(),
                a1.contiguous(),
                g1.contiguous(),
                v1.contiguous(),
                weights[param_prefix + "w2.t"],
                weights[param_prefix + "a2.t"],
                weights[param_prefix + "g2.t"],
                weights[param_prefix + "v2.t"],
                v.contiguous(),
                v_first.contiguous(),
                weights[param_prefix + "v0"],
            )
            v_done = True
        elif self.lowrank_weight != "orig" and _can_use_lowrank_out_fused(path.rows, C):
            w, a, g = linear_wag_rank_out_f16(
                w1.contiguous(),
                a1.contiguous(),
                g1.contiguous(),
                weights[param_prefix + "w2.t"],
                weights[param_prefix + "a2.t"],
                weights[param_prefix + "g2.t"],
            )
        else:
            w = self.linear_dispatcher.linear_rank_out_act(w1, weights.get(param_prefix + "w2"), weights.get(param_prefix + "w2.t"), path.rows, 1)
            a = self.linear_dispatcher.linear_rank_out(a1, weights.get(param_prefix + "a2"), weights.get(param_prefix + "a2.t"), path.rows)
            g = self.linear_dispatcher.linear_rank_out_act(g1, weights.get(param_prefix + "g2"), weights.get(param_prefix + "g2.t"), path.rows, 2)
        k, neg_kk, kka = tmix_kk_a_gate(
            B, T, C, H, k.contiguous(), weights[param_prefix + "k_k"], weights[param_prefix + "a0"], a.contiguous(), weights[param_prefix + "k_a"]
        )

        if layer == 0:
            v_first = v
        elif not v_done:
            if self.lowrank_weight != "orig" and _can_use_lowrank_out_fused(path.rows, C):
                warnings.warn(
                    "This branch was previously dead code introduced by an upstream project and has since been cleaned up",
                    RuntimeWarning,
                )
                # if v1 is None:
                #     v1 = self.dispatcher.linear_rank_in(xv, weights.get(param_prefix + "v1"), weights.get(param_prefix + "v1.t"), path.rows)
                # v = linear_t_vres_f16(v1.contiguous(), weights[param_prefix + "v2.t"], v.contiguous(), v_first.contiguous(), weights[param_prefix + "v0"])
            else:
                v12 = self.linear_dispatcher.linear_rank_out(
                    self.linear_dispatcher.linear_rank_in(xv, weights.get(param_prefix + "v1"), weights.get(param_prefix + "v1.t"), path.rows),
                    weights.get(param_prefix + "v2"),
                    weights.get(param_prefix + "v2.t"),
                    path.rows,
                )
                v = tmix_vres_gate(B, T, C, v.contiguous(), v_first.contiguous(), weights[param_prefix + "v0"], v12.contiguous())

        y = torch.empty_like(r)
        if self.wkv_mode == "fp32io16":
            w_raw = add_vec(C, w.contiguous(), weights[param_prefix + "w0"])
            wkv_forward_fp32(
                B, T, C, H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y
            )
        elif T <= 16:
            wkv_seq_w0_fp16(
                B,
                T,
                C,
                H,
                wkv_state,
                r.contiguous(),
                w.contiguous(),
                weights[param_prefix + "w0"],
                k.contiguous(),
                v.contiguous(),
                neg_kk.contiguous(),
                kka.contiguous(),
                y,
                elapsed_t,
            )
        else:
            w_raw = add_vec(C, w.contiguous(), weights[param_prefix + "w0"])
            wkv_seq_fp16(
                B, T, C, H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t
            )

        y = tmix_lnx_rkvres_xg(
            B,
            T,
            C,
            H,
            y.contiguous(),
            r.contiguous(),
            k.contiguous(),
            v.contiguous(),
            weights[param_prefix + "r_k"],
            weights[param_prefix + "ln_x.weight"],
            weights[param_prefix + "ln_x.bias"],
            g.contiguous(),
        )

        out = self.linear_dispatcher.linear_orig_layout(y, weights[param_prefix + "output.weight"], path, "att_c2c")
        return out, v_first

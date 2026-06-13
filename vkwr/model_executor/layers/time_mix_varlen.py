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
from vkwr._ops.v1.v1_wkv_ops import wkv_forward_fp32, wkv_one_w0_fp16, wkv_seq_fp16, wkv_seq_w0_fp16
from vkwr._ops.v1_5.v1_5_mix_ops import tmix_mix6_varlen
from vkwr._ops.v1_5.v1_5_wkv_fp32_ops import wkv_forward_fp32_varlen
from vkwr._ops.v1_5.v1_5_wkv_ops import wkv_seq_fp16_varlen, wkv_seq_w0_fp16_varlen
from vkwr.config.model import RWKV7InferenceConfig
from vkwr.model_executor.layers.linear import RWKV7LinearDispatcher
from vkwr.model_executor.layers.path_dispatcher_config_varlen import PathConfig, lorank_cfg


class RWKV7TimeMixDispatcher(nn.Module):
    def __init__(
        self,
        C: int,
        H: int,
        linear_dispatcher: RWKV7LinearDispatcher,
        model_dict: dict | None = None,
        inference_config: RWKV7InferenceConfig | None = None,
    ):
        super().__init__()
        self.C = C
        self.H = H
        self.linear_dispatcher = linear_dispatcher
        self.model_dict = model_dict
        self.inference_config = inference_config if inference_config is not None else RWKV7InferenceConfig()

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
        query_start_loc: torch.Tensor,
        req_id: torch.Tensor,
        max_t: int,
        B: int,
        is_uniform: bool,
        weights: dict | None = None,
        pre_mix=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weights is None:
            weights = self.model_dict
        total_tokens = x.size(0)
        C = self.C
        H = self.H

        if pre_mix is not None:
            xr, xw, xk, xv, xa, xg = pre_mix
        else:
            x_in = x.contiguous()
            w_r = weights[param_prefix + "x_r"]
            w_w = weights[param_prefix + "x_w"]
            w_k = weights[param_prefix + "x_k"]
            w_v = weights[param_prefix + "x_v"]
            w_a = weights[param_prefix + "x_a"]
            w_g = weights[param_prefix + "x_g"]
            if is_uniform:
                xr, xw, xk, xv, xa, xg = tmix_mix6(
                    B,
                    max_t,
                    C,
                    x_in.view(B, max_t, C).contiguous(),
                    shift_state[0],
                    w_r,
                    w_w,
                    w_k,
                    w_v,
                    w_a,
                    w_g,
                )
            else:
                xr, xw, xk, xv, xa, xg = tmix_mix6_varlen(
                    B,
                    total_tokens,
                    C,
                    x_in,
                    shift_state[0],
                    w_r,
                    w_w,
                    w_k,
                    w_v,
                    w_a,
                    w_g,
                    query_start_loc,
                    req_id,
                )

        if path.use_batched_rkv:
            if is_uniform:
                flat = torch.stack((xr.view(B, -1, C), xk.view(B, -1, C), xv.view(B, -1, C)))
                rkv = torch.bmm(flat, weights[param_prefix + "rkv.weight"])
                r, k, v = [t.view(B, max_t, C) for t in rkv.unbind(0)]
            else:
                flat = torch.stack((xr.reshape(-1, C), xk.reshape(-1, C), xv.reshape(-1, C)))
                rkv = torch.bmm(flat, weights[param_prefix + "rkv.weight"])
                r, k, v = [t.reshape(total_tokens, C) for t in rkv.unbind(0)]
        elif is_uniform:
            r = self.linear_dispatcher.linear_orig_layout(xr.view(B, max_t, C), weights[param_prefix + "receptance.weight"], path, "att_c2c")
            k = self.linear_dispatcher.linear_orig_layout(xk.view(B, max_t, C), weights[param_prefix + "key.weight"], path, "att_c2c")
            v = self.linear_dispatcher.linear_orig_layout(xv.view(B, max_t, C), weights[param_prefix + "value.weight"], path, "att_c2c")
        else:
            r = self.linear_dispatcher.linear_orig_layout(xr, weights[param_prefix + "receptance.weight"], path, "att_c2c")
            k = self.linear_dispatcher.linear_orig_layout(xk, weights[param_prefix + "key.weight"], path, "att_c2c")
            v = self.linear_dispatcher.linear_orig_layout(xv, weights[param_prefix + "value.weight"], path, "att_c2c")

        v1 = None
        if (
            self.inference_config.lowrank_weight != "orig"
            and lorank_cfg.can_use_lowrank_fused(path.rows, C)
            and lorank_cfg.can_use_lowrank_out_fused(path.rows, C)
            and layer != 0
        ):
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
        elif self.inference_config.lowrank_weight != "orig" and lorank_cfg.can_use_lowrank_fused(path.rows, C):
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
        if self.inference_config.lowrank_weight != "orig" and lorank_cfg.can_use_lowrank_out_fused(path.rows, C) and layer != 0 and v1 is not None:
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
        elif self.inference_config.lowrank_weight != "orig" and lorank_cfg.can_use_lowrank_out_fused(path.rows, C):
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

        if is_uniform:
            k_out, neg_kk, kka = tmix_kk_a_gate(
                B,
                max_t,
                C,
                H,
                k.contiguous().view(B, max_t, C),
                weights[param_prefix + "k_k"],
                weights[param_prefix + "a0"],
                a.contiguous().view(B, max_t, C),
                weights[param_prefix + "k_a"],
            )
            k = k_out.view(total_tokens, C)
            neg_kk = neg_kk.view(total_tokens, C)
            kka = kka.view(total_tokens, C)
        else:
            k_3d = k.contiguous().reshape(total_tokens, 1, C)
            a12 = a.contiguous().reshape(total_tokens, 1, C)
            k_out, neg_kk, kka = tmix_kk_a_gate(
                total_tokens, 1, C, H, k_3d, weights[param_prefix + "k_k"], weights[param_prefix + "a0"], a12, weights[param_prefix + "k_a"]
            )
            k = k_out.reshape(total_tokens, C)
            neg_kk = neg_kk.reshape(total_tokens, C)
            kka = kka.reshape(total_tokens, C)

        if layer == 0:
            v_first = v
        elif not v_done:
            if self.inference_config.lowrank_weight != "orig" and lorank_cfg.can_use_lowrank_out_fused(path.rows, C):
                warnings.warn(
                    "This branch was previously dead code introduced by an upstream project and has since been cleaned up",
                    RuntimeWarning,
                )
            else:
                v12 = self.linear_dispatcher.linear_rank_out(
                    self.linear_dispatcher.linear_rank_in(xv, weights.get(param_prefix + "v1"), weights.get(param_prefix + "v1.t"), path.rows),
                    weights.get(param_prefix + "v2"),
                    weights.get(param_prefix + "v2.t"),
                    path.rows,
                )
                if is_uniform:
                    v = tmix_vres_gate(
                        B,
                        max_t,
                        C,
                        v.contiguous().view(B, max_t, C),
                        v_first.contiguous().view(B, max_t, C),
                        weights[param_prefix + "v0"],
                        v12.contiguous().view(B, max_t, C),
                    ).view(total_tokens, C)
                else:
                    v = tmix_vres_gate(
                        total_tokens,
                        1,
                        C,
                        v.contiguous().reshape(total_tokens, 1, C),
                        v_first.contiguous().reshape(total_tokens, 1, C),
                        weights[param_prefix + "v0"],
                        v12.contiguous().reshape(total_tokens, 1, C),
                    ).reshape(total_tokens, C)

        y = torch.empty_like(r)
        if max_t == 1:
            r1 = r.view(B, C) if r.dim() == 3 else r
            w1 = w.view(B, C) if w.dim() == 3 else w
            k1 = k.view(B, C) if k.dim() == 3 else k
            v1 = v.view(B, C) if v.dim() == 3 else v
            neg_kk1 = neg_kk.view(B, C) if neg_kk.dim() == 3 else neg_kk
            kka1 = kka.view(B, C) if kka.dim() == 3 else kka
            y1 = y.view(B, C) if y.dim() == 3 else y
            wkv_one_w0_fp16(
                B,
                C,
                H,
                wkv_state,
                r1.contiguous(),
                w1.contiguous(),
                weights[param_prefix + "w0"],
                k1.contiguous(),
                v1.contiguous(),
                neg_kk1.contiguous(),
                kka1.contiguous(),
                y1,
                elapsed_t,
            )
        elif is_uniform:
            r_3d = r.view(B, max_t, C).contiguous()
            k_3d = k.view(B, max_t, C).contiguous()
            v_3d = v.view(B, max_t, C).contiguous()
            neg_kk_3d = neg_kk.view(B, max_t, C).contiguous()
            kka_3d = kka.view(B, max_t, C).contiguous()
            y_3d = y.view(B, max_t, C).contiguous()
            if self.inference_config.wkv_mode == "fp32io16":
                w_raw = add_vec(C, w.contiguous(), weights[param_prefix + "w0"])
                wkv_forward_fp32(
                    B,
                    max_t,
                    C,
                    H,
                    wkv_state,
                    r_3d,
                    w_raw.view(B, max_t, C).contiguous(),
                    k_3d,
                    v_3d,
                    neg_kk_3d,
                    kka_3d,
                    y_3d,
                )
            elif max_t <= 16:
                wkv_seq_w0_fp16(
                    B,
                    max_t,
                    C,
                    H,
                    wkv_state,
                    r_3d,
                    w.view(B, max_t, C).contiguous(),
                    weights[param_prefix + "w0"],
                    k_3d,
                    v_3d,
                    neg_kk_3d,
                    kka_3d,
                    y_3d,
                    elapsed_t,
                )
            else:
                w_raw = add_vec(C, w.contiguous(), weights[param_prefix + "w0"])
                wkv_seq_fp16(
                    B,
                    max_t,
                    C,
                    H,
                    wkv_state,
                    r_3d,
                    w_raw.view(B, max_t, C).contiguous(),
                    k_3d,
                    v_3d,
                    neg_kk_3d,
                    kka_3d,
                    y_3d,
                    elapsed_t,
                )
        elif self.inference_config.wkv_mode == "fp32io16":
            w_raw = add_vec(C, w.contiguous(), weights[param_prefix + "w0"])
            wkv_forward_fp32_varlen(
                B,
                total_tokens,
                max_t,
                C,
                H,
                query_start_loc,
                wkv_state,
                r.contiguous(),
                w_raw.contiguous(),
                k.contiguous(),
                v.contiguous(),
                neg_kk.contiguous(),
                kka.contiguous(),
                y,
            )
        elif max_t <= 16:
            wkv_seq_w0_fp16_varlen(
                B,
                total_tokens,
                max_t,
                C,
                H,
                query_start_loc,
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
            wkv_seq_fp16_varlen(
                B,
                total_tokens,
                max_t,
                C,
                H,
                query_start_loc,
                wkv_state,
                r.contiguous(),
                w_raw.contiguous(),
                k.contiguous(),
                v.contiguous(),
                neg_kk.contiguous(),
                kka.contiguous(),
                y,
                elapsed_t,
            )

        if is_uniform:
            y = tmix_lnx_rkvres_xg(
                B,
                max_t,
                C,
                H,
                y.view(B, max_t, C).contiguous(),
                r.view(B, max_t, C).contiguous(),
                k.view(B, max_t, C).contiguous(),
                v.view(B, max_t, C).contiguous(),
                weights[param_prefix + "r_k"],
                weights[param_prefix + "ln_x.weight"],
                weights[param_prefix + "ln_x.bias"],
                g.view(B, max_t, C).contiguous(),
            ).view(total_tokens, C)
        else:
            y = tmix_lnx_rkvres_xg(
                total_tokens,
                1,
                C,
                H,
                y.contiguous().reshape(total_tokens, 1, C),
                r.contiguous().reshape(total_tokens, 1, C),
                k.contiguous().reshape(total_tokens, 1, C),
                v.contiguous().reshape(total_tokens, 1, C),
                weights[param_prefix + "r_k"],
                weights[param_prefix + "ln_x.weight"],
                weights[param_prefix + "ln_x.bias"],
                g.contiguous().reshape(total_tokens, 1, C),
            ).reshape(total_tokens, C)

        out = self.linear_dispatcher.linear_orig_layout(y, weights[param_prefix + "output.weight"], path, "att_c2c")
        return out, v_first

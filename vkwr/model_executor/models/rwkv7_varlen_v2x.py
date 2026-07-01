#!/usr/bin/env python3
import torch

from vkwr._ops.v1.v1_norm_ops import (
    add_f16,
    add_last_layer_norm_f16,
    add_layer_norm_f16,
    emb_ln0_bf16_to_f16,
    layer_norm_f16,
)
from vkwr._ops.v1.v1_wkv_ops import HEAD_SIZE
from vkwr._ops.v1_5.v1_5_norm_ops import add_last_layer_norm_f16_varlen
from vkwr._ops.v2 import import_all_v2_ops
from vkwr._ops.v2.v2_norm_ops import add_layer_norm_tmix_mix6_f16
from vkwr._ops.v2_5 import import_all_v2_5_ops
from vkwr._ops.v2_5.v2_5_wkv_ops import advance_i32_varlen
from vkwr.config.model import RWKV7Config, RWKV7InferenceConfig, WeightConfig
from vkwr.model_executor.layers.channel_mix_varlen_v2x import RWKV7ChannelMixDispatcher
from vkwr.model_executor.layers.linear import RWKV7LinearDispatcher
from vkwr.model_executor.layers.path_dispatcher_config_varlen import CmixConfig, CmixThresholds, PathConfig, PathSelector
from vkwr.model_executor.layers.time_mix_varlen_v2x import RWKV7TimeMixDispatcher
from vkwr.model_executor.utils import cuda_mem, log


class RWKV7:
    """RWKV7 varlen model using v2/v2_5 ops with slot_indices mapping.

    Differences from rwkv7_varlen.py (v1/v1_5):
    - State is global [max_slots, ...] instead of per-request [B, ...]
    - slot_indices: [B] (int32) maps request batch index -> physical GPU slot
    - Ops with state dependency use v2x series (v2/v2_5) with slot_indices mapping
    - Ops without state dependency use v1/v1.5 equivalents
    - No max_slots parameter — runner manages state buffer size

    Production state buffer shapes (managed by GPUModelRunner):
        state[0] (shift_state): [L, 2, max_slots, C] - fp16
        state[1] (wkv_state): [L, max_slots, H, N, N] - fp16/fp32
        state[2] (elapsed): [max_slots] - int32
    """

    def __init__(
        self,
        model_path: str,
        weight_config: WeightConfig | None = None,
        inference_config: RWKV7InferenceConfig | None = None,
    ) -> None:
        import_all_v2_ops()
        import_all_v2_5_ops()
        self.weight_config = weight_config if weight_config is not None else WeightConfig()
        self.inference_config = inference_config if inference_config is not None else RWKV7InferenceConfig()
        self.cmix_thresholds = CmixThresholds()
        self.model_path = model_path

        self.config, z, emb_src, ln0_w_src, ln0_b_src, emb_cpu = self._load_and_detect_dims()
        self._preprocess_weights(z, emb_cpu)
        self._build_fused_emb_ln0(z, emb_src, ln0_w_src, ln0_b_src, emb_cpu)
        if self.inference_config.rkv_mode != "off" and not self.weight_config.use_orig_linear("att_c2c"):
            self._stack_rkv_weights(z)

        self.z = z
        self.emb_cpu = self.inference_config.emb_device == "cpu"
        self.emb_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        self.linear_dispatcher = RWKV7LinearDispatcher(self.config.C, self.weight_config)
        self.cmix_dispatcher = RWKV7ChannelMixDispatcher(self.config.C, self.linear_dispatcher, model_dict=z, inference_config=self.inference_config)
        self.tmix_dispatcher = RWKV7TimeMixDispatcher(
            self.config.C, self.config.H, self.linear_dispatcher, model_dict=z, inference_config=self.inference_config
        )
        self.cmix_config = CmixConfig()
        self.path_selector = PathSelector(cmix_thresholds=self.cmix_thresholds, weight_config=self.weight_config, inference_config=self.inference_config)

        log(f"model ready L={self.config.L} C={self.config.C} H={self.config.H} N={self.config.N} V={self.config.V}")
        log(cuda_mem())

    @staticmethod
    def _setup_torch_backend():
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch._C._jit_set_autocast_mode(False)

    def _load_and_detect_dims(self) -> tuple[RWKV7Config, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        log(f"loading weights from {self.model_path}")
        z = torch.load(self.model_path, map_location="cpu", mmap=True)
        log(f"weights mmap loaded, tensors={len(z)}")

        H, N = z["blocks.0.att.r_k"].shape
        C, V = H * N, z["emb.weight"].shape[0]
        assert N == HEAD_SIZE
        max_layer = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks."))
        L = max_layer + 1
        config = RWKV7Config(L=L, C=C, H=H, N=N, V=V)
        log(f"detected model C={C} H={H} N={N} V={V}")
        log(f"cmix no-fc path: rows<={self.cmix_thresholds.nofc_max_rows} row20_t<={self.cmix_thresholds.nofc_row20_max_t}")

        emb_src = z["emb.weight"].squeeze()
        ln0_w_src = z["blocks.0.ln0.weight"].squeeze()
        ln0_b_src = z["blocks.0.ln0.bias"].squeeze()
        emb_cpu = emb_src if self.inference_config.emb_device == "cpu" else None
        return config, z, emb_src, ln0_w_src, ln0_b_src, emb_cpu

    def _should_transpose(self, key: str, is_lowrank: bool) -> bool:
        if is_lowrank:
            return False
        suffixes = ("key.weight", "value.weight", "receptance.weight", "output.weight", "head.weight")
        return any(s in key for s in suffixes) and not self.weight_config.is_orig_linear_weight(key)

    def _setup_cmix_sparse_fc(self, key: str, value: torch.Tensor, dev: torch.device, z: dict) -> None:
        if ".ffn.key.weight" in key and self.inference_config.cmix_sparse == "auto":
            z[key + ".fc"] = value.to(device=dev, dtype=self.inference_config.dtype).contiguous()

    def _store_lowrank_weight(self, key: str, value: torch.Tensor, z: dict) -> None:
        if self.inference_config.lowrank_weight in ("orig", "both"):
            z[key] = value
        else:
            del z[key]
        if self.inference_config.lowrank_weight in ("transpose", "both"):
            z[key + ".t"] = value.t().contiguous()

    def _preprocess_weights(self, z: dict, emb_cpu) -> None:
        log(f"moving and preprocessing weights to CUDA emb={self.inference_config.emb_device}")
        for key in list(z.keys()):
            if key == "emb.weight" and emb_cpu is not None:
                continue
            value = z[key].squeeze()
            is_lowrank = self.weight_config.is_lowrank_weight(key)
            self._setup_cmix_sparse_fc(key, value, torch.device("cuda"), z)
            if self._should_transpose(key, is_lowrank):
                value = value.t()
            value = value.to(device=torch.device("cuda"), dtype=self.inference_config.dtype).contiguous()

            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            if is_lowrank:
                self._store_lowrank_weight(key, value, z)
            else:
                z[key] = value

    @staticmethod
    def _fuse_emb_ln0_gpu(emb_src, ln0_w_bf16, ln0_b_bf16, emb_dev):
        with torch.cuda.device(emb_dev):
            return emb_ln0_bf16_to_f16(emb_src.to(device=emb_dev).contiguous(), ln0_w_bf16, ln0_b_bf16)

    def _fuse_emb_ln0_cpu(self, emb_cpu, ln0_w_bf16, ln0_b_bf16, emb_dev):
        emb = torch.empty((self.config.V, self.config.C), dtype=self.inference_config.dtype, pin_memory=True)
        with torch.cuda.device(emb_dev):
            for start in range(0, self.config.V, 4096):
                end = min(start + 4096, self.config.V)
                chunk = emb_cpu[start:end].to(device=emb_dev).contiguous()
                chunk = emb_ln0_bf16_to_f16(chunk, ln0_w_bf16, ln0_b_bf16)
                emb[start:end].copy_(chunk)
        return emb

    def _build_fused_emb_ln0(self, z, emb_src, ln0_w_src, ln0_b_src, emb_cpu):
        emb_dev = torch.device("cuda")
        ln0_w_bf16 = ln0_w_src.to(device=emb_dev).contiguous()
        ln0_b_bf16 = ln0_b_src.to(device=emb_dev).contiguous()
        if emb_cpu is None:
            z["emb.weight"] = self._fuse_emb_ln0_gpu(emb_src, ln0_w_bf16, ln0_b_bf16, emb_dev)
        else:
            z["emb.weight"] = self._fuse_emb_ln0_cpu(emb_cpu, ln0_w_bf16, ln0_b_bf16, emb_dev)

    def _stack_rkv_weights(self, z) -> None:
        for layer in range(self.config.L):
            p = f"blocks.{layer}.att."
            z[p + "rkv.weight"] = torch.stack((z[p + "receptance.weight"], z[p + "key.weight"], z[p + "value.weight"])).contiguous()

    @property
    def device(self) -> torch.device:
        return self.z["blocks.0.att.r_k"].device

    def zero_state(self, B: int) -> list[torch.Tensor]:
        """Create zero-initialized state for B requests (used for warmup/testing).

        Production state is managed by GPUModelRunner with [max_slots, ...] shape.
        This method returns [B, ...]-shaped tensors suitable for single-batch
        warmup forward passes or small-scale tests.
        """
        cfg = self.config
        wkv_dtype = torch.float32 if self.inference_config.wkv_mode == "fp32io16" else self.inference_config.dtype
        return [
            torch.zeros((cfg.L, 2, B, cfg.C), dtype=self.inference_config.dtype, device="cuda"),
            torch.zeros((cfg.L, B, cfg.H, cfg.N, cfg.N), dtype=wkv_dtype, device="cuda"),
            torch.zeros((B,), dtype=torch.int32, device="cuda"),
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        query_start_loc: torch.Tensor,
        max_t: int,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        B = query_start_loc.size(0) - 1
        path = self.path_selector.select(B, max_t, total_tokens=tokens.size(0))
        x = self.embed(tokens)
        req_id = self._build_req_id(tokens, query_start_loc)
        return self.forward_from_x(x, state, path, query_start_loc, req_id, max_t, B, slot_indices)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.emb_cpu:
            if tokens.device != self.z["emb.weight"].device:
                tokens = tokens.to(self.z["emb.weight"].device, non_blocking=True)
            return self.z["emb.weight"][tokens]
        flat = tokens.reshape(-1)
        if flat.device.type != "cpu":
            flat = flat.cpu()
        total_tokens = flat.size(0)
        host = torch.empty((total_tokens, self.config.C), dtype=self.inference_config.dtype, pin_memory=True)
        torch.index_select(self.z["emb.weight"], 0, flat, out=host)
        dev = host.to(device=torch.device("cuda"), dtype=self.inference_config.dtype)
        return dev

    @staticmethod
    def _build_req_id(tokens: torch.Tensor, query_start_loc: torch.Tensor) -> torch.Tensor:
        B = query_start_loc.size(0) - 1
        total_tokens = tokens.size(0)
        req_id = torch.empty(total_tokens, dtype=torch.int32, device=query_start_loc.device)
        for i in range(B):
            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()
            req_id[start:end].fill_(i)
        return req_id

    def forward_from_x(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor],
        path: PathConfig,
        query_start_loc: torch.Tensor,
        req_id: torch.Tensor,
        max_t: int,
        B: int,
        slot_indices: torch.Tensor,
        all_logits: bool = False,
    ) -> torch.Tensor:
        z = self.z
        total_tokens = x.size(0)
        is_uniform = total_tokens == B * max_t
        v_first = x
        xx = layer_norm_f16(x.contiguous(), z["blocks.0.ln1.weight"], z["blocks.0.ln1.bias"])
        pre_mix = None

        for layer in range(self.config.L):
            param_prefix = f"blocks.{layer}."
            x, xx, v_first = self._forward_layer_body(
                layer,
                x,
                xx,
                v_first,
                state,
                param_prefix,
                path,
                query_start_loc,
                req_id,
                max_t,
                B,
                is_uniform,
                pre_mix,
                slot_indices,
            )
            pre_mix = None

            if layer + 1 < self.config.L:
                x, xx, pre_mix = self._compute_next_layer_pre_mix(layer, x, xx, state, is_uniform, max_t, B, slot_indices)
            elif not all_logits:
                return self._finalize_last_layer(x, xx, state, query_start_loc, max_t, is_uniform, B, slot_indices)
            else:
                x = self.add(x, xx)

        x = (x.contiguous(), z["ln_out.weight"], z["ln_out.bias"])
        advance_i32_varlen(state[2], query_start_loc, slot_indices)
        return self.linear_head(x)

    def _forward_layer_body(
        self,
        layer: int,
        x: torch.Tensor,
        xx: torch.Tensor,
        v_first: torch.Tensor,
        state: list[torch.Tensor],
        param_prefix: str,
        path: PathConfig,
        query_start_loc: torch.Tensor,
        req_id: torch.Tensor,
        max_t: int,
        B: int,
        is_uniform: bool,
        pre_mix=None,
        slot_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.z
        p = param_prefix

        # Time mix
        xx, v_first = self.tmix_dispatcher.forward(
            layer,
            xx,
            state[0][layer],
            state[1][layer],
            state[2],
            v_first,
            p + "att.",
            path,
            query_start_loc,
            req_id,
            max_t,
            B,
            is_uniform,
            pre_mix=pre_mix,
            slot_indices=slot_indices,
        )

        # Channel mix: add_ln + cmix (split path for varlen)
        x, xx_ln = self.add_ln(x, xx, z[p + "ln2.weight"], z[p + "ln2.bias"])
        xx = self.cmix_dispatcher.forward(
            xx_ln,
            state[0][layer],
            p,
            path,
            query_start_loc,
            req_id,
            B,
            is_uniform,
            max_t,
            slot_indices=slot_indices,
        )

        return x, xx, v_first

    def _compute_next_layer_pre_mix(
        self,
        layer: int,
        x: torch.Tensor,
        xx: torch.Tensor,
        state: list[torch.Tensor],
        is_uniform: bool,
        T: int,
        B: int,
        slot_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.z
        p_next = f"blocks.{layer + 1}."
        if self.inference_config.ln1_tmix_fuse and is_uniform and T == 1 and B == 1:
            x3d = x.view(1, 1, self.config.C).contiguous()
            xx3d = xx.view(1, 1, self.config.C).contiguous()
            outs = add_layer_norm_tmix_mix6_f16(
                x3d,
                xx3d,
                state[0][layer + 1][0],
                z[p_next + "ln1.weight"],
                z[p_next + "ln1.bias"],
                z[p_next + "att.x_r"],
                z[p_next + "att.x_w"],
                z[p_next + "att.x_k"],
                z[p_next + "att.x_v"],
                z[p_next + "att.x_a"],
                z[p_next + "att.x_g"],
                slot_indices,
            )
            x = outs[0].view(B, self.config.C)
            pre_mix = outs[1:]
            xx = x
        else:
            x, xx = self.add_ln(x, xx, z[p_next + "ln1.weight"], z[p_next + "ln1.bias"])
            pre_mix = None
        return x, xx, pre_mix

    def _finalize_last_layer(
        self,
        x: torch.Tensor,
        xx: torch.Tensor,
        state: list[torch.Tensor],
        query_start_loc: torch.Tensor,
        max_t: int,
        is_uniform: bool,
        B: int,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        z = self.z
        if is_uniform:
            x_out = add_last_layer_norm_f16(
                x.view(B, max_t, self.config.C).contiguous(),
                xx.view(B, max_t, self.config.C).contiguous(),
                z["ln_out.weight"],
                z["ln_out.bias"],
            )
        else:
            total_tokens = x.size(0)
            x_out = add_last_layer_norm_f16_varlen(total_tokens, x.contiguous(), xx.contiguous(), z["ln_out.weight"], z["ln_out.bias"], query_start_loc)
        advance_i32_varlen(state[2], query_start_loc, slot_indices)
        return self.linear_head(x_out)

    def linear_head(self, x):
        z = self.z
        if not self.weight_config.use_orig_linear("head"):
            return self.linear_dispatcher.linear(x, z["head.weight"])
        rows = x.numel() // self.config.C
        return self.linear_dispatcher.linear_orig_layout(x, z["head.weight"], PathConfig(rows, False, self.cmix_config.CMIX_DENSE), "head")

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return add_f16(x.contiguous(), y.contiguous())

    def add_ln(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outs = add_layer_norm_f16(x.contiguous(), residual.contiguous(), weight, bias)
        return outs[0], outs[1]

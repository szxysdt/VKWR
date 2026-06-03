#!/usr/bin/env python3
import torch

from vkwr._ops.v1.v1_norm_ops import (
    add_f16,
    add_last_layer_norm_f16,
    add_layer_norm_cmix_mix_f16,
    add_layer_norm_f16,
    add_layer_norm_tmix_mix6_f16,
    emb_ln0_bf16_to_f16,
    layer_norm_f16,
)
from vkwr._ops.v1.v1_wkv_ops import HEAD_SIZE, advance_i32
from vkwr.config.model import ModelConfig, RWKV7InferenceConfig, WeightConfig
from vkwr.model_executor.layers.channel_mix import RWKV7ChannelMixDispatcher
from vkwr.model_executor.layers.linear import RWKV7LinearDispatcher
from vkwr.model_executor.layers.path_dispatcher_config import CmixConfig, CmixThresholds, PathConfig, PathSelector
from vkwr.model_executor.layers.time_mix import RWKV7TimeMixDispatcher
from vkwr.model_executor.utils import cuda_mem, log


class RWKV7:
    def __init__(
        self,
        model_path: str,
        weight_config: WeightConfig | None = None,
        inference_config: RWKV7InferenceConfig | None = None,
    ) -> None:
        # ---------------------------------- set up ---------------------------------- #
        self._setup_torch_backend()
        # ---------------------------------- config ---------------------------------- #
        self.weight_config = weight_config if weight_config is not None else WeightConfig()
        self.inference_config = inference_config if inference_config is not None else RWKV7InferenceConfig()
        self.cmix_thresholds = CmixThresholds()
        self.model_path = model_path
        # --------------------------------- run init --------------------------------- #

        # Load model weights and infer dimensionality
        self.config, z, emb_src, ln0_w_src, ln0_b_src, emb_cpu = self._load_and_detect_dims()
        # Preprocess weights: transpose, cast, device transfer, low-rank decomposition
        self._preprocess_weights(z, emb_cpu)
        # Build fused emb+ln0 module (GPU or CPU-pinned path)
        self._build_fused_emb_ln0(z, emb_src, ln0_w_src, ln0_b_src, emb_cpu)
        # Stack R/K/V weights into batched tensors (if batched_rkv enabled)
        self._stack_rkv_weights(z)

        # -------------------------------- init others ------------------------------- #
        self.z = z
        self.emb_cpu = self.inference_config.emb_device == "cpu"
        self.emb_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        self.linear_dispatcher = RWKV7LinearDispatcher(self.config.C, self.weight_config)
        self.cmix_dispatcher = RWKV7ChannelMixDispatcher(self.config.C, self.linear_dispatcher, model_dict=z)
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

    def _load_and_detect_dims(self) -> tuple[ModelConfig, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:

        log(f"loading weights from {self.model_path}")
        z = torch.load(self.model_path, map_location="cpu", mmap=True)
        log(f"weights mmap loaded, tensors={len(z)}")

        H, N = z["blocks.0.att.r_k"].shape
        C, V = H * N, z["emb.weight"].shape[0]
        assert N == HEAD_SIZE
        max_layer = max(int(k.split(".")[1]) for k in z.keys() if k.startswith("blocks."))
        L = max_layer + 1
        config = ModelConfig(L=L, C=C, H=H, N=N, V=V)
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
            # cmix weight
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

    def zero_state(self, B: int) -> list[torch.Tensor]:
        cfg = self.config
        wkv_dtype = torch.float32 if self.inference_config.wkv_mode == "fp32io16" else self.inference_config.dtype
        return [
            torch.zeros((cfg.L, 2, B, cfg.C), dtype=self.inference_config.dtype, device="cuda"),
            torch.zeros((cfg.L, B, cfg.H, cfg.N, cfg.N), dtype=wkv_dtype, device="cuda"),
            torch.zeros((B,), dtype=torch.int32, device="cuda"),
        ]

    def forward(self, tokens: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = self.path_selector.select(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.emb_cpu:
            if tokens.device != self.z["emb.weight"].device:
                tokens = tokens.to(self.z["emb.weight"].device, non_blocking=True)
            return self.z["emb.weight"][tokens]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        host, dev = self.emb_cache.get((B, T), (None, None))
        if host is None:
            host = torch.empty((B * T, self.config.C), dtype=self.inference_config.dtype, pin_memory=True)
            dev = torch.empty((B, T, self.config.C), dtype=self.inference_config.dtype, device=torch.device("cuda"))
            self.emb_cache[(B, T)] = (host, dev)
        flat = tokens.reshape(-1)
        if flat.device.type != "cpu":
            flat = flat.cpu()
        torch.index_select(self.z["emb.weight"], 0, flat, out=host)
        dev.copy_(host.view(B, T, self.config.C), non_blocking=True)
        return dev

    def forward_from_x(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor],
        path: PathConfig,
        all_logits: bool = False,
        last_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass given pre-computed embeddings.

        Callers must obtain a PathConfig via path_selector.select() BEFORE calling this function.

        Example:
            path = model.path_selector.select(B, T)
            out = model.forward_from_x(x, state, path, all_logits=all_logits)
        """
        z = self.z
        B, T, _ = x.shape
        v_first = x
        xx = self.ln(x, z["blocks.0.ln1.weight"], z["blocks.0.ln1.bias"])
        pre_mix = None

        for layer in range(self.config.L):
            param_prefix = f"blocks.{layer}."
            # (1) Layer body: tmix + cmix
            x, xx, v_first = self._forward_layer_body(layer, x, xx, v_first, state, param_prefix, path, pre_mix)
            pre_mix = None
            # (2) Next layer prep or finalize
            if layer + 1 < self.config.L:
                x, xx, pre_mix = self._compute_next_layer_pre_mix(layer, x, xx, state, B, T)
            elif not all_logits:
                return self._finalize_last_layer(x, xx, state, T, last_indices)
            else:
                x = self.add(x, xx)

        # Fallback for all_logits=True (evaluation)
        x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
        advance_i32(state[2], T)
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
        pre_mix=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute single-layer tmix + cmix. Returns (new_x, new_xx, new_v_first)."""
        z = self.z
        B, T, _ = x.shape
        p = param_prefix

        # Time mix
        xx, v_first = self.tmix_dispatcher.forward(layer, xx, state[0][layer], state[1][layer], state[2], v_first, p + "att.", path, pre_mix=pre_mix)

        # Channel mix + residual add&norm
        if T == 1 and path.cmix_mode not in (self.cmix_config.CMIX_B1T1_SPARSE, self.cmix_config.CMIX_ROWS2_SPARSE):
            x, mixed = add_layer_norm_cmix_mix_f16(
                x.contiguous(), xx.contiguous(), state[0][layer][1], z[p + "ln2.weight"], z[p + "ln2.bias"], z[p + "ffn.x_k"]
            )
            xx = self.cmix_dispatcher.forward_from_mixed(mixed, p, path)
        else:
            x, xx = self.add_ln(x, xx, z[p + "ln2.weight"], z[p + "ln2.bias"])
            xx = self.cmix_dispatcher.forward(xx, state[0][layer], p, path)

        return x, xx, v_first

    def _compute_next_layer_pre_mix(
        self,
        layer: int,
        x: torch.Tensor,
        xx: torch.Tensor,
        state: list[torch.Tensor],
        B: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare for the next layer: add_ln(x, xx) or fused LN1+TMIX mix6.
        Returns (next_x, next_xx, pre_mix).
        """
        z = self.z
        p_next = f"blocks.{layer + 1}."
        if self.inference_config.ln1_tmix_fuse and B == 1 and T == 1:
            outs = add_layer_norm_tmix_mix6_f16(
                x.contiguous(),
                xx.contiguous(),
                state[0][layer + 1][0],
                z[p_next + "ln1.weight"],
                z[p_next + "ln1.bias"],
                z[p_next + "att.x_r"],
                z[p_next + "att.x_w"],
                z[p_next + "att.x_k"],
                z[p_next + "att.x_v"],
                z[p_next + "att.x_a"],
                z[p_next + "att.x_g"],
            )
            x = outs[0]
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
        T: int,
        last_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Final layer output: add + ln + head."""
        z = self.z
        if last_indices is not None:
            x = self.ln(self.add(x, xx), z["ln_out.weight"], z["ln_out.bias"])
            B = x.shape[0]
            x = x[torch.arange(B, device=x.device), last_indices].contiguous()
        else:
            x = self.add_last_ln(x, xx, z["ln_out.weight"], z["ln_out.bias"])
        advance_i32(state[2], T)
        return self.linear_head(x)

    def ln(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return layer_norm_f16(x.contiguous(), weight, bias)

    def forward_all_logits(self, tokens: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = self.path_selector.select(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path, all_logits=True)

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

    def add_last_ln(self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return add_last_layer_norm_f16(x.contiguous(), residual.contiguous(), weight, bias)

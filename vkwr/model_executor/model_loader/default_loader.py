from typing import Any

import torch

from vkwr.model_executor.model_loader.base_loader import BaseModelLoader
from vkwr.model_executor.model_loader.weight_utils import load_pytorch_state_dict, maybe_squeeze

LOWRANK_SUFFIXES = ("att.w1", "att.w2", "att.a1", "att.a2", "att.g1", "att.g2", "att.v1", "att.v2")
TRANSPOSE_SUFFIXES = (
    "att.receptance.weight",
    "att.key.weight",
    "att.value.weight",
    "att.output.weight",
    "ffn.key.weight",
    "ffn.value.weight",
)


class DefaultModelLoader(BaseModelLoader):
    def __init__(self, load_format: str = "auto"):
        self.load_format = load_format

    def load_model(self, model_path: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        z = load_pytorch_state_dict(model_path, str(device))
        z = maybe_squeeze(z)
        result = {}
        for key, value in z.items():
            if key == "emb.weight":
                result[key] = value
                continue
            if not key.startswith("blocks."):
                if key == "head.weight":
                    result[key] = value.to(device=device, dtype=dtype, non_blocking=True).contiguous()
                elif key in ("ln_out.weight", "ln_out.bias"):
                    result[key] = value.to(device=device, dtype=dtype, non_blocking=True).contiguous()
                else:
                    result[key] = value.to(device=device, dtype=dtype, non_blocking=True).contiguous()
                continue
            if key.endswith(".r_k"):
                result[key] = value.flatten().contiguous().to(device=device, dtype=dtype, non_blocking=True)
                continue
            is_lowrank = any(key.endswith(suf) for suf in LOWRANK_SUFFIXES)
            needs_transpose = any(key.endswith(suf) for suf in TRANSPOSE_SUFFIXES)
            if is_lowrank:
                w = value.to(device=device, dtype=dtype, non_blocking=True).contiguous()
                result[key] = w
                result[key + ".t"] = w.T.contiguous()
            elif needs_transpose:
                result[key] = value.T.contiguous().to(device=device, dtype=dtype, non_blocking=True)
            else:
                result[key] = value.to(device=device, dtype=dtype, non_blocking=True).contiguous()
        return result

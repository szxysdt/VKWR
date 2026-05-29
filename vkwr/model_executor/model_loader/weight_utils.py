from typing import Any

import torch


def load_pytorch_state_dict(model_path: str, device: str = "cpu") -> dict[str, Any]:
    return torch.load(model_path, map_location=device, mmap=True, weights_only=True)


def maybe_squeeze(z: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in z.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.squeeze()
        else:
            result[key] = value
    return result


def detect_model_dims(z: dict[str, Any]) -> tuple[int, int, int, int, int]:
    r_k = z["blocks.0.att.r_k"]
    H = r_k.shape[0]
    N = r_k.shape[1] if r_k.ndim > 1 else 64
    C = H * N
    V = z["emb.weight"].shape[0]
    max_block = 0
    for key in z:
        if key.startswith("blocks."):
            parts = key.split(".")
            idx = int(parts[1])
            if idx > max_block:
                max_block = idx
    L = max_block + 1
    return L, C, H, N, V

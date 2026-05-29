from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vkwr._ops.sampling_ops import sample_temperature_topk_topp, setup_rand

if TYPE_CHECKING:
    from vkwr.engine.request import SamplingParams


class RWKV7Sampler(nn.Module):
    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = seed
        self._states: torch.Tensor | None = None

    def forward(
        self,
        logits: torch.Tensor,
        sampling_params_list: list["SamplingParams"],
    ):
        B, _ = logits.shape
        if self._states is None or self._states.shape[0] // 64 != B:
            self._states = setup_rand(self.seed, B)
        params = sampling_params_list[0]
        sampled = sample_temperature_topk_topp(
            logits,
            self._states,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
        )
        return sampled, None

import torch
import torch.library

from vkwr import _sampling_C  # noqa: F401 — triggers TORCH_LIBRARY registration


@torch.library.register_fake("vkwr_sampling::temperature_topk_topp")
def _(logits, states, temperature, top_k, top_p):
    B = logits.size(0)
    return torch.empty((B,), dtype=torch.int32, device=logits.device)


@torch.library.register_fake("vkwr_sampling::repetition_topk_topp")
def _(logits, penalties, states, presence_penalty, repetition_penalty, penalty_decay, temperature, top_k, top_p):
    B = logits.size(0)
    return torch.empty((B,), dtype=torch.int32, device=logits.device)


# setup_rand takes only scalar args — its C++ impl is registered as CatchAll
# (no dispatch key), so PyTorch can dispatch without tensor device inference.
# CatchAll also covers FakeTensor, so registering an explicit fake impl
# conflicts on newer PyTorch versions. We guard with try/except for
# cross-version compatibility; the fake is only needed for torch.compile.
try:

    @torch.library.register_fake("vkwr_sampling::setup_rand")
    def _setup_rand(seed, B):
        return torch.empty((B * 64,), dtype=torch.int8, device="cuda")
except RuntimeError:
    pass


def setup_rand(seed, B):
    return torch.ops.vkwr_sampling.setup_rand(seed, B)


def sample_temperature_topk_topp(logits, states, temperature=1.0, top_k=-1, top_p=1.0):
    return torch.ops.vkwr_sampling.temperature_topk_topp(logits, states, temperature, top_k, top_p)


def sample_repetition_topk_topp(
    logits, penalties, states, presence_penalty=0.0, repetition_penalty=1.0, penalty_decay=0.996, temperature=1.0, top_k=-1, top_p=1.0
):
    return torch.ops.vkwr_sampling.repetition_topk_topp(
        logits, penalties, states, presence_penalty, repetition_penalty, penalty_decay, temperature, top_k, top_p
    )

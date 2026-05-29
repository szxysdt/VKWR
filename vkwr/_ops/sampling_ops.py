import torch
import torch.library
from vkwr import _sampling_C


@torch.library.register_fake("vkwr_sampling::temperature_topk_topp")
def _(logits, states, temperature, top_k, top_p):
    B = logits.size(0)
    return torch.empty((B,), dtype=torch.int32, device=logits.device)


@torch.library.register_fake("vkwr_sampling::repetition_topk_topp")
def _(logits, penalties, states, presence_penalty, repetition_penalty, penalty_decay, temperature, top_k, top_p):
    B = logits.size(0)
    return torch.empty((B,), dtype=torch.int32, device=logits.device)


@torch.library.register_fake("vkwr_sampling::setup_rand")
def _setup_rand(seed, B):
    return torch.empty((B * 64,), dtype=torch.int8, device="cuda")


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

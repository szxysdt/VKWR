import time

import pytest
import torch

from vkwr._ops.sampling_ops import (
    sample_repetition_topk_topp,
    sample_temperature_topk_topp,
    setup_rand,
)


def _make_states(seed, B):
    return setup_rand(seed, B)


def _logits(V, dtype=torch.float32):
    return torch.randn(V, dtype=dtype, device="cuda")


@pytest.mark.parametrize(
    "B,seed",
    [
        (1, 42),
        (4, 123),
        (16, 0),
        (1, 2**40),
    ],
)
def test_setup_rand_basic(B, seed):
    states = setup_rand(seed, B)
    assert states.dtype == torch.int8
    assert states.device.type == "cuda"


def test_setup_rand_determinism():
    s1 = setup_rand(12345, 4)
    s2 = setup_rand(12345, 4)
    torch.testing.assert_close(s1, s2, atol=0, rtol=0)


def test_setup_rand_different_seed():
    s1 = setup_rand(12345, 4)
    s2 = setup_rand(54321, 4)
    assert not torch.equal(s1, s2)


@pytest.mark.parametrize("V", [4, 16, 64, 256, 50256, 100000])
def test_sample_temperature_topk_topp_basic(V):
    B = 2
    logits = torch.randn(B, V, device="cuda")
    states = _make_states(42, B)
    out = sample_temperature_topk_topp(logits, states)
    assert out.shape == (B,)
    assert out.dtype == torch.int32
    assert out.device.type == "cuda"
    assert (out >= 0).all()
    assert (out < V).all()


@pytest.mark.parametrize("V", [4, 64, 50256])
def test_sample_repetition_topk_topp_basic(V):
    B = 2
    logits = torch.randn(B, V, device="cuda")
    penalties = torch.zeros(B, V, device="cuda")
    states = _make_states(42, B)
    out = sample_repetition_topk_topp(logits, penalties, states)
    assert out.shape == (B,)
    assert out.dtype == torch.int32
    assert (out >= 0).all()
    assert (out < V).all()


@pytest.mark.parametrize("V", [4, 64, 256, 50256])
def test_sample_temperature_topk_topp_3d_logits(V):
    B, T = 2, 4
    logits = torch.randn(B, T, V, device="cuda")
    states = _make_states(42, B)
    out = sample_temperature_topk_topp(logits, states)
    assert out.shape == (B,)
    assert (out >= 0).all()
    assert (out < V).all()


@pytest.mark.parametrize("V", [4, 64, 256])
def test_sample_repetition_topk_topp_3d_logits(V):
    B, T = 2, 4
    logits = torch.randn(B, T, V, device="cuda")
    penalties = torch.zeros(B, V, device="cuda")
    states = _make_states(42, B)
    out = sample_repetition_topk_topp(logits, penalties, states)
    assert out.shape == (B,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_temperature_topk_topp_temperature_zero_path():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    out = sample_temperature_topk_topp(logits, states, temperature=1.0, top_k=-1, top_p=1.0)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_temperature_topk_topp_high_temperature():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    out = sample_temperature_topk_topp(logits, states, temperature=100.0)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_temperature_topk_topp_low_temperature():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    out = sample_temperature_topk_topp(logits, states, temperature=0.001)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_temperature_topk_topp_temperature_out_of_range_low():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    with pytest.raises((RuntimeError, ValueError)):
        sample_temperature_topk_topp(logits, states, temperature=0.0)


def test_sample_temperature_topk_topp_temperature_out_of_range_high():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    with pytest.raises((RuntimeError, ValueError)):
        sample_temperature_topk_topp(logits, states, temperature=1001.0)


def test_sample_temperature_topk_topp_negative_temperature():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    with pytest.raises((RuntimeError, ValueError)):
        sample_temperature_topk_topp(logits, states, temperature=-1.0)


def test_sample_temperature_topk_topp_topk_effect():
    V = 256
    logits = torch.zeros(V, device="cuda")
    logits[:10] = 10.0
    states = _make_states(42, 1)
    out = sample_temperature_topk_topp(logits.unsqueeze(0), states, temperature=1.0, top_k=10, top_p=1.0)
    assert out[0] < 10


def test_sample_temperature_topk_topp_topp_effect():
    V = 256
    logits = torch.zeros(V, device="cuda")
    logits[:5] = 10.0
    states = _make_states(42, 1)
    out = sample_temperature_topk_topp(logits.unsqueeze(0), states, temperature=1.0, top_k=-1, top_p=0.5)
    assert out[0] < 5


def test_sample_temperature_topk_topp_topk_zero_clamped():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    out = sample_temperature_topk_topp(logits, states, top_k=0)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_temperature_topk_topp_topk_negative_clamped():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    out = sample_temperature_topk_topp(logits, states, top_k=-1)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_temperature_topk_topp_topp_zero_becomes_argmax():
    V = 256
    logits = torch.zeros(V, device="cuda")
    logits[42] = 10.0
    states = _make_states(42, 1)
    out = sample_temperature_topk_topp(logits.unsqueeze(0), states, top_p=0.0)
    assert out[0] == 42


def test_sample_temperature_topk_topp_topp_clamped():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    out1 = sample_temperature_topk_topp(logits, states, top_p=-0.5)
    out2 = sample_temperature_topk_topp(logits.clone(), states, top_p=1.5)
    assert out1.shape == (2,)
    assert out2.shape == (2,)


def test_sample_temperature_topk_topp_unsupported_dtype():
    V = 256
    logits = torch.randint(0, 100, (2, V), dtype=torch.int32, device="cuda")
    states = _make_states(42, 2)
    with pytest.raises((RuntimeError, ValueError)):
        sample_temperature_topk_topp(logits, states)


def test_sample_temperature_topk_topp_fp16_dtype():
    V = 256
    logits = torch.randn(2, V, dtype=torch.float16, device="cuda")
    states = _make_states(42, 2)
    with pytest.raises((RuntimeError, ValueError)):
        sample_temperature_topk_topp(logits, states)


def test_sample_temperature_topk_topp_invalid_vocab_not_multiple_of_4():
    V = 255
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    with pytest.raises((RuntimeError, ValueError)):
        sample_temperature_topk_topp(logits, states)


def test_sample_temperature_topk_topp_invalid_vocab_zero():
    V = 0
    logits = torch.randn(2, V, device="cuda")
    states = _make_states(42, 2)
    with pytest.raises((RuntimeError, ValueError)):
        sample_temperature_topk_topp(logits, states)


def test_sample_temperature_topk_topp_invalid_vocab_too_large():
    V = 2000000
    logits = torch.randn(1, V, device="cuda")
    states = _make_states(42, 1)
    with pytest.raises((RuntimeError, ValueError)):
        sample_temperature_topk_topp(logits, states)


def test_sample_temperature_topk_topp_nan_logits():
    V = 256
    logits = torch.full((2, V), float("nan"), device="cuda")
    states = _make_states(42, 2)
    out = sample_temperature_topk_topp(logits, states)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_temperature_topk_topp_inf_logits():
    V = 256
    logits = torch.full((2, V), float("inf"), device="cuda")
    logits[:, 42] = float("-inf")
    logits[0, 100] = 10.0
    logits[1, 200] = 10.0
    states = _make_states(42, 2)
    out = sample_temperature_topk_topp(logits, states)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_temperature_topk_topp_mixed_inf_nan():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    logits[0, ::10] = float("inf")
    logits[1, ::20] = float("nan")
    states = _make_states(42, 2)
    out = sample_temperature_topk_topp(logits, states)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_repetition_topk_topp_penalty_effect():
    V = 256
    logits = torch.zeros(2, V, device="cuda")
    logits[:, 10] = 5.0
    penalties = torch.zeros(2, V, device="cuda")
    penalties[:, 10] = 10.0
    states = _make_states(42, 2)
    out = sample_repetition_topk_topp(logits, penalties, states, repetition_penalty=1.0, presence_penalty=0.0)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_repetition_topk_topp_presence_penalty():
    V = 256
    logits = torch.zeros(2, V, device="cuda")
    logits[:, 10] = 5.0
    penalties = torch.zeros(2, V, device="cuda")
    penalties[0, 10] = 1.0
    states = _make_states(42, 2)
    out = sample_repetition_topk_topp(logits, penalties, states, presence_penalty=2.0, repetition_penalty=1.0)
    assert out.shape == (2,)


def test_sample_repetition_topk_topp_penalty_decay():
    V = 256
    logits = torch.zeros(1, V, device="cuda")
    logits[:, 10] = 5.0
    penalties = torch.zeros(1, V, device="cuda")
    penalties[:, 10] = 1.0
    states = _make_states(42, 1)
    out = sample_repetition_topk_topp(logits, penalties, states, penalty_decay=0.5)
    assert out.shape == (1,)


def test_sample_repetition_topk_topp_no_repetition():
    V = 64
    logits = torch.zeros(V, device="cuda")
    logits[0] = 10.0
    logits[1] = 10.0
    penalties = torch.zeros(V, device="cuda")
    penalties[0] = 1.0
    states = _make_states(42, 1)
    out = sample_repetition_topk_topp(logits.unsqueeze(0), penalties.unsqueeze(0), states, repetition_penalty=2.0)
    assert out[0] == 1


def test_sample_repetition_topk_topp_high_repetition_penalty():
    V = 256
    logits = torch.zeros(1, V, device="cuda")
    logits[:, 10] = 10.0
    penalties = torch.zeros(1, V, device="cuda")
    penalties[:, 10] = 100.0
    states = _make_states(42, 1)
    out = sample_repetition_topk_topp(logits, penalties, states)
    assert out[0] != 10


def test_sample_repetition_topk_topp_1d_penalties():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    penalties = torch.zeros(2, V, device="cuda")
    states = _make_states(42, 2)
    out = sample_repetition_topk_topp(logits, penalties, states)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_temperature_topk_topp_determinism_with_seed():
    V = 256
    logits = torch.randn(1, V, device="cuda")
    states1 = _make_states(42, 1)
    states2 = _make_states(42, 1)
    out1 = sample_temperature_topk_topp(logits.clone(), states1, temperature=1.0)
    out2 = sample_temperature_topk_topp(logits.clone(), states2, temperature=1.0)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


def test_sample_temperature_topk_topp_different_seed_different_result():
    V = 256
    logits = torch.full((2, V), 0.0, device="cuda")
    states1 = _make_states(1, 1)
    states2 = _make_states(2, 1)
    out1 = sample_temperature_topk_topp(logits, states1)
    out2 = sample_temperature_topk_topp(logits.clone(), states2)
    assert out1[0] != out2[0]


def test_sample_temperature_topk_topp_repeated_calls():
    V = 256
    logits_base = torch.randn(1, V, device="cuda")
    seen = set()
    states = _make_states(42, 1)
    for _ in range(20):
        out = sample_temperature_topk_topp(logits_base.clone(), states)
        seen.add(int(out[0]))
    assert len(seen) > 1


def test_sample_repetition_topk_topp_repeated_calls():
    V = 256
    logits_base = torch.randn(1, V, device="cuda")
    seen = set()
    states = _make_states(42, 1)
    for _ in range(20):
        penalties = torch.zeros(1, V, device="cuda")
        out = sample_repetition_topk_topp(logits_base.clone(), penalties, states)
        seen.add(int(out[0]))
    assert len(seen) > 1


def test_sample_temperature_topk_topp_topp_1_is_no_filter():
    V = 256
    logits = torch.randn(2, V, device="cuda")
    out1 = sample_temperature_topk_topp(logits.clone(), _make_states(42, 2), top_p=1.0)
    out2 = sample_temperature_topk_topp(logits.clone(), _make_states(42, 2), top_p=1.0)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


def test_sample_temperature_topk_topp_argmax_temperature_zero_limit():
    V = 256
    logits = torch.zeros(V, device="cuda")
    logits[42] = 100.0
    states = _make_states(42, 1)
    out = sample_temperature_topk_topp(logits.unsqueeze(0), states, temperature=0.001)
    assert out[0] == 42


def test_sample_temperature_topk_topp_uniform_logits_distribution():
    V = 1000
    logits = torch.zeros(1, V, device="cuda")
    counts = {}
    for i in range(100):
        states = _make_states(42 + i, 1)
        out = sample_temperature_topk_topp(logits.clone(), states)
        idx = int(out[0])
        counts[idx] = counts.get(idx, 0) + 1
    avg = sum(counts.values()) / len(counts)
    max_dev = max(abs(v - avg) for v in counts.values())
    assert max_dev < avg * 3


def test_sample_repetition_topk_topp_penalties_mutation():
    V = 256
    logits = torch.zeros(1, V, device="cuda")
    logits[:, 10] = 5.0
    penalties = torch.zeros(1, V, device="cuda")
    penalties_before = penalties.clone()
    states = _make_states(42, 1)
    sample_repetition_topk_topp(logits, penalties, states, presence_penalty=1.0)
    assert not torch.equal(penalties, penalties_before)


def test_sample_repetition_topk_topp_invalid_vocab():
    V = 255
    logits = torch.randn(2, V, device="cuda")
    penalties = torch.zeros(2, V, device="cuda")
    states = _make_states(42, 2)
    with pytest.raises((RuntimeError, ValueError)):
        sample_repetition_topk_topp(logits, penalties, states)


def test_sample_repetition_topk_topp_nan_logits():
    V = 256
    logits = torch.full((2, V), float("nan"), device="cuda")
    penalties = torch.zeros(2, V, device="cuda")
    states = _make_states(42, 2)
    out = sample_repetition_topk_topp(logits, penalties, states)
    assert out.shape == (2,)
    assert (out >= 0).all()
    assert (out < V).all()


def test_sample_temperature_topk_topp_performance():
    V = 50256
    B = 8
    logits = torch.randn(B, V, device="cuda")
    states = _make_states(42, B)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(30):
        sample_temperature_topk_topp(logits.clone(), states, temperature=0.8, top_k=40, top_p=0.95)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / 30 * 1000
    print(f"  temperature_topk_topp perf: {avg_ms:.3f}ms/call, V={V}, B={B}")
    assert avg_ms < 100.0


def test_sample_repetition_topk_topp_performance():
    V = 50256
    B = 8
    logits = torch.randn(B, V, device="cuda")
    penalties = torch.zeros(B, V, device="cuda")
    states = _make_states(42, B)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(30):
        sample_repetition_topk_topp(logits.clone(), penalties.clone(), states, temperature=0.8, top_k=40, top_p=0.95)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / 30 * 1000
    print(f"  repetition_topk_topp perf: {avg_ms:.3f}ms/call, V={V}, B={B}")
    assert avg_ms < 100.0

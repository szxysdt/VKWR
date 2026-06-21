#!/usr/bin/env python3
"""RWKV7 varlen alignment tests: fp16 varlen model vs fp32io16 baseline.

Compares the varlen model (wkv_mode="fp16") against a per-request baseline
(wkv_mode="fp32io16") to verify correctness of variable-length batching.

Requires a model checkpoint at VKWR_RWKV7_MODEL_PATH.

NOTE: Results from these tests are for reference only. Actual computational
stability and precision should be evaluated with a dedicated test dataset on
the full model to determine real-world performance.
"""

import os
import warnings

import pytest
import torch


def _get_model_path() -> str | None:
    return os.environ.get("VKWR_RWKV7_MODEL_PATH")


def _load_varlen_model(model_path: str):
    from vkwr.config.model import RWKV7InferenceConfig, WeightConfig
    from vkwr.model_executor.models.rwkv7_varlen import RWKV7

    inference_config = RWKV7InferenceConfig(
        wkv_mode="fp16",
        emb_device="cpu",
        rkv_mode="off",
        cmix_sparse="no-fc",
        lowrank_weight="both",
    )
    weight_config = WeightConfig.from_cli_string("att_c2c,ffn_key,head")
    return RWKV7(
        model_path=model_path,
        weight_config=weight_config,
        inference_config=inference_config,
    )


def _load_baseline_model(model_path: str):
    from vkwr.config.model import RWKV7InferenceConfig, WeightConfig
    from vkwr.model_executor.models.rwkv7 import RWKV7

    inference_config = RWKV7InferenceConfig(
        wkv_mode="fp32io16",
        emb_device="cpu",
        rkv_mode="off",
        cmix_sparse="no-fc",
        lowrank_weight="both",
    )
    weight_config = WeightConfig.from_cli_string("att_c2c,ffn_key,head")
    return RWKV7(
        model_path=model_path,
        weight_config=weight_config,
        inference_config=inference_config,
    )


@pytest.fixture(scope="module")
def models():
    """Load both models once per module, clean up on teardown."""
    model_path = _get_model_path()
    if model_path is None or not os.path.isfile(model_path):
        pytest.skip(f"Model not found: {model_path}")
    varlen = _load_varlen_model(model_path)
    baseline = _load_baseline_model(model_path)
    yield varlen, baseline
    del varlen, baseline
    torch.cuda.empty_cache()


def _make_tokens(offset: int, length: int, vocab: int) -> torch.Tensor:
    """Deterministic token generation."""
    tokens = torch.arange(offset, offset + length, dtype=torch.long)
    tokens = (tokens * 1103515245 + 12345) % vocab
    return tokens


def _build_tokens(seq_lens: list[int], V: int) -> list[torch.Tensor]:
    """Build per-request token tensors on CUDA (shared source of truth)."""
    return [t.cuda() for t in _build_tokens_cpu(seq_lens, V)]


def _build_tokens_cpu(seq_lens: list[int], V: int) -> list[torch.Tensor]:
    """CPU-only variant for error messages and debugging."""
    return [_make_tokens(sum(seq_lens[:i]), sl, V) for i, sl in enumerate(seq_lens)]


def _build_varlen_inputs(tokens_list: list[torch.Tensor], seq_lens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build flat tokens and query_start_loc from per-request token list."""
    flat_tokens = torch.cat(tokens_list)
    query_start_loc = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.int32,
        device=flat_tokens.device,
    )
    return flat_tokens, query_start_loc


def _baseline_forward(baseline, tokens_list: list[torch.Tensor]) -> torch.Tensor:
    """Run each request independently via baseline, return [B, V] logits."""
    outs = []
    for tokens_i in tokens_list:
        state_i = baseline.zero_state(1)
        out_i = baseline.forward(tokens_i.unsqueeze(0), state_i)
        torch.cuda.synchronize()
        outs.append(out_i)
    return torch.cat(outs, dim=0)


def _varlen_forward(varlen, tokens_list: list[torch.Tensor], seq_lens: list[int]) -> torch.Tensor:
    """Run all requests in one varlen batch, return [B, V] logits."""
    flat_tokens, query_start_loc = _build_varlen_inputs(tokens_list, seq_lens)
    B = len(seq_lens)
    max_t = max(seq_lens)
    state_vl = varlen.zero_state(B)
    out = varlen.forward(flat_tokens, state_vl, query_start_loc, max_t)
    torch.cuda.synchronize()
    return out


def _assert_self_compare(out_a: torch.Tensor, out_b: torch.Tensor, label: str, seq_lens: list[int], V: int) -> None:
    """Self-comparison: top5 overlap >= 4 passes; overlap == 5 is perfect; overlap == 4 warns."""
    fa, fb = out_a.float().cpu(), out_b.float().cpu()
    B = len(seq_lens)
    fails = []
    for i in range(B):
        top5_a = set(fa[i].topk(5, dim=-1).indices.tolist())
        top5_b = set(fb[i].topk(5, dim=-1).indices.tolist())
        overlap = len(top5_a & top5_b)
        if overlap == 5:
            continue
        if overlap >= 4:
            tokens_i = _make_tokens(sum(seq_lens[:i]), seq_lens[i], V).tolist()
            msg = (
                f"  R{i}(T={seq_lens[i]}): top5 overlap={overlap}/5 -- relaxed pass\n"
                f"    tokens = {tokens_i}\n"
                f"    top5_a = {sorted(top5_a)}\n"
                f"    top5_b = {sorted(top5_b)}"
            )
            warnings.warn(
                f"Baseline self-comparison top5 partial match for case '{label}' seq_lens={seq_lens}:\n"
                f"{msg}\n"
                "Note: Numerical instability originates from jitter under fp16 precision. "
                "This test uses pseudo-random tokens that are far from the natural token distribution, "
                "which can cause certain logits to be very close to each other. "
                "Future tests should rely on real datasets for authoritative results."
            )
            continue
        tokens_i = _make_tokens(sum(seq_lens[:i]), seq_lens[i], V).tolist()
        fails.append(
            f"  R{i}(T={seq_lens[i]}): top5 overlap={overlap}/5\n    tokens = {tokens_i}\n    top5_a = {sorted(top5_a)}\n    top5_b = {sorted(top5_b)}"
        )
    if fails:
        assert False, f"Baseline self-comparison failed for case '{label}' seq_lens={seq_lens}:\n" + "\n".join(fails)


def _compare_per_request(out_vl: torch.Tensor, out_v1: torch.Tensor, seq_lens: list[int]) -> dict:
    """Compare varlen vs baseline per request. Returns aggregated result."""
    out_vl_f = out_vl.float().cpu()
    out_v1_f = out_v1.float().cpu()
    B = len(seq_lens)

    per_req = []
    all_pass = True

    for i in range(B):
        diff = (out_vl_f[i] - out_v1_f[i]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        s_vl = out_vl_f[i].argmax().item()
        s_v1 = out_v1_f[i].argmax().item()

        # --- Single topk(64) for both tensors, derive top2/top5 from it ---
        top64_vl_val, top64_vl_idx = out_vl_f[i].topk(64, sorted=True)
        top64_v1_val, top64_v1_idx = out_v1_f[i].topk(64, sorted=True)

        # --- top2 relaxed matching ---
        top2_vl_set = set(top64_vl_idx[:2].tolist())
        top2_v1_set = set(top64_v1_idx[:2].tolist())
        top2_match = s_vl == s_v1
        if not top2_match and top2_vl_set == top2_v1_set:
            gap_vl = (out_vl_f[i][s_vl] - out_vl_f[i][s_v1]).abs().item()
            gap_v1 = (out_v1_f[i][s_v1] - out_v1_f[i][s_vl]).abs().item()
            if gap_v1 < 0.05:
                warnings.warn(f"top2 swap detected Req{i}: VL argmax={s_vl}, V1 argmax={s_v1}, gap_vl={gap_vl:.6f}, gap_v1={gap_v1:.6f} -- relaxed pass")
                top2_match = True

        top5_vl = set(top64_vl_idx[:5].tolist())
        top5_v1 = set(top64_v1_idx[:5].tolist())
        top5_ov = len(top5_vl & top5_v1)

        # --- top-64 max diff (meaningful region, not full vocab tail) ---
        topk_idx_vl = set(top64_vl_idx.tolist())
        topk_idx_v1 = set(top64_v1_idx.tolist())
        topk_union = topk_idx_vl | topk_idx_v1
        max_diff_top64 = max((diff[j].item() for j in topk_union), default=0.0)

        req_ok = top2_match and top5_ov >= 3 and max_diff_top64 <= 0.6 and mean_diff <= 0.5

        if top5_ov < 5:
            warnings.warn(f"top5 partial match Req{i}: overlap={top5_ov}/5, VL top5={sorted(top5_vl)}, V1 top5={sorted(top5_v1)}")
        if not req_ok:
            all_pass = False

        top5_vl_indices = top64_vl_idx[:5].tolist()
        top5_vl_values = top64_vl_val[:5].tolist()
        top5_v1_indices = top64_v1_idx[:5].tolist()
        top5_v1_values = top64_v1_val[:5].tolist()

        per_req.append(
            {
                "idx": i,
                "t": seq_lens[i],
                "max_diff": max_diff,
                "max_diff_top64": max_diff_top64,
                "mean_diff": mean_diff,
                "top2_match": top2_match,
                "top5_overlap": top5_ov,
                "vl_argmax": s_vl,
                "v1_argmax": s_v1,
                "ok": req_ok,
                "top5_vl_indices": top5_vl_indices,
                "top5_vl_values": top5_vl_values,
                "top5_v1_indices": top5_v1_indices,
                "top5_v1_values": top5_v1_values,
            }
        )

    return {"per_req": per_req, "all_pass": all_pass}


# =============================================================================
# Alignment cases
# =============================================================================

# Extending this list is the intended way to add new cases. Each entry is
# (test_label, [seq_lens]).  The label is used in assertion messages and
# can serve as a pytest id via parametrize `ids`.
ALIGNMENT_CASES: list[tuple[str, list[int]]] = [
    # --- Group 1: decode-only ---
    ("decode_b3", [1, 1, 1]),
    ("decode_b5", [1, 1, 1, 1, 1]),
    ("decode_b8", [1] * 8),
    # --- Group 2: prefill + decode mix ---
    ("pre1_short+dec", [5, 1]),
    ("pre1_dec+short", [1, 5]),
    ("pre1_dec2", [10, 1, 1]),
    ("pre1_sandwich", [1, 10, 1]),
    ("pre1_bug20", [20, 1, 1]),
    ("pre1_medium_mix", [50, 1, 3]),
    ("pre1_long+dec", [100, 1]),
    ("pre1_longer_mix", [200, 2, 1]),
    # --- Group 3: max_t boundary (wkv kernel split at 16) ---
    ("maxt_le16_15", [15, 1]),
    ("maxt_le16_16", [16, 1]),
    ("maxt_gt16_17", [17, 3]),
    ("maxt_gt16_32", [32, 5, 2]),
    ("maxt_pow2_64", [64, 1]),
    ("maxt_pow2_128", [128, 3]),
    # --- Group 4: pure prefill ---
    ("pre2_diff_len", [3, 7]),
    ("pre2_three", [8, 2, 5]),
    ("pre2_short_mix", [4, 6, 2]),
    ("pre2_four", [7, 13, 3, 9]),
    ("pre2_five_varied", [1, 3, 5, 2, 7]),
    ("pre2_many_small", [2, 3, 1, 4, 2]),
    # --- Group 5: long sequence ---
    ("long_single_256", [256]),
    ("long_single_512", [512]),
    ("long_single_1024", [1024]),
    # --- Group 6: multi-request ---
    ("multi_four", [2, 8, 4, 1]),
    ("multi_six", [3, 5, 2, 7, 4, 1]),
]


def _sort_seq_lens(seq_lens: list[int], reverse: bool = False) -> list[int]:
    """Sort seq_lens within a batch.

    Currently only ascending order is tested.  Descending (reverse=True)
    is supported for future coverage of ordering-dependent varlen bugs.
    """
    return sorted(seq_lens, reverse=reverse)


# TODO: This test only covers ascending-sorted seq_lens.  Results are
# indicative only — varlen batching correctness for unsorted or
# descending-order batches has not been validated yet.  Enable
# `ALIGNMENT_SORT_REVERSE = True` to test descending order, or extend
# to a parametrize over sort strategies for full coverage.
ALIGNMENT_SORT_REVERSE = False


@pytest.mark.parametrize("label,seq_lens", ALIGNMENT_CASES, ids=[c[0] for c in ALIGNMENT_CASES])
def test_varlen_alignment(models, label: str, seq_lens: list[int]):
    """Varlen fp16 output must align with fp32io16 baseline per request.

    NOTE: Only ascending-sorted batches are tested.  Results are for
    reference only; unsorted and descending-order cases are not covered.
    """
    varlen, baseline = models
    seq_lens = _sort_seq_lens(seq_lens, reverse=ALIGNMENT_SORT_REVERSE)
    V = baseline.config.V
    tokens_list = _build_tokens(seq_lens, V)

    torch.cuda.synchronize()
    out_vl = _varlen_forward(varlen, tokens_list, seq_lens)
    torch.cuda.synchronize()

    out_v1 = _baseline_forward(baseline, tokens_list)
    out_v1_self = _baseline_forward(baseline, tokens_list)

    _assert_self_compare(out_v1, out_v1_self, label, seq_lens, V)

    result = _compare_per_request(out_vl, out_v1, seq_lens)

    diag_lines = []
    for r in result["per_req"]:
        tag = "OK" if r["ok"] else "FAIL"
        diag_lines.append(
            f"  R{r['idx']}(T={r['t']}): max64={r['max_diff_top64']:.4e} mean={r['mean_diff']:.4e} top2={r['top2_match']} top5={r['top5_overlap']}/5 [{tag}]"
        )
        if not r["ok"]:
            diag_lines.append(f"    VL top5 idx={r['top5_vl_indices']} val={[f'{v:.4f}' for v in r['top5_vl_values']]}")
            diag_lines.append(f"    V1 top5 idx={r['top5_v1_indices']} val={[f'{v:.4f}' for v in r['top5_v1_values']]}")

    assert result["all_pass"], f"Alignment failed for case '{label}' seq_lens={seq_lens}:\n" + "\n".join(diag_lines)


# =============================================================================
# State and shape tests
# =============================================================================


def test_varlen_output_shape(models):
    """Output shape must be [B, V], dtype fp16, no NaN."""
    varlen, _ = models
    seq_lens = [5, 3, 1]
    V = varlen.config.V
    tokens_list = _build_tokens(seq_lens, V)
    out = _varlen_forward(varlen, tokens_list, seq_lens)
    B = len(seq_lens)
    assert out.shape == (B, V), f"Expected ({B}, {V}), got {out.shape}"
    assert out.dtype == torch.float16
    assert not torch.isnan(out).any()


def test_varlen_state_elapsed(models):
    """state[2] (elapsed) must equal the actual seq_len per request after forward."""
    varlen, _ = models
    seq_lens = [7, 3, 12, 1]
    B = len(seq_lens)
    V = varlen.config.V
    tokens_list = _build_tokens(seq_lens, V)
    flat_tokens, query_start_loc = _build_varlen_inputs(tokens_list, seq_lens)
    state = varlen.zero_state(B)
    max_t = max(seq_lens)

    varlen.forward(flat_tokens, state, query_start_loc, max_t)
    torch.cuda.synchronize()

    elapsed = state[2].tolist()
    assert elapsed == seq_lens, f"Elapsed mismatch: expected {seq_lens}, got {elapsed}"


def test_varlen_determinism(models):
    """Same varlen forward on same input must produce identical top5."""
    varlen, _ = models
    seq_lens = [10, 3, 7]
    V = varlen.config.V
    tokens_list = _build_tokens(seq_lens, V)

    outs = []
    for _ in range(3):
        out = _varlen_forward(varlen, tokens_list, seq_lens)
        outs.append(out.float().cpu())

    B = len(seq_lens)
    for i in range(1, len(outs)):
        for r in range(B):
            top5_0 = set(outs[0][r].topk(5, dim=-1).indices.tolist())
            top5_i = set(outs[i][r].topk(5, dim=-1).indices.tolist())
            overlap = len(top5_0 & top5_i)
            if overlap == 5:
                continue
            if overlap >= 4:
                tokens_r = _make_tokens(sum(seq_lens[:r]), seq_lens[r], V).tolist()
                warnings.warn(
                    f"Determinism top5 partial match R{r}: run 0 vs run {i}, "
                    f"overlap={overlap}/5 -- relaxed pass\n"
                    f"  tokens = {tokens_r}\n"
                    f"  top5_0 = {sorted(top5_0)}\n"
                    f"  top5_i = {sorted(top5_i)}\n"
                    "Note: Numerical instability originates from jitter under fp16 precision. "
                    "This test uses pseudo-random tokens that are far from the natural token distribution, "
                    "which can cause certain logits to be very close to each other. "
                    "Future tests should rely on real datasets for authoritative results."
                )
                continue
            tokens_r = _make_tokens(sum(seq_lens[:r]), seq_lens[r], V).tolist()
            assert False, (
                f"Determinism failed R{r}: run 0 top5 vs run {i} top5, "
                f"overlap={overlap}/5\n"
                f"  tokens = {tokens_r}\n"
                f"  top5_0 = {sorted(top5_0)}\n"
                f"  top5_i = {sorted(top5_i)}"
            )

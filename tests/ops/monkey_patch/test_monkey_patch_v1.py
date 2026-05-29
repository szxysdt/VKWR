from pathlib import Path

import pytest
import torch

# Import V1 ops to register torch.ops
from vkwr._ops.v1 import import_all_v1_ops

import_all_v1_ops()

GOLDEN_DIR = Path(__file__).parents[2] / "third_party" / "Albatross_faster3a" / "ops" / "monkey_patch"

PRECISION_TIERS = [
    (1e-1, 1e-1, "loose"),
    (5e-2, 5e-2, "medium"),
    (1e-2, 1e-2, "tight"),
    (1e-3, 1e-3, "very_tight"),
    (1e-4, 1e-4, "extreme"),
]


def _get_files(op_name):
    d = GOLDEN_DIR / op_name
    if not d.exists():
        return []
    return sorted(d.glob("layer_*_call_*.pt"))


def _fp16(t):
    return t.cuda().to(torch.float16).contiguous()


def _test_with_tolerances(op_name, files, compute_fn, tiers=None):
    if not files:
        print(f"  {op_name}: SKIPPED (no golden data)")
        pytest.skip(f"No golden data for {op_name}")
        return

    if tiers is None:
        tiers = PRECISION_TIERS

    total = len(files)
    for atol, rtol, label in tiers:
        failures = []
        passed = 0
        for fpath in files:
            data = torch.load(fpath, weights_only=False)
            meta = data.get("meta", {})
            layer_id = meta.get("layer_id", "?")
            call_idx = meta.get("call_index", "?")
            try:
                result, expected = compute_fn(data)
                for i, (r, e) in enumerate(zip(result, expected)):
                    if r is not None and e is not None:
                        suffix = f"[{i}]" if len(result) > 1 else ""
                        torch.testing.assert_close(r, e, atol=atol, rtol=rtol, msg=f"output{suffix} mismatch")
                passed += 1
            except Exception as e:
                failures.append(f"layer={layer_id} call={call_idx}: {e}")
            finally:
                torch.cuda.empty_cache()

        status = "PASS" if not failures else "FAIL"
        print(f"  {op_name} [{label} atol={atol}, rtol={rtol}]: {passed}/{total} [{status}]")

        if failures:
            print(f"    Stopping at tier {label}. First failures:")
            for f in failures[:5]:
                print(f"      - {f}")
            if len(failures) > 5:
                print(f"      ... and {len(failures) - 5} more")
            fail_msg = f"{op_name} failed at {label} (atol={atol}, rtol={rtol}): {len(failures)}/{total} failures"
            pytest.fail(fail_msg)

    max_tier = tiers[-1][2]
    print(f"  {op_name}: ALL TIERS PASSED ({passed}/{total}, max={max_tier})")


# ==================== advance_i32 ====================


def _compute_advance_i32(data):
    amount = data["amount"]
    x = data["elapsed_t_in"].cuda().to(torch.int32).contiguous()
    x_ref = data["elapsed_t_after"].cuda().to(torch.int32)
    torch.ops.vkwr_v1_wkv.advance_i32(x, amount)
    return (x,), (x_ref,)


@pytest.mark.skipif(not _get_files("advance_i32"), reason="No advance_i32 golden data")
def test_advance_i32():
    _test_with_tolerances("advance_i32", _get_files("advance_i32"), _compute_advance_i32)


# ==================== wkv_seq_w0_fp16 ====================


def _compute_wkv_seq_w0_fp16(data):
    B, T, C, H = data["B"], data["T"], data["C"], data["H"]
    state = data["wkv_state_in"].cuda().half().contiguous()
    r = _fp16(data["r"])
    w = _fp16(data["w"])
    w0 = _fp16(data["w0"])
    k = _fp16(data["k"])
    v = _fp16(data["v"])
    a = _fp16(data["a"])
    b = _fp16(data["b"])
    elapsed_t = data["elapsed_t_in"].cuda().to(torch.int32).contiguous()
    y_ref = data["y"].cuda().half()
    state_ref = data["wkv_state_after"].cuda().half()
    elapsed_t_ref = data["elapsed_t_after"].cuda().to(torch.int32)
    y_new = torch.empty(B, T, C, dtype=torch.float16, device="cuda")
    torch.ops.vkwr_v1_wkv.wkv_seq_w0_fp16(B, T, C, H, state, r, w, w0, k, v, a, b, y_new, elapsed_t)
    return (y_new, state, elapsed_t), (y_ref, state_ref, elapsed_t_ref)


# Precision cap: tight (1e-2). FP16 atomicAdd reduction order is non-deterministic across runs,
# so very_tight (1e-3) is not reproducible for some layer/call combinations.
@pytest.mark.skipif(not _get_files("wkv_seq_w0_fp16"), reason="No wkv_seq_w0_fp16 golden data")
def test_wkv_seq_w0_fp16():
    _test_with_tolerances("wkv_seq_w0_fp16", _get_files("wkv_seq_w0_fp16"), _compute_wkv_seq_w0_fp16, PRECISION_TIERS[:3])


# ==================== wkv_seq_fp16 ====================


def _compute_wkv_seq_fp16(data):
    B, T, C, H = data["B"], data["T"], data["C"], data["H"]
    state = data["wkv_state_in"].cuda().half().contiguous()
    r = _fp16(data["r"])
    w = _fp16(data["w"])
    k = _fp16(data["k"])
    v = _fp16(data["v"])
    a = _fp16(data["a"])
    b = _fp16(data["b"])
    elapsed_t = data["elapsed_t_in"].cuda().to(torch.int32).contiguous()
    y_ref = data["y"].cuda().half()
    state_ref = data["wkv_state_after"].cuda().half()
    elapsed_t_ref = data["elapsed_t_after"].cuda().to(torch.int32)
    y_new = torch.empty(B, T, C, dtype=torch.float16, device="cuda")
    torch.ops.vkwr_v1_wkv.wkv_seq_fp16(B, T, C, H, state, r, w, k, v, a, b, y_new, elapsed_t)
    return (y_new, state, elapsed_t), (y_ref, state_ref, elapsed_t_ref)


# Precision cap: very_tight (1e-3). This op is more stable than wkv_seq_w0 but still has
# non-deterministic fp16 reduction at extreme (1e-4) for some layers.
@pytest.mark.skipif(not _get_files("wkv_seq_fp16"), reason="No wkv_seq_fp16 golden data")
def test_wkv_seq_fp16():
    _test_with_tolerances("wkv_seq_fp16", _get_files("wkv_seq_fp16"), _compute_wkv_seq_fp16, PRECISION_TIERS[:4])


# ==================== add_vec ====================


def _compute_add_vec(data):
    C = data["C"]
    x = _fp16(data["x"])
    vec = _fp16(data["vec"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_mix.add_vec(C, x, vec)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("add_vec"), reason="No add_vec golden data")
def test_add_vec():
    _test_with_tolerances("add_vec", _get_files("add_vec"), _compute_add_vec)


# ==================== tmix_mix6 ====================


def _compute_tmix_mix6(data):
    B, T, C = data["B"], data["T"], data["C"]
    shift_state = _fp16(data["shift_state_in"])
    x = _fp16(data["x"])
    x_r = _fp16(data["x_r"])
    x_w = _fp16(data["x_w"])
    x_k = _fp16(data["x_k"])
    x_v = _fp16(data["x_v"])
    x_a = _fp16(data["x_a"])
    x_g = _fp16(data["x_g"])
    xr_ref = data["xr"].cuda().half()
    xw_ref = data["xw"].cuda().half()
    xk_ref = data["xk"].cuda().half()
    xv_ref = data["xv"].cuda().half()
    xa_ref = data["xa"].cuda().half()
    xg_ref = data["xg"].cuda().half()
    shift_state_ref = data["shift_state_after"].cuda().half()
    xr, xw, xk, xv, xa, xg = torch.ops.vkwr_v1_mix.tmix_mix6(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g)
    return (xr, xw, xk, xv, xa, xg, shift_state), (xr_ref, xw_ref, xk_ref, xv_ref, xa_ref, xg_ref, shift_state_ref)


@pytest.mark.skipif(not _get_files("tmix_mix6"), reason="No tmix_mix6 golden data")
def test_tmix_mix6():
    _test_with_tolerances("tmix_mix6", _get_files("tmix_mix6"), _compute_tmix_mix6)


# ==================== tmix_kk_a_gate ====================


def _compute_tmix_kk_a_gate(data):
    B, T, C, H = data["B"], data["T"], data["C"], data["H"]
    k = _fp16(data["k_in"])
    k_k = _fp16(data["k_k"])
    a0 = _fp16(data["a0"])
    a12 = _fp16(data["a12"])
    k_a = _fp16(data["k_a"])
    new_k_ref = data["new_k"].cuda().half()
    neg_kk_ref = data["neg_kk"].cuda().half()
    kka_ref = data["kka"].cuda().half()
    new_k, neg_kk, kka = torch.ops.vkwr_v1_mix.tmix_kk_a_gate(B, T, C, H, k, k_k, a0, a12, k_a)
    return (new_k, neg_kk, kka), (new_k_ref, neg_kk_ref, kka_ref)


@pytest.mark.skipif(not _get_files("tmix_kk_a_gate"), reason="No tmix_kk_a_gate golden data")
def test_tmix_kk_a_gate():
    _test_with_tolerances("tmix_kk_a_gate", _get_files("tmix_kk_a_gate"), _compute_tmix_kk_a_gate)


# ==================== tmix_lnx_rkvres_xg ====================


def _compute_tmix_lnx_rkvres_xg(data):
    B, T, C, H = data["B"], data["T"], data["C"], data["H"]
    x = _fp16(data["x"])
    r = _fp16(data["r"])
    k = _fp16(data["k"])
    v = _fp16(data["v"])
    r_k = _fp16(data["r_k"])
    weight = _fp16(data["weight"])
    bias = _fp16(data["bias"])
    g = _fp16(data["g"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_mix.tmix_lnx_rkvres_xg(B, T, C, H, x, r, k, v, r_k, weight, bias, g)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("tmix_lnx_rkvres_xg"), reason="No tmix_lnx_rkvres_xg golden data")
def test_tmix_lnx_rkvres_xg():
    _test_with_tolerances("tmix_lnx_rkvres_xg", _get_files("tmix_lnx_rkvres_xg"), _compute_tmix_lnx_rkvres_xg)


# ==================== tmix_vres_gate ====================


def _compute_tmix_vres_gate(data):
    B, T, C = data["B"], data["T"], data["C"]
    v = _fp16(data["v"])
    v_first = _fp16(data["v_first"])
    v0 = _fp16(data["v0"])
    v12 = _fp16(data["v12"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_mix.tmix_vres_gate(B, T, C, v, v_first, v0, v12)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("tmix_vres_gate"), reason="No tmix_vres_gate golden data")
def test_tmix_vres_gate():
    _test_with_tolerances("tmix_vres_gate", _get_files("tmix_vres_gate"), _compute_tmix_vres_gate)


# ==================== cmix_mix ====================


def _compute_cmix_mix(data):
    B, T, C = data["B"], data["T"], data["C"]
    shift_state = _fp16(data["shift_state_in"])
    x = _fp16(data["x"])
    x_k = _fp16(data["x_k"])
    mixed_ref = data["mixed"].cuda().half()
    shift_state_ref = data["shift_state_after"].cuda().half()
    mixed_new = torch.ops.vkwr_v1_mix.cmix_mix(B, T, C, x, shift_state, x_k)
    return (mixed_new, shift_state), (mixed_ref, shift_state_ref)


@pytest.mark.skipif(not _get_files("cmix_mix"), reason="No cmix_mix golden data")
def test_cmix_mix():
    _test_with_tolerances("cmix_mix", _get_files("cmix_mix"), _compute_cmix_mix)


# ==================== cmix_sparse_down_relu_one ====================


def _compute_cmix_sparse_down_relu_one(data):
    C, F = data["C"], data["F"]
    preact = _fp16(data["preact"])
    value_fc = _fp16(data["value_fc"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_mix.cmix_sparse_down_relu_one(C, F, preact, value_fc)
    return (out_new,), (out_ref,)


# Precision cap: tight (1e-2). Multiple thread blocks accumulate into the same output via
# atomicAdd(__half2); fp16 addition is non-associative, so reduction order varies between runs.
@pytest.mark.skipif(not _get_files("cmix_sparse_down_relu_one"), reason="No cmix_sparse_down_relu_one golden data")
def test_cmix_sparse_down_relu_one():
    _test_with_tolerances("cmix_sparse_down_relu_one", _get_files("cmix_sparse_down_relu_one"), _compute_cmix_sparse_down_relu_one, PRECISION_TIERS[:3])


# ==================== cmix_sparse_down_relu_rows ====================


def _compute_cmix_sparse_down_relu_rows(data):
    B, T, C, F = data["B"], data["T"], data["C"], data["F"]
    preact = _fp16(data["preact"])
    value_fc = _fp16(data["value_fc"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_mix.cmix_sparse_down_relu_rows(B, T, C, F, preact, value_fc)
    return (out_new,), (out_ref,)


# Precision cap: tight (1e-2). Same atomicAdd(__half2) non-determinism as cmix_sparse_down_relu_one.
@pytest.mark.skipif(not _get_files("cmix_sparse_down_relu_rows"), reason="No cmix_sparse_down_relu_rows golden data")
def test_cmix_sparse_down_relu_rows():
    _test_with_tolerances("cmix_sparse_down_relu_rows", _get_files("cmix_sparse_down_relu_rows"), _compute_cmix_sparse_down_relu_rows, PRECISION_TIERS[:3])


# ==================== cmix_sparse_down_relu_rows_t512 ====================


def _compute_cmix_sparse_down_relu_rows_t512(data):
    B, T, C, F = data["B"], data["T"], data["C"], data["F"]
    preact = _fp16(data["preact"])
    value_fc = _fp16(data["value_fc"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_mix.cmix_sparse_down_relu_rows_t512(B, T, C, F, preact, value_fc)
    return (out_new,), (out_ref,)


# Precision cap: tight (1e-2). Same atomicAdd(__half2) non-determinism as cmix_sparse_down_relu_one.
@pytest.mark.skipif(not _get_files("cmix_sparse_down_relu_rows_t512"), reason="No cmix_sparse_down_relu_rows_t512 golden data")
def test_cmix_sparse_down_relu_rows_t512():
    _test_with_tolerances(
        "cmix_sparse_down_relu_rows_t512", _get_files("cmix_sparse_down_relu_rows_t512"), _compute_cmix_sparse_down_relu_rows_t512, PRECISION_TIERS[:3]
    )


# ==================== relu_square ====================


def _compute_relu_square(data):
    x = _fp16(data["x"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_mix.relu_square(x)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("relu_square"), reason="No relu_square golden data")
def test_relu_square():
    _test_with_tolerances("relu_square", _get_files("relu_square"), _compute_relu_square)


# ==================== layer_norm_f16 ====================


def _compute_layer_norm_f16(data):
    x = _fp16(data["x"])
    weight = _fp16(data["weight"])
    bias = _fp16(data["bias"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_norm.layer_norm_f16(x, weight, bias)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("layer_norm_f16"), reason="No layer_norm_f16 golden data")
def test_layer_norm_f16():
    _test_with_tolerances("layer_norm_f16", _get_files("layer_norm_f16"), _compute_layer_norm_f16)


# ==================== add_layer_norm_f16 ====================


def _compute_add_layer_norm_f16(data):
    x = _fp16(data["x"])
    residual = _fp16(data["residual"])
    weight = _fp16(data["weight"])
    bias = _fp16(data["bias"])
    x_out_ref = data["x_out"].cuda().half()
    normed_ref = data["normed"].cuda().half()
    x_out_new, normed_new = torch.ops.vkwr_v1_norm.add_layer_norm_f16(x, residual, weight, bias)
    return (x_out_new, normed_new), (x_out_ref, normed_ref)


@pytest.mark.skipif(not _get_files("add_layer_norm_f16"), reason="No add_layer_norm_f16 golden data")
def test_add_layer_norm_f16():
    _test_with_tolerances("add_layer_norm_f16", _get_files("add_layer_norm_f16"), _compute_add_layer_norm_f16)


# ==================== add_last_layer_norm_f16 ====================


def _compute_add_last_layer_norm_f16(data):
    x = _fp16(data["x"])
    residual = _fp16(data["residual"])
    weight = _fp16(data["weight"])
    bias = _fp16(data["bias"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_norm.add_last_layer_norm_f16(x, residual, weight, bias)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("add_last_layer_norm_f16"), reason="No add_last_layer_norm_f16 golden data")
def test_add_last_layer_norm_f16():
    _test_with_tolerances("add_last_layer_norm_f16", _get_files("add_last_layer_norm_f16"), _compute_add_last_layer_norm_f16)


# ==================== add_layer_norm_cmix_mix_f16 ====================


def _compute_add_layer_norm_cmix_mix_f16(data):
    x = _fp16(data["x"])
    residual = _fp16(data["residual"])
    shift_state = _fp16(data["shift_state_in"])
    weight = _fp16(data["weight"])
    bias = _fp16(data["bias"])
    x_k = _fp16(data["x_k"])
    x_out_ref = data["x_out"].cuda().half()
    mixed_ref = data["mixed"].cuda().half()
    shift_state_ref = data["shift_state_after"].cuda().half()
    x_out_new, mixed_new = torch.ops.vkwr_v1_norm.add_layer_norm_cmix_mix_f16(x, residual, shift_state, weight, bias, x_k)
    return (x_out_new, mixed_new, shift_state), (x_out_ref, mixed_ref, shift_state_ref)


@pytest.mark.skipif(not _get_files("add_layer_norm_cmix_mix_f16"), reason="No add_layer_norm_cmix_mix_f16 golden data")
def test_add_layer_norm_cmix_mix_f16():
    _test_with_tolerances("add_layer_norm_cmix_mix_f16", _get_files("add_layer_norm_cmix_mix_f16"), _compute_add_layer_norm_cmix_mix_f16)


# ==================== add_layer_norm_tmix_mix6_f16 ====================


def _compute_add_layer_norm_tmix_mix6_f16(data):
    x = _fp16(data["x"])
    residual = _fp16(data["residual"])
    shift_state = _fp16(data["shift_state_in"])
    weight = _fp16(data["weight"])
    bias = _fp16(data["bias"])
    x_r = _fp16(data["x_r"])
    x_w = _fp16(data["x_w"])
    x_k = _fp16(data["x_k"])
    x_v = _fp16(data["x_v"])
    x_a = _fp16(data["x_a"])
    x_g = _fp16(data["x_g"])
    x_out_ref = data["x_out"].cuda().half()
    r_ref = data["r"].cuda().half()
    w_ref = data["w"].cuda().half()
    k_ref = data["k"].cuda().half()
    v_ref = data["v"].cuda().half()
    a_ref = data["a"].cuda().half()
    g_ref = data["g"].cuda().half()
    shift_state_ref = data["shift_state_after"].cuda().half()
    outs = torch.ops.vkwr_v1_norm.add_layer_norm_tmix_mix6_f16(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g)
    return (outs[0], outs[1], outs[2], outs[3], outs[4], outs[5], outs[6], shift_state), (x_out_ref, r_ref, w_ref, k_ref, v_ref, a_ref, g_ref, shift_state_ref)


@pytest.mark.skipif(not _get_files("add_layer_norm_tmix_mix6_f16"), reason="No add_layer_norm_tmix_mix6_f16 golden data")
def test_add_layer_norm_tmix_mix6_f16():
    _test_with_tolerances("add_layer_norm_tmix_mix6_f16", _get_files("add_layer_norm_tmix_mix6_f16"), _compute_add_layer_norm_tmix_mix6_f16)


# ==================== linear_f16 ====================


def _compute_linear_f16(data):
    x = _fp16(data["x"])
    weight = _fp16(data["weight"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_linear.linear_f16(x, weight)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("linear_f16"), reason="No linear_f16 golden data")
def test_linear_f16():
    _test_with_tolerances("linear_f16", _get_files("linear_f16"), _compute_linear_f16)


# ==================== linear_f16_orig ====================


def _compute_linear_f16_orig(data):
    x = _fp16(data["x"])
    weight_orig = _fp16(data["weight_orig"])
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_linear.linear_f16_orig(x, weight_orig)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("linear_f16_orig"), reason="No linear_f16_orig golden data")
def test_linear_f16_orig():
    _test_with_tolerances("linear_f16_orig", _get_files("linear_f16_orig"), _compute_linear_f16_orig)


# ==================== linear_f16_orig_lt_cfg ====================


def _compute_linear_f16_orig_lt_cfg(data):
    x = _fp16(data["x"])
    weight_orig = _fp16(data["weight_orig"])
    params = data["params"]
    workspace_mb, algo_index = params[0], params[1]
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_linear.linear_f16_orig_lt_cfg(x, weight_orig, workspace_mb, algo_index)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("linear_f16_orig_lt_cfg"), reason="No linear_f16_orig_lt_cfg golden data")
def test_linear_f16_orig_lt_cfg():
    _test_with_tolerances("linear_f16_orig_lt_cfg", _get_files("linear_f16_orig_lt_cfg"), _compute_linear_f16_orig_lt_cfg)


# ==================== linear_orig_rows_exact_f16 ====================


def _compute_linear_orig_rows_exact_f16(data):
    x = _fp16(data["x"])
    weight_orig = _fp16(data["weight_orig"])
    params = data["params"]
    threads, out_tile, use4 = params[0], params[1], params[2]
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_linear.linear_orig_rows_exact_f16(x, weight_orig, threads, out_tile, use4)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("linear_orig_rows_exact_f16"), reason="No linear_orig_rows_exact_f16 golden data")
def test_linear_orig_rows_exact_f16():
    _test_with_tolerances("linear_orig_rows_exact_f16", _get_files("linear_orig_rows_exact_f16"), _compute_linear_orig_rows_exact_f16)


# ==================== linear_orig_rows_f16 ====================


def _compute_linear_orig_rows_f16(data):
    x = _fp16(data["x"])
    weight_orig = _fp16(data["weight_orig"])
    params = data["params"]
    row_tile, out_tile = params[0], params[1]
    out_ref = data["out"].cuda().half()
    out_new = torch.ops.vkwr_v1_linear.linear_orig_rows_f16(x, weight_orig, row_tile, out_tile)
    return (out_new,), (out_ref,)


@pytest.mark.skipif(not _get_files("linear_orig_rows_f16"), reason="No linear_orig_rows_f16 golden data")
def test_linear_orig_rows_f16():
    _test_with_tolerances("linear_orig_rows_f16", _get_files("linear_orig_rows_f16"), _compute_linear_orig_rows_f16)


# ==================== linear_wag_rank_in_f16 ====================


def _compute_linear_wag_rank_in_f16(data):
    M = data["M"]
    K = data["K"]
    Rw, Ra, Rg = data["Rw"], data["Ra"], data["Rg"]
    xw = data["xw"].cuda().half().contiguous().view(M, K)
    xa = data["xa"].cuda().half().contiguous().view(M, K)
    xg = data["xg"].cuda().half().contiguous().view(M, K)
    w1_t = _fp16(data["w1_t"])
    a1_t = _fp16(data["a1_t"])
    g1_t = _fp16(data["g1_t"])
    w1_ref = data["w1"].cuda().half().view(M, Rw)
    a1_ref = data["a1"].cuda().half().view(M, Ra)
    g1_ref = data["g1"].cuda().half().view(M, Rg)
    w1_new, a1_new, g1_new = torch.ops.vkwr_v1_rank.linear_wag_rank_in_f16(M, K, Rw, Ra, Rg, xw, xa, xg, w1_t, a1_t, g1_t)
    return (w1_new, a1_new, g1_new), (w1_ref, a1_ref, g1_ref)


@pytest.mark.skipif(not _get_files("linear_wag_rank_in_f16"), reason="No linear_wag_rank_in_f16 golden data")
def test_linear_wag_rank_in_f16():
    _test_with_tolerances("linear_wag_rank_in_f16", _get_files("linear_wag_rank_in_f16"), _compute_linear_wag_rank_in_f16)


# ==================== linear_wagv_rank_in_f16 ====================


def _compute_linear_wagv_rank_in_f16(data):
    M = data["M"]
    K = data["K"]
    Rw, Ra, Rg, Rv = data["Rw"], data["Ra"], data["Rg"], data["Rv"]
    xw = data["xw"].cuda().half().contiguous().view(M, K)
    xa = data["xa"].cuda().half().contiguous().view(M, K)
    xg = data["xg"].cuda().half().contiguous().view(M, K)
    xv = data["xv"].cuda().half().contiguous().view(M, K)
    w1_t = _fp16(data["w1_t"])
    a1_t = _fp16(data["a1_t"])
    g1_t = _fp16(data["g1_t"])
    v1_t = _fp16(data["v1_t"])
    w1_ref = data["w1"].cuda().half().view(M, Rw)
    a1_ref = data["a1"].cuda().half().view(M, Ra)
    g1_ref = data["g1"].cuda().half().view(M, Rg)
    v1_ref = data["v1"].cuda().half().view(M, Rv)
    w1_new, a1_new, g1_new, v1_new = torch.ops.vkwr_v1_rank.linear_wagv_rank_in_f16(M, K, Rw, Ra, Rg, Rv, xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t)
    return (w1_new, a1_new, g1_new, v1_new), (w1_ref, a1_ref, g1_ref, v1_ref)


@pytest.mark.skipif(not _get_files("linear_wagv_rank_in_f16"), reason="No linear_wagv_rank_in_f16 golden data")
def test_linear_wagv_rank_in_f16():
    _test_with_tolerances("linear_wagv_rank_in_f16", _get_files("linear_wagv_rank_in_f16"), _compute_linear_wagv_rank_in_f16)


# ==================== linear_wag_rank_out_f16 ====================


def _compute_linear_wag_rank_out_f16(data):
    M = data["M"]
    C = data["C"]
    Kw, Ka, Kg = data["Kw"], data["Ka"], data["Kg"]
    w1 = data["w1"].cuda().half().contiguous().view(M, Kw)
    a1 = data["a1"].cuda().half().contiguous().view(M, Ka)
    g1 = data["g1"].cuda().half().contiguous().view(M, Kg)
    w2_t = _fp16(data["w2_t"])
    a2_t = _fp16(data["a2_t"])
    g2_t = _fp16(data["g2_t"])
    w_ref = data["w"].cuda().half().view(M, C)
    a_ref = data["a"].cuda().half().view(M, C)
    g_ref = data["g"].cuda().half().view(M, C)
    w_new, a_new, g_new = torch.ops.vkwr_v1_rank.linear_wag_rank_out_f16(M, C, Kw, Ka, Kg, w1, a1, g1, w2_t, a2_t, g2_t)
    return (w_new, a_new, g_new), (w_ref, a_ref, g_ref)


@pytest.mark.skipif(not _get_files("linear_wag_rank_out_f16"), reason="No linear_wag_rank_out_f16 golden data")
def test_linear_wag_rank_out_f16():
    _test_with_tolerances("linear_wag_rank_out_f16", _get_files("linear_wag_rank_out_f16"), _compute_linear_wag_rank_out_f16)


# ==================== linear_wagv_rank_out_f16 ====================


def _compute_linear_wagv_rank_out_f16(data):
    M = data["M"]
    C = data["C"]
    Kw, Ka, Kg, Kv = data["Kw"], data["Ka"], data["Kg"], data["Kv"]
    w1 = data["w1"].cuda().half().contiguous().view(M, Kw)
    a1 = data["a1"].cuda().half().contiguous().view(M, Ka)
    g1 = data["g1"].cuda().half().contiguous().view(M, Kg)
    v1 = data["v1"].cuda().half().contiguous().view(M, Kv)
    w2_t = _fp16(data["w2_t"])
    a2_t = _fp16(data["a2_t"])
    g2_t = _fp16(data["g2_t"])
    v2_t = _fp16(data["v2_t"])
    v = data["v"].cuda().half().contiguous().view(M, C)
    v_first = data["v_first"].cuda().half().contiguous().view(M, C)
    v0 = _fp16(data["v0"])
    w_ref = data["w"].cuda().half().view(M, C)
    a_ref = data["a"].cuda().half().view(M, C)
    g_ref = data["g"].cuda().half().view(M, C)
    v_out_ref = data["v_out"].cuda().half().view(M, C)
    w_new, a_new, g_new, v_new = torch.ops.vkwr_v1_rank.linear_wagv_rank_out_f16(M, C, Kw, Ka, Kg, Kv, w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0)
    return (w_new, a_new, g_new, v_new), (w_ref, a_ref, g_ref, v_out_ref)


@pytest.mark.skipif(not _get_files("linear_wagv_rank_out_f16"), reason="No linear_wagv_rank_out_f16 golden data")
def test_linear_wagv_rank_out_f16():
    _test_with_tolerances("linear_wagv_rank_out_f16", _get_files("linear_wagv_rank_out_f16"), _compute_linear_wagv_rank_out_f16)

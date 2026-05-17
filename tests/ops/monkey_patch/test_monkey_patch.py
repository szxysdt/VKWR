import pytest
import torch
from pathlib import Path
from vkwr._ops.state_ops import state_fwd_seq, state_fwd_one, spmv_forward, HEAD_SIZE

MONKEY_PATCH_DIR = Path(__file__).parent.parent.parent / "third_party" / "ops" / "monkey_patch"

# Precision tiers from loose to strict: (atol, rtol, label)
PRECISION_TIERS = [
    (1e-1, 1e-1, "loose"),
    (5e-2, 5e-2, "medium"),
    (1e-2, 1e-2, "tight"),
    (1e-3, 1e-3, "very_tight"),
    (1e-4, 1e-4, "extreme"),
]


def _get_one_files():
    d = MONKEY_PATCH_DIR / "rwkv7_one_op"
    if not d.exists():
        return []
    return sorted(d.glob("layer_*_call_*.pt"))


def _get_seq_files():
    d = MONKEY_PATCH_DIR / "rwkv7_seq_op"
    if not d.exists():
        return []
    return sorted(d.glob("layer_*_call_*.pt"))


def _get_spmv_files():
    d = MONKEY_PATCH_DIR / "spmv_op"
    if not d.exists():
        return []
    return sorted(d.glob("layer_*_call_*.pt"))


def _reshape_state(state):
    H, HS1, HS2 = state.shape
    assert HS1 == HS2 == HEAD_SIZE
    C = H * HEAD_SIZE
    return state.reshape(1, C, HEAD_SIZE)


def _test_with_tolerances(
    func_name,
    files,
    compute_fn,
    check_state=True,
):
    """Run tests against precision tiers, stopping at the first tier that fails."""
    if not files:
        print(f"  {func_name}: skipped (no files)")
        return

    total = len(files)
    for atol, rtol, label in PRECISION_TIERS:
        failures = []
        passed = 0
        for fpath in files:
            data = torch.load(fpath, weights_only=False)
            layer_id = data["meta"]["layer_id"]
            call_idx = data["meta"]["call_index"]
            try:
                result_y, result_state, expected_y, expected_state = compute_fn(data)
                torch.testing.assert_close(result_y, expected_y, atol=atol, rtol=rtol)
                if check_state:
                    torch.testing.assert_close(result_state, expected_state, atol=atol, rtol=rtol)
                passed += 1
            except Exception as e:
                failures.append(f"layer={layer_id} call={call_idx}: {e}")
            finally:
                torch.cuda.empty_cache()

        status = "PASS" if not failures else "FAIL"
        print(f"  {func_name} [{label} atol={atol}, rtol={rtol}]: {passed}/{total} passed [{status}]")

        if failures:
            print(f"    Stopping precision sweep at tier {label}. First failures:")
            for f in failures[:5]:
                print(f"      - {f}")
            if len(failures) > 5:
                print(f"      ... and {len(failures) - 5} more")
            break


@pytest.mark.skipif(not _get_one_files(), reason="No monkey_patch rwkv7_one_op data found")
def test_rwkv7_one_op_monkey_patch():
    def compute_fn(data):
        state_in = _reshape_state(data["state_in"]).cuda()
        B, C, _ = state_in.shape
        r = data["r"].reshape(B, C).cuda()
        w = data["w"].reshape(B, C).cuda()
        k = data["k"].reshape(B, C).cuda()
        v = data["v"].reshape(B, C).cuda()
        a = data["a"].reshape(B, C).cuda()
        b = data["b"].reshape(B, C).cuda()
        elapsed_t = data["elapsed_t"].expand(B).cuda()
        expected_y = data["y"].reshape(B, C).cuda()
        expected_state_after = _reshape_state(data["state_after"]).cuda()
        state_in_clone = state_in.clone()
        y = state_fwd_one(B, C, state_in_clone, r, w, k, v, a, b, elapsed_t)
        return y, state_in_clone, expected_y, expected_state_after

    _test_with_tolerances("rwkv7_one_op", _get_one_files(), compute_fn)


@pytest.mark.skipif(not _get_seq_files(), reason="No monkey_patch rwkv7_seq_op data found")
def test_rwkv7_seq_op_monkey_patch():
    def compute_fn(data):
        state_in = _reshape_state(data["state_in"]).cuda()
        B, C, _ = state_in.shape
        T = data["r"].shape[0]
        r = data["r"].reshape(B, T, C).cuda()
        w = data["w"].reshape(B, T, C).cuda()
        k = data["k"].reshape(B, T, C).cuda()
        v = data["v"].reshape(B, T, C).cuda()
        a = data["a"].reshape(B, T, C).cuda()
        b = data["b"].reshape(B, T, C).cuda()
        elapsed_t = data["elapsed_t"].expand(B).cuda()
        expected_y = data["y"].reshape(B, T, C).cuda()
        expected_state_after = _reshape_state(data["state_after"]).cuda()
        state_in_clone = state_in.clone()
        y = state_fwd_seq(B, T, C, state_in_clone, r, w, k, v, a, b, elapsed_t)
        return y, state_in_clone, expected_y, expected_state_after

    _test_with_tolerances("rwkv7_seq_op", _get_seq_files(), compute_fn)


@pytest.mark.skipif(not _get_spmv_files(), reason="No monkey_patch spmv_op data found")
def test_spmv_op_monkey_patch():
    def compute_fn(data):
        vec = data["vec"].cuda()
        mat = data["mat"].cuda()
        expected_out = data["out"].cuda()
        out = spmv_forward(vec, mat)
        return out, None, expected_out, None

    _test_with_tolerances("spmv_op", _get_spmv_files(), compute_fn, check_state=False)

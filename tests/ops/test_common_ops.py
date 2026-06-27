import pytest
import torch

from vkwr._ops.common_ops import gather_decode_state, scatter_decode_state


@pytest.fixture
def state_buffers():
    """Create global and decode state buffers for gather/scatter tests."""
    L, C, H, N, max_bsz = 3, 256, 4, 64, 8

    global_shift = torch.randn(L, 2, max_bsz, C, dtype=torch.float16, device="cuda")
    global_wkv = torch.randn(L, max_bsz, H, N, N, dtype=torch.float16, device="cuda")
    global_elapsed = torch.randint(0, 1000, (max_bsz,), dtype=torch.int32, device="cuda")

    decode_shift = torch.empty(L, 2, max_bsz, C, dtype=torch.float16, device="cuda")
    decode_wkv = torch.empty(L, max_bsz, H, N, N, dtype=torch.float16, device="cuda")
    decode_elapsed = torch.empty(max_bsz, dtype=torch.int32, device="cuda")

    return L, C, H, N, max_bsz, global_shift, global_wkv, global_elapsed, decode_shift, decode_wkv, decode_elapsed


def test_gather_shift_correctness(state_buffers):
    L, C, H, N, max_bsz = state_buffers[:5]
    global_shift, global_wkv, global_elapsed = state_buffers[5:8]
    decode_shift, decode_wkv, decode_elapsed = state_buffers[8:]

    B = 4
    slot_indices = torch.tensor([3, 0, 5, 1], dtype=torch.long, device="cuda")

    gather_decode_state(
        L,
        C,
        H,
        N,
        B,
        global_shift,
        global_wkv,
        global_elapsed,
        decode_shift,
        decode_wkv,
        decode_elapsed,
        slot_indices,
    )

    for i in range(B):
        s = slot_indices[i].item()
        torch.testing.assert_close(decode_shift[:, :, i], global_shift[:, :, s])
        torch.testing.assert_close(decode_wkv[:, i], global_wkv[:, s])
        assert decode_elapsed[i].item() == global_elapsed[s].item()


def test_scatter_shift_correctness(state_buffers):
    L, C, H, N, max_bsz = state_buffers[:5]
    global_shift, global_wkv, global_elapsed = state_buffers[5:8]
    decode_shift, decode_wkv, decode_elapsed = state_buffers[8:]

    B = 5
    slot_indices = torch.tensor([7, 2, 4, 1, 6], dtype=torch.long, device="cuda")

    decode_shift[:, :, :B].copy_(torch.arange(L * 2 * B * C, dtype=torch.float16, device="cuda").view(L, 2, B, C))
    decode_wkv[:, :B].copy_(torch.arange(L * B * H * N * N, dtype=torch.float16, device="cuda").view(L, B, H, N, N))
    decode_elapsed[:B].copy_(torch.arange(B, dtype=torch.int32, device="cuda") * 100)

    global_shift_copy = global_shift.clone()
    global_wkv_copy = global_wkv.clone()
    global_elapsed_copy = global_elapsed.clone()

    scatter_decode_state(
        L,
        C,
        H,
        N,
        B,
        global_shift,
        global_wkv,
        global_elapsed,
        decode_shift,
        decode_wkv,
        decode_elapsed,
        slot_indices,
    )

    for i in range(B):
        s = slot_indices[i].item()
        torch.testing.assert_close(global_shift[:, :, s], decode_shift[:, :, i])
        torch.testing.assert_close(global_wkv[:, s], decode_wkv[:, i])
        assert global_elapsed[s].item() == decode_elapsed[i].item()

    for i in range(max_bsz):
        if i not in slot_indices.tolist():
            torch.testing.assert_close(global_shift[:, :, i], global_shift_copy[:, :, i])
            torch.testing.assert_close(global_wkv[:, i], global_wkv_copy[:, i])
            assert global_elapsed[i].item() == global_elapsed_copy[i].item()


def test_gather_scatter_roundtrip(state_buffers):
    L, C, H, N, max_bsz = state_buffers[:5]
    global_shift, global_wkv, global_elapsed = state_buffers[5:8]
    decode_shift, decode_wkv, decode_elapsed = state_buffers[8:]

    B = 6
    slot_indices = torch.tensor([2, 5, 0, 7, 3, 1], dtype=torch.long, device="cuda")

    global_shift_copy = global_shift.clone()
    global_wkv_copy = global_wkv.clone()
    global_elapsed_copy = global_elapsed.clone()

    gather_decode_state(
        L,
        C,
        H,
        N,
        B,
        global_shift,
        global_wkv,
        global_elapsed,
        decode_shift,
        decode_wkv,
        decode_elapsed,
        slot_indices,
    )

    decode_shift[:, :, :B] *= 2.0
    decode_wkv[:, :B] += 1.0
    decode_elapsed[:B] += 999

    scatter_decode_state(
        L,
        C,
        H,
        N,
        B,
        global_shift,
        global_wkv,
        global_elapsed,
        decode_shift,
        decode_wkv,
        decode_elapsed,
        slot_indices,
    )

    for i in range(B):
        s = slot_indices[i].item()
        expected_shift = global_shift_copy[:, :, s] * 2.0
        expected_wkv = global_wkv_copy[:, s] + 1.0
        expected_elapsed = global_elapsed_copy[s] + 999
        torch.testing.assert_close(global_shift[:, :, s], expected_shift)
        torch.testing.assert_close(global_wkv[:, s], expected_wkv)
        assert global_elapsed[s].item() == expected_elapsed.item()


def test_gather_single_slot(state_buffers):
    L, C, H, N, max_bsz = state_buffers[:5]
    global_shift, global_wkv, global_elapsed = state_buffers[5:8]
    decode_shift, decode_wkv, decode_elapsed = state_buffers[8:]

    B = 1
    slot_indices = torch.tensor([4], dtype=torch.long, device="cuda")

    gather_decode_state(
        L,
        C,
        H,
        N,
        B,
        global_shift,
        global_wkv,
        global_elapsed,
        decode_shift,
        decode_wkv,
        decode_elapsed,
        slot_indices,
    )

    torch.testing.assert_close(decode_shift[:, :, 0], global_shift[:, :, 4])
    torch.testing.assert_close(decode_wkv[:, 0], global_wkv[:, 4])
    assert decode_elapsed[0].item() == global_elapsed[4].item()


def test_gather_full_batch(state_buffers):
    L, C, H, N, max_bsz = state_buffers[:5]
    global_shift, global_wkv, global_elapsed = state_buffers[5:8]
    decode_shift, decode_wkv, decode_elapsed = state_buffers[8:]

    B = max_bsz
    slot_indices = torch.arange(max_bsz, dtype=torch.long, device="cuda")

    gather_decode_state(
        L,
        C,
        H,
        N,
        B,
        global_shift,
        global_wkv,
        global_elapsed,
        decode_shift,
        decode_wkv,
        decode_elapsed,
        slot_indices,
    )

    torch.testing.assert_close(decode_shift[:, :, :B], global_shift[:, :, :B])
    torch.testing.assert_close(decode_wkv[:, :B], global_wkv[:, :B])
    torch.testing.assert_close(decode_elapsed[:B], global_elapsed[:B])


def test_gather_duplicate_slots(state_buffers):
    L, C, H, N, max_bsz = state_buffers[:5]
    global_shift, global_wkv, global_elapsed = state_buffers[5:8]
    decode_shift, decode_wkv, decode_elapsed = state_buffers[8:]

    B = 3
    slot_indices = torch.tensor([2, 2, 2], dtype=torch.long, device="cuda")

    gather_decode_state(
        L,
        C,
        H,
        N,
        B,
        global_shift,
        global_wkv,
        global_elapsed,
        decode_shift,
        decode_wkv,
        decode_elapsed,
        slot_indices,
    )

    for i in range(B):
        torch.testing.assert_close(decode_shift[:, :, i], global_shift[:, :, 2])
        torch.testing.assert_close(decode_wkv[:, i], global_wkv[:, 2])
        assert decode_elapsed[i].item() == global_elapsed[2].item()


@pytest.mark.parametrize(
    "L,C,H,N",
    [
        (1, 128, 2, 32),
        (6, 1024, 8, 128),
    ],
)
def test_gather_various_shapes(L, C, H, N):
    max_bsz = 4
    B = 3
    global_shift = torch.randn(L, 2, max_bsz, C, dtype=torch.float16, device="cuda")
    global_wkv = torch.randn(L, max_bsz, H, N, N, dtype=torch.float16, device="cuda")
    global_elapsed = torch.randint(0, 500, (max_bsz,), dtype=torch.int32, device="cuda")
    decode_shift = torch.empty(L, 2, B, C, dtype=torch.float16, device="cuda")
    decode_wkv = torch.empty(L, B, H, N, N, dtype=torch.float16, device="cuda")
    decode_elapsed = torch.empty(B, dtype=torch.int32, device="cuda")
    slot_indices = torch.tensor([1, 3, 0], dtype=torch.long, device="cuda")

    gather_decode_state(
        L,
        C,
        H,
        N,
        B,
        global_shift,
        global_wkv,
        global_elapsed,
        decode_shift,
        decode_wkv,
        decode_elapsed,
        slot_indices,
    )

    for i in range(B):
        s = slot_indices[i].item()
        torch.testing.assert_close(decode_shift[:, :, i], global_shift[:, :, s])
        torch.testing.assert_close(decode_wkv[:, i], global_wkv[:, s])
        assert decode_elapsed[i].item() == global_elapsed[s].item()


def test_gather_invalid_dtype_shift():
    L, C, H, N, B = 2, 256, 4, 64, 2
    max_bsz = 4
    global_shift = torch.randn(L, 2, max_bsz, C, dtype=torch.float32, device="cuda")
    global_wkv = torch.randn(L, max_bsz, H, N, N, dtype=torch.float16, device="cuda")
    global_elapsed = torch.zeros(max_bsz, dtype=torch.int32, device="cuda")
    decode_shift = torch.empty(L, 2, B, C, dtype=torch.float16, device="cuda")
    decode_wkv = torch.empty(L, B, H, N, N, dtype=torch.float16, device="cuda")
    decode_elapsed = torch.empty(B, dtype=torch.int32, device="cuda")
    slot_indices = torch.zeros(B, dtype=torch.long, device="cuda")

    with pytest.raises(RuntimeError):
        gather_decode_state(
            L,
            C,
            H,
            N,
            B,
            global_shift,
            global_wkv,
            global_elapsed,
            decode_shift,
            decode_wkv,
            decode_elapsed,
            slot_indices,
        )


def test_gather_invalid_dtype_elapsed():
    L, C, H, N, B = 2, 256, 4, 64, 2
    max_bsz = 4
    global_shift = torch.randn(L, 2, max_bsz, C, dtype=torch.float16, device="cuda")
    global_wkv = torch.randn(L, max_bsz, H, N, N, dtype=torch.float16, device="cuda")
    global_elapsed = torch.zeros(max_bsz, dtype=torch.int64, device="cuda")
    decode_shift = torch.empty(L, 2, B, C, dtype=torch.float16, device="cuda")
    decode_wkv = torch.empty(L, B, H, N, N, dtype=torch.float16, device="cuda")
    decode_elapsed = torch.empty(B, dtype=torch.int32, device="cuda")
    slot_indices = torch.zeros(B, dtype=torch.long, device="cuda")

    with pytest.raises(RuntimeError):
        gather_decode_state(
            L,
            C,
            H,
            N,
            B,
            global_shift,
            global_wkv,
            global_elapsed,
            decode_shift,
            decode_wkv,
            decode_elapsed,
            slot_indices,
        )


def test_gather_invalid_slot_indices_dtype():
    L, C, H, N, B = 2, 256, 4, 64, 2
    max_bsz = 4
    global_shift = torch.randn(L, 2, max_bsz, C, dtype=torch.float16, device="cuda")
    global_wkv = torch.randn(L, max_bsz, H, N, N, dtype=torch.float16, device="cuda")
    global_elapsed = torch.zeros(max_bsz, dtype=torch.int32, device="cuda")
    decode_shift = torch.empty(L, 2, B, C, dtype=torch.float16, device="cuda")
    decode_wkv = torch.empty(L, B, H, N, N, dtype=torch.float16, device="cuda")
    decode_elapsed = torch.empty(B, dtype=torch.int32, device="cuda")
    slot_indices = torch.zeros(B, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError):
        gather_decode_state(
            L,
            C,
            H,
            N,
            B,
            global_shift,
            global_wkv,
            global_elapsed,
            decode_shift,
            decode_wkv,
            decode_elapsed,
            slot_indices,
        )

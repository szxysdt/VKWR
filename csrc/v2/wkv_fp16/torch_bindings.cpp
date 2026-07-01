#include <torch/extension.h>
#include <torch/library.h>
#include <limits.h>

#include "core/register.h"
#include "v1_check.h"
#include "wkv_fp16_v2.h"

void advance_i32_cuda(torch::Tensor x, int64_t amount, torch::Tensor slot_indices);

namespace {

static bool has_slot_indices(torch::Tensor slot_indices) {
  return slot_indices.defined() && slot_indices.numel() > 0;
}

static void check_elapsed_t(int64_t B, torch::Tensor elapsed_t, torch::Tensor slot_indices) {
  TORCH_CHECK(elapsed_t.dim() == 1, "elapsed_t must be 1D");
  if (has_slot_indices(slot_indices)) {
    TORCH_CHECK(elapsed_t.size(0) > 0, "elapsed_t must be non-empty when slot_indices provided");
  } else {
    TORCH_CHECK(elapsed_t.size(0) == B, "elapsed_t.size(0) must equal B: %lld == %lld", elapsed_t.size(0), B);
  }
  TORCH_CHECK(elapsed_t.scalar_type() == torch::kInt32, "elapsed_t must be int32");
  TORCH_CHECK(elapsed_t.is_cuda() && elapsed_t.is_contiguous(), "elapsed_t must be CUDA contiguous");
}

void check_wkv_fp16_inputs(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor slot_indices) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_half_cuda_contig(r, "r");
  check_half_cuda_contig(w, "w");
  check_half_cuda_contig(k, "k");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(a, "a");
  check_half_cuda_contig(b, "b");
  TORCH_CHECK(r.size(0) == B && r.size(1) == T && r.size(2) == C,
              "r must have shape [B,T,C]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
              r.sizes() == v.sizes() && r.sizes() == a.sizes() && r.sizes() == b.sizes(),
              "r,w,k,v,a,b shape mismatch");
  TORCH_CHECK(state.dim() == 4 && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
               "state must have shape [S,H,64,64]");
  if (has_slot_indices(slot_indices)) {
    TORCH_CHECK(state.size(0) > 0, "state must be non-empty when slot_indices provided");
  } else {
    TORCH_CHECK(state.size(0) == B, "state.size(0) must equal B: %lld == %lld", state.size(0), B);
  }
  TORCH_CHECK(state.scalar_type() == torch::kFloat16, "state must be fp16");
  TORCH_CHECK(state.is_cuda() && state.is_contiguous(), "state must be CUDA contiguous");
}

void check_wkv_one_fp16_inputs(
    int64_t B, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor slot_indices) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_half_cuda_contig(r, "r");
  check_half_cuda_contig(w, "w");
  check_half_cuda_contig(k, "k");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(a, "a");
  check_half_cuda_contig(b, "b");
  TORCH_CHECK(r.size(0) == B && r.size(1) == C,
              "r must have shape [B,C]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
              r.sizes() == v.sizes() && r.sizes() == a.sizes() && r.sizes() == b.sizes(),
              "r,w,k,v,a,b shape mismatch");
  TORCH_CHECK(state.dim() == 4 && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
               "state must have shape [S,H,64,64]");
  if (has_slot_indices(slot_indices)) {
    TORCH_CHECK(state.size(0) > 0, "state must be non-empty when slot_indices provided");
  } else {
    TORCH_CHECK(state.size(0) == B, "state.size(0) must equal B: %lld == %lld", state.size(0), B);
  }
  TORCH_CHECK(state.scalar_type() == torch::kFloat16, "state must be fp16");
  TORCH_CHECK(state.is_cuda() && state.is_contiguous(), "state must be CUDA contiguous");
}

}  // namespace

void wkv_seq_fp16(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor slot_indices, torch::Tensor elapsed_t) {
  check_wkv_fp16_inputs(B, T, C, H, state, r, w, k, v, a, b, slot_indices);
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(y.size(0) == B && y.size(1) == T && y.size(2) == C,
              "y must have shape [B,T,C]");
  check_elapsed_t(B, elapsed_t, slot_indices);
  wkv_seq_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      state, r, w, k, v, a, b, y, slot_indices, elapsed_t);
}

void wkv_seq_w0_fp16(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w, torch::Tensor w0,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor slot_indices, torch::Tensor elapsed_t) {
  check_wkv_fp16_inputs(B, T, C, H, state, r, w, k, v, a, b, slot_indices);
  check_half_cuda_contig(w0, "w0");
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(y.size(0) == B && y.size(1) == T && y.size(2) == C,
              "y must have shape [B,T,C]");
  check_elapsed_t(B, elapsed_t, slot_indices);
  wkv_seq_w0_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      state, r, w, w0, k, v, a, b, y, slot_indices, elapsed_t);
}

void wkv_one_fp16(
    int64_t B, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor slot_indices, torch::Tensor elapsed_t) {
  check_wkv_one_fp16_inputs(B, C, H, state, r, w, k, v, a, b, slot_indices);
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(y.size(0) == B && y.size(1) == C,
              "y must have shape [B,C]");
  check_elapsed_t(B, elapsed_t, slot_indices);
  wkv_one_v2_cuda(
      static_cast<int>(B), static_cast<int>(C), static_cast<int>(H),
      state, r, w, k, v, a, b, y, slot_indices, elapsed_t);
}

void wkv_one_w0_fp16(
    int64_t B, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w, torch::Tensor w0,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor slot_indices, torch::Tensor elapsed_t) {
  check_wkv_one_fp16_inputs(B, C, H, state, r, w, k, v, a, b, slot_indices);
  check_half_cuda_contig(w0, "w0");
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(y.size(0) == B && y.size(1) == C,
              "y must have shape [B,C]");
  check_elapsed_t(B, elapsed_t, slot_indices);
  wkv_one_w0_v2_cuda(
      static_cast<int>(B), static_cast<int>(C), static_cast<int>(H),
      state, r, w, w0, k, v, a, b, y, slot_indices, elapsed_t);
}

// ===== advance_i32 binding =====
void advance_i32(torch::Tensor x, int64_t amount, torch::Tensor slot_indices) {
  check_i32_cuda_contig(x, "x");
  TORCH_CHECK(x.dim() == 1, "x must have shape [B]");
  advance_i32_cuda(x, amount, slot_indices);
}

// ===== TORCH_LIBRARY registration =====
TORCH_LIBRARY_FRAGMENT(vkwr_v2_wkv, m) {
  m.def("wkv_seq_fp16(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor slot_indices, Tensor elapsed_t) -> ()");
  m.impl("wkv_seq_fp16", c10::kCUDA, &wkv_seq_fp16);

  m.def("wkv_seq_w0_fp16(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor slot_indices, Tensor elapsed_t) -> ()");
  m.impl("wkv_seq_w0_fp16", c10::kCUDA, &wkv_seq_w0_fp16);

  m.def("wkv_one_fp16(int B, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor slot_indices, Tensor elapsed_t) -> ()");
  m.impl("wkv_one_fp16", c10::kCUDA, &wkv_one_fp16);

  m.def("wkv_one_w0_fp16(int B, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor slot_indices, Tensor elapsed_t) -> ()");
  m.impl("wkv_one_w0_fp16", c10::kCUDA, &wkv_one_w0_fp16);

  m.def("advance_i32(Tensor(a!) x, int amount, Tensor slot_indices) -> ()");
  m.impl("advance_i32", c10::kCUDA, &advance_i32);
}

REGISTER_EXTENSION(_v2_wkv_fp16_C)

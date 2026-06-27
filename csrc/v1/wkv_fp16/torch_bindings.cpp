#include <torch/extension.h>
#include <torch/library.h>
#include <limits.h>

#include "core/register.h"
#include "v1_check.h"
#include "wkv_fp16_v2.h"

void advance_i32_cuda(torch::Tensor x, int64_t amount);

namespace {

void check_wkv_fp16_inputs(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b) {
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
  TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
              "state must have shape [B,H,64,64]");
  TORCH_CHECK(state.scalar_type() == torch::kFloat16, "state must be fp16");
  TORCH_CHECK(state.is_cuda() && state.is_contiguous(), "state must be CUDA contiguous");
}

void check_wkv_one_fp16_inputs(
    int64_t B, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b) {
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
  TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
              "state must have shape [B,H,64,64]");
  TORCH_CHECK(state.scalar_type() == torch::kFloat16, "state must be fp16");
  TORCH_CHECK(state.is_cuda() && state.is_contiguous(), "state must be CUDA contiguous");
}

}  // namespace

// ===== fp16 WKV bindings =====
void wkv_seq_fp16(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor elapsed_t) {
  check_wkv_fp16_inputs(B, T, C, H, state, r, w, k, v, a, b);
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(y.size(0) == B && y.size(1) == T && y.size(2) == C,
              "y must have shape [B,T,C]");
  TORCH_CHECK(elapsed_t.dim() == 1 && elapsed_t.size(0) == B,
              "elapsed_t must have shape [B]");
  TORCH_CHECK(elapsed_t.scalar_type() == torch::kInt32, "elapsed_t must be int32");
  TORCH_CHECK(elapsed_t.is_cuda() && elapsed_t.is_contiguous(), "elapsed_t must be CUDA contiguous");
  wkv_seq_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      state, r, w, k, v, a, b, y, elapsed_t);
}

void wkv_seq_w0_fp16(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w, torch::Tensor w0,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor elapsed_t) {
  check_wkv_fp16_inputs(B, T, C, H, state, r, w, k, v, a, b);
  check_half_cuda_contig(w0, "w0");
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(y.size(0) == B && y.size(1) == T && y.size(2) == C,
              "y must have shape [B,T,C]");
  TORCH_CHECK(elapsed_t.dim() == 1 && elapsed_t.size(0) == B,
              "elapsed_t must have shape [B]");
  TORCH_CHECK(elapsed_t.scalar_type() == torch::kInt32, "elapsed_t must be int32");
  TORCH_CHECK(elapsed_t.is_cuda() && elapsed_t.is_contiguous(), "elapsed_t must be CUDA contiguous");
  wkv_seq_w0_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      state, r, w, w0, k, v, a, b, y, elapsed_t);
}

void wkv_one_fp16(
    int64_t B, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor elapsed_t) {
  check_wkv_one_fp16_inputs(B, C, H, state, r, w, k, v, a, b);
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(y.size(0) == B && y.size(1) == C,
              "y must have shape [B,C]");
  TORCH_CHECK(elapsed_t.dim() == 1 && elapsed_t.size(0) == B,
              "elapsed_t must have shape [B]");
  TORCH_CHECK(elapsed_t.scalar_type() == torch::kInt32, "elapsed_t must be int32");
  TORCH_CHECK(elapsed_t.is_cuda() && elapsed_t.is_contiguous(), "elapsed_t must be CUDA contiguous");
  wkv_one_v2_cuda(
      static_cast<int>(B), static_cast<int>(C), static_cast<int>(H),
      state, r, w, k, v, a, b, y, elapsed_t);
}

void wkv_one_w0_fp16(
    int64_t B, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w, torch::Tensor w0,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor elapsed_t) {
  check_wkv_one_fp16_inputs(B, C, H, state, r, w, k, v, a, b);
  check_half_cuda_contig(w0, "w0");
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(y.size(0) == B && y.size(1) == C,
              "y must have shape [B,C]");
  TORCH_CHECK(elapsed_t.dim() == 1 && elapsed_t.size(0) == B,
              "elapsed_t must have shape [B]");
  TORCH_CHECK(elapsed_t.scalar_type() == torch::kInt32, "elapsed_t must be int32");
  TORCH_CHECK(elapsed_t.is_cuda() && elapsed_t.is_contiguous(), "elapsed_t must be CUDA contiguous");
  wkv_one_w0_v2_cuda(
      static_cast<int>(B), static_cast<int>(C), static_cast<int>(H),
      state, r, w, w0, k, v, a, b, y, elapsed_t);
}

// ===== advance_i32 binding =====
void advance_i32(torch::Tensor x, int64_t amount) {
  check_i32_cuda_contig(x, "x");
  TORCH_CHECK(x.dim() == 1, "x must have shape [B]");
  advance_i32_cuda(x, amount);
}

// ===== TORCH_LIBRARY registration =====
TORCH_LIBRARY_FRAGMENT(vkwr_v1_wkv, m) {
  m.def("wkv_seq_fp16(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()");
  m.impl("wkv_seq_fp16", c10::kCUDA, &wkv_seq_fp16);

  m.def("wkv_seq_w0_fp16(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()");
  m.impl("wkv_seq_w0_fp16", c10::kCUDA, &wkv_seq_w0_fp16);

  m.def("wkv_one_fp16(int B, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()");
  m.impl("wkv_one_fp16", c10::kCUDA, &wkv_one_fp16);

  m.def("wkv_one_w0_fp16(int B, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()");
  m.impl("wkv_one_w0_fp16", c10::kCUDA, &wkv_one_w0_fp16);

  m.def("advance_i32(Tensor(a!) x, int amount) -> ()");
  m.impl("advance_i32", c10::kCUDA, &advance_i32);
}

REGISTER_EXTENSION(_v1_wkv_fp16_C)

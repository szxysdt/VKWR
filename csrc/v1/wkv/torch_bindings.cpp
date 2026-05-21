#include <torch/extension.h>
#include <torch/library.h>
#include <limits.h>

#include "core/register.h"
#include "v1/common/v1_check.h"
#include "wkv_fp16_v2.h"
#include "wkv_fp32_v2.h"

void advance_i32_cuda(torch::Tensor x, int64_t amount);

// ===== fp16 WKV: input validation =====
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
              r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
              r.sizes() == b.sizes(),
              "r,w,k,v,a,b shape mismatch");
  TORCH_CHECK(state.dim() == 3 && state.size(0) == B && state.size(1) == C && state.size(2) == 64,
              "state must have shape [B,C,64]");
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
              r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
              r.sizes() == b.sizes(),
              "r,w,k,v,a,b shape mismatch");
  TORCH_CHECK(state.dim() == 3 && state.size(0) == B && state.size(1) == C && state.size(2) == 64,
              "state must have shape [B,C,64]");
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

// ===== fp32 WKV: input validation =====
namespace {

void check_fp32_io(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, name, " must be fp16");
}

void check_fp32_state(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, name, " must be fp32");
}

void check_wkv_fp32_inputs(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_fp32_state(state, "state");
  check_fp32_io(r, "r");
  check_fp32_io(w, "w");
  check_fp32_io(k, "k");
  check_fp32_io(v, "v");
  check_fp32_io(a, "a");
  check_fp32_io(b, "b");
  check_fp32_io(y, "y");
  TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
              "state must have shape [B,H,64,64]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() && r.sizes() == v.sizes() &&
              r.sizes() == a.sizes() && r.sizes() == b.sizes() && r.sizes() == y.sizes(),
              "r,w,k,v,a,b,y shape mismatch");
  TORCH_CHECK(r.dim() == 3 && r.size(0) == B && r.size(1) == T && r.size(2) == C,
              "r must have shape [B,T,C]");
}

}  // namespace

// ===== fp32 WKV bindings =====
void wkv_forward_fp32(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y) {
  check_wkv_fp32_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      0, state, r, w, k, v, a, b, y);
}

void wkv_forward_seq_fp32(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y) {
  check_wkv_fp32_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      1, state, r, w, k, v, a, b, y);
}

void wkv_forward_small_fp32(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y) {
  check_wkv_fp32_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      2, state, r, w, k, v, a, b, y);
}

void wkv_forward_block_fp32(
    int64_t B, int64_t T, int64_t C, int64_t H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y) {
  check_wkv_fp32_inputs(B, T, C, H, state, r, w, k, v, a, b, y);
  wkv_fp32_v2_cuda(
      static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H),
      3, state, r, w, k, v, a, b, y);
}

// ===== advance_i32 binding =====
void advance_i32(torch::Tensor x, int64_t amount) {
  check_i32_cuda_contig(x, "x");
  TORCH_CHECK(x.dim() == 1, "x must have shape [B]");
  advance_i32_cuda(x, amount);
}

// ===== TORCH_LIBRARY registration (VKWR Pattern C) =====
TORCH_LIBRARY(vkwr_v1_wkv, m) {
  m.def("wkv_seq_fp16(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(a!) elapsed_t) -> ()");
  m.impl("wkv_seq_fp16", c10::kCUDA, &wkv_seq_fp16);

  m.def("wkv_seq_w0_fp16(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(a!) elapsed_t) -> ()");
  m.impl("wkv_seq_w0_fp16", c10::kCUDA, &wkv_seq_w0_fp16);

  m.def("wkv_one_fp16(int B, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(a!) elapsed_t) -> ()");
  m.impl("wkv_one_fp16", c10::kCUDA, &wkv_one_fp16);

  m.def("wkv_one_w0_fp16(int B, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor(a!) elapsed_t) -> ()");
  m.impl("wkv_one_w0_fp16", c10::kCUDA, &wkv_one_w0_fp16);

  m.def("wkv_forward_fp32(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.impl("wkv_forward_fp32", c10::kCUDA, &wkv_forward_fp32);

  m.def("wkv_forward_seq_fp32(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.impl("wkv_forward_seq_fp32", c10::kCUDA, &wkv_forward_seq_fp32);

  m.def("wkv_forward_small_fp32(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.impl("wkv_forward_small_fp32", c10::kCUDA, &wkv_forward_small_fp32);

  m.def("wkv_forward_block_fp32(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.impl("wkv_forward_block_fp32", c10::kCUDA, &wkv_forward_block_fp32);

  m.def("advance_i32(Tensor(a!) x, int amount) -> ()");
  m.impl("advance_i32", c10::kCUDA, &advance_i32);
}

REGISTER_EXTENSION(_v1_wkv_C)

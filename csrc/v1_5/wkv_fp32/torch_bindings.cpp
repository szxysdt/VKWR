#include <torch/extension.h>
#include <torch/library.h>

#include "core/register.h"
#include "v1_check.h"
#include "wkv_fp32_v2.h"

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

void check_wkv_fp32_varlen_inputs(int64_t B, int64_t C, int64_t H,
                                  int64_t total_tokens,
                                  torch::Tensor query_start_loc,
                                  torch::Tensor state, torch::Tensor r,
                                  torch::Tensor w, torch::Tensor k,
                                  torch::Tensor v, torch::Tensor a,
                                  torch::Tensor b, torch::Tensor y) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_fp32_state(state, "state");
  check_fp32_io(r, "r");
  check_fp32_io(w, "w");
  check_fp32_io(k, "k");
  check_fp32_io(v, "v");
  check_fp32_io(a, "a");
  check_fp32_io(b, "b");
  check_fp32_io(y, "y");
  TORCH_CHECK(r.size(0) == total_tokens && r.size(1) == C,
              "r must have shape [total_tokens, C]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
                  r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
                  r.sizes() == b.sizes() && r.sizes() == y.sizes(),
              "r,w,k,v,a,b,y shape mismatch");
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be CUDA tensor (device-side read)");
  TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(1) == H &&
                  state.size(2) == 64 && state.size(3) == 64,
              "state must have shape [B,H,64,64]");
}

}  // namespace

// ===== fp32 WKV varlen bindings =====
void wkv_forward_fp32_varlen(int64_t B, int64_t total_tokens, int64_t max_t,
                             int64_t C, int64_t H,
                             torch::Tensor query_start_loc, torch::Tensor state,
                             torch::Tensor r, torch::Tensor w, torch::Tensor k,
                             torch::Tensor v, torch::Tensor a, torch::Tensor b,
                             torch::Tensor y) {
  check_wkv_fp32_varlen_inputs(B, C, H, total_tokens, query_start_loc, state, r,
                               w, k, v, a, b, y);
  const int* qsl_ptr = query_start_loc.data_ptr<int>();
  wkv_fp32_v2_cuda_varlen(static_cast<int>(B), static_cast<int>(max_t), qsl_ptr,
                          static_cast<int>(C), static_cast<int>(H), 0, state, r,
                          w, k, v, a, b, y);
}

void wkv_forward_seq_fp32_varlen(int64_t B, int64_t total_tokens, int64_t max_t,
                                 int64_t C, int64_t H,
                                 torch::Tensor query_start_loc,
                                 torch::Tensor state, torch::Tensor r,
                                 torch::Tensor w, torch::Tensor k,
                                 torch::Tensor v, torch::Tensor a,
                                 torch::Tensor b, torch::Tensor y) {
  check_wkv_fp32_varlen_inputs(B, C, H, total_tokens, query_start_loc, state, r,
                               w, k, v, a, b, y);
  const int* qsl_ptr = query_start_loc.data_ptr<int>();
  wkv_fp32_v2_cuda_varlen(static_cast<int>(B), static_cast<int>(max_t), qsl_ptr,
                          static_cast<int>(C), static_cast<int>(H), 1, state, r,
                          w, k, v, a, b, y);
}

void wkv_forward_small_fp32_varlen(int64_t B, int64_t total_tokens,
                                   int64_t max_t, int64_t C, int64_t H,
                                   torch::Tensor query_start_loc,
                                   torch::Tensor state, torch::Tensor r,
                                   torch::Tensor w, torch::Tensor k,
                                   torch::Tensor v, torch::Tensor a,
                                   torch::Tensor b, torch::Tensor y) {
  check_wkv_fp32_varlen_inputs(B, C, H, total_tokens, query_start_loc, state, r,
                               w, k, v, a, b, y);
  const int* qsl_ptr = query_start_loc.data_ptr<int>();
  wkv_fp32_v2_cuda_varlen(static_cast<int>(B), static_cast<int>(max_t), qsl_ptr,
                          static_cast<int>(C), static_cast<int>(H), 2, state, r,
                          w, k, v, a, b, y);
}

void wkv_forward_block_fp32_varlen(int64_t B, int64_t total_tokens,
                                   int64_t max_t, int64_t C, int64_t H,
                                   torch::Tensor query_start_loc,
                                   torch::Tensor state, torch::Tensor r,
                                   torch::Tensor w, torch::Tensor k,
                                   torch::Tensor v, torch::Tensor a,
                                   torch::Tensor b, torch::Tensor y) {
  check_wkv_fp32_varlen_inputs(B, C, H, total_tokens, query_start_loc, state, r,
                               w, k, v, a, b, y);
  const int* qsl_ptr = query_start_loc.data_ptr<int>();
  wkv_fp32_v2_cuda_varlen(static_cast<int>(B), static_cast<int>(max_t), qsl_ptr,
                          static_cast<int>(C), static_cast<int>(H), 3, state, r,
                          w, k, v, a, b, y);
}

// ===== TORCH_LIBRARY registration (new namespace) =====
TORCH_LIBRARY_FRAGMENT(vkwr_v1_5_wkv, m) {
  m.def(
      "wkv_forward_fp32_varlen(int B, int total_tokens, int max_t, int C, int "
      "H, Tensor "
      "query_start_loc, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor "
      "v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.impl("wkv_forward_fp32_varlen", c10::kCUDA, &wkv_forward_fp32_varlen);

  m.def(
      "wkv_forward_seq_fp32_varlen(int B, int total_tokens, int max_t, int C, "
      "int H, Tensor "
      "query_start_loc, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor "
      "v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.impl("wkv_forward_seq_fp32_varlen", c10::kCUDA,
         &wkv_forward_seq_fp32_varlen);

  m.def(
      "wkv_forward_small_fp32_varlen(int B, int total_tokens, int max_t, int "
      "C, int H, Tensor "
      "query_start_loc, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor "
      "v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.impl("wkv_forward_small_fp32_varlen", c10::kCUDA,
         &wkv_forward_small_fp32_varlen);

  m.def(
      "wkv_forward_block_fp32_varlen(int B, int total_tokens, int max_t, int "
      "C, int H, Tensor "
      "query_start_loc, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor "
      "v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.impl("wkv_forward_block_fp32_varlen", c10::kCUDA,
         &wkv_forward_block_fp32_varlen);
}

REGISTER_EXTENSION(_v1_5_wkv_fp32_C)

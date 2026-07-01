#include <torch/extension.h>
#include <torch/library.h>

#include "core/register.h"
#include "v1_check.h"
#include "wkv_fp16_v2.h"

void advance_i32_varlen_cuda(at::Tensor elapsed, at::Tensor query_start_loc);

// Note: wkv_one series are not called across .so at C++ layer
// (MODULE .so do not link each other, extern produces undefined reference).
// All-one scenario is dispatched from Python wrapper layer via
// torch.ops.vkwr_v1_wkv.wkv_one_fp16().

namespace {

void check_wkv_varlen_inputs(int64_t B, int64_t C, int64_t H,
                             int64_t total_tokens,
                             torch::Tensor query_start_loc, torch::Tensor state,
                             torch::Tensor r, torch::Tensor w, torch::Tensor k,
                             torch::Tensor v, torch::Tensor a,
                             torch::Tensor b) {
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be CUDA tensor (device-side read)");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_half_cuda_contig(r, "r");
  check_half_cuda_contig(w, "w");
  check_half_cuda_contig(k, "k");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(a, "a");
  check_half_cuda_contig(b, "b");
  TORCH_CHECK(r.size(0) == total_tokens && r.size(1) == C,
              "r must have shape [total_tokens, C]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() &&
                  r.sizes() == v.sizes() && r.sizes() == a.sizes() &&
                  r.sizes() == b.sizes(),
              "r,w,k,v,a,b shape mismatch");
  TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(1) == H,
              "state must have shape [B,H,64,64]");
}

}  // namespace

void wkv_seq_fp16_varlen(int64_t B, int64_t total_tokens, int64_t max_t,
                         int64_t C, int64_t H, torch::Tensor query_start_loc,
                         torch::Tensor state, torch::Tensor r, torch::Tensor w,
                         torch::Tensor k, torch::Tensor v, torch::Tensor a,
                         torch::Tensor b, torch::Tensor y,
                         torch::Tensor elapsed_t) {
  check_wkv_varlen_inputs(B, C, H, total_tokens, query_start_loc, state, r, w,
                          k, v, a, b);
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(y.size(0) == total_tokens && y.size(1) == C,
              "y must have shape [total_tokens, C]");
  TORCH_CHECK(elapsed_t.dim() == 1 && elapsed_t.size(0) == B,
              "elapsed_t must have shape [B]");
  TORCH_CHECK(elapsed_t.scalar_type() == torch::kInt32,
              "elapsed_t must be int32");
  TORCH_CHECK(elapsed_t.is_cuda() && elapsed_t.is_contiguous(),
              "elapsed_t must be CUDA contiguous");

  const int* qsl_ptr = query_start_loc.data_ptr<int>();
  wkv_seq_v2_cuda_varlen(static_cast<int>(B), static_cast<int>(max_t), qsl_ptr,
                         static_cast<int>(C), static_cast<int>(H), state, r, w,
                         k, v, a, b, y, elapsed_t);
}

void wkv_seq_w0_fp16_varlen(int64_t B, int64_t total_tokens, int64_t max_t,
                            int64_t C, int64_t H, torch::Tensor query_start_loc,
                            torch::Tensor state, torch::Tensor r,
                            torch::Tensor w, torch::Tensor w0, torch::Tensor k,
                            torch::Tensor v, torch::Tensor a, torch::Tensor b,
                            torch::Tensor y, torch::Tensor elapsed_t) {
  check_wkv_varlen_inputs(B, C, H, total_tokens, query_start_loc, state, r, w,
                          k, v, a, b);
  check_half_cuda_contig(w0, "w0");
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(y.size(0) == total_tokens && y.size(1) == C,
              "y must have shape [total_tokens, C]");
  TORCH_CHECK(elapsed_t.dim() == 1 && elapsed_t.size(0) == B,
              "elapsed_t must have shape [B]");
  TORCH_CHECK(elapsed_t.scalar_type() == torch::kInt32,
              "elapsed_t must be int32");
  TORCH_CHECK(elapsed_t.is_cuda() && elapsed_t.is_contiguous(),
              "elapsed_t must be CUDA contiguous");

  const int* qsl_ptr = query_start_loc.data_ptr<int>();
  wkv_seq_w0_v2_cuda_varlen(static_cast<int>(B), static_cast<int>(max_t),
                            qsl_ptr, static_cast<int>(C), static_cast<int>(H),
                            state, r, w, w0, k, v, a, b, y, elapsed_t);
}

// ===== advance_i32_varlen binding =====
void advance_i32_varlen(torch::Tensor elapsed, torch::Tensor query_start_loc) {
  check_i32_cuda_contig(elapsed, "elapsed");
  TORCH_CHECK(elapsed.dim() == 1, "elapsed must have shape [B]");
  check_i32_cuda_contig(query_start_loc, "query_start_loc");
  TORCH_CHECK(query_start_loc.dim() == 1,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(
      query_start_loc.size(0) == elapsed.size(0) + 1,
      "query_start_loc must have shape [B+1] where elapsed has shape [B]");
  advance_i32_varlen_cuda(elapsed, query_start_loc);
}

// ===== TORCH_LIBRARY registration (new namespace) =====
TORCH_LIBRARY_FRAGMENT(vkwr_v1_5_wkv, m) {
  m.def(
      "wkv_seq_fp16_varlen(int B, int total_tokens, int max_t, int C, int H, "
      "Tensor "
      "query_start_loc, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor "
      "v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()");
  m.impl("wkv_seq_fp16_varlen", c10::kCUDA, &wkv_seq_fp16_varlen);

  m.def(
      "wkv_seq_w0_fp16_varlen(int B, int total_tokens, int max_t, int C, int "
      "H, Tensor "
      "query_start_loc, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, "
      "Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) "
      "-> ()");
  m.impl("wkv_seq_w0_fp16_varlen", c10::kCUDA, &wkv_seq_w0_fp16_varlen);

  m.def("advance_i32_varlen(Tensor(a!) elapsed, Tensor query_start_loc) -> ()");
  m.impl("advance_i32_varlen", c10::kCUDA, &advance_i32_varlen);
}

REGISTER_EXTENSION(_v1_5_wkv_fp16_C)

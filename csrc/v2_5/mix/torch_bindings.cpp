#include <torch/extension.h>
#include <torch/library.h>

#include "core/register.h"
#include "mix_ops.h"
#include "v1_check.h"

std::vector<torch::Tensor> tmix_mix6_varlen_fn(
    int64_t B, int64_t total_tokens, int64_t C, torch::Tensor x,
    torch::Tensor slot_indices, torch::Tensor shift_state, torch::Tensor x_r,
    torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a,
    torch::Tensor x_g, torch::Tensor query_start_loc, torch::Tensor req_id) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  check_vec(x_r, C, "x_r");
  check_vec(x_w, C, "x_w");
  check_vec(x_k, C, "x_k");
  check_vec(x_v, C, "x_v");
  check_vec(x_a, C, "x_a");
  check_vec(x_g, C, "x_g");
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be CUDA tensor (device-side read)");
  TORCH_CHECK(req_id.is_cuda(),
              "req_id must be CUDA tensor (device-side read)");
  TORCH_CHECK(slot_indices.dim() == 1 && slot_indices.is_cuda() &&
                  slot_indices.scalar_type() == torch::kInt32,
              "slot_indices must be CUDA int32 tensor");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == total_tokens && x.size(1) == C,
              "x must have shape [total_tokens, C]");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(1) == C,
              "shift_state must have shape [S, C] (S >= 1)");
  TORCH_CHECK(req_id.dim() == 1 && req_id.size(0) == total_tokens &&
                  req_id.scalar_type() == torch::kInt32,
              "req_id must have shape [total_tokens] with int32 dtype");
  return tmix_mix6_cuda_varlen(static_cast<int>(B), static_cast<int>(C), x,
                                shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
                                query_start_loc, req_id, slot_indices);
}

std::vector<torch::Tensor> tmix_mix6_t1_c4096_varlen_fn(
    int64_t B, int64_t total_tokens, torch::Tensor x, torch::Tensor slot_indices,
    torch::Tensor shift_state, torch::Tensor x_r, torch::Tensor x_w,
    torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g,
    torch::Tensor query_start_loc, torch::Tensor req_id, int64_t threads,
    int64_t vec, bool half_math) {
  TORCH_CHECK(
      threads == 128 || threads == 256 || threads == 512 || threads == 1024,
      "unsupported threads");
  TORCH_CHECK(vec == 1 || vec == 2 || vec == 4 || vec == 8, "unsupported vec");
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  check_vec(x_r, 4096, "x_r");
  check_vec(x_w, 4096, "x_w");
  check_vec(x_k, 4096, "x_k");
  check_vec(x_v, 4096, "x_v");
  check_vec(x_a, 4096, "x_a");
  check_vec(x_g, 4096, "x_g");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be CUDA tensor (device-side read)");
  TORCH_CHECK(req_id.is_cuda(),
              "req_id must be CUDA tensor (device-side read)");
  TORCH_CHECK(slot_indices.dim() == 1 && slot_indices.is_cuda() &&
                  slot_indices.scalar_type() == torch::kInt32,
              "slot_indices must be CUDA int32 tensor");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == total_tokens && x.size(1) == 4096,
              "x must have shape [total_tokens, 4096]");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(1) == 4096,
              "shift_state must have shape [S, 4096]");
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(req_id.dim() == 1 && req_id.size(0) == total_tokens &&
                  req_id.scalar_type() == torch::kInt32,
              "req_id must have shape [total_tokens] with int32 dtype");
  return tmix_mix6_t1_c4096_cuda_varlen(
      static_cast<int>(B), x, slot_indices, shift_state, x_r, x_w, x_k, x_v, x_a, x_g,
      query_start_loc, req_id, static_cast<int>(threads), static_cast<int>(vec),
      half_math);
}

torch::Tensor cmix_mix_varlen_fn(int64_t B, int64_t total_tokens, int64_t C,
                                  torch::Tensor x, torch::Tensor slot_indices,
                                  torch::Tensor shift_state, torch::Tensor x_k,
                                  torch::Tensor query_start_loc,
                                  torch::Tensor req_id) {
  TORCH_CHECK((C % 2) == 0, "C must be even");
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  check_vec(x_k, C, "x_k");
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be CUDA tensor (device-side read)");
  TORCH_CHECK(req_id.is_cuda(),
              "req_id must be CUDA tensor (device-side read)");
  TORCH_CHECK(slot_indices.dim() == 1 && slot_indices.is_cuda() &&
                  slot_indices.scalar_type() == torch::kInt32,
              "slot_indices must be CUDA int32 tensor");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == total_tokens && x.size(1) == C,
              "x must have shape [total_tokens, C]");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(1) == C,
              "shift_state must have shape [S, C] (S >= 1)");
  TORCH_CHECK(req_id.dim() == 1 && req_id.size(0) == total_tokens &&
                  req_id.scalar_type() == torch::kInt32,
              "req_id must have shape [total_tokens] with int32 dtype");
  return cmix_mix_cuda_varlen(static_cast<int>(B), static_cast<int>(C), x,
                               shift_state, x_k, query_start_loc, req_id, slot_indices);
}

torch::Tensor cmix_sparse_rows_varlen_fn(
    int64_t B, int64_t total_tokens, int64_t C, int64_t F, torch::Tensor x,
    torch::Tensor slot_indices, torch::Tensor shift_state, torch::Tensor x_k,
    torch::Tensor key_fc, torch::Tensor value_fc, torch::Tensor query_start_loc,
    torch::Tensor req_id) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(shift_state, "shift_state");
  check_vec(x_k, C, "x_k");
  check_half_cuda_contig(key_fc, "key_fc");
  TORCH_CHECK(key_fc.dim() == 2 && key_fc.size(0) == F && key_fc.size(1) == C,
              "key_fc must have shape [F,C]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 256) == 0, "C must be divisible by 256");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) == B + 1,
              "query_start_loc must have shape [B+1]");
  TORCH_CHECK(query_start_loc.scalar_type() == torch::kInt32,
              "query_start_loc must be int32");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "query_start_loc must be CUDA tensor (device-side read)");
  TORCH_CHECK(req_id.is_cuda(),
              "req_id must be CUDA tensor (device-side read)");
  TORCH_CHECK(slot_indices.dim() == 1 && slot_indices.is_cuda() &&
                  slot_indices.scalar_type() == torch::kInt32,
              "slot_indices must be CUDA int32 tensor");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == total_tokens && x.size(1) == C,
              "x must have shape [total_tokens, C]");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(1) == C,
              "shift_state must have shape [S, C] (S >= 1)");
  TORCH_CHECK(req_id.dim() == 1 && req_id.size(0) == total_tokens &&
                  req_id.scalar_type() == torch::kInt32,
              "req_id must have shape [total_tokens] with int32 dtype");
  return cmix_sparse_rows_cuda_varlen(
      static_cast<int>(B), static_cast<int>(C), static_cast<int>(F), x,
      shift_state, x_k, key_fc, value_fc, query_start_loc, req_id, slot_indices);
}

torch::Tensor cmix_sparse_down_relu_rows_varlen_fn(int64_t rows, int64_t C,
                                                   int64_t F,
                                                   torch::Tensor preact,
                                                   torch::Tensor value_fc) {
  check_half_cuda_contig(preact, "preact");
  TORCH_CHECK(
      preact.dim() == 2 && preact.size(0) == rows && preact.size(1) == F,
      "preact must have shape [rows, F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 256) == 0, "C must be divisible by 256");
  TORCH_CHECK((F % 128) == 0, "F must be divisible by 128");
  return cmix_sparse_down_relu_rows_cuda_varlen(
      static_cast<int>(rows), static_cast<int>(C), static_cast<int>(F), preact,
      value_fc);
}

torch::Tensor cmix_sparse_down_relu_rows_t512_varlen_fn(
    int64_t rows, int64_t C, int64_t F, torch::Tensor preact,
    torch::Tensor value_fc) {
  check_half_cuda_contig(preact, "preact");
  TORCH_CHECK(
      preact.dim() == 2 && preact.size(0) == rows && preact.size(1) == F,
      "preact must have shape [rows, F]");
  check_half_cuda_contig(value_fc, "value_fc");
  TORCH_CHECK(
      value_fc.dim() == 2 && value_fc.size(0) == F && value_fc.size(1) == C,
      "value_fc must have shape [F,C]");
  TORCH_CHECK((C % 512) == 0, "C must be divisible by 512");
  TORCH_CHECK((F % 512) == 0, "F must be divisible by 512");
  return cmix_sparse_down_relu_rows_t512_cuda_varlen(
      static_cast<int>(rows), static_cast<int>(C), static_cast<int>(F), preact,
      value_fc);
}

TORCH_LIBRARY_FRAGMENT(vkwr_v2_5_mix, m) {
  m.def(
      "tmix_mix6_varlen(int B, int total_tokens, int C, Tensor x, Tensor "
      "slot_indices, Tensor(a!) shift_state, Tensor x_r, Tensor x_w, Tensor x_k, "
      "Tensor x_v, Tensor x_a, Tensor x_g, Tensor query_start_loc, Tensor "
      "req_id) -> Tensor[]");
  m.impl("tmix_mix6_varlen", c10::kCUDA, &tmix_mix6_varlen_fn);

  m.def(
      "tmix_mix6_t1_c4096_varlen(int B, int total_tokens, Tensor x, Tensor "
      "slot_indices, Tensor(a!) shift_state, Tensor x_r, Tensor x_w, Tensor x_k, "
      "Tensor x_v, Tensor x_a, Tensor x_g, Tensor query_start_loc, Tensor "
      "req_id, int threads, int vec, bool half_math=False) -> Tensor[]");
  m.impl("tmix_mix6_t1_c4096_varlen", c10::kCUDA,
         &tmix_mix6_t1_c4096_varlen_fn);

  m.def(
      "cmix_mix_varlen(int B, int total_tokens, int C, Tensor x, Tensor "
      "slot_indices, Tensor(a!) shift_state, Tensor x_k, Tensor query_start_loc, "
      "Tensor req_id) -> Tensor");
  m.impl("cmix_mix_varlen", c10::kCUDA, &cmix_mix_varlen_fn);

  m.def(
      "cmix_sparse_rows_varlen(int B, int total_tokens, int C, int F, Tensor "
      "x, Tensor slot_indices, Tensor(a!) shift_state, Tensor x_k, Tensor key_fc, "
      "Tensor value_fc, Tensor query_start_loc, Tensor req_id) -> Tensor");
  m.impl("cmix_sparse_rows_varlen", c10::kCUDA, &cmix_sparse_rows_varlen_fn);

  m.def(
      "cmix_sparse_down_relu_rows_varlen(int rows, int C, int F, Tensor "
      "preact, Tensor value_fc) -> Tensor");
  m.impl("cmix_sparse_down_relu_rows_varlen", c10::kCUDA,
         &cmix_sparse_down_relu_rows_varlen_fn);

  m.def(
      "cmix_sparse_down_relu_rows_t512_varlen(int rows, int C, int F, Tensor "
      "preact, Tensor value_fc) -> Tensor");
  m.impl("cmix_sparse_down_relu_rows_t512_varlen", c10::kCUDA,
         &cmix_sparse_down_relu_rows_t512_varlen_fn);
}

REGISTER_EXTENSION(_v2_5_mix_C)

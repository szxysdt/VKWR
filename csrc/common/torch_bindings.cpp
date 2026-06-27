#include <torch/extension.h>
#include <torch/library.h>

#include "core/register.h"
#include "v1_check.h"

void gather_shift_cuda(
    int64_t L, int64_t C, int64_t B,
    int64_t src_stride_d, int64_t src_stride_b, int64_t src_stride_l,
    int64_t dst_stride_d, int64_t dst_stride_b, int64_t dst_stride_l,
    at::Tensor src, at::Tensor dst, at::Tensor slot_indices);

void gather_wkv_cuda(
    int64_t L, int64_t H, int64_t N, int64_t B,
    int64_t src_stride_b, int64_t src_stride_l,
    int64_t dst_stride_b, int64_t dst_stride_l,
    at::Tensor src, at::Tensor dst, at::Tensor slot_indices);

void gather_elapsed_cuda(int64_t B, at::Tensor src, at::Tensor dst, at::Tensor slot_indices);

void scatter_shift_cuda(
    int64_t L, int64_t C, int64_t B,
    int64_t src_stride_d, int64_t src_stride_b, int64_t src_stride_l,
    int64_t dst_stride_d, int64_t dst_stride_b, int64_t dst_stride_l,
    at::Tensor src, at::Tensor dst, at::Tensor slot_indices);

void scatter_wkv_cuda(
    int64_t L, int64_t H, int64_t N, int64_t B,
    int64_t src_stride_b, int64_t src_stride_l,
    int64_t dst_stride_b, int64_t dst_stride_l,
    at::Tensor src, at::Tensor dst, at::Tensor slot_indices);

void scatter_elapsed_cuda(int64_t B, at::Tensor src, at::Tensor dst, at::Tensor slot_indices);

namespace {

void check_gather_scatter_shift(
    int64_t L, int64_t C, int64_t B,
    torch::Tensor global_state, torch::Tensor decode_state,
    torch::Tensor slot_indices) {
  check_half_cuda_contig(global_state, "global_state");
  check_half_cuda_contig(decode_state, "decode_state");
  check_i64_cuda_contig(slot_indices, "slot_indices");
  TORCH_CHECK(global_state.dim() == 4 && global_state.size(0) == L && global_state.size(1) == 2 && global_state.size(3) == C,
              "global_state must have shape [L,2,max_bsz,C]");
  TORCH_CHECK(decode_state.dim() == 4 && decode_state.size(0) == L && decode_state.size(1) == 2 && decode_state.size(3) == C,
              "decode_state must have shape [L,2,max_bsz,C]");
  TORCH_CHECK(slot_indices.dim() == 1 && slot_indices.size(0) >= B,
              "slot_indices must have shape [>=B]");
  TORCH_CHECK(decode_state.size(2) >= B, "decode_state batch dim must be >= B");
}

void check_gather_scatter_wkv(
    int64_t L, int64_t H, int64_t N, int64_t B,
    torch::Tensor global_state, torch::Tensor decode_state,
    torch::Tensor slot_indices) {
  check_half_cuda_contig(global_state, "global_state");
  check_half_cuda_contig(decode_state, "decode_state");
  check_i64_cuda_contig(slot_indices, "slot_indices");
  TORCH_CHECK(global_state.dim() == 5 && global_state.size(0) == L && global_state.size(2) == H &&
              global_state.size(3) == N && global_state.size(4) == N,
              "global_state must have shape [L,max_bsz,H,N,N]");
  TORCH_CHECK(decode_state.dim() == 5 && decode_state.size(0) == L && decode_state.size(2) == H &&
              decode_state.size(3) == N && decode_state.size(4) == N,
              "decode_state must have shape [L,max_bsz,H,N,N]");
  TORCH_CHECK(slot_indices.dim() == 1 && slot_indices.size(0) >= B,
              "slot_indices must have shape [>=B]");
  TORCH_CHECK(decode_state.size(1) >= B, "decode_state batch dim must be >= B");
}

void check_gather_scatter_elapsed(
    int64_t B,
    torch::Tensor global_state, torch::Tensor decode_state,
    torch::Tensor slot_indices) {
  check_i32_cuda_contig(global_state, "global_elapsed");
  check_i32_cuda_contig(decode_state, "decode_elapsed");
  check_i64_cuda_contig(slot_indices, "slot_indices");
  TORCH_CHECK(decode_state.dim() == 1 && decode_state.size(0) >= B,
              "decode_elapsed must have shape [>=B]");
  TORCH_CHECK(slot_indices.dim() == 1 && slot_indices.size(0) >= B,
              "slot_indices must have shape [>=B]");
}

inline int64_t elem_stride(const torch::Tensor& t, int64_t dim) {
  return t.stride(dim);
}

}  // namespace

void gather_decode_state(
    int64_t L, int64_t C, int64_t H, int64_t N, int64_t B,
    torch::Tensor global_shift, torch::Tensor global_wkv, torch::Tensor global_elapsed,
    torch::Tensor decode_shift, torch::Tensor decode_wkv, torch::Tensor decode_elapsed,
    torch::Tensor slot_indices) {
  check_gather_scatter_shift(L, C, B, global_shift, decode_shift, slot_indices);
  check_gather_scatter_wkv(L, H, N, B, global_wkv, decode_wkv, slot_indices);
  check_gather_scatter_elapsed(B, global_elapsed, decode_elapsed, slot_indices);

  gather_shift_cuda(L, C, B,
      elem_stride(global_shift, 1), elem_stride(global_shift, 2), elem_stride(global_shift, 0),
      elem_stride(decode_shift, 1), elem_stride(decode_shift, 2), elem_stride(decode_shift, 0),
      global_shift, decode_shift, slot_indices);
  gather_wkv_cuda(L, H, N, B,
      elem_stride(global_wkv, 1), elem_stride(global_wkv, 0),
      elem_stride(decode_wkv, 1), elem_stride(decode_wkv, 0),
      global_wkv, decode_wkv, slot_indices);
  gather_elapsed_cuda(B, global_elapsed, decode_elapsed, slot_indices);
}

void scatter_decode_state(
    int64_t L, int64_t C, int64_t H, int64_t N, int64_t B,
    torch::Tensor global_shift, torch::Tensor global_wkv, torch::Tensor global_elapsed,
    torch::Tensor decode_shift, torch::Tensor decode_wkv, torch::Tensor decode_elapsed,
    torch::Tensor slot_indices) {
  check_gather_scatter_shift(L, C, B, global_shift, decode_shift, slot_indices);
  check_gather_scatter_wkv(L, H, N, B, global_wkv, decode_wkv, slot_indices);
  check_gather_scatter_elapsed(B, global_elapsed, decode_elapsed, slot_indices);

  scatter_shift_cuda(L, C, B,
      elem_stride(decode_shift, 1), elem_stride(decode_shift, 2), elem_stride(decode_shift, 0),
      elem_stride(global_shift, 1), elem_stride(global_shift, 2), elem_stride(global_shift, 0),
      decode_shift, global_shift, slot_indices);
  scatter_wkv_cuda(L, H, N, B,
      elem_stride(decode_wkv, 1), elem_stride(decode_wkv, 0),
      elem_stride(global_wkv, 1), elem_stride(global_wkv, 0),
      decode_wkv, global_wkv, slot_indices);
  scatter_elapsed_cuda(B, decode_elapsed, global_elapsed, slot_indices);
}

TORCH_LIBRARY(vkwr_common, m) {
  m.def("gather_decode_state(int L, int C, int H, int N, int B, Tensor global_shift, Tensor global_wkv, Tensor global_elapsed, Tensor decode_shift, Tensor decode_wkv, Tensor decode_elapsed, Tensor slot_indices) -> ()");
  m.impl("gather_decode_state", c10::kCUDA, &gather_decode_state);

  m.def("scatter_decode_state(int L, int C, int H, int N, int B, Tensor global_shift, Tensor global_wkv, Tensor global_elapsed, Tensor decode_shift, Tensor decode_wkv, Tensor decode_elapsed, Tensor slot_indices) -> ()");
  m.impl("scatter_decode_state", c10::kCUDA, &scatter_decode_state);
}

REGISTER_EXTENSION(_common_C)

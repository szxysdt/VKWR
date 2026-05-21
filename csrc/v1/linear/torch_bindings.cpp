#include <torch/extension.h>
#include <torch/library.h>

#include "core/register.h"
#include "v1/common/v1_check.h"
#include "linear_ops.h"

torch::Tensor linear_f16_fn(torch::Tensor x, torch::Tensor weight) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16 shape mismatch");
  return linear_f16_cuda(x, weight);
}

torch::Tensor linear_f16_orig_fn(torch::Tensor x, torch::Tensor weight_orig) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_orig, "weight_orig");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_orig.dim() == 2, "weight_orig must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_orig.size(1), "linear_f16_orig shape mismatch");
  return linear_f16_orig_cuda(x, weight_orig);
}

torch::Tensor linear_orig_rows_f16_fn(torch::Tensor x, torch::Tensor weight_orig, int64_t row_tile, int64_t out_tile) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_orig, "weight_orig");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_orig.dim() == 2, "weight_orig must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_orig.size(1), "linear_orig_rows_f16 shape mismatch");
  return linear_orig_rows_f16_cuda(x, weight_orig, row_tile, out_tile);
}

torch::Tensor linear_orig_rows_cfg_f16_fn(torch::Tensor x, torch::Tensor weight_orig, int64_t threads, int64_t row_tile, int64_t out_tile) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_orig, "weight_orig");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_orig.dim() == 2, "weight_orig must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_orig.size(1), "linear_orig_rows_cfg_f16 shape mismatch");
  return linear_orig_rows_cfg_f16_cuda(x, weight_orig, threads, row_tile, out_tile);
}

torch::Tensor linear_orig_rows_exact_f16_fn(torch::Tensor x, torch::Tensor weight_orig, int64_t threads, int64_t out_tile, bool use4) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_orig, "weight_orig");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_orig.dim() == 2, "weight_orig must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_orig.size(1), "linear_orig_rows_exact_f16 shape mismatch");
  return linear_orig_rows_exact_f16_cuda(x, weight_orig, threads, out_tile, use4);
}

torch::Tensor linear_orig_wmma16_f16_fn(torch::Tensor x, torch::Tensor weight_orig) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_orig, "weight_orig");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_orig.dim() == 2, "weight_orig must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_orig.size(1), "linear_orig_wmma16_f16 shape mismatch");
  return linear_orig_wmma16_f16_cuda(x, weight_orig);
}

torch::Tensor linear_f16_orig_lt_fn(torch::Tensor x, torch::Tensor weight_orig) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_orig, "weight_orig");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_orig.dim() == 2, "weight_orig must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_orig.size(1), "linear_f16_orig_lt shape mismatch");
  return linear_f16_orig_lt_cuda(x, weight_orig);
}

torch::Tensor linear_f16_orig_lt_cfg_fn(torch::Tensor x, torch::Tensor weight_orig, int64_t workspace_mb, int64_t algo_index) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_orig, "weight_orig");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_orig.dim() == 2, "weight_orig must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_orig.size(1), "linear_f16_orig_lt_cfg shape mismatch");
  TORCH_CHECK(workspace_mb >= 0 && workspace_mb <= 1024, "workspace_mb out of range");
  TORCH_CHECK(algo_index >= 0 && algo_index < 64, "algo_index out of range");
  return linear_f16_orig_lt_cfg_cuda(x, weight_orig, workspace_mb, algo_index);
}

torch::Tensor linear_f16_lt_fn(torch::Tensor x, torch::Tensor weight) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_lt shape mismatch");
  return linear_f16_lt_cuda(x, weight);
}

torch::Tensor linear_f16_m1_splitk_fn(torch::Tensor x, torch::Tensor weight) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_m1_splitk shape mismatch");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_f16_m1_splitk requires M=1");
  return linear_f16_m1_splitk_cuda(x, weight);
}

torch::Tensor linear_f16_m1_splitk_cfg_fn(torch::Tensor x, torch::Tensor weight, int64_t chunk_k) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_m1_splitk_cfg shape mismatch");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_f16_m1_splitk_cfg requires M=1");
  return linear_f16_m1_splitk_cfg_cuda(x, weight, chunk_k);
}

torch::Tensor linear_f16_m1_splitk_tile_fn(torch::Tensor x, torch::Tensor weight, int64_t chunk_k, int64_t tile_cols) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_m1_splitk_tile shape mismatch");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_f16_m1_splitk_tile requires M=1");
  return linear_f16_m1_splitk_tile_cuda(x, weight, chunk_k, tile_cols);
}

torch::Tensor linear_f16_m1_splitk_warpred_tile_fn(torch::Tensor x, torch::Tensor weight, int64_t chunk_k, int64_t tile_cols) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_m1_splitk_warpred_tile shape mismatch");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_f16_m1_splitk_warpred_tile requires M=1");
  return linear_f16_m1_splitk_warpred_tile_cuda(x, weight, chunk_k, tile_cols);
}

torch::Tensor linear_f16_rows_splitk_fn(torch::Tensor x, torch::Tensor weight, int64_t chunk_k) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_rows_splitk shape mismatch");
  return linear_f16_rows_splitk_cuda(x, weight, chunk_k);
}

torch::Tensor linear_t_f16_fn(torch::Tensor x, torch::Tensor weight_t) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_t, "weight_t");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_t.dim() == 2, "weight_t must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_t.size(1), "linear_t_f16 shape mismatch");
  return linear_t_f16_cuda(x, weight_t);
}

torch::Tensor linear_t_act_f16_fn(torch::Tensor x, torch::Tensor weight_t, int64_t act) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_t, "weight_t");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_t.dim() == 2, "weight_t must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_t.size(1), "linear_t_act_f16 shape mismatch");
  TORCH_CHECK(act == 1 || act == 2, "act must be 1=tanh or 2=sigmoid");
  return linear_t_act_f16_cuda(x, weight_t, act);
}

torch::Tensor linear_t_vres_f16_fn(torch::Tensor x, torch::Tensor weight_t, torch::Tensor v, torch::Tensor v_first, torch::Tensor v0) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_t, "weight_t");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(v_first, "v_first");
  check_half_cuda_contig(v0, "v0");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_t.dim() == 2, "weight_t must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_t.size(1), "linear_t_vres_f16 shape mismatch");
  TORCH_CHECK(v.sizes() == v_first.sizes(), "v/v_first shape mismatch");
  TORCH_CHECK(v.dim() >= 2 && v.size(-1) == weight_t.size(0), "v shape mismatch");
  TORCH_CHECK(v0.dim() == 1 && v0.size(0) == weight_t.size(0), "v0 shape mismatch");
  return linear_t_vres_f16_cuda(x, weight_t, v, v_first, v0);
}

TORCH_LIBRARY(vkwr_v1_linear, m) {
  m.def("linear_f16(Tensor x, Tensor weight) -> Tensor");
  m.impl("linear_f16", c10::kCUDA, &linear_f16_fn);

  m.def("linear_f16_orig(Tensor x, Tensor weight_orig) -> Tensor");
  m.impl("linear_f16_orig", c10::kCUDA, &linear_f16_orig_fn);

  m.def("linear_orig_rows_f16(Tensor x, Tensor weight_orig, int row_tile, int out_tile) -> Tensor");
  m.impl("linear_orig_rows_f16", c10::kCUDA, &linear_orig_rows_f16_fn);

  m.def("linear_orig_rows_cfg_f16(Tensor x, Tensor weight_orig, int threads, int row_tile, int out_tile) -> Tensor");
  m.impl("linear_orig_rows_cfg_f16", c10::kCUDA, &linear_orig_rows_cfg_f16_fn);

  m.def("linear_orig_rows_exact_f16(Tensor x, Tensor weight_orig, int threads, int out_tile, bool use4) -> Tensor");
  m.impl("linear_orig_rows_exact_f16", c10::kCUDA, &linear_orig_rows_exact_f16_fn);

  m.def("linear_orig_wmma16_f16(Tensor x, Tensor weight_orig) -> Tensor");
  m.impl("linear_orig_wmma16_f16", c10::kCUDA, &linear_orig_wmma16_f16_fn);

  m.def("linear_f16_orig_lt(Tensor x, Tensor weight_orig) -> Tensor");
  m.impl("linear_f16_orig_lt", c10::kCUDA, &linear_f16_orig_lt_fn);

  m.def("linear_f16_orig_lt_cfg(Tensor x, Tensor weight_orig, int workspace_mb, int algo_index) -> Tensor");
  m.impl("linear_f16_orig_lt_cfg", c10::kCUDA, &linear_f16_orig_lt_cfg_fn);

  m.def("linear_f16_lt(Tensor x, Tensor weight) -> Tensor");
  m.impl("linear_f16_lt", c10::kCUDA, &linear_f16_lt_fn);

  m.def("linear_f16_m1_splitk(Tensor x, Tensor weight) -> Tensor");
  m.impl("linear_f16_m1_splitk", c10::kCUDA, &linear_f16_m1_splitk_fn);

  m.def("linear_f16_m1_splitk_cfg(Tensor x, Tensor weight, int chunk_k) -> Tensor");
  m.impl("linear_f16_m1_splitk_cfg", c10::kCUDA, &linear_f16_m1_splitk_cfg_fn);

  m.def("linear_f16_m1_splitk_tile(Tensor x, Tensor weight, int chunk_k, int tile_cols) -> Tensor");
  m.impl("linear_f16_m1_splitk_tile", c10::kCUDA, &linear_f16_m1_splitk_tile_fn);

  m.def("linear_f16_m1_splitk_warpred_tile(Tensor x, Tensor weight, int chunk_k, int tile_cols) -> Tensor");
  m.impl("linear_f16_m1_splitk_warpred_tile", c10::kCUDA, &linear_f16_m1_splitk_warpred_tile_fn);

  m.def("linear_f16_rows_splitk(Tensor x, Tensor weight, int chunk_k) -> Tensor");
  m.impl("linear_f16_rows_splitk", c10::kCUDA, &linear_f16_rows_splitk_fn);

  m.def("linear_t_f16(Tensor x, Tensor weight_t) -> Tensor");
  m.impl("linear_t_f16", c10::kCUDA, &linear_t_f16_fn);

  m.def("linear_t_act_f16(Tensor x, Tensor weight_t, int act) -> Tensor");
  m.impl("linear_t_act_f16", c10::kCUDA, &linear_t_act_f16_fn);

  m.def("linear_t_vres_f16(Tensor x, Tensor weight_t, Tensor v, Tensor v_first, Tensor v0) -> Tensor");
  m.impl("linear_t_vres_f16", c10::kCUDA, &linear_t_vres_f16_fn);
}

REGISTER_EXTENSION(_v1_linear_C)

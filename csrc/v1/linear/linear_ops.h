#pragma once

#include <torch/extension.h>

at::Tensor linear_f16_cuda(at::Tensor x, at::Tensor weight);
at::Tensor linear_f16_orig_cuda(at::Tensor x, at::Tensor weight_orig);
at::Tensor linear_orig_rows_f16_cuda(at::Tensor x, at::Tensor weight_orig, int64_t row_tile, int64_t out_tile);
at::Tensor linear_orig_rows_cfg_f16_cuda(at::Tensor x, at::Tensor weight_orig, int64_t threads, int64_t row_tile, int64_t out_tile);
at::Tensor linear_orig_rows_exact_f16_cuda(at::Tensor x, at::Tensor weight_orig, int64_t threads, int64_t out_tile, bool use4);
at::Tensor linear_orig_wmma16_f16_cuda(at::Tensor x, at::Tensor weight_orig);
at::Tensor linear_f16_orig_lt_cuda(at::Tensor x, at::Tensor weight_orig);
at::Tensor linear_f16_orig_lt_cfg_cuda(at::Tensor x, at::Tensor weight_orig, int64_t workspace_mb, int64_t algo_index);
at::Tensor linear_f16_lt_cuda(at::Tensor x, at::Tensor weight);
at::Tensor linear_f16_m1_splitk_cuda(at::Tensor x, at::Tensor weight);
at::Tensor linear_f16_m1_splitk_cfg_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k);
at::Tensor linear_f16_m1_splitk_tile_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k, int64_t tile_cols);
at::Tensor linear_f16_m1_splitk_warpred_tile_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k, int64_t tile_cols);
at::Tensor linear_f16_rows_splitk_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k);
at::Tensor linear_t_f16_cuda(at::Tensor x, at::Tensor weight_t);
at::Tensor linear_t_act_f16_cuda(at::Tensor x, at::Tensor weight_t, int64_t act);
at::Tensor linear_t_vres_f16_cuda(at::Tensor x, at::Tensor weight_t, at::Tensor v, at::Tensor v_first, at::Tensor v0);

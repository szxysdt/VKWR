#pragma once

#include <torch/extension.h>

#include <vector>

// Varlen fused add_ln + tmix_mix6
std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_cuda_varlen(
    at::Tensor x, at::Tensor residual, at::Tensor shift_state,
    at::Tensor weight, at::Tensor bias, at::Tensor x_r, at::Tensor x_w,
    at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g,
    at::Tensor query_start_loc, at::Tensor req_id, double eps);

// Varlen fused add_ln + cmix_mix
std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_cuda_varlen(
    at::Tensor x, at::Tensor residual, at::Tensor shift_state,
    at::Tensor weight, at::Tensor bias, at::Tensor x_k,
    at::Tensor query_start_loc, at::Tensor req_id, double eps);

// Varlen add_last_layer_norm (reads last token per request)
at::Tensor add_last_layer_norm_f16_cuda_varlen(
    at::Tensor x, at::Tensor residual, at::Tensor weight, at::Tensor bias,
    at::Tensor query_start_loc, double eps);

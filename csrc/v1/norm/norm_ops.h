#pragma once

#include <torch/extension.h>
#include <vector>

at::Tensor layer_norm_f16_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps);
at::Tensor emb_ln0_bf16_to_f16_cuda(at::Tensor emb, at::Tensor weight, at::Tensor bias, double eps);
at::Tensor layer_norm_f16_small_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps);
at::Tensor layer_norm_f16_small512_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps);
at::Tensor add_f16_cuda(at::Tensor x, at::Tensor y);
std::vector<at::Tensor> add_layer_norm_f16_cuda(at::Tensor x, at::Tensor residual, at::Tensor weight, at::Tensor bias, double eps);
at::Tensor add_last_layer_norm_f16_cuda(at::Tensor x, at::Tensor residual, at::Tensor weight, at::Tensor bias, double eps);
std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_cuda(at::Tensor x, at::Tensor residual, at::Tensor shift_state, at::Tensor weight, at::Tensor bias, at::Tensor x_k, double eps);
std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_cfg_cuda(at::Tensor x, at::Tensor residual, at::Tensor shift_state, at::Tensor weight, at::Tensor bias, at::Tensor x_k, double eps, int threads);
std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_scalar_stats_cuda(at::Tensor x, at::Tensor residual, at::Tensor shift_state, at::Tensor weight, at::Tensor bias, at::Tensor x_k, double eps);
std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_cuda(at::Tensor x, at::Tensor residual, at::Tensor shift_state, at::Tensor weight, at::Tensor bias, at::Tensor x_r, at::Tensor x_w, at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g, double eps);
std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_cfg_cuda(at::Tensor x, at::Tensor residual, at::Tensor shift_state, at::Tensor weight, at::Tensor bias, at::Tensor x_r, at::Tensor x_w, at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g, double eps, int threads);
std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_scalar_stats_cuda(at::Tensor x, at::Tensor residual, at::Tensor shift_state, at::Tensor weight, at::Tensor bias, at::Tensor x_r, at::Tensor x_w, at::Tensor x_k, at::Tensor x_v, at::Tensor x_a, at::Tensor x_g, double eps);

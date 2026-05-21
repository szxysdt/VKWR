#pragma once

#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> linear_wag_rank_in_f16_cuda(
    int64_t M, int64_t K, int64_t Rw, int64_t Ra, int64_t Rg,
    at::Tensor xw, at::Tensor xa, at::Tensor xg,
    at::Tensor w1_t, at::Tensor a1_t, at::Tensor g1_t);

std::vector<at::Tensor> linear_wagv_rank_in_f16_cuda(
    int64_t M, int64_t K, int64_t Rw, int64_t Ra, int64_t Rg, int64_t Rv,
    at::Tensor xw, at::Tensor xa, at::Tensor xg, at::Tensor xv,
    at::Tensor w1_t, at::Tensor a1_t, at::Tensor g1_t, at::Tensor v1_t);

std::vector<at::Tensor> linear_wag_rank_out_f16_cuda(
    int64_t M, int64_t C, int64_t Kw, int64_t Ka, int64_t Kg,
    at::Tensor w1, at::Tensor a1, at::Tensor g1,
    at::Tensor w2_t, at::Tensor a2_t, at::Tensor g2_t);

std::vector<at::Tensor> linear_wagv_rank_out_f16_cuda(
    int64_t M, int64_t C, int64_t Kw, int64_t Ka, int64_t Kg, int64_t Kv,
    at::Tensor w1, at::Tensor a1, at::Tensor g1, at::Tensor v1,
    at::Tensor w2_t, at::Tensor a2_t, at::Tensor g2_t, at::Tensor v2_t,
    at::Tensor v, at::Tensor v_first, at::Tensor v0);

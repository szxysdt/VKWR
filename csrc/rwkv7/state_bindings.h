#pragma once

#include <torch/extension.h>

void forward_seq(int64_t B, int64_t T, int64_t C, int64_t H,
                 torch::Tensor &state, torch::Tensor &r, torch::Tensor &w,
                 torch::Tensor &k, torch::Tensor &v, torch::Tensor &a,
                 torch::Tensor &b, torch::Tensor &y, torch::Tensor &elapsed_t);

void forward_one(int64_t B, int64_t C, int64_t H,
                 torch::Tensor &state, torch::Tensor &r, torch::Tensor &w,
                 torch::Tensor &k, torch::Tensor &v, torch::Tensor &a,
                 torch::Tensor &b, torch::Tensor &y, torch::Tensor &elapsed_t);

void spmv_forward(int64_t D, int64_t C,
                  torch::Tensor &vec, torch::Tensor &mat, torch::Tensor &out);

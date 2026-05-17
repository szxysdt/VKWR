#pragma once

#include <torch/extension.h>

// RWKV op binding declarations
void wkv_forward(int64_t B, int64_t T, int64_t C,
                 torch::Tensor &w, torch::Tensor &u,
                 torch::Tensor &k, torch::Tensor &v, torch::Tensor &y,
                 torch::Tensor &aa, torch::Tensor &bb, torch::Tensor &pp);

void mm8_seq(int64_t B, int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y);

void mm8_one(int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &mx, torch::Tensor &rx,
             torch::Tensor &my, torch::Tensor &ry,
             torch::Tensor &y);

#ifdef ENABLE_CUBLAS_GEMM
void gemm_fp16_cublas_impl(torch::Tensor a, torch::Tensor b, torch::Tensor c);
#endif

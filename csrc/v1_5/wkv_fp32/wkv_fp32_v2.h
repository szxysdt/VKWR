#pragma once

#include <torch/extension.h>

// varlen version: query_start_loc replaces T, max_t passed from Python side
void wkv_fp32_v2_cuda_varlen(
    int B, int max_t,
    const int* query_start_loc,  // [B+1] device pointer
    int C, int H, int mode, torch::Tensor state, torch::Tensor r,
    torch::Tensor w, torch::Tensor k, torch::Tensor v, torch::Tensor a,
    torch::Tensor b, torch::Tensor y);

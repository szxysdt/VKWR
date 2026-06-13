#pragma once

#include <torch/extension.h>

// varlen version: query_start_loc replaces T, max_t passed from Python side
void wkv_seq_v2_cuda_varlen(int B, int max_t,
                            const int* query_start_loc,  // [B+1] device pointer
                            int C, int H, torch::Tensor state, torch::Tensor r,
                            torch::Tensor w, torch::Tensor k, torch::Tensor v,
                            torch::Tensor a, torch::Tensor b, torch::Tensor y,
                            torch::Tensor elapsed_t);

void wkv_seq_w0_v2_cuda_varlen(
    int B, int max_t,
    const int* query_start_loc,  // [B+1] device pointer
    int C, int H, torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor w0, torch::Tensor k, torch::Tensor v, torch::Tensor a,
    torch::Tensor b, torch::Tensor y, torch::Tensor elapsed_t);

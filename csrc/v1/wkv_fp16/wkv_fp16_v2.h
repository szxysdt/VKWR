#pragma once

#include <torch/extension.h>

void wkv_seq_v2_cuda(
    int B, int T, int C, int H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor elapsed_t);

void wkv_seq_w0_v2_cuda(
    int B, int T, int C, int H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w, torch::Tensor w0,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor elapsed_t);

void wkv_one_v2_cuda(
    int B, int C, int H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor elapsed_t);

void wkv_one_w0_v2_cuda(
    int B, int C, int H,
    torch::Tensor state, torch::Tensor r, torch::Tensor w, torch::Tensor w0,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y, torch::Tensor elapsed_t);

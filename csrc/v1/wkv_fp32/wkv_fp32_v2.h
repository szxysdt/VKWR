#pragma once

#include <torch/extension.h>

void wkv_fp32_v2_cuda(
    int B, int T, int C, int H, int mode,
    torch::Tensor state, torch::Tensor r, torch::Tensor w,
    torch::Tensor k, torch::Tensor v, torch::Tensor a, torch::Tensor b,
    torch::Tensor y);

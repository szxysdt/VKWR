#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "v1/common/warp_primitives.cuh"

namespace {

__global__ void advance_i32_kernel(int* __restrict__ x, int amount, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += amount;
  }
}

}  // namespace

void advance_i32_cuda(at::Tensor x, int64_t amount) {
  TORCH_CHECK(amount >= INT_MIN && amount <= INT_MAX, "advance_i32 amount out of int range");
  constexpr int threads = 256;
  const int64_t n = x.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  advance_i32_kernel<<<static_cast<int>(ceil_div(n, threads)), threads, 0, stream>>>(
      x.data_ptr<int>(), static_cast<int>(amount), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

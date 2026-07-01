#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "common/warp_primitives.cuh"

namespace {

__global__ void advance_i32_varlen_kernel(
    int* __restrict__ elapsed, const int* __restrict__ query_start_loc,
    const int* __restrict__ slot_indices, int64_t B) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < B) {
    elapsed[slot_indices[i]] += query_start_loc[i + 1] - query_start_loc[i];
  }
}

}  // namespace

void advance_i32_varlen_cuda(at::Tensor elapsed, at::Tensor query_start_loc,
                             at::Tensor slot_indices) {
  constexpr int threads = 256;
  const int64_t B = slot_indices.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  advance_i32_varlen_kernel<<<static_cast<int>(ceil_div(B, threads)), threads,
                              0, stream>>>(
      elapsed.data_ptr<int>(), query_start_loc.data_ptr<int>(),
      slot_indices.data_ptr<int>(), B);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

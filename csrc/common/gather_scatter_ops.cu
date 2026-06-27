#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include "common/warp_primitives.cuh"

using F = half;

namespace {

// ===== Unified half-precision gather kernel =====
// Each block handles one (b, l) pair. Resolves slot index once, then
// copies `d` contiguous chunks of `chunk_size` half elements using half2
// vectorized loads/stores. No per-element div/mod arithmetic.
//
// Matches vLLM mamba_utils batch_memcpy pattern: ptr resolution per block,
// linear copy loop inside.
//
// For shift [L, 2, S, C]: d=2, chunk_size=C, chunk_stride=C
// For wkv  [L, S, H, N, N]: d=1, chunk_size=H*N*N, chunk_stride=H*N*N

__global__ void kernel_gather_half(
    int64_t B,
    int64_t L,
    int64_t d,            // number of chunks per (l, b) pair (2 for shift, 1 for wkv)
    int64_t chunk_size,   // contiguous half elements per chunk
    int64_t src_stride_d, // element stride between consecutive chunks in src
    int64_t dst_stride_d, // element stride between consecutive chunks in dst
    int64_t src_stride_l, // element stride for +1 in L dim of src
    int64_t src_stride_b, // element stride for +1 in slot/batch dim of src
    int64_t dst_stride_l, // element stride for +1 in L dim of dst
    int64_t dst_stride_b, // element stride for +1 in batch dim of dst
    const F* __restrict__ src,
    F* __restrict__ dst,
    const int64_t* __restrict__ slot_indices) {

  const int64_t b = static_cast<int64_t>(blockIdx.y);
  const int64_t l = static_cast<int64_t>(blockIdx.x);
  const int64_t s = slot_indices[b];

  const half2* __restrict__ src_base = reinterpret_cast<const half2*>(
      src + l * src_stride_l + s * src_stride_b);
  half2* __restrict__ dst_base = reinterpret_cast<half2*>(
      dst + l * dst_stride_l + b * dst_stride_b);

  for (int64_t di = 0; di < d; di++) {
    const half2* __restrict__ src_chunk = reinterpret_cast<const half2*>(
        reinterpret_cast<const char*>(src_base) + di * src_stride_d * sizeof(F));
    half2* __restrict__ dst_chunk = reinterpret_cast<half2*>(
        reinterpret_cast<char*>(dst_base) + di * dst_stride_d * sizeof(F));

    const int64_t half2_per_chunk = chunk_size / 2;
    const int64_t tid = static_cast<int64_t>(threadIdx.x);

    for (int64_t i = tid; i < half2_per_chunk; i += blockDim.x) {
      dst_chunk[i] = src_chunk[i];
    }
  }
}

// ===== Unified half-precision scatter kernel =====
// Inverse of gather. Same structure: each block handles one (b, l) pair.

__global__ void kernel_scatter_half(
    int64_t B,
    int64_t L,
    int64_t d,
    int64_t chunk_size,
    int64_t src_stride_d,
    int64_t dst_stride_d,
    int64_t src_stride_l,
    int64_t src_stride_b,
    int64_t dst_stride_l,
    int64_t dst_stride_b,
    const F* __restrict__ src,
    F* __restrict__ dst,
    const int64_t* __restrict__ slot_indices) {

  const int64_t b = static_cast<int64_t>(blockIdx.y);
  const int64_t l = static_cast<int64_t>(blockIdx.x);
  const int64_t s = slot_indices[b];

  const half2* __restrict__ src_base = reinterpret_cast<const half2*>(
      src + l * src_stride_l + b * src_stride_b);
  half2* __restrict__ dst_base = reinterpret_cast<half2*>(
      dst + l * dst_stride_l + s * dst_stride_b);

  for (int64_t di = 0; di < d; di++) {
    const half2* __restrict__ src_chunk = reinterpret_cast<const half2*>(
        reinterpret_cast<const char*>(src_base) + di * src_stride_d * sizeof(F));
    half2* __restrict__ dst_chunk = reinterpret_cast<half2*>(
        reinterpret_cast<char*>(dst_base) + di * dst_stride_d * sizeof(F));

    const int64_t half2_per_chunk = chunk_size / 2;
    const int64_t tid = static_cast<int64_t>(threadIdx.x);

    for (int64_t i = tid; i < half2_per_chunk; i += blockDim.x) {
      dst_chunk[i] = src_chunk[i];
    }
  }
}

// ===== Elapsed gather/scatter (int32, B elements) =====
// Already optimal: 1 thread per element, single index lookup.

__global__ void kernel_gather_elapsed(
    int64_t B,
    const int32_t* __restrict__ src,
    int32_t* __restrict__ dst,
    const int64_t* __restrict__ slot_indices) {

  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < B) {
    dst[idx] = src[slot_indices[idx]];
  }
}

__global__ void kernel_scatter_elapsed(
    int64_t B,
    const int32_t* __restrict__ src,
    int32_t* __restrict__ dst,
    const int64_t* __restrict__ slot_indices) {

  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < B) {
    dst[slot_indices[idx]] = src[idx];
  }
}

constexpr int BLOCK = 256;

}  // namespace

// ===== Host wrappers =====

void gather_shift_cuda(
    int64_t L, int64_t C, int64_t B,
    int64_t src_stride_d, int64_t src_stride_b, int64_t src_stride_l,
    int64_t dst_stride_d, int64_t dst_stride_b, int64_t dst_stride_l,
    at::Tensor src, at::Tensor dst, at::Tensor slot_indices) {

  dim3 grid(L, B);
  auto stream = at::cuda::getCurrentCUDAStream();
  kernel_gather_half<<<grid, BLOCK, 0, stream>>>(
      B, L,
      /*d=*/2, /*chunk_size=*/C,
      src_stride_d, dst_stride_d,
      src_stride_l, src_stride_b,
      dst_stride_l, dst_stride_b,
      reinterpret_cast<const F*>(src.data_ptr()),
      reinterpret_cast<F*>(dst.data_ptr()),
      slot_indices.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_wkv_cuda(
    int64_t L, int64_t H, int64_t N, int64_t B,
    int64_t src_stride_b, int64_t src_stride_l,
    int64_t dst_stride_b, int64_t dst_stride_l,
    at::Tensor src, at::Tensor dst, at::Tensor slot_indices) {

  const int64_t plane_size = H * N * N;
  dim3 grid(L, B);
  auto stream = at::cuda::getCurrentCUDAStream();
  kernel_gather_half<<<grid, BLOCK, 0, stream>>>(
      B, L,
      /*d=*/1, /*chunk_size=*/plane_size,
      /*src_stride_d=*/plane_size, /*dst_stride_d=*/plane_size,
      src_stride_l, src_stride_b,
      dst_stride_l, dst_stride_b,
      reinterpret_cast<const F*>(src.data_ptr()),
      reinterpret_cast<F*>(dst.data_ptr()),
      slot_indices.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_elapsed_cuda(int64_t B, at::Tensor src, at::Tensor dst, at::Tensor slot_indices) {
  auto stream = at::cuda::getCurrentCUDAStream();
  kernel_gather_elapsed<<<
      static_cast<int>(ceil_div(B, BLOCK)), BLOCK, 0, stream>>>(
          B,
          src.data_ptr<int32_t>(),
          dst.data_ptr<int32_t>(),
          slot_indices.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_shift_cuda(
    int64_t L, int64_t C, int64_t B,
    int64_t src_stride_d, int64_t src_stride_b, int64_t src_stride_l,
    int64_t dst_stride_d, int64_t dst_stride_b, int64_t dst_stride_l,
    at::Tensor src, at::Tensor dst, at::Tensor slot_indices) {

  dim3 grid(L, B);
  auto stream = at::cuda::getCurrentCUDAStream();
  kernel_scatter_half<<<grid, BLOCK, 0, stream>>>(
      B, L,
      /*d=*/2, /*chunk_size=*/C,
      src_stride_d, dst_stride_d,
      src_stride_l, src_stride_b,
      dst_stride_l, dst_stride_b,
      reinterpret_cast<const F*>(src.data_ptr()),
      reinterpret_cast<F*>(dst.data_ptr()),
      slot_indices.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_wkv_cuda(
    int64_t L, int64_t H, int64_t N, int64_t B,
    int64_t src_stride_b, int64_t src_stride_l,
    int64_t dst_stride_b, int64_t dst_stride_l,
    at::Tensor src, at::Tensor dst, at::Tensor slot_indices) {

  const int64_t plane_size = H * N * N;
  dim3 grid(L, B);
  auto stream = at::cuda::getCurrentCUDAStream();
  kernel_scatter_half<<<grid, BLOCK, 0, stream>>>(
      B, L,
      /*d=*/1, /*chunk_size=*/plane_size,
      /*src_stride_d=*/plane_size, /*dst_stride_d=*/plane_size,
      src_stride_l, src_stride_b,
      dst_stride_l, dst_stride_b,
      reinterpret_cast<const F*>(src.data_ptr()),
      reinterpret_cast<F*>(dst.data_ptr()),
      slot_indices.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_elapsed_cuda(int64_t B, at::Tensor src, at::Tensor dst, at::Tensor slot_indices) {
  auto stream = at::cuda::getCurrentCUDAStream();
  kernel_scatter_elapsed<<<
      static_cast<int>(ceil_div(B, BLOCK)), BLOCK, 0, stream>>>(
          B,
          src.data_ptr<int32_t>(),
          dst.data_ptr<int32_t>(),
          slot_indices.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

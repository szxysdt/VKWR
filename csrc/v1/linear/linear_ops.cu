/*
 * VKWR v1 Linear operators
 * Migrated from faster3a rwkv7_v3a_ops.cu
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <climits>
#include <vector>

#include "v1/common/warp_primitives.cuh"

using dtype = at::Half;
namespace wmma = nvcuda::wmma;

namespace {

inline void check_cublas(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cublas status ", static_cast<int>(status));
}

inline void check_cublaslt(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cublasLt status ", static_cast<int>(status));
}

template <int Act>
__device__ __forceinline__ float apply_act(float x) {
  if constexpr (Act == 1) {
    return tanhf(x);
  } else {
    return 1.0f / (1.0f + expf(-x));
  }
}

} // namespace

/* ===== Linear device kernels ===== */

template <int ChunkK, int Warps>
__global__ __launch_bounds__(128, 2) void linear_f16_m1_splitk_partial_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    float* __restrict__ partial) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = (blockIdx.x * Warps + warp) * 32 + lane;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  const int k0 = blockIdx.y * ChunkK;
  const int k1 = min(k0 + ChunkK, K);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int k = k0; k < k1; ++k) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight + static_cast<int64_t>(k) * N + n));
    acc0 = fmaf(xv, wv.x, acc0);
    acc1 = fmaf(xv, wv.y, acc1);
  }
  reinterpret_cast<float2*>(partial + static_cast<int64_t>(blockIdx.y) * N)[pair] = make_float2(acc0, acc1);
}

__global__ void linear_f16_m1_splitk_reduce_kernel(
    int chunks,
    int N,
    const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = 0; c < chunks; ++c) {
    const float2 v = reinterpret_cast<const float2*>(partial + static_cast<int64_t>(c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
  reinterpret_cast<__half2*>(y)[pair] = __floats2half2_rn(acc0, acc1);
}

__global__ void linear_f16_m1_splitk_reduce_warp_kernel(
    int chunks,
    int N,
    const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = blockIdx.x * 4 + warp;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = lane; c < chunks; c += 32) {
    const float2 v = reinterpret_cast<const float2*>(partial + static_cast<int64_t>(c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc0 += __shfl_down_sync(0xffffffffu, acc0, offset);
    acc1 += __shfl_down_sync(0xffffffffu, acc1, offset);
  }
  if (lane == 0) {
    reinterpret_cast<__half2*>(y)[pair] = __floats2half2_rn(acc0, acc1);
  }
}

template <int ChunkK, int Warps>
__global__ __launch_bounds__(128, 2) void linear_f16_rows_splitk_partial_kernel(
    int K,
    int N,
    int chunks,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    float* __restrict__ partial) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = (blockIdx.x * Warps + warp) * 32 + lane;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  const int chunk = blockIdx.y;
  const int m = blockIdx.z;
  const int k0 = chunk * ChunkK;
  const int k1 = min(k0 + ChunkK, K);
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int k = k0; k < k1; ++k) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight + static_cast<int64_t>(k) * N + n));
    acc0 = fmaf(xv, wv.x, acc0);
    acc1 = fmaf(xv, wv.y, acc1);
  }
  reinterpret_cast<float2*>(partial + (static_cast<int64_t>(m) * chunks + chunk) * N)[pair] = make_float2(acc0, acc1);
}

__global__ void linear_f16_rows_splitk_reduce_kernel(
    int chunks,
    int N,
    const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = 0; c < chunks; ++c) {
    const float2 v = reinterpret_cast<const float2*>(partial + (static_cast<int64_t>(m) * chunks + c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
  reinterpret_cast<__half2*>(y + static_cast<int64_t>(m) * N)[pair] = __floats2half2_rn(acc0, acc1);
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n = blockIdx.x;
  const int m = blockIdx.y;
  if (m >= M || n >= N) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = weight_t + static_cast<int64_t>(n) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + (k2 << 1)));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + (k2 << 1)));
    acc = fmaf(xv.x, wv.x, acc);
    acc = fmaf(xv.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(acc);
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + K - 1));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + K - 1)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int RowTile, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_rows_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m0 = blockIdx.y * RowTile;
  float acc[RowTile][OutTile];
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      acc[r][j] = 0.0f;
    }
  }
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    float2 wv[OutTile];
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      wv[j] = (n < N)
          ? __half22float2(*reinterpret_cast<const __half2*>(weight_orig + static_cast<int64_t>(n) * K + k))
          : make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + static_cast<int64_t>(m) * K + k));
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          acc[r][j] = fmaf(xv.x, wv[j].x, acc[r][j]);
          acc[r][j] = fmaf(xv.y, wv[j].y, acc[r][j]);
        }
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float wv = __half2float(*reinterpret_cast<const __half*>(weight_orig + static_cast<int64_t>(n) * K + K - 1));
#pragma unroll
        for (int r = 0; r < RowTile; ++r) {
          const int m = m0 + r;
          if (m < M) {
            const float xv = __half2float(*reinterpret_cast<const __half*>(x + static_cast<int64_t>(m) * K + K - 1));
            acc[r][j] = fmaf(xv, wv, acc[r][j]);
          }
        }
      }
    }
  }
  __shared__ float partial[Threads / 32][RowTile][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float v = warp_sum(acc[r][j]);
      if (lane == 0) {
        partial[warp][r][j] = v;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          const int n = n0 + j;
          if (n < N) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < Threads / 32; ++w) {
              sum += partial[w][r][j];
            }
            *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
          }
        }
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row1_exact_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  for (int k2 = threadIdx.x; k2 < (K >> 1); k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_orig + static_cast<int64_t>(n0 + j) * K + k));
      acc[j] = fmaf(xv.x, wv.x, acc[j]);
      acc[j] = fmaf(xv.y, wv.y, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      y[n0 + j] = __float2half_rn(sum);
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row1_exact4_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight_orig + static_cast<int64_t>(n0 + j) * K + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(wj));
      const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(wj + 2));
      acc[j] = fmaf(x0.x, w0.x, acc[j]);
      acc[j] = fmaf(x0.y, w0.y, acc[j]);
      acc[j] = fmaf(x1.x, w1.x, acc[j]);
      acc[j] = fmaf(x1.y, w1.y, acc[j]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      y[n0 + j] = __float2half_rn(sum);
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row2_exact_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k2 = threadIdx.x; k2 < (K >> 1); k2 += Threads) {
    const int k = k2 << 1;
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_orig + static_cast<int64_t>(n0 + j) * K + k));
      acc0[j] = fmaf(x0.x, wv.x, acc0[j]);
      acc0[j] = fmaf(x0.y, wv.y, acc0[j]);
      acc1[j] = fmaf(x1.x, wv.x, acc1[j]);
      acc1[j] = fmaf(x1.y, wv.y, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row2_exact4_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x00 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x01 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const float2 x10 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
    const float2 x11 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 2));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight_orig + static_cast<int64_t>(n0 + j) * K + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(wj));
      const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(wj + 2));
      acc0[j] = fmaf(x00.x, w0.x, acc0[j]);
      acc0[j] = fmaf(x00.y, w0.y, acc0[j]);
      acc0[j] = fmaf(x01.x, w1.x, acc0[j]);
      acc0[j] = fmaf(x01.y, w1.y, acc0[j]);
      acc1[j] = fmaf(x10.x, w0.x, acc1[j]);
      acc1[j] = fmaf(x10.y, w0.y, acc1[j]);
      acc1[j] = fmaf(x11.x, w1.x, acc1[j]);
      acc1[j] = fmaf(x11.y, w1.y, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}

__global__ __launch_bounds__(32, 8) void linear_orig_wmma16_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * 16;
  const int m0 = blockIdx.y * 16;
  __shared__ __half a_tile[16 * 16];
  __shared__ __half b_tile[16 * 16];
  __shared__ float c_tile[16 * 16];

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += 16) {
    for (int idx = threadIdx.x; idx < 16 * 16; idx += 32) {
      const int r = idx >> 4;
      const int kk = idx & 15;
      const int m = m0 + r;
      a_tile[idx] = (m < M && k0 + kk < K)
          ? *reinterpret_cast<const __half*>(x + static_cast<int64_t>(m) * K + k0 + kk)
          : __float2half(0.0f);
      const int n = n0 + r;
      b_tile[r * 16 + kk] = (n < N && k0 + kk < K)
          ? *reinterpret_cast<const __half*>(weight_orig + static_cast<int64_t>(n) * K + k0 + kk)
          : __float2half(0.0f);
    }
    __syncwarp();
    wmma::load_matrix_sync(a_frag, a_tile, 16);
    wmma::load_matrix_sync(b_frag, b_tile, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncwarp();
  }

  wmma::store_matrix_sync(c_tile, c_frag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int idx = threadIdx.x; idx < 16 * 16; idx += 32) {
    const int r = idx >> 4;
    const int j = idx & 15;
    const int m = m0 + r;
    const int n = n0 + j;
    if (m < M && n < N) {
      *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(c_tile[idx]);
    }
  }
}

template <int Threads, int OutTile, int Act>
__global__ __launch_bounds__(Threads, 2) void linear_t_act_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = apply_act<Act>(__half2float(*reinterpret_cast<const __half*>(x_row + k)));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile, int Act>
__global__ __launch_bounds__(Threads, 2) void linear_t_act_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    xv.x = apply_act<Act>(xv.x);
    xv.y = apply_act<Act>(xv.y);
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = apply_act<Act>(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + K - 1)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

/* linear_t_vres kernels (L1240-L1377 in source) */

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_vres_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        const int64_t idx = static_cast<int64_t>(m) * N + n;
        const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
        const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
        const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
        *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_vres_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + K - 1));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + K - 1)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        const int64_t idx = static_cast<int64_t>(m) * N + n;
        const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
        const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
        const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
        *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
      }
    }
  }
}

/* ===== Host-side entry functions ===== */

at::Tensor linear_f16_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16 K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16 M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      weight.data_ptr<dtype>(),
      CUDA_R_16F,
      n,
      x.data_ptr<dtype>(),
      CUDA_R_16F,
      k,
      &beta,
      y.data_ptr<dtype>(),
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "linear_f16 cublasGemmEx");
  return y;
}

at::Tensor linear_f16_orig_cuda(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_orig K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_orig M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasGemmEx(
      handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      weight_orig.data_ptr<dtype>(),
      CUDA_R_16F,
      k,
      x.data_ptr<dtype>(),
      CUDA_R_16F,
      k,
      &beta,
      y.data_ptr<dtype>(),
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "linear_f16_orig cublasGemmEx");
  return y;
}

template <int RowTile, int OutTile>
at::Tensor linear_orig_rows_f16_cuda_impl(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_orig_rows_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_orig_rows_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_orig_rows_f16_kernel<128, RowTile, OutTile><<<dim3(ceil_div(N, OutTile), ceil_div(M, RowTile), 1), 128, 0, stream>>>(
      M, K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Threads, int RowTile, int OutTile>
at::Tensor linear_orig_rows_cfg_f16_cuda_impl(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_orig_rows_cfg_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_orig_rows_cfg_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_orig_rows_f16_kernel<Threads, RowTile, OutTile><<<dim3(ceil_div(N, OutTile), ceil_div(M, RowTile), 1), Threads, 0, stream>>>(
      M, K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_orig_rows_f16_cuda(at::Tensor x, at::Tensor weight_orig, int64_t row_tile, int64_t out_tile) {
  if (row_tile == 1 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<1, 2>(x, weight_orig);
  if (row_tile == 1 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<1, 4>(x, weight_orig);
  if (row_tile == 1 && out_tile == 8) return linear_orig_rows_f16_cuda_impl<1, 8>(x, weight_orig);
  if (row_tile == 1 && out_tile == 16) return linear_orig_rows_f16_cuda_impl<1, 16>(x, weight_orig);
  if (row_tile == 2 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<2, 2>(x, weight_orig);
  if (row_tile == 2 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<2, 4>(x, weight_orig);
  if (row_tile == 2 && out_tile == 8) return linear_orig_rows_f16_cuda_impl<2, 8>(x, weight_orig);
  if (row_tile == 3 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<3, 2>(x, weight_orig);
  if (row_tile == 3 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<3, 4>(x, weight_orig);
  if (row_tile == 3 && out_tile == 8) return linear_orig_rows_f16_cuda_impl<3, 8>(x, weight_orig);
  if (row_tile == 4 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<4, 2>(x, weight_orig);
  if (row_tile == 4 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<4, 4>(x, weight_orig);
  if (row_tile == 4 && out_tile == 8) return linear_orig_rows_f16_cuda_impl<4, 8>(x, weight_orig);
  if (row_tile == 8 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<8, 2>(x, weight_orig);
  if (row_tile == 8 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<8, 4>(x, weight_orig);
  if (row_tile == 16 && out_tile == 1) return linear_orig_rows_f16_cuda_impl<16, 1>(x, weight_orig);
  if (row_tile == 16 && out_tile == 2) return linear_orig_rows_f16_cuda_impl<16, 2>(x, weight_orig);
  if (row_tile == 16 && out_tile == 4) return linear_orig_rows_f16_cuda_impl<16, 4>(x, weight_orig);
  TORCH_CHECK(false, "unsupported linear_orig_rows_f16 row_tile/out_tile");
}

at::Tensor linear_orig_rows_cfg_f16_cuda(at::Tensor x, at::Tensor weight_orig, int64_t threads, int64_t row_tile, int64_t out_tile) {
  if (threads == 64 && row_tile == 1 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<64, 1, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 1 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<64, 1, 8>(x, weight_orig);
  if (threads == 128 && row_tile == 1 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<128, 1, 8>(x, weight_orig);
  if (threads == 256 && row_tile == 1 && out_tile == 1) return linear_orig_rows_cfg_f16_cuda_impl<256, 1, 1>(x, weight_orig);
  if (threads == 32 && row_tile == 4 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<32, 4, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 4 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<64, 4, 4>(x, weight_orig);
  if (threads == 96 && row_tile == 4 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<96, 4, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 4 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<32, 4, 8>(x, weight_orig);
  if (threads == 64 && row_tile == 4 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<64, 4, 8>(x, weight_orig);
  if (threads == 32 && row_tile == 8 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<32, 8, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 8 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<64, 8, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 2 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<32, 2, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 2 && out_tile == 2) return linear_orig_rows_cfg_f16_cuda_impl<64, 2, 2>(x, weight_orig);
  if (threads == 64 && row_tile == 2 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<64, 2, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 3 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<32, 3, 4>(x, weight_orig);
  if (threads == 64 && row_tile == 3 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<64, 3, 4>(x, weight_orig);
  if (threads == 96 && row_tile == 3 && out_tile == 4) return linear_orig_rows_cfg_f16_cuda_impl<96, 3, 4>(x, weight_orig);
  if (threads == 32 && row_tile == 3 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<32, 3, 8>(x, weight_orig);
  if (threads == 64 && row_tile == 3 && out_tile == 8) return linear_orig_rows_cfg_f16_cuda_impl<64, 3, 8>(x, weight_orig);
  TORCH_CHECK(false, "unsupported linear_orig_rows_cfg_f16 threads/row_tile/out_tile");
}

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_orig_row1_exact_f16_cuda_impl(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_orig_row1_exact_f16 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_orig_row1_exact_f16 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0, "linear_orig_row1_exact_f16 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 1, "linear_orig_row1_exact_f16 requires one row");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if constexpr (Use4) {
    linear_orig_row1_exact4_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_orig_row1_exact_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Threads, int OutTile, bool Use4>
at::Tensor linear_orig_row2_exact_f16_cuda_impl(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_orig_row2_exact_f16 K/N too large");
  TORCH_CHECK((n64 % OutTile) == 0, "linear_orig_row2_exact_f16 requires N divisible by out_tile");
  TORCH_CHECK((k64 % (Use4 ? 4 : 2)) == 0, "linear_orig_row2_exact_f16 unsupported K alignment");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 == 2, "linear_orig_row2_exact_f16 requires two rows");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if constexpr (Use4) {
    linear_orig_row2_exact4_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_orig_row2_exact_f16_kernel<Threads, OutTile><<<N / OutTile, Threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_orig_rows_exact_f16_cuda(at::Tensor x, at::Tensor weight_orig, int64_t threads, int64_t out_tile, bool use4) {
  const int64_t rows = x.numel() / x.size(-1);
  if (rows == 1) {
    if (!use4 && threads == 128 && out_tile == 2) return linear_orig_row1_exact_f16_cuda_impl<128, 2, false>(x, weight_orig);
    if (use4 && threads == 128 && out_tile == 2) return linear_orig_row1_exact_f16_cuda_impl<128, 2, true>(x, weight_orig);
  }
  if (rows == 2) {
    if (use4 && threads == 64 && out_tile == 2) return linear_orig_row2_exact_f16_cuda_impl<64, 2, true>(x, weight_orig);
    if (use4 && threads == 256 && out_tile == 1) return linear_orig_row2_exact_f16_cuda_impl<256, 1, true>(x, weight_orig);
    if (!use4 && threads == 128 && out_tile == 2) return linear_orig_row2_exact_f16_cuda_impl<128, 2, false>(x, weight_orig);
  }
  TORCH_CHECK(false, "unsupported linear_orig_rows_exact_f16 rows/threads/out_tile/use4");
}

at::Tensor linear_orig_wmma16_f16_cuda(at::Tensor x, at::Tensor weight_orig) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_orig_wmma16_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_orig_wmma16_f16 M too large");
  const int M = static_cast<int>(m64);
  TORCH_CHECK((K % 16) == 0 && (N % 16) == 0, "linear_orig_wmma16_f16 requires K/N multiple of 16");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_orig_wmma16_f16_kernel<<<dim3(N / 16, ceil_div(M, 16), 1), 32, 0, stream>>>(
      M, K, N, x.data_ptr<dtype>(), weight_orig.data_ptr<dtype>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_orig_lt_cfg_cuda(at::Tensor x, at::Tensor weight_orig, int64_t workspace_mb, int64_t algo_index);

at::Tensor linear_f16_orig_lt_cuda(at::Tensor x, at::Tensor weight_orig) {
  return linear_f16_orig_lt_cfg_cuda(x, weight_orig, 0, 0);
}

at::Tensor linear_f16_orig_lt_cfg_cuda(at::Tensor x, at::Tensor weight_orig, int64_t workspace_mb, int64_t algo_index) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_orig.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_orig_lt_cfg K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_orig_lt_cfg M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  const size_t workspace_size = static_cast<size_t>(workspace_mb) << 20;
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)}, x.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }

  static cublasLtHandle_t lt_handle = nullptr;
  if (lt_handle == nullptr) {
    check_cublaslt(cublasLtCreate(&lt_handle), "cublasLtCreate");
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  check_cublaslt(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "linear_f16_orig_lt desc");
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  check_cublaslt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)), "linear_f16_orig_lt transa");
  check_cublaslt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)), "linear_f16_orig_lt transb");
  check_cublaslt(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, k, n, k), "linear_f16_orig_lt a layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, k, m, k), "linear_f16_orig_lt b layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, n, m, n), "linear_f16_orig_lt c layout");
  check_cublaslt(cublasLtMatmulPreferenceCreate(&pref), "linear_f16_orig_lt preference");
  check_cublaslt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)),
                 "linear_f16_orig_lt workspace");

  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(64);
  int returned = 0;
  check_cublaslt(cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, static_cast<int>(heuristics.size()), heuristics.data(), &returned),
                 "linear_f16_orig_lt heuristic");
  TORCH_CHECK(returned > 0, "linear_f16_orig_lt found no algorithm");
  const int selected_algo = algo_index < returned ? static_cast<int>(algo_index) : 0;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublaslt(cublasLtMatmul(
      lt_handle,
      op_desc,
      &alpha,
      weight_orig.data_ptr<dtype>(),
      a_desc,
      x.data_ptr<dtype>(),
      b_desc,
      &beta,
      y.data_ptr<dtype>(),
      c_desc,
      y.data_ptr<dtype>(),
      c_desc,
      &heuristics[selected_algo].algo,
      workspace_ptr,
      workspace_size,
      at::cuda::getCurrentCUDAStream()),
      "linear_f16_orig_lt matmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
  return y;
}

at::Tensor linear_f16_lt_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_lt K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_lt M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  static cublasLtHandle_t lt_handle = nullptr;
  if (lt_handle == nullptr) {
    check_cublaslt(cublasLtCreate(&lt_handle), "cublasLtCreate");
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  check_cublaslt(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
  const cublasOperation_t trans = CUBLAS_OP_N;
  check_cublaslt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)), "cublasLt set transa");
  check_cublaslt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)), "cublasLt set transb");
  check_cublaslt(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, n, k, n), "cublasLt a layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, k, m, k), "cublasLt b layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, n, m, n), "cublasLt c layout");
  check_cublaslt(cublasLtMatmulPreferenceCreate(&pref), "cublasLt preference");
  const size_t workspace_size = 0;
  check_cublaslt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)),
                 "cublasLt set workspace");

  cublasLtMatmulHeuristicResult_t heuristic = {};
  int returned = 0;
  check_cublaslt(cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heuristic, &returned),
                 "cublasLt heuristic");
  TORCH_CHECK(returned > 0, "cublasLt found no algorithm");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublaslt(cublasLtMatmul(
      lt_handle,
      op_desc,
      &alpha,
      weight.data_ptr<dtype>(),
      a_desc,
      x.data_ptr<dtype>(),
      b_desc,
      &beta,
      y.data_ptr<dtype>(),
      c_desc,
      y.data_ptr<dtype>(),
      c_desc,
      &heuristic.algo,
      nullptr,
      0,
      at::cuda::getCurrentCUDAStream()),
      "cublasLtMatmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
  return y;
}

template <int ChunkK, int Warps, bool WarpReduce = false>
at::Tensor linear_f16_m1_splitk_cuda_impl(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_m1_splitk K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  TORCH_CHECK(x.numel() == k64, "linear_f16_m1_splitk requires M=1");
  TORCH_CHECK((N % 64) == 0, "linear_f16_m1_splitk requires N multiple of 64");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (K == 0 || N == 0) {
    return y;
  }
  const int chunks = static_cast<int>(ceil_div(K, ChunkK));
  auto partial = at::empty({chunks, n64}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_f16_m1_splitk_partial_kernel<ChunkK, Warps><<<dim3(ceil_div(N, Warps * 64), chunks, 1), Warps * 32, 0, stream>>>(
      K, N, x.data_ptr<dtype>(), weight.data_ptr<dtype>(), partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if constexpr (WarpReduce) {
    linear_f16_m1_splitk_reduce_warp_kernel<<<static_cast<int>(ceil_div(N / 2, 4)), 128, 0, stream>>>(
        chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  } else {
    linear_f16_m1_splitk_reduce_kernel<<<static_cast<int>(ceil_div(N / 2, 128)), 128, 0, stream>>>(
        chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_m1_splitk_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t K = x.size(-1);
  const int64_t N = weight.size(1);
  if (K == 4096 && N == 4096) {
    return linear_f16_m1_splitk_cuda_impl<160, 1, true>(x, weight);
  }
  if (N >= 65536) {
    return linear_f16_m1_splitk_cuda_impl<768, 2>(x, weight);
  }
  if (K == 4096 && N == 16384) {
    return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
  }
  if (K >= 8192) {
    return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
  }
  return linear_f16_m1_splitk_cuda_impl<256, 4>(x, weight);
}

at::Tensor linear_f16_m1_splitk_cfg_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k) {
  switch (chunk_k) {
    case 64:
      return linear_f16_m1_splitk_cuda_impl<64, 4>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 4>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 4>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 4>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 4>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 4>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 4>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 4>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 4>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 4>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 4>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 4>(x, weight);
    case 224:
      return linear_f16_m1_splitk_cuda_impl<224, 4>(x, weight);
    case 256:
      return linear_f16_m1_splitk_cuda_impl<256, 4>(x, weight);
    case 384:
      return linear_f16_m1_splitk_cuda_impl<384, 4>(x, weight);
    case 512:
      return linear_f16_m1_splitk_cuda_impl<512, 4>(x, weight);
    case 640:
      return linear_f16_m1_splitk_cuda_impl<640, 4>(x, weight);
    case 768:
      return linear_f16_m1_splitk_cuda_impl<768, 4>(x, weight);
    case 896:
      return linear_f16_m1_splitk_cuda_impl<896, 4>(x, weight);
    case 1024:
      return linear_f16_m1_splitk_cuda_impl<1024, 4>(x, weight);
    case 2048:
      return linear_f16_m1_splitk_cuda_impl<2048, 4>(x, weight);
    case 4096:
      return linear_f16_m1_splitk_cuda_impl<4096, 4>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported chunk_k");
  }
}

at::Tensor linear_f16_m1_splitk_tile_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k, int64_t tile_cols) {
  if (tile_cols == 64) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 1>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 1>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 1>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 1>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 1>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 1>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 1>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 1>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 1>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 1>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 1>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 1>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 1>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 1>(x, weight);
      case 384:
        return linear_f16_m1_splitk_cuda_impl<384, 1>(x, weight);
      case 512:
        return linear_f16_m1_splitk_cuda_impl<512, 1>(x, weight);
      case 640:
        return linear_f16_m1_splitk_cuda_impl<640, 1>(x, weight);
      case 768:
        return linear_f16_m1_splitk_cuda_impl<768, 1>(x, weight);
      case 896:
        return linear_f16_m1_splitk_cuda_impl<896, 1>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported chunk_k");
    }
  }
  if (tile_cols == 128) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 2>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 2>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 2>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 2>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 2>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 2>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 2>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 2>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 2>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 2>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 2>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 2>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 2>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 2>(x, weight);
      case 384:
        return linear_f16_m1_splitk_cuda_impl<384, 2>(x, weight);
      case 512:
        return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
      case 640:
        return linear_f16_m1_splitk_cuda_impl<640, 2>(x, weight);
      case 768:
        return linear_f16_m1_splitk_cuda_impl<768, 2>(x, weight);
      case 896:
        return linear_f16_m1_splitk_cuda_impl<896, 2>(x, weight);
      case 1024:
        return linear_f16_m1_splitk_cuda_impl<1024, 2>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported chunk_k");
    }
  }
  TORCH_CHECK(tile_cols == 256, "unsupported tile_cols");
  return linear_f16_m1_splitk_cfg_cuda(x, weight, chunk_k);
}

at::Tensor linear_f16_m1_splitk_warpred_tile_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k, int64_t tile_cols) {
  if (tile_cols == 64) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 1, true>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 1, true>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 1, true>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 1, true>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 1, true>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 1, true>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 1, true>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 1, true>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 1, true>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 1, true>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 1, true>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 1, true>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 1, true>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 1, true>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported warpred chunk_k");
    }
  }
  if (tile_cols == 128) {
    switch (chunk_k) {
      case 64:
        return linear_f16_m1_splitk_cuda_impl<64, 2, true>(x, weight);
      case 96:
        return linear_f16_m1_splitk_cuda_impl<96, 2, true>(x, weight);
      case 112:
        return linear_f16_m1_splitk_cuda_impl<112, 2, true>(x, weight);
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 2, true>(x, weight);
      case 144:
        return linear_f16_m1_splitk_cuda_impl<144, 2, true>(x, weight);
      case 152:
        return linear_f16_m1_splitk_cuda_impl<152, 2, true>(x, weight);
      case 160:
        return linear_f16_m1_splitk_cuda_impl<160, 2, true>(x, weight);
      case 168:
        return linear_f16_m1_splitk_cuda_impl<168, 2, true>(x, weight);
      case 176:
        return linear_f16_m1_splitk_cuda_impl<176, 2, true>(x, weight);
      case 184:
        return linear_f16_m1_splitk_cuda_impl<184, 2, true>(x, weight);
      case 192:
        return linear_f16_m1_splitk_cuda_impl<192, 2, true>(x, weight);
      case 208:
        return linear_f16_m1_splitk_cuda_impl<208, 2, true>(x, weight);
      case 224:
        return linear_f16_m1_splitk_cuda_impl<224, 2, true>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 2, true>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported warpred chunk_k");
    }
  }
  TORCH_CHECK(tile_cols == 256, "unsupported warpred tile_cols");
  switch (chunk_k) {
    case 64:
      return linear_f16_m1_splitk_cuda_impl<64, 4, true>(x, weight);
    case 96:
      return linear_f16_m1_splitk_cuda_impl<96, 4, true>(x, weight);
    case 112:
      return linear_f16_m1_splitk_cuda_impl<112, 4, true>(x, weight);
    case 128:
      return linear_f16_m1_splitk_cuda_impl<128, 4, true>(x, weight);
    case 144:
      return linear_f16_m1_splitk_cuda_impl<144, 4, true>(x, weight);
    case 152:
      return linear_f16_m1_splitk_cuda_impl<152, 4, true>(x, weight);
    case 160:
      return linear_f16_m1_splitk_cuda_impl<160, 4, true>(x, weight);
    case 168:
      return linear_f16_m1_splitk_cuda_impl<168, 4, true>(x, weight);
    case 176:
      return linear_f16_m1_splitk_cuda_impl<176, 4, true>(x, weight);
    case 184:
      return linear_f16_m1_splitk_cuda_impl<184, 4, true>(x, weight);
    case 192:
      return linear_f16_m1_splitk_cuda_impl<192, 4, true>(x, weight);
    case 208:
      return linear_f16_m1_splitk_cuda_impl<208, 4, true>(x, weight);
    case 224:
      return linear_f16_m1_splitk_cuda_impl<224, 4, true>(x, weight);
    case 256:
      return linear_f16_m1_splitk_cuda_impl<256, 4, true>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported warpred chunk_k");
  }
}

template <int ChunkK, int Warps>
at::Tensor linear_f16_rows_splitk_cuda_impl(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_rows_splitk K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_rows_splitk M too large");
  const int M = static_cast<int>(m64);
  TORCH_CHECK((N % 64) == 0, "linear_f16_rows_splitk requires N multiple of 64");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || K == 0 || N == 0) {
    return y;
  }
  const int chunks = static_cast<int>(ceil_div(K, ChunkK));
  auto partial = at::empty({m64, chunks, n64}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_f16_rows_splitk_partial_kernel<ChunkK, Warps><<<dim3(ceil_div(N, Warps * 64), chunks, M), Warps * 32, 0, stream>>>(
      K, N, chunks, x.data_ptr<dtype>(), weight.data_ptr<dtype>(), partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_f16_rows_splitk_reduce_kernel<<<dim3(static_cast<int>(ceil_div(N / 2, 128)), M, 1), 128, 0, stream>>>(
      chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_rows_splitk_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k) {
  switch (chunk_k) {
    case 128:
      return linear_f16_rows_splitk_cuda_impl<128, 2>(x, weight);
    case 256:
      return linear_f16_rows_splitk_cuda_impl<256, 2>(x, weight);
    case 512:
      return linear_f16_rows_splitk_cuda_impl<512, 2>(x, weight);
    case 1024:
      return linear_f16_rows_splitk_cuda_impl<1024, 2>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported chunk_k");
  }
}

at::Tensor linear_t_f16_cuda(at::Tensor x, at::Tensor weight_t) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_t_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (K <= 512 && N >= 1024 && M <= 4) {
    if (M == 1) {
      linear_t_f16_ntile_scalar_kernel<128, 2><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
          M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
    } else {
      linear_t_f16_ntile_kernel<128, 4><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
          M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
    }
  } else if (K >= 1024) {
    linear_t_f16_kernel<256><<<dim3(N, M, 1), 256, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_t_f16_kernel<128><<<dim3(N, M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Act>
at::Tensor linear_t_act_f16_cuda_impl(at::Tensor x, at::Tensor weight_t) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_t_act_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_act_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(K <= 512 && N >= 1024 && M <= 4, "linear_t_act_f16 currently supports only small-rank rank-out");
  if (M == 1) {
    linear_t_act_f16_ntile_scalar_kernel<128, 2, Act><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_t_act_f16_ntile_kernel<128, 4, Act><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_t_act_f16_cuda(at::Tensor x, at::Tensor weight_t, int64_t act) {
  if (act == 1) {
    return linear_t_act_f16_cuda_impl<1>(x, weight_t);
  }
  return linear_t_act_f16_cuda_impl<2>(x, weight_t);
}

at::Tensor linear_t_vres_f16_cuda(at::Tensor x, at::Tensor weight_t, at::Tensor v, at::Tensor v_first, at::Tensor v0) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_t_vres_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_vres_f16 M too large");
  const int M = static_cast<int>(m64);
  auto y = at::empty_like(v);
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(K <= 512 && N >= 1024 && M <= 4, "linear_t_vres_f16 currently supports only small-rank rank-out");
  if (M == 1) {
    linear_t_vres_f16_ntile_scalar_kernel<128, 2><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_t_vres_f16_ntile_kernel<128, 4><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
